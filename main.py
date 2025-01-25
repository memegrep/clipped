import asyncio
import logging
import io
import base64
import os
import requests
import onnxruntime as ort
import numpy as np
import time
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoConfig
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.logging import RichHandler
from PIL import Image
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter("clip_requests_total", "Total requests processed", ["type"])
BATCH_SIZE = Histogram("clip_batch_size", "Batch sizes processed", ["type"])
PROCESSING_TIME = Histogram(
    "clip_processing_seconds", "Time spent processing", ["type"]
)
QUEUE_SIZE = Gauge("clip_queue_size", "Current size of request queue", ["type"])
ERROR_COUNT = Counter("clip_errors_total", "Total errors encountered", ["type"])

FORMAT = "%(message)s"
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("rich")


class ServerConfig(BaseSettings):
    """Server configuration with environment variables"""

    MAX_QUEUE_SIZE: int = 1000
    BATCH_TIMEOUT: float = 0.01
    BATCH_SIZE: int = 32
    MAX_RETRIES: int = 3
    HEALTH_CHECK_INTERVAL: int = 30
    MODEL_NAME: str = "jinaai/jina-clip-v1"
    API_KEY: Optional[str] = None


config = ServerConfig()


@dataclass
class Task:
    type: Literal["text", "image"]
    inputs: List[Union[str, bytes]]
    future: asyncio.Future
    created_at: float = field(default_factory=time.monotonic)


class BatchProcessingError(Exception):
    pass


@dataclass
class BatchHandler:
    def __init__(
        self,
        model_path: str,
        batch_size: int = config.BATCH_SIZE,
        max_queue_size: int = config.MAX_QUEUE_SIZE,
        batch_timeout: float = config.BATCH_TIMEOUT,
    ):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.batch_timeout = batch_timeout
        self._shutdown = False
        self._batch_queues: Dict[str, asyncio.Queue] = {
            "text": asyncio.Queue(maxsize=max_queue_size),
            "image": asyncio.Queue(maxsize=max_queue_size),
        }
        try:
            logger.info("Initializing model...")
            self._init_model(model_path)
            logger.info("[bold green]Model initialized successfully[/bold green]")
        except Exception as e:
            logger.error(f"[bold red]Model initialization failed: {e}[/bold red]")
            raise

        # Health check task
        self._healthy = True
        self._last_successful_inference = time.monotonic()
        logger.info("Model ready")

    def _init_model(self, model_path: str):
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        from huggingface_hub import hf_hub_download

        text_path = hf_hub_download(
            model_path, "text_model_quantized.onnx", subfolder="onnx"
        )
        vision_path = hf_hub_download(
            model_path, "vision_model_quantized.onnx", subfolder="onnx"
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # session_options.intra_op_num_threads = os.cpu_count()
        providers = [
            ("CPUExecutionProvider", {"arena_extend_strategy": "kNextPowerOfTwo"})
        ]
        self.text_model = ort.InferenceSession(
            text_path, providers=providers, sess_options=session_options
        )
        self.vision_model = ort.InferenceSession(
            vision_path, providers=providers, sess_options=session_options
        )

    async def health_check(self):
        """Periodic health check"""
        while not self._shutdown:
            try:
                await self._process_text_batch(["test query"])
                self._healthy = True
                self._last_successful_inference = time.monotonic()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self._healthy = False
            await asyncio.sleep(config.HEALTH_CHECK_INTERVAL)

    async def process_batches(self):
        """Process batches for both text and image queues concurrently"""
        tasks = [
            asyncio.create_task(self._process_queue("text")),
            asyncio.create_task(self._process_queue("image")),
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for task in tasks:
                task.cancel()
            raise

    async def _process_queue(self, queue_type: Literal["text", "image"]):
        """Process a single modality queue"""
        while not self._shutdown:
            try:
                batch_tasks = []
                batch_inputs = []

                first_task = await self._batch_queues[queue_type].get()
                batch_tasks.append(first_task)
                batch_inputs.extend(first_task.inputs)
                batch_start = time.monotonic()

                while (
                    len(batch_inputs) < self.batch_size
                    and time.monotonic() - batch_start < self.batch_timeout
                ):
                    try:
                        task = await asyncio.wait_for(
                            self._batch_queues[queue_type].get(),
                            timeout=self.batch_timeout,
                        )
                        if len(batch_inputs) + len(task.inputs) <= self.batch_size:
                            batch_tasks.append(task)
                            batch_inputs.extend(task.inputs)
                        else:
                            await self._batch_queues[queue_type].put(task)
                            break
                    except asyncio.TimeoutError:
                        break

                BATCH_SIZE.labels(type=queue_type).observe(len(batch_inputs))
                try:
                    with PROCESSING_TIME.labels(type=queue_type).time():
                        if queue_type == "text":
                            embeddings = await self._process_text_batch(batch_inputs)
                        else:
                            embeddings = await self._process_image_batch(batch_inputs)

                    if embeddings is None:
                        raise BatchProcessingError("Failed to process batch")

                    start_idx = 0
                    for task in batch_tasks:
                        end_idx = start_idx + len(task.inputs)
                        task.future.set_result(embeddings[start_idx:end_idx])
                        start_idx = end_idx
                        REQUEST_COUNT.labels(type=queue_type).inc(len(task.inputs))

                except Exception as e:
                    ERROR_COUNT.labels(type=queue_type).inc()
                    logger.error(f"Error processing {queue_type} batch: {e}")
                    for task in batch_tasks:
                        if not task.future.done():
                            task.future.set_exception(e)

            except Exception as e:
                logger.error(f"Queue processing error for {queue_type}: {e}")
                await asyncio.sleep(1)

    async def _process_text_batch(self, texts: List[str]) -> np.ndarray | None:
        """Process a batch of texts with retries and logging"""
        batch_size = len(texts)
        logger.info(
            f"[bold blue]Processing text batch[/bold blue] 📝 [green]size={batch_size}[/green]"
        )

        start = time.perf_counter()

        for attempt in range(config.MAX_RETRIES):
            try:
                inputs = self.processor(
                    text=texts, padding=True, truncation=True, return_tensors="np"
                )
                inputs_dict = {"input_ids": inputs["input_ids"].astype(np.int64)}
                outputs = self.text_model.run(None, inputs_dict)
                embeddings = outputs[0]
                normalized = torch.nn.functional.normalize(
                    torch.from_numpy(embeddings), p=2, dim=1
                ).numpy()

                duration = (time.perf_counter() - start) * 1000
                logger.info(
                    f"[bold green]✓[/bold green] Processed {batch_size} texts in {duration:.1f}ms "
                    f"({batch_size/duration*1000:.1f} texts/sec)"
                )
                return normalized

            except Exception as e:
                if attempt == config.MAX_RETRIES - 1:
                    raise BatchProcessingError(
                        f"Text batch processing failed after {config.MAX_RETRIES} attempts: {e}"
                    )
                logger.warning(f"Text batch attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(0.1 * (attempt + 1))

    async def _process_image_batch(
        self, images: List[Union[str, bytes]]
    ) -> np.ndarray | None:
        batch_size = len(images)
        logger.info(
            f"[bold blue]Processing image batch[/bold blue] 📝 [green]size={batch_size}[/green]"
        )

        start = time.perf_counter()

        processed_images = []
        for img in images:
            if isinstance(img, str):
                if img.startswith("data:"):
                    img_data = base64.b64decode(img.split(",")[1])
                    image = Image.open(io.BytesIO(img_data))
                else:
                    response = requests.get(img)
                    image = Image.open(io.BytesIO(response.content))
            else:
                raise ValueError("Invalid image type")
            processed_images.append(image)

        for attempt in range(config.MAX_RETRIES):
            try:
                inputs = self.processor(images=processed_images, return_tensors="np")
                outputs = self.vision_model.run(
                    None, {"pixel_values": inputs["pixel_values"]}
                )
                embeddings = outputs[0]  # image embeddings
                normalized = torch.nn.functional.normalize(
                    torch.from_numpy(embeddings), p=2, dim=1
                ).numpy()

                duration = (time.perf_counter() - start) * 1000
                logger.info(
                    f"[bold green]✓[/bold green] Processed {batch_size} images in {duration:.1f}ms ({batch_size/duration*1000:.1f} images/sec)"
                )

                return normalized
            except Exception as e:
                if attempt == config.MAX_RETRIES - 1:
                    raise BatchProcessingError(
                        f"Image batch processing failed after {config.MAX_RETRIES} attempts: {e}"
                    )
                logger.warning(f"Image batch attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(0.1 * (attempt + 1))

    async def add_task(self, task: Task) -> Any:
        """Add a task to the appropriate queue with backpressure"""
        queue = self._batch_queues[task.type]

        if queue.qsize() >= self.max_queue_size:
            raise HTTPException(
                status_code=503, detail=f"Server overloaded. {task.type} queue is full."
            )

        if not self._healthy:
            raise HTTPException(
                status_code=503, detail="Server is unhealthy. Please try again later."
            )

        QUEUE_SIZE.labels(type=task.type).set(queue.qsize())
        await queue.put(task)
        return await task.future


class ModalityInput(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[ModalityInput]]
    model: str = "clip"
    dimensions: Optional[int] = None
    normalized: bool = True
    embedding_type: Literal["float", "base64"] = "float"


class EmbeddingResponse(BaseModel):
    data: List[dict]
    model: str
    usage: dict


app = FastAPI()


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if not config.API_KEY:
        return await call_next(request)

    if request.url.path == "/health":
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return await call_next(request)


handler: Optional[BatchHandler] = None


@app.on_event("startup")
async def startup():
    global handler
    handler = BatchHandler("jinaai/jina-clip-v1", batch_size=32)
    logger.info("[bold green]Starting CLIP ONNX Server[/bold green]")
    asyncio.create_task(handler.process_batches())
    asyncio.create_task(handler.health_check())


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not handler or not handler._healthy:
        raise HTTPException(status_code=503, detail="Server unhealthy")
    return {
        "status": "healthy",
        "last_successful_inference": handler._last_successful_inference,
        "queue_sizes": {
            "text": handler._batch_queues["text"].qsize(),
            "image": handler._batch_queues["image"].qsize(),
        },
    }


@app.post("/embeddings")
async def create_embedding(request: EmbeddingRequest):
    if not handler:
        return JSONResponse({"error": "Model not ready"}, 500)

    inputs = request.input if isinstance(request.input, list) else [request.input]

    normalized_inputs = []
    for inp in inputs:
        if isinstance(inp, str):
            normalized_inputs.append(ModalityInput(text=inp))
        else:
            normalized_inputs.append(inp)

    # separate text and images
    texts, text_indices = [], []
    images, image_indices = [], []
    for idx, item in enumerate(normalized_inputs):
        if item.text:
            texts.append(item.text)
            text_indices.append(idx)
        elif item.image:
            images.append(item.image)
            image_indices.append(idx)

    loop = asyncio.get_running_loop()
    text_task = image_task = None

    text_future = None
    image_future = None

    if texts:
        text_future = loop.create_future()
        text_task = Task("text", texts, text_future)
        await handler._batch_queues["text"].put(text_task)

    if images:
        image_future = loop.create_future()
        image_task = Task("image", images, image_future)
        await handler._batch_queues["image"].put(image_task)

    # Await results
    text_embeds = await text_future if text_future else []
    image_embeds = await image_future if image_future else []

    # Reconstruct original order
    embeddings = [None] * len(normalized_inputs)
    for i, idx in enumerate(text_indices):
        embeddings[idx] = text_embeds[i].tolist()
    for i, idx in enumerate(image_indices):
        embeddings[idx] = image_embeds[i].tolist()

    return JSONResponse(
        {
            "data": [{"embedding": e, "index": i} for i, e in enumerate(embeddings)],
            "model": request.model,
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
        }
    )
