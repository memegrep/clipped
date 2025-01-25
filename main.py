import asyncio
import logging
import io
import base64
import requests
import onnxruntime as ort
import numpy as np
import time
import torch
from dataclasses import dataclass
from typing import Optional, List, Union, Literal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoConfig
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler
from PIL import Image

FORMAT = "%(message)s"
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("rich")


@dataclass
class BatchHandler:
    def __init__(self, model_path: str, batch_size: int = 32):
        logger.info("Initializing model")
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

        providers = ["CPUExecutionProvider"]
        self.text_model = ort.InferenceSession(text_path, providers=providers)
        self.vision_model = ort.InferenceSession(vision_path, providers=providers)

        self.batch_size = batch_size
        self._shutdown = False
        self._batch_queue = asyncio.Queue()
        self._result_queue = asyncio.Queue()
        logger.info("Model ready")

    async def process_batches(self):
        while not self._shutdown:
            try:
                batch = await self._batch_queue.get()
                if batch["type"] == "text":
                    results = await self._process_text_batch(batch["inputs"])
                else:
                    results = await self._process_image_batch(batch["inputs"])
                await self._result_queue.put(results)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")

    async def _process_text_batch(self, texts: List[str]):
        batch_size = len(texts)
        logger.info(
            f"[bold blue]Processing text batch[/bold blue] 📝 [green]size={batch_size}[/green]"
        )

        start = time.perf_counter()
        inputs = self.processor(
            text=texts, padding=True, truncation=True, return_tensors="np"
        )
        inputs_dict = {
            "input_ids": inputs["input_ids"].astype(np.int64),
        }
        outputs = self.text_model.run(None, inputs_dict)
        embeddings = outputs[0]
        normalized = torch.nn.functional.normalize(
            torch.from_numpy(embeddings), p=2, dim=1
        ).numpy()

        duration = (time.perf_counter() - start) * 1000
        logger.info(
            f"[bold green]✓[/bold green] Processed {batch_size} texts in {duration:.1f}ms ({batch_size/duration*1000:.1f} texts/sec)"
        )

        return normalized

    async def _process_image_batch(self, images: List[Union[str, bytes]]):
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

        inputs = self.processor(images=processed_images, return_tensors="np")
        outputs = self.vision_model.run(None, {"pixel_values": inputs["pixel_values"]})
        embeddings = outputs[0]  # image embeddings
        normalized = torch.nn.functional.normalize(
            torch.from_numpy(embeddings), p=2, dim=1
        ).numpy()

        duration = (time.perf_counter() - start) * 1000
        logger.info(
            f"[bold green]✓[/bold green] Processed {batch_size} images in {duration:.1f}ms ({batch_size/duration*1000:.1f} images/sec)"
        )

        return normalized


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
handler: Optional[BatchHandler] = None


@app.on_event("startup")
async def startup():
    global handler
    handler = BatchHandler("jinaai/jina-clip-v1", batch_size=32)
    logger.info("[bold green]Starting CLIP ONNX Server[/bold green]")
    asyncio.create_task(handler.process_batches())


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
    texts = [inp.text for inp in normalized_inputs if inp.text is not None]
    images = [inp.image for inp in normalized_inputs if inp.image is not None]

    results = []
    if texts:
        text_embeddings = await handler._process_text_batch(texts)
        results.extend(text_embeddings)
    if images:
        image_embeddings = await handler._process_image_batch(images)
        results.extend(image_embeddings)

    return JSONResponse(
        {
            "data": [
                {"embedding": emb.tolist(), "index": i} for i, emb in enumerate(results)
            ],
            "model": request.model,
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
        }
    )
