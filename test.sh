#!/bin/bash

URL="http://localhost:8000/embeddings"
TEXT_PAYLOAD='{"input": ["spiderman lookin sus", "luigi doing pushups", "warren buffet doin the griddy"], "model": "clip"}'
IMAGE_PAYLOAD='{"input": [{"image": "https://picsum.photos/200"}, {"image": "https://picsum.photos/300"}, {"image": "https://picsum.photos/400"}], "model": "clip"}'
MIXED_PAYLOAD='{"input": [{"text": "the meme lord supreme"}, {"image": "https://picsum.photos/200"}, {"text": "elon musk smoking a blunt"}], "model": "clip"}'

echo $TEXT_PAYLOAD > text.json
echo $IMAGE_PAYLOAD > image.json
echo $MIED_PAYLOAD > mixed.json

echo "Testing text embeddings..."
hey -n 1000 -c 50 -m POST -H "Content-Type: application/json" -D text.json $URL

echo "Testing image embeddings..."
hey -n 1000 -c 50 -m POST -H "Content-Type: application/json" -D image.json $URL

echo "Testing mixed embeddings..."
hey -n 1000 -c 50 -m POST -H "Content-Type: application/json" -D mixed.json $URL

rm text.json image.json mixed.json
