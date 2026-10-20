import base64
import os

import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
COLLECTION_NAME = "multimodal-embeddings"

documents = [
    {"caption": "An image about plane emergency safety.", "image": "images/image-1.png"},
    {"caption": "An image about airplane components.", "image": "images/image-2.png"},
    {"caption": "An image about COVID safety restrictions.", "image": "images/image-3.png"},
    {"caption": "A confidential image about UFO sightings.", "image": "images/image-4.png"},
    {"caption": "An image about unusual footprints on Aralar 2011.", "image": "images/image-5.png"},
]


def image_input(path):
    data = base64.b64encode(open(path, "rb").read()).decode()
    return {"content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]}


def embed(inputs):
    # inputs: list of strings (text) and/or image_input(...) dicts
    # ponytail: 3 tries because the :free provider often 504s; switch to a paid model if it keeps failing
    for attempt in range(3):
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": EMBED_MODEL, "input": inputs, "encoding_format": "float"},
            timeout=120,
        )
        if resp.status_code < 500:
            break
        print(f"OpenRouter {resp.status_code}, retrying ({attempt + 1}/3)...")
    resp.raise_for_status()
    return [d["embedding"] for d in sorted(resp.json()["data"], key=lambda d: d["index"])]


text_vecs = embed([doc["caption"] for doc in documents])
image_vecs = embed([image_input(doc["image"]) for doc in documents])
dim = len(text_vecs[0])  # model decides the size; read it instead of hardcoding

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "image": models.VectorParams(size=dim, distance=models.Distance.COSINE),
        "text": models.VectorParams(size=dim, distance=models.Distance.COSINE),
    },
)
client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(id=idx, vector={"text": t, "image": i}, payload=doc)
        for idx, (doc, t, i) in enumerate(zip(documents, text_vecs, image_vecs))
    ],
)

# text -> image
hit = client.query_points(COLLECTION_NAME, query=embed(["Plane components"])[0], using="image", limit=1).points[0]
print("text->image:", hit.payload["image"], hit.score)

# image -> text
hit = client.query_points(COLLECTION_NAME, query=embed([image_input("images/image-2.png")])[0], using="text", limit=1).points[0]
print("image->text:", hit.payload["caption"], hit.score)
