"""
Local version of https://qdrant.tech/documentation/tutorials-basics/cloud-inference-hybrid-search/
- dense: OpenRouter embeddings (instead of Cloud Inference)
- sparse: BM25 via fastembed, computed locally (instead of the server-side BM25 text index)
- fusion: RRF, same as the tutorial
"""
import os

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-minilm-l6-v2")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
COLLECTION_NAME = "hybrid-search"

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
bm25 = SparseTextEmbedding("Qdrant/bm25")

# ponytail: one parquet shard (~33MB) out of 49, cached in ~/.cache/huggingface after the first run.
# The tutorial uses streaming=True instead, which re-downloads every run and was slow/flaky here.
ds = (
    load_dataset("miriad/miriad-4.4M", data_files="data/train-00000-of-00049.parquet", split="train")
    # each passage has several QA pairs (qa_id "<id>_1", "<id>_2", ...); keep one so passages are distinct
    .filter(lambda row: row["qa_id"].endswith("_1"))
    .shuffle(seed=42)
    .select(range(100))
)
rows: list[dict] = ds.to_list()
passages: list[str] = list(ds["passage_text"])


def embed(texts):
    # ponytail: 3 tries because the :free provider often 504s; switch to a paid model if it keeps failing
    for attempt in range(3):
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": EMBED_MODEL, "input": texts, "encoding_format": "float"},
            timeout=120,
        )
        if resp.status_code < 500:
            break
        print(f"OpenRouter {resp.status_code}, retrying ({attempt + 1}/3)...")
    resp.raise_for_status()
    return [d["embedding"] for d in sorted(resp.json()["data"], key=lambda d: d["index"])]


def sparse(text_vec):
    return models.SparseVector(indices=text_vec.indices.tolist(), values=text_vec.values.tolist())


dense_vecs = embed(passages)
sparse_vecs = list(bm25.embed(passages))

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense_vector": models.VectorParams(size=len(dense_vecs[0]), distance=models.Distance.COSINE),
    },
    # IDF modifier: Qdrant computes IDF server-side, fastembed's bm25 only gives term frequencies
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(id=i, vector={"dense_vector": d, "bm25": sparse(s)}, payload=row)
        for i, (row, d, s) in enumerate(zip(rows, dense_vecs, sparse_vecs))
    ],
)

query_text = "Why does stoma closure fail?"
results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(query=embed([query_text])[0], using="dense_vector", limit=5),
        models.Prefetch(query=sparse(list(bm25.query_embed(query_text))[0]), using="bm25", limit=5),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=5,
    with_payload=True,
)
for p in results.points:
    print(f"{p.score:.4f}  {p.payload['passage_text'][:150]}...")
