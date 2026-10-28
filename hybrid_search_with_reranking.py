"""
Local version of https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/
- dense: OpenRouter embeddings (instead of Cloud Inference)
- full-text: BM25 via fastembed, computed locally (instead of the server-side BM25 text index)
- rerank: ColBERT (answerai-colbert-small-v1) via fastembed, computed locally;
  OpenRouter has no late-interaction/multivector models
"""
import csv
import os
import urllib.request

import httpx
from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-minilm-l6-v2")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
COLLECTION_NAME = "hybrid-search-rerank"
CSV_URL = "https://raw.githubusercontent.com/qdrant/examples/refs/heads/master/sci-fi-books/top_100_scifi_books_full.csv"

# location accepts a URL or ":memory:" (handy for trying it without a server)
client = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=os.getenv("QDRANT_API_KEY"))
bm25 = SparseTextEmbedding("Qdrant/bm25")
colbert = LateInteractionTextEmbedding("answerdotai/answerai-colbert-small-v1")


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


with urllib.request.urlopen(CSV_URL) as response:
    rows = list(csv.DictReader(line.decode("utf-8") for line in response))
descriptions = [row["Description"] for row in rows]

dense_vecs = embed(descriptions)
sparse_vecs = list(bm25.embed(descriptions))
multi_vecs = list(colbert.embed(descriptions))

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=len(dense_vecs[0]), distance=models.Distance.COSINE),
        "multi": models.VectorParams(
            size=multi_vecs[0].shape[1],  # 96 for answerai-colbert-small-v1
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
            hnsw_config=models.HnswConfigDiff(m=0),  # only used for reranking, no index needed
        ),
    },
    # IDF modifier: Qdrant computes IDF server-side, fastembed's bm25 only gives term frequencies
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
client.upload_points(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=i,
            vector={"dense": d, "bm25": sparse(s), "multi": m.tolist()},
            payload={"title": row["Title"], "author": row["Author"], "description": row["Description"]},
        )
        for i, (row, d, s, m) in enumerate(zip(rows, dense_vecs, sparse_vecs, multi_vecs))
    ],
    batch_size=25,
)

query = "time travel"
dense_q = embed([query])[0]
sparse_q = sparse(list(bm25.query_embed(query))[0])
multi_q = list(colbert.query_embed(query))[0].tolist()
prefetch = [
    models.Prefetch(query=dense_q, using="dense", limit=20),
    models.Prefetch(query=sparse_q, using="bm25", limit=20),
]


def show(title, points):
    print(f"\n== {title} ==")
    for p in points:
        print(f"{p.score:.4f}  {p.payload['title']} — {p.payload['author']}")


show("dense", client.query_points(COLLECTION_NAME, query=dense_q, using="dense", limit=10).points)
show("bm25", client.query_points(COLLECTION_NAME, query=sparse_q, using="bm25", limit=10).points)
show(
    "hybrid (RRF)",
    client.query_points(
        COLLECTION_NAME, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=10
    ).points,
)
show(
    "hybrid + ColBERT rerank",
    client.query_points(COLLECTION_NAME, prefetch=prefetch, query=multi_q, using="multi", limit=10).points,
)
