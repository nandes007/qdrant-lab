from typing import List
import numpy as np
from fastembed import TextEmbedding

documents: List[str] = [
    "Hello, World!",
    "fastembed is supported by and maintained by Qdrant."
]
embedding_model = TextEmbedding()
embeddings: List[np.ndarray] = list(embedding_model.embed(documents))

from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

docs = [
    "Qdrant has Langchain integrations",
    "Qdrant also has Llama Index integrations"
]
metadata = [
    {"source": "Langchain-docs"},
    {"source": "LlamaIndex-docs"},
]
ids = [42, 2]

model_name = "BAAI/bge-small-en-v1.5"

client.create_collection(
    collection_name="demo_collection",
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name),
        distance=models.Distance.COSINE,
    ),
)

metadata_with_docs = [
    {"document": doc, **meta} for doc, meta in zip(docs, metadata)
]

client.upload_collection(
    collection_name="demo_collection",
    vectors=[models.Document(text=doc, model=model_name) for doc in docs],
    payload=metadata_with_docs,
    ids=ids,
)

search_result = client.query_points(
    collection_name="demo_collection",
    query=models.Document(text="This is a query document", model=model_name),
).points
print(search_result)