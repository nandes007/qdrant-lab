"""
Qdrant local quickstart test — follows https://qdrant.tech/documentation/quickstart/
Run:  python qdrant_quickstart.py
Requires: Qdrant running on localhost:6333, `pip install qdrant-client`
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

COLLECTION = "test_collection"

# 1. Connect
client = QdrantClient(url="http://localhost:6333")
print("Connected. Existing collections:", [c.name for c in client.get_collections().collections])

# 2. Create collection (recreate if it exists, so the script is re-runnable)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=4, distance=Distance.DOT),
)
print(f"\n[1] Created collection '{COLLECTION}'")

# 3. Upsert points with payloads
op = client.upsert(
    collection_name=COLLECTION,
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ],
)
print("\n[2] Upsert:", op)  # expect: status=completed

# 4. Plain similarity search
query = [0.2, 0.1, 0.9, 0.7]
hits = client.query_points(
    collection_name=COLLECTION, query=query, with_payload=True, limit=3
).points
print("\n[3] Top 3 nearest (expect ids 4, 1, 3):")
for h in hits:
    print(f"   id={h.id}  score={h.score:.3f}  payload={h.payload}")

# 5. Filtered search
hits = client.query_points(
    collection_name=COLLECTION,
    query=query,
    query_filter=Filter(must=[FieldCondition(key="city", match=MatchValue(value="London"))]),
    with_payload=True,
    limit=3,
).points
print("\n[4] Filtered to city=London (expect id 2, score ~0.871):")
for h in hits:
    print(f"   id={h.id}  score={h.score:.3f}  payload={h.payload}")

# 6. Extra: confirm point count
print("\n[5] Point count:", client.count(COLLECTION).count)  # expect 6