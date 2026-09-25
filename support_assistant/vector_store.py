"""ChromaDB: ingestion (chunk -> embed -> store) and retrieval (query -> top-k chunks).

Run `python vector_store.py` to (re)build the index and try a sample query.
"""
import chromadb
from chromadb.config import Settings

import embeddings
from config import CHROMA_DIR, COLLECTION_NAME, DOCS_DIR, TOP_K

DOC_TITLES = {
    "doc_01": "Delivery Policy",
    "doc_02": "Returns & Refunds",
    "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking",
    "doc_05": "Order Cancellation Policy",
    "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards",
    "doc_08": "Customer Support Hours",
}


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR),
                                       settings=Settings(anonymized_telemetry=False))
    # "hnsw:space": "cosine" -> ChromaDB ranks results by cosine distance (= 1 - cosine similarity).
    # embedding_function=None because we always pass our own MiniLM vectors.
    return client.get_or_create_collection(name=COLLECTION_NAME,
                                           metadata={"hnsw:space": "cosine"},
                                           embedding_function=None)


# ---------------------------------------------------------------- ingestion
def load_documents() -> list[dict]:
    docs = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        doc_id = path.stem                                  # e.g. "doc_02"
        docs.append({"doc_id": doc_id, "title": DOC_TITLES.get(doc_id, doc_id),
                     "text": path.read_text(encoding="utf-8").strip()})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """One chunk per document. Each policy is a single short paragraph (~70-90 words),
    well under MiniLM's 256-token input limit, so splitting it further would only
    separate sentences that belong together."""
    return [{"chunk_id": f"{d['doc_id']}_chunk_0", "doc_id": d["doc_id"],
             "title": d["title"], "text": d["text"]} for d in docs]


def build_index() -> int:
    chunks = chunk_documents(load_documents())
    collection = get_collection()
    collection.upsert(                                      # upsert = safe to run many times
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings.embed([c["text"] for c in chunks]),
        metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks],
    )
    return collection.count()


def ensure_index(expected: int = 8) -> int:
    """Build the index only if it is missing or incomplete."""
    count = get_collection().count()
    return count if count >= expected else build_index()


# ---------------------------------------------------------------- retrieval
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed the query and return the k most similar chunks (most similar first)."""
    result = get_collection().query(query_embeddings=embeddings.embed([query]), n_results=k,
                                    include=["documents", "metadatas", "distances"])
    hits = []
    for chunk_id, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                          result["metadatas"][0], result["distances"][0]):
        hits.append({"chunk_id": chunk_id, "doc_id": meta["doc_id"], "title": meta["title"],
                     "text": text, "similarity": round(1.0 - float(dist), 4)})
    return hits


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
    for q in ["How long do refunds take?", "Can I cancel my order after it is packed?",
              "What does Zepto Pass+ cost?"]:
        print(f"\nQ: {q}")
        for h in retrieve(q):
            print(f"   {h['chunk_id']:<16} {h['title']:<28} similarity={h['similarity']:.3f}")
