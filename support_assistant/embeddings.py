"""Local embeddings with sentence-transformers (all-MiniLM-L6-v2). No API key, no cost.

The model (~90 MB) is downloaded once from Hugging Face on first use and cached.
"""
from functools import lru_cache

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model():
    # Imported here so that simply importing this file stays fast.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed(texts: list[str]) -> list[list[float]]:
    """Turn texts into 384-number vectors. normalize_embeddings=True makes every
    vector length 1, which is what cosine similarity expects."""
    vectors = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()
