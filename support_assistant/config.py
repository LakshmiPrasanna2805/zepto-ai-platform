"""Central settings for the support assistant (read from environment variables)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = "zepto_policies"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
SNIPPET_CHARS = 200


def mock_llm_enabled() -> bool:
    """MOCK_LLM unset or "1" -> offline mock mode (the graded default).
    Only an explicit MOCK_LLM=0 switches on real LLM calls."""
    return os.getenv("MOCK_LLM", "1").strip() != "0"


# ---- only used when MOCK_LLM=0 (optional extension) ----
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")   # any model listed in your Groq console
LLM_API_KEY_ENV = "GROQ_API_KEY"                              # the key is read from this env var, never hard-coded
MAX_SCHEMA_RETRIES = 2                                        # retries AFTER the first attempt
