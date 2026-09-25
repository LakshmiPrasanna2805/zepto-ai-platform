"""FastAPI wrapper.   Run:  uvicorn main:app --port 8000
Then open http://localhost:8000/docs to try POST /ask in the browser."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import vector_store
from config import COLLECTION_NAME, mock_llm_enabled
from graph import answer_query, build_graph
from schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: make sure the ChromaDB index exists and load the embedding model once.
    n = vector_store.ensure_index()
    build_graph()
    print(f"[startup] {n} chunks in '{COLLECTION_NAME}', MOCK_LLM={'on' if mock_llm_enabled() else 'off'}")
    yield


app = FastAPI(title="Zepto Support Assistant", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": mock_llm_enabled(),
            "indexed_chunks": vector_store.get_collection().count()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        state = answer_query(request.query)
    except Exception as err:          # e.g. MOCK_LLM=0 without an API key
        raise HTTPException(status_code=500, detail=str(err))
    return state["response"]
