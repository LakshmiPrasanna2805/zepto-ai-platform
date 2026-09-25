# Module 3 — Support Assistant (`/support_assistant`)

This is a small retrieval-augmented (RAG) support service for Zepto's policies. It embeds 8 policy documents locally, routes each question through a LangGraph flow, and returns a Pydantic-validated JSON answer from a FastAPI endpoint.

**Graded mode = offline mock.** With `MOCK_LLM` unset (or `MOCK_LLM=1`), no LLM is called and no API key is needed. Retrieval (embeddings + ChromaDB) still runs for real.

| File | Role |
|---|---|
| `docs/doc_01.txt … doc_08.txt` | The 8 policy documents, copied exactly from the brief |
| `config.py` | Settings, including the `MOCK_LLM` switch (`mock_llm_enabled()`) |
| `embeddings.py` | `embed()`: local `all-MiniLM-L6-v2` embeddings via sentence-transformers |
| `vector_store.py` | Ingestion (`load_documents` → `chunk_documents` → `build_index`) and retrieval (`retrieve`) with ChromaDB |
| `prompts.py` | Structured prompt templates (role–context–task–format–length, negative constraint, few-shot example) |
| `schemas.py` | Pydantic `AskRequest` / `AskResponse` |
| `graph.py` | LangGraph `StateGraph`: `classify_intent` → `retrieve_and_answer` or `direct_answer` |
| `llm.py` | Optional real-LLM calls (Groq) and the retry-on-invalid-JSON logic, used only when `MOCK_LLM=0` |
| `main.py` | FastAPI app with `POST /ask` and `GET /health` |
| `run_examples.py` | Calls the running API and records the raw JSON responses |
| `Dockerfile` | Container that serves the API on port 7860 |

## How to run (local)

```bash
cd support_assistant
pip install -r requirements.txt
python vector_store.py                  # builds the ChromaDB index (downloads the ~90 MB model the first time)
uvicorn main:app --port 8000            # MOCK_LLM left unset = mock mode
```
In a second terminal:
```bash
cd support_assistant
python run_examples.py                  # sends example queries, fills in the section below
```
Interactive docs: http://localhost:8000/docs

## How to run (Docker)

```bash
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
# then, from another terminal:
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d "{\"query\": \"How long do refunds take?\"}"
python run_examples.py http://localhost:7860
```
The model download and index build happen during `docker build`, so the running container needs no internet in mock mode. PyTorch is installed as the CPU-only build to keep the image small.

## Example calls (`MOCK_LLM` at its default)

<!-- EXAMPLES:START -->
Recorded against `http://localhost:7860` with `MOCK_LLM` left at its default (server reports `mock_llm = True`, `indexed_chunks = 8`).


### policy question -> should route to retrieve_and_answer

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "How long does a refund take after I return an item?"}'
```
Response (HTTP 200):
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in...",
  "sources": [
    "doc_02_chunk_0",
    "doc_06_chunk_0",
    "doc_05_chunk_0"
  ],
  "confidence": 0.619
}
```

### policy question -> should route to retrieve_and_answer

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "Can I cancel my order after it has been packed?"}'
```
Response (HTTP 200):
```json
{
  "answer": "Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer...",
  "sources": [
    "doc_05_chunk_0",
    "doc_02_chunk_0",
    "doc_06_chunk_0"
  ],
  "confidence": 0.677
}
```

### general question -> should route to direct_answer

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "What is the capital of France?"}'
```
Response (HTTP 200):
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
<!-- EXAMPLES:END -->

## Architecture: ingestion → embedding → retrieval → generation

```
 docs/*.txt
     │  (1) INGESTION   vector_store.load_documents() + chunk_documents()   -> 8 chunks (1 per doc)
     ▼
 embeddings.embed()     (2) EMBEDDING   all-MiniLM-L6-v2, 384-dim, normalized
     ▼
 ChromaDB  collection "zepto_policies"  (persistent, ./chroma_db, cosine space)
     ▲
     │  (3) RETRIEVAL   vector_store.retrieve(query) -> top-3 chunks        [always real]
     │
 POST /ask ─► LangGraph:  classify_intent ──policy_question──► retrieve_and_answer ─► AskResponse
                                          └─general_question─► direct_answer ───────► AskResponse
                          (4) GENERATION happens inside these nodes           [branches on MOCK_LLM]
```

1. **Ingestion.** `vector_store.load_documents()` reads `docs/doc_*.txt` and attaches each document's title. `chunk_documents()` makes **one chunk per document** with ids like `doc_02_chunk_0`. Each policy is one short paragraph (~70–90 words), well under MiniLM's 256-token limit, so splitting further would only separate sentences that belong together.
2. **Embedding.** `embeddings.embed()` turns each chunk into a 384-number vector with `sentence-transformers/all-MiniLM-L6-v2`, running locally on the CPU. `vector_store.build_index()` upserts the ids, texts, vectors and metadata (`doc_id`, `title`) into the persistent ChromaDB collection **`zepto_policies`**, configured with `hnsw:space = cosine`. The FastAPI startup hook (`ensure_index`) rebuilds the index if it is missing.
3. **Retrieval.** In the LangGraph node **`retrieve_and_answer`**, `vector_store.retrieve()` embeds the query with the same model and asks ChromaDB for the **top-3** chunks by cosine similarity (similarity = 1 − cosine distance). This step runs for real in both modes.
4. **Generation.** Each node produces the final `AskResponse` (`answer`, `sources`, `confidence`), validated by Pydantic:
   - **`classify_intent`.** Mock: keyword heuristic, so the lowercased query containing *delivery, return, refund, membership, tracking, cancel, gift card* or *support hours* → `policy_question`, otherwise `general_question`. Real: the LLM classifies with `CLASSIFY_PROMPT_TEMPLATE`.
   - **Conditional edge.** `route_by_intent` sends `policy_question` → `retrieve_and_answer` and `general_question` → `direct_answer`. Routing does not depend on `MOCK_LLM`.
   - **`retrieve_and_answer`.** Mock: `"Based on the retrieved context: <first ~200 chars of the top chunk>"`, with `sources` = the 3 retrieved chunk ids and `confidence` = the top chunk's cosine similarity (clipped to 0–1, deterministic). Real: `ANSWER_PROMPT_TEMPLATE` is filled with the 3 chunks and sent to the LLM.
   - **`direct_answer`.** Mock: the fixed string *“I can only answer questions about Zepto policies right now.”*, `sources = []`, `confidence = 1.0`. Real: `DIRECT_PROMPT_TEMPLATE` is sent to the LLM with no retrieval.

**What `MOCK_LLM` changes.** Only the generation step in each of the 3 nodes (step 4). Ingestion, embedding, retrieval and routing are identical in both modes.
- **Default (mock):** deterministic Python rules, with zero network calls to any LLM provider.
- **`MOCK_LLM=0`:** `llm.generate_structured()` calls Groq's free OpenAI-compatible API (key read from the `GROQ_API_KEY` environment variable, never committed). It parses the reply as JSON and validates it against `AskResponse`. If validation fails, it retries **up to 2 more times** with a corrective instruction, then returns a clearly marked `[ERROR] …` response with `confidence = 0`.

## Structured prompt (used only when `MOCK_LLM=0`)
Every template in `prompts.py` follows **ROLE → CONTEXT → TASK → FORMAT → LENGTH**. Each includes explicit negative constraints (e.g. *“Do NOT answer using information that is not present in the provided context”*, *“Do NOT invent prices, time limits or policies”*) and a few-shot **EXAMPLE** of the expected JSON reply.

## Optional real-LLM mode (ungraded)
```bash
# Windows PowerShell:  $env:MOCK_LLM="0"; $env:GROQ_API_KEY="<your free Groq key>"
export MOCK_LLM=0 GROQ_API_KEY=<your free Groq key>
uvicorn main:app --port 8000
```
This uses the free tier at console.groq.com (no credit card). To pick a different model, set `LLM_MODEL` to any model listed in your Groq console.
