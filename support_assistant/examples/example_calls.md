# Example calls to `POST /ask`

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
