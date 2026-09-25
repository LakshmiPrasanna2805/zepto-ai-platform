"""Send example questions to the RUNNING API and save the raw JSON responses.

1) In one terminal:   uvicorn main:app --port 8000          (MOCK_LLM left unset = mock mode)
2) In another:        python run_examples.py
   -> prints each call and writes examples/example_calls.md
"""
import json
import sys
from pathlib import Path

import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
EXAMPLES = [
    ("policy question -> should route to retrieve_and_answer", "How long does a refund take after I return an item?"),
    ("policy question -> should route to retrieve_and_answer", "Can I cancel my order after it has been packed?"),
    ("general question -> should route to direct_answer", "What is the capital of France?"),
]

health = requests.get(f"{BASE_URL}/health", timeout=10).json()
lines = ["# Example calls to `POST /ask`\n",
         f"Recorded against `{BASE_URL}` with `MOCK_LLM` left at its default "
         f"(server reports `mock_llm = {health['mock_llm']}`, `indexed_chunks = {health['indexed_chunks']}`).\n"]

for label, query in EXAMPLES:
    body = {"query": query}
    resp = requests.post(f"{BASE_URL}/ask", json=body, timeout=60)
    print(f"\n{label}\nPOST /ask {json.dumps(body)}\n-> HTTP {resp.status_code}\n{json.dumps(resp.json(), indent=2)}")
    curl = f"curl -X POST {BASE_URL}/ask -H \"Content-Type: application/json\" -d '{json.dumps(body)}'"
    lines += [f"\n### {label}\n", "```bash", curl, "```",
              f"Response (HTTP {resp.status_code}):", "```json", json.dumps(resp.json(), indent=2), "```"]

out = Path("examples") / "example_calls.md"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nSaved {out}")

# Also paste the transcripts into README.md between the two marker comments
readme = Path("README.md")
start, end = "<!-- EXAMPLES:START -->", "<!-- EXAMPLES:END -->"
text = readme.read_text(encoding="utf-8")
if start in text and end in text:
    block = "\n".join(lines[1:])          # skip the file's own H1 title
    text = text.split(start)[0] + start + "\n" + block + "\n" + end + text.split(end)[1]
    readme.write_text(text, encoding="utf-8")
    print(f"Updated {readme}")
