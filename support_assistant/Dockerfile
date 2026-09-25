# Zepto Support Assistant - FastAPI + LangGraph + ChromaDB, CPU only
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    MOCK_LLM=1

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel ships ~3 GB of GPU libraries we don't need)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download all-MiniLM-L6-v2 and build the ChromaDB index at build time,
# so the container starts fast and needs no internet at run time (in mock mode).
RUN python vector_store.py && chmod -R a+rwX /app

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
