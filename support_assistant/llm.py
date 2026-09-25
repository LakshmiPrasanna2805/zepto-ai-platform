"""Real LLM calls — ONLY used when MOCK_LLM=0 (optional, ungraded extension).

Uses Groq's OpenAI-compatible chat endpoint (free tier, no credit card).
The API key is read from the GROQ_API_KEY environment variable; it is never stored in code.
"""
import json
import os
import re

import requests
from pydantic import ValidationError

from config import LLM_API_KEY_ENV, LLM_API_URL, LLM_MODEL, MAX_SCHEMA_RETRIES
from prompts import CORRECTIVE_INSTRUCTION
from schemas import AskResponse


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    api_key = os.getenv(LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"MOCK_LLM=0 but the {LLM_API_KEY_ENV} environment variable is not set")
    resp = requests.post(
        LLM_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": LLM_MODEL, "messages": messages, "temperature": temperature},
        timeout=30,
    )
    if resp.status_code != 200:                        # explicit status-code check
        raise RuntimeError(f"LLM API returned HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


def _parse(raw: str) -> AskResponse:
    """Pull the JSON object out of the reply (tolerating code fences) and validate it."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in the reply")
    return AskResponse.model_validate(json.loads(match.group()))


def generate_structured(prompt: str) -> AskResponse:
    """Ask the LLM for a JSON answer. If the reply fails schema validation, retry up to
    MAX_SCHEMA_RETRIES more times with a corrective instruction, then give up and
    return a clearly marked error response."""
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(1 + MAX_SCHEMA_RETRIES):          # 1 try + 2 retries = 3 attempts
        raw = chat(messages)
        try:
            return _parse(raw)
        except (ValueError, json.JSONDecodeError, ValidationError) as err:
            last_error = str(err).splitlines()[0]
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": CORRECTIVE_INSTRUCTION.format(error=last_error)}]
    return AskResponse(
        answer=f"[ERROR] The language model did not return valid structured output after "
               f"{1 + MAX_SCHEMA_RETRIES} attempts ({last_error}).",
        sources=[], confidence=0.0)
