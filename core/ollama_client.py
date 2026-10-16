"""🔒 Ollama client — student code may call embed() and chat() but must not change this file.

    from core.ollama_client import embed, chat
    vectors = embed(["ข้อความ 1", "ข้อความ 2"])        # -> list[list[float]] (EMBED_MODEL)
    reply = chat([{"role": "user", "content": "..."}])  # -> str (LLM_MODEL, temperature 0)
"""
import httpx

from . import config

_http = httpx.Client(base_url=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT)

EMBED_BATCH = 32


def embed(texts: list[str]) -> list[list[float]]:
    """Embed many texts with EMBED_MODEL (sent in batches of 32)."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        resp = _http.post("/api/embed", json={"model": config.EMBED_MODEL, "input": texts[start:start + EMBED_BATCH]})
        resp.raise_for_status()
        vectors.extend(resp.json()["embeddings"])
    return vectors


def chat(messages: list[dict], response_format: dict | None = None) -> str:
    """Send chat messages to LLM_MODEL and return the reply text."""
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": config.NUM_CTX},
    }
    if response_format is not None:
        payload["format"] = response_format
    resp = _http.post("/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json()["message"]["content"]
