"""🔒 Settings — read from manifest.json and environment variables (set by the grader at evaluation time)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


def _normalize_host(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    if "://" not in raw:
        raw = "http://" + raw
    return raw


OLLAMA_HOST = _normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
LLM_MODEL = os.environ.get("LLM_MODEL") or MANIFEST["llm_model"]
EMBED_MODEL = os.environ.get("EMBED_MODEL") or MANIFEST["embed_model"]

TOP_K = int(os.environ.get("TOP_K", "5"))
NUM_CTX = int(os.environ.get("NUM_CTX", "8192"))
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "150"))

VERSION = "starter-1.0"
