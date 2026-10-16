"""🔒 Helpers that student code may use."""
import numpy as np


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (1 = same direction, 0 = unrelated)."""
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(va @ vb / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-10))


def cosine_many(query: list[float], vectors: list[list[float]]) -> list[float]:
    """Cosine similarity of one query vector against many vectors (fast, vectorised)."""
    if not vectors:
        return []
    m = np.asarray(vectors, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    scores = (m @ q) / ((np.linalg.norm(m, axis=1) * np.linalg.norm(q)) + 1e-10)
    return [float(s) for s in scores]
