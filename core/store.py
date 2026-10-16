"""🔒 In-memory knowledge store: namespace -> list[Chunk]. Re-sending a doc_id replaces its chunks."""
import threading

from .types import Chunk


class KnowledgeStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spaces: dict[str, list[Chunk]] = {}

    def replace(self, namespace: str, doc_ids: set[str], chunks: list[Chunk]) -> None:
        with self._lock:
            kept = [c for c in self._spaces.get(namespace, []) if c.doc_id not in doc_ids]
            self._spaces[namespace] = kept + chunks

    def delete(self, namespace: str) -> bool:
        with self._lock:
            return self._spaces.pop(namespace, None) is not None

    def chunks(self, namespace: str | None) -> list[Chunk]:
        if not namespace:
            return []
        with self._lock:
            return list(self._spaces.get(namespace, []))
