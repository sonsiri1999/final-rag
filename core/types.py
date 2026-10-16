"""🔒 Data types shared by core/ and student/."""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One piece of a knowledge document.

    doc_id  which document it came from
    text    the chunk text (what student/chunking.py returned)
    vector  its embedding (what student/embedding.py returned), or None if embedding is not done yet
    meta    a free dict for your own data (e.g. heading, position) — core never reads it
    """
    doc_id: str
    text: str
    vector: list[float] | None = None
    meta: dict = field(default_factory=dict)


# a search result: (chunk, score) — higher score = more relevant
Hit = tuple[Chunk, float]
