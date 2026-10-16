"""🔒 Defaults used while a student/ function is not done yet (or fails).

They are the weakest behaviour that still satisfies the API contract: the service runs and every response
has the right shape, but nothing is actually solved — so the score of an untouched starter is close to 0
and every point above it comes from your work in student/.
"""
from .types import Chunk

# returned by the prompt defaults: the server answers without calling the LLM at all
NO_LLM = "__NO_LLM__"


def split(doc_id: str, text: str) -> list[str]:
    """No chunking: the whole document is one chunk."""
    return [text]


def embed_documents(texts: list[str]) -> list[None]:
    """No embedding."""
    return [None] * len(texts)


def embed_query(text: str) -> None:
    return None


def make_query(question: str, options: list[str]) -> str:
    return question


def search(chunks: list[Chunk], query_text: str, query_vector, k: int) -> list[tuple[Chunk, float]]:
    """No retrieval: nothing is passed on to the prompt."""
    return []


def build_mc_prompt(question: str, options: list[tuple[str, str]], hits) -> str:
    """No prompt: the LLM is not called and the first option key is returned."""
    return NO_LLM


def build_code_prompt(requirement: str, signature: str | None, entry_point: str | None, hits) -> str:
    """No prompt: the LLM is not called and an empty function is returned."""
    return NO_LLM


def empty_function(signature: str | None, entry_point: str | None) -> str:
    header = (signature or f"def {entry_point or 'solution'}(*args, **kwargs):").rstrip()
    if not header.endswith(":"):
        header += ":"
    return f'{header}\n    raise NotImplementedError("TODO HERE: Prompt (student/prompts.py)")\n'
