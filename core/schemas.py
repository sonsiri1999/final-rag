"""🔒 Request/response models — the API contract (spec/openapi.yaml)."""
from pydantic import BaseModel, Field

NAMESPACE_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"


class Document(BaseModel):
    doc_id: str
    content: str
    format: str = "markdown"


class KnowledgeRequest(BaseModel):
    namespace: str = Field(pattern=NAMESPACE_PATTERN)
    documents: list[Document] = Field(min_length=1)


class KnowledgeResponse(BaseModel):
    namespace: str
    documents: int
    chunks: int


class Option(BaseModel):
    key: str
    text: str


class AskRequest(BaseModel):
    namespace: str = Field(pattern=NAMESPACE_PATTERN)
    question_id: str
    question: str
    options: list[Option] = Field(min_length=2)


class Source(BaseModel):
    doc_id: str
    text: str
    score: float | None = None


class AskResponse(BaseModel):
    question_id: str
    answer: str
    sources: list[Source] = []


class CodeRequest(BaseModel):
    task_id: str
    requirement: str
    language: str = "python"
    entry_point: str | None = None
    signature: str | None = None
    namespace: str | None = None


class CodeResponse(BaseModel):
    task_id: str
    language: str
    code: str
