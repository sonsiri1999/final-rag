"""🔒 The HTTP service (API contract: GET /health, POST /knowledge, DELETE /knowledge/{ns}, POST /ask, POST /code).

Flow:  /knowledge  doc -> student.chunking.split -> student.embedding.embed_documents -> store
       /ask        student.retrieval.make_query -> student.embedding.embed_query -> student.retrieval.search
                   -> student.prompts.build_mc_prompt -> LLM (JSON answer forced) -> option key
       /code       retrieval (only when a namespace is given) -> student.prompts.build_code_prompt -> LLM
                   -> extract code (one retry if it does not compile)
"""
import ast
import inspect
import logging

import httpx
from fastapi import FastAPI, HTTPException

from student import chunking, embedding, prompts, retrieval

from . import config, fallbacks, hooks, ollama_client
from .answers import extract_code, parse_answer
from .schemas import (AskRequest, AskResponse, CodeRequest, CodeResponse, KnowledgeRequest, KnowledgeResponse,
                      Source)
from .store import KnowledgeStore
from .types import Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("starter")
app = FastAPI(title="Student RAG Service (starter)", version=config.VERSION)
store = KnowledgeStore()

HOOKS = {
    "chunking.split": chunking.split,
    "embedding.embed_documents": embedding.embed_documents,
    "embedding.embed_query": embedding.embed_query,
    "retrieval.make_query": retrieval.make_query,
    "retrieval.search": retrieval.search,
    "prompts.build_mc_prompt": prompts.build_mc_prompt,
    "prompts.build_code_prompt": prompts.build_code_prompt,
}
CODE_RETRY = "Your previous code does not compile: {error}\nFix it and reply again with a single ```python code block."


@app.on_event("startup")
def _todo_report() -> None:
    todo = [name for name, fn in HOOKS.items() if "TODO HERE" in inspect.getsource(fn)]
    log.info("starter %s — LLM %s, embedding %s, Ollama %s", config.VERSION, config.LLM_MODEL,
             config.EMBED_MODEL, config.OLLAMA_HOST)
    if todo:
        log.warning("ยังมี TODO ที่ยังไม่ได้ทำ: %s (ระหว่างนี้ใช้ค่าตั้งต้นแบบพื้นฐาน)", ", ".join(todo))
    else:
        log.info("ทำครบทุก TODO แล้ว ✓")


def _ollama_error(exc: Exception) -> HTTPException:
    log.exception("Ollama call failed")
    return HTTPException(status_code=502, detail=f"LLM backend error: {exc}")


def _retrieve(namespace: str | None, question: str, options: list[str]) -> list[tuple[Chunk, float]]:
    chunks = store.chunks(namespace)
    if not chunks:
        return []
    query = hooks.call("retrieval.make_query", retrieval.make_query, (question, options),
                       fallbacks.make_query, hooks.check_make_query)
    query_vector = None
    if any(c.vector is not None for c in chunks):
        query_vector = hooks.call("embedding.embed_query", embedding.embed_query, (query,),
                                  fallbacks.embed_query, hooks.check_embed_query)
    hits = hooks.call("retrieval.search", retrieval.search, (chunks, query, query_vector, config.TOP_K),
                      fallbacks.search, hooks.check_search)
    return hits[:max(config.TOP_K, 1) * 4]  # guard against returning the whole store


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "student_id": config.MANIFEST["student_id"],
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "version": config.VERSION,
        "student_functions": {name: hooks.STATUS.get(name, {"state": "not called yet"}) for name in HOOKS},
    }


@app.post("/knowledge", response_model=KnowledgeResponse)
def add_knowledge(req: KnowledgeRequest) -> KnowledgeResponse:
    new_chunks: list[Chunk] = []
    for doc in req.documents:
        pieces = hooks.call("chunking.split", chunking.split, (doc.doc_id, doc.content),
                            fallbacks.split, hooks.check_split)
        new_chunks += [Chunk(doc_id=doc.doc_id, text=p) for p in pieces]
    try:
        texts = [c.text for c in new_chunks]
        vectors = hooks.call("embedding.embed_documents", embedding.embed_documents, (texts,),
                             fallbacks.embed_documents, hooks.check_embed_documents)
    except httpx.HTTPError as exc:
        raise _ollama_error(exc)
    for chunk, vector in zip(new_chunks, vectors):
        chunk.vector = vector
    store.replace(req.namespace, {d.doc_id for d in req.documents}, new_chunks)
    return KnowledgeResponse(namespace=req.namespace, documents=len(req.documents), chunks=len(new_chunks))


@app.delete("/knowledge/{namespace}")
def delete_knowledge(namespace: str) -> dict:
    return {"namespace": namespace, "deleted": store.delete(namespace)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    options = [(o.key, o.text) for o in req.options]
    keys = [k for k, _ in options]
    schema = {"type": "object",
              "properties": {"reason": {"type": "string"}, "answer": {"type": "string", "enum": keys}},
              "required": ["reason", "answer"]}
    try:
        hits = _retrieve(req.namespace, req.question, [t for _, t in options])
        messages = hooks.call("prompts.build_mc_prompt", prompts.build_mc_prompt, (req.question, options, hits),
                              fallbacks.build_mc_prompt, hooks.check_messages)
        if messages == fallbacks.NO_LLM:  # prompt not written yet: no LLM call, always the first option
            answer = keys[0]
        else:
            raw = ollama_client.chat(messages, response_format=schema)  # JSON with an allowed key is enforced here
            answer = parse_answer(raw, keys)
    except httpx.HTTPError as exc:
        raise _ollama_error(exc)
    return AskResponse(question_id=req.question_id, answer=answer,
                       sources=[Source(doc_id=c.doc_id, text=c.text[:2000], score=round(s, 4)) for c, s in hits])


@app.post("/code", response_model=CodeResponse)
def write_code(req: CodeRequest) -> CodeResponse:
    if req.language.lower() != "python":
        raise HTTPException(status_code=400, detail="Only python is supported")
    try:
        hits = _retrieve(req.namespace, req.requirement, [])
        args = (req.requirement, req.signature, req.entry_point, hits)
        messages = hooks.call("prompts.build_code_prompt", prompts.build_code_prompt, args,
                              fallbacks.build_code_prompt, hooks.check_messages)
        if messages == fallbacks.NO_LLM:  # prompt not written yet: no LLM call, an empty function
            return CodeResponse(task_id=req.task_id, language="python",
                                code=fallbacks.empty_function(req.signature, req.entry_point))
        raw = ollama_client.chat(messages)
        code = extract_code(raw)
        try:
            ast.parse(code)
        except SyntaxError as err:
            retry = messages + [{"role": "assistant", "content": raw},
                                {"role": "user", "content": CODE_RETRY.format(error=err)}]
            code = extract_code(ollama_client.chat(retry))
    except httpx.HTTPError as exc:
        raise _ollama_error(exc)
    return CodeResponse(task_id=req.task_id, language="python", code=code)
