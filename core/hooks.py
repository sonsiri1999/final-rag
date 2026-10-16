"""🔒 Calls the student/ functions safely.

Every student function goes through call(): if it is not written yet (returns None), raises an error,
or returns something of the wrong shape, the service does NOT crash — it logs a warning (in Thai) and
uses the basic default from core/fallbacks.py instead. The status of each function is shown in /health.
"""
import logging
import time
import traceback
from collections import Counter

from .types import Chunk

log = logging.getLogger("starter")
STATUS: dict[str, dict] = {}
_seen: Counter = Counter()


def _report(name: str, state: str, message: str, detail: str = "") -> None:
    STATUS[name] = {"state": state, "message": message, "at": time.strftime("%H:%M:%S")}
    _seen[(name, message)] += 1
    if _seen[(name, message)] == 1:  # once per distinct problem, so the log stays readable
        log.warning("[student.%s] %s", name, message)
        if detail:
            log.warning(detail)


def call(name: str, fn, args: tuple, fallback, check):
    """Run fn(*args); on None / exception / invalid result use fallback(*args).

    check(value) returns None when the value is fine, a (Thai) problem description otherwise,
    and may also return ("ok", normalised_value) to accept a value after normalising it.
    """
    try:
        value = fn(*args)
    except Exception as exc:  # noqa: BLE001 — any student error falls back, never a 500
        _report(name, "error", f"เกิด error: {type(exc).__name__}: {exc} — ใช้ค่าตั้งต้นแทน",
                traceback.format_exc(limit=4))
        return fallback(*args)
    if value is None:
        _report(name, "todo", "ยังไม่ได้ทำ (TODO) — ใช้ค่าตั้งต้นที่ไม่ได้แก้ปัญหาอะไรเลย (คะแนนส่วนนี้จะเป็น 0)")
        return fallback(*args)
    try:
        verdict = check(value, *args)
    except Exception as exc:  # noqa: BLE001
        verdict = f"ตรวจรูปแบบไม่ได้ ({type(exc).__name__}: {exc})"
    if isinstance(verdict, tuple) and verdict[0] == "ok":
        value, verdict = verdict[1], None
    if verdict:
        _report(name, "invalid", f"ค่าที่คืนมาไม่ถูกรูปแบบ: {verdict} — ใช้ค่าตั้งต้นแทน")
        return fallback(*args)
    if STATUS.get(name, {}).get("state") != "ok":
        log.info("[student.%s] ทำงานได้ ✓", name)
    STATUS[name] = {"state": "ok", "at": time.strftime("%H:%M:%S")}
    return value


# ------------------------------------------------------------------ result checks
def _is_vector(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, (int, float)) for x in v)


def check_split(value, doc_id, text):
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return "ต้องคืน list[str]"
    pieces = [x for x in value if x.strip()]
    if not pieces:
        return "ได้ list ว่าง (ต้องมีอย่างน้อย 1 chunk)"
    return ("ok", pieces)


def check_embed_documents(value, texts):
    if not isinstance(value, list) or len(value) != len(texts):
        return f"ต้องคืน list ของ vector จำนวน {len(texts)} ตัว (เท่ากับจำนวนข้อความ)"
    if not all(_is_vector(v) for v in value):
        return "ทุกตัวต้องเป็น list[float]"
    if len({len(v) for v in value}) > 1:
        return "vector ต้องยาวเท่ากันทุกตัว"
    return None


def check_embed_query(value, text):
    return None if _is_vector(value) else "ต้องคืน list[float] 1 vector"


def check_make_query(value, *args):
    return None if isinstance(value, str) and value.strip() else "ต้องคืนสตริงที่ไม่ว่าง"


def check_search(value, chunks, query_text, query_vector, k):
    if not isinstance(value, list):
        return "ต้องคืน list ของ (Chunk, score)"
    hits = []
    for item in value:
        if isinstance(item, Chunk):
            hits.append((item, 0.0))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], Chunk) \
                and isinstance(item[1], (int, float)):
            hits.append((item[0], float(item[1])))
        else:
            return "ทุกตัวต้องเป็น (Chunk, score) — Chunk ต้องมาจาก list chunks ที่ได้รับ"
    return ("ok", hits)


def check_messages(value, *args):
    if isinstance(value, str) and value.strip():
        return ("ok", [{"role": "user", "content": value}])
    if isinstance(value, list) and value and all(
            isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
            and isinstance(m.get("content"), str) for m in value):
        return None
    return 'ต้องคืนสตริง หรือ list ของ {"role": "system"|"user"|"assistant", "content": "..."}'
