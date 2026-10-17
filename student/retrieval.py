"""✏️ ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

ค่าตั้งต้นตอนยังไม่ทำ: ไม่คืน chunk ใดเลย LLM จึงไม่เห็นเอกสาร
"""
import math
import re
from collections import Counter

from core.types import Chunk
from core.utils import cosine, cosine_many  # noqa: F401 — ตัวช่วยคำนวณความคล้ายของ vector
from student import settings

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except ImportError:
    _BM25Okapi = None

try:
    from pythainlp.tokenize import word_tokenize as _th_word_tokenize
except ImportError:
    _th_word_tokenize = None

_EN_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "with", "as", "by", "at", "from", "or", "and", "that", "this",
    "it", "what", "which", "who", "how", "when", "where", "why", "does", "do",
}
_TH_STOP = {
    "และ", "หรือ", "ของ", "ใน", "ที่", "จาก", "ให้", "ได้", "เป็น", "คือ", "มี",
    "จะ", "ต้อง", "การ", "ความ", "ตาม", "ข้อใด", "อะไร", "อย่างไร", "เมื่อใด",
}

_CODE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)+")
_EN_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_TH_RUN_RE = re.compile(r"[\u0e00-\u0e7f]+")


def tokenize_multilingual(text: str) -> list[str]:
    """แยก token สำหรับทั้งภาษาไทย ภาษาอังกฤษ ตัวเลข และรหัสเฉพาะ (เช่น BB-SAVE-26, O(log n))"""
    if not text:
        return []
    tokens: list[str] = []

    # 1. รหัสที่มีขีดหรือจุด เช่น BB-SAVE-26, 3.2, v2.x
    for m in _CODE_TOKEN_RE.finditer(text):
        raw = m.group(0).lower()
        tokens.append(raw)
        for part in re.split(r"[-_.]", raw):
            if len(part) >= 2 and part not in _EN_STOP:
                tokens.append(part)

    # 2. คำภาษาอังกฤษและชื่อตัวแปร (แยก snake_case / camelCase)
    for m in _EN_WORD_RE.finditer(text):
        w = m.group(0)
        wl = w.lower()
        if wl not in _EN_STOP and len(wl) > 1:
            tokens.append(wl)
        parts = [p.lower() for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", w) if p]
        if len(parts) > 1:
            for p in parts:
                if len(p) > 1 and p not in _EN_STOP:
                    tokens.append(p)

    # 3. ตัวเลข (เก็บทั้งแบบมีลูกน้ำและไม่มีลูกน้ำ เช่น 1,000 -> 1000)
    for m in _NUM_RE.finditer(text):
        num = m.group(0)
        norm_num = num.replace(",", "")
        tokens.append(norm_num)

    # 4. ภาษาไทย: ใช้ทั้งคำตัดด้วยพจนานุกรม (pythainlp newmm + word bigrams) และ character n-grams (3-gram, 4-gram)
    for m in _TH_RUN_RE.finditer(text):
        run = m.group(0)
        if run in _TH_STOP:
            continue
        if _th_word_tokenize is not None:
            words = [w.strip() for w in _th_word_tokenize(run, engine="newmm") if w.strip()]
            meaningful = [w for w in words if w not in _TH_STOP and len(w) >= 2]
            tokens.extend(meaningful)
            for i in range(len(meaningful) - 1):
                tokens.append(f"{meaningful[i]}_{meaningful[i + 1]}")
        if len(run) <= 4:
            tokens.append(run)
            continue
        tokens.append(run)
        # สร้าง character 3-gram และ 4-gram เพื่อให้ BM25 จับคำไทยที่เขียนติดกันได้แม่นยำ
        for n in (3, 4):
            if len(run) >= n:
                for i in range(len(run) - n + 1):
                    sub = run[i:i + n]
                    if sub not in _TH_STOP:
                        tokens.append(sub)

    return tokens


def _bm25_scores(corpus_tokens: list[list[str]], query_tokens: list[str]) -> list[float]:
    """คำนวณคะแนน BM25 ด้วย rank_bm25 (หรือ fallback ในตัวหากไม่ได้ติดตั้ง)"""
    if not corpus_tokens or not query_tokens:
        return [0.0] * len(corpus_tokens)

    if _BM25Okapi is not None:
        try:
            bm25 = _BM25Okapi(corpus_tokens)
            return [float(s) for s in bm25.get_scores(query_tokens)]
        except Exception:
            pass

    # Pure-Python BM25Okapi fallback (k1=1.5, b=0.75)
    n_docs = len(corpus_tokens)
    avgdl = sum(len(d) for d in corpus_tokens) / max(n_docs, 1)
    df: Counter = Counter()
    doc_freqs: list[Counter] = []
    for doc in corpus_tokens:
        c = Counter(doc)
        doc_freqs.append(c)
        for term in c:
            df[term] += 1

    k1, b = 1.5, 0.75
    scores = [0.0] * n_docs
    for q in query_tokens:
        if q not in df:
            continue
        idf = math.log(1.0 + (n_docs - df[q] + 0.5) / (df[q] + 0.5))
        for i, c in enumerate(doc_freqs):
            tf = c.get(q, 0)
            if tf > 0:
                dl = len(corpus_tokens[i])
                denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-6))
                scores[i] += idf * (tf * (k1 + 1.0)) / max(denom, 1e-6)
    return scores


def _exact_match_bonus(chunk_text: str, query_text: str) -> float:
    """ให้คะแนนพิเศษเมื่อพบรหัสเฉพาะ ชื่อฟังก์ชัน หรือตัวเลขสำคัญจากคำถามใน chunk ตรงๆ"""
    cl = chunk_text.lower()
    cl_nocomma = cl.replace(",", "")
    bonus = 0.0

    # รหัสเฉพาะ เช่น BB-SAVE-26, Silver, Gold
    specials = set(_CODE_TOKEN_RE.findall(query_text)) | set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b", query_text))
    matched_special = 0
    for sp in specials:
        spl = sp.lower()
        if spl not in _EN_STOP and spl in cl:
            matched_special += 1
    if specials:
        bonus += 0.20 * min(matched_special / max(len(specials), 1), 1.0)

    # ตัวเลขในคำถาม เช่น 1,000, 3.2, 2026
    nums = {n.replace(",", "") for n in _NUM_RE.findall(query_text)}
    if nums:
        matched_nums = sum(1 for n in nums if n in cl_nocomma)
        bonus += 0.10 * min(matched_nums / len(nums), 1.0)

    return bonus


def make_query(question: str, options: list[str]) -> str:
    """สร้างข้อความที่จะใช้ค้นหา

    รับ:  question  คำถาม (หรือ requirement ของโจทย์เขียนโค้ด ซึ่งกรณีนี้ options จะเป็น list ว่าง)
          options   ข้อความของตัวเลือก ก–จ (ไม่มีตัวอักษร key นำหน้า)
    คืน:  สตริงคำค้น (ห้ามว่าง)
    """
    q = (question or "").strip()
    if getattr(settings, "QUERY_WITH_OPTIONS", True) and options:
        opts_str = " ".join(o.strip() for o in options if o and o.strip())
        if opts_str:
            q = f"{q}\n{opts_str}"
    return q or "query"


def search(chunks: list[Chunk], query_text: str, query_vector: list[float] | None, k: int) -> list[tuple[Chunk, float]]:
    """เลือก chunk ที่เกี่ยวข้องที่สุด k ชิ้น ด้วย Hybrid Search (Vector Cosine + BM25 + RRF)"""
    if not chunks:
        return []

    top_k = max(int(getattr(settings, "TOP_K", k) or k), int(k or 5))
    n = len(chunks)

    # 1. คำนวณคะแนน Vector Cosine Similarity
    has_vectors = query_vector is not None and all(c.vector is not None for c in chunks)
    if has_vectors:
        vec_scores = cosine_many(query_vector, [c.vector for c in chunks])
    elif query_vector is not None and any(c.vector is not None for c in chunks):
        vec_scores = [cosine(query_vector, c.vector) if c.vector is not None else 0.0 for c in chunks]
        has_vectors = True
    else:
        vec_scores = [0.0] * n

    # 2. คำนวณคะแนน BM25 + Exact Match Bonus
    corpus_tokens = [tokenize_multilingual(f"{c.doc_id} {c.text}") for c in chunks]
    q_tokens = tokenize_multilingual(query_text)
    bm25_raw = _bm25_scores(corpus_tokens, q_tokens)
    bonuses = [_exact_match_bonus(c.text, query_text) for c in chunks]

    # Normalize BM25 scores ให้อยู่ในช่วง [0, 1]
    max_bm25 = max(bm25_raw, default=0.0)
    bm25_norm = [s / max_bm25 if max_bm25 > 1e-9 else 0.0 for s in bm25_raw]

    # Normalize Vector scores ให้อยู่ในช่วง [0, 1]
    if has_vectors:
        min_v, max_v = min(vec_scores), max(vec_scores)
        span_v = max_v - min_v
        vec_norm = [(v - min_v) / span_v if span_v > 1e-9 else max(0.0, v) for v in vec_scores]
    else:
        vec_norm = [0.0] * n

    # 3. คำนวณ Reciprocal Rank Fusion (RRF) ร่วมกับ Weighted Hybrid Score
    vec_order = sorted(range(n), key=lambda i: vec_scores[i], reverse=True)
    bm25_order = sorted(range(n), key=lambda i: bm25_raw[i] + bonuses[i], reverse=True)
    vec_rank = {idx: r for r, idx in enumerate(vec_order, 1)}
    bm25_rank = {idx: r for r, idx in enumerate(bm25_order, 1)}

    w_kw = float(getattr(settings, "HYBRID_WEIGHT", 0.45) or 0.0)
    if not has_vectors:
        w_kw = 1.0

    combined: list[tuple[int, float]] = []
    for i in range(n):
        rrf = (1.0 / (60 + vec_rank[i])) + (1.0 / (60 + bm25_rank[i])) if has_vectors else (1.0 / (60 + bm25_rank[i]))
        linear = (1.0 - w_kw) * vec_norm[i] + w_kw * min(1.0, bm25_norm[i] + bonuses[i])
        score = 0.85 * linear + 0.15 * (rrf * 30.0)
        combined.append((i, float(score)))

    combined.sort(key=lambda item: item[1], reverse=True)

    # กรองด้วย MIN_SCORE (ถ้าตั้งค่าไว้) แต่ต้องมีผลลัพธ์อย่างน้อย 1 ชิ้นเสมอ
    min_score = getattr(settings, "MIN_SCORE", None)
    if min_score is not None:
        filtered = [item for item in combined if item[1] >= float(min_score)]
        if filtered:
            combined = filtered

    selected = combined[:top_k]

    # 4. ถ้ายังมีที่ว่างหรือ chunk ข้างเคียงในเอกสารเดียวกันมีเนื้อหาต่อเนื่อง ให้ดึงข้างเคียงของอันดับ 1 มาช่วยถ้ายังไม่อยู่ใน selected
    selected_indices = {i for i, _ in selected}
    if selected and len(selected) < top_k and n > len(selected):
        best_idx, best_score = selected[0]
        for neighbor in (best_idx - 1, best_idx + 1):
            if 0 <= neighbor < n and neighbor not in selected_indices:
                if chunks[neighbor].doc_id == chunks[best_idx].doc_id:
                    selected.append((neighbor, round(best_score * 0.85, 4)))
                    selected_indices.add(neighbor)
                    if len(selected) >= top_k:
                        break

    return [(chunks[i], float(sc)) for i, sc in selected]
