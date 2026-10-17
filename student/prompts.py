"""✏️ ส่วนที่ 4: Prompt — บอก LLM ว่าจะให้ทำอะไร

คืนค่าเป็น list ของ message แบบ chat:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
(หรือคืนเป็นสตริงก็ได้ ระบบจะถือว่าเป็นข้อความของ user)

hits คือผลจาก retrieval.search เป็น list ของ (chunk, score) อ่านเนื้อหาได้จาก chunk.text และ chunk.doc_id

ค่าตั้งต้นตอนยังไม่ทำ: ไม่เรียก LLM เลย ข้อสอบปรนัยตอบ "ก" ทุกข้อ และโจทย์โค้ดได้ฟังก์ชันว่างที่ raise NotImplementedError
(ทำส่วนนี้ก่อน เพราะถ้ายังไม่ทำ ส่วนอื่นจะไม่มีผลต่อคะแนนเลย)
"""
import re
from student import settings


_NEGATION_RE = re.compile(r"(ไม่ถูกต้อง|ไม่ใช่|ไม่ได้|ไม่เป็น|ไม่ควร|ยกเว้น|ข้อใดผิด|ไม่มีข้อใด)")


def _enrich_markdown_tables(text: str) -> str:
    """แปลงตาราง Markdown ให้มีบรรทัดสรุป Header: Value กำกับทุกแถว เพื่อป้องกัน LLM 7B อ่านคอลัมน์สลับกัน"""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if (
            line.startswith("|")
            and line.endswith("|")
            and i + 1 < n
            and lines[i + 1].strip().startswith("|")
            and "---" in lines[i + 1]
        ):
            headers = [c.strip() for c in line.strip("|").split("|")]
            out.append(lines[i])
            out.append(lines[i + 1])
            i += 2
            row_summaries: list[str] = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                out.append(lines[i])
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cells) == len(headers) and any(cells):
                    row_key = cells[0]
                    pairs = [f'{h}="{c}"' for h, c in zip(headers, cells) if h and c]
                    if pairs:
                        row_summaries.append(f"  • [{row_key}] -> " + ", ".join(pairs))
                i += 1
            if row_summaries:
                out.append("สรุปข้อมูลตารางแบบชัดเจนรายรหัส (ใช้ข้อมูลบรรทัดของรหัสนั้นโดยตรง ห้ามสลับบรรทัด):")
                out.extend(row_summaries)
            continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def _format_context(hits, max_chars: int) -> str:
    """จัดรูปแบบรายการ chunk ที่ค้นได้เป็นข้อความอ้างอิง โดยไม่เกิดการซ้ำซ้อนและไม่เกิน max_chars"""
    if not hits:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    total = 0

    for idx, item in enumerate(hits, 1):
        chunk = item[0] if isinstance(item, (tuple, list)) else item
        raw_text = getattr(chunk, "text", str(chunk)).strip()
        doc_id = getattr(chunk, "doc_id", f"doc_{idx}")
        if not raw_text or raw_text in seen:
            continue
        seen.add(raw_text)
        enriched = _enrich_markdown_tables(raw_text)
        block = f"--- [เอกสารอ้างอิง #{idx} | แหล่งที่มา: {doc_id}] ---\n{enriched}"
        if total + len(block) > max_chars and parts:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(block[:remaining] + "\n... (ตัดทอน)")
            break
        parts.append(block)
        total += len(block) + 2

    return "\n\n".join(parts)


def build_mc_prompt(question: str, options: list[tuple[str, str]], hits) -> list[dict]:
    """prompt สำหรับข้อสอบปรนัย

    รับ:  question  คำถาม
          options   list ของ (key, text) เช่น [("ก", "O(1)"), ("ข", "O(log n)"), ...]
          hits      chunk ที่ค้นได้ (list ของ (chunk, score)) อาจเป็น list ว่าง
    คืน:  list ของ message
    """
    max_chars = int(getattr(settings, "MAX_CONTEXT_CHARS", 8000) or 8000)
    context_str = _format_context(hits, max_chars)
    allowed_keys = [k for k, _ in options]
    options_str = "\n".join(f"{k}. {t}" for k, t in options)

    is_negative = bool(_NEGATION_RE.search(question or ""))
    neg_warning = (
        "\n⚠️ ข้อควรระวังพิเศษ: คำถามนี้ถามในเชิงปฏิเสธ (เช่น 'ไม่ถูกต้อง', 'ใช้ไม่ได้', 'ยกเว้น') "
        "ให้ตรวจสอบทีละตัวเลือกกับเอกสารอ้างอิงว่าข้อใดทำได้/ถูกต้อง และเลือกเฉพาะข้อที่ใช้ไม่ได้หรือผิดตามที่โจทย์ถาม!"
        if is_negative else ""
    )

    system_msg = (
        "You are an expert exam solver for strict document-grounded Multiple Choice Questions (Thai & English).\n"
        "Follow these rules strictly:\n"
        "1. Rely on the provided Reference Documents (เอกสารอ้างอิง), even if their rules differ from general knowledge. "
        "(If the Reference Documents do not mention the topic at all, use standard Computer Science knowledge.)\n"
        "2. In the \"reason\" field, think step-by-step:\n"
        "   - Step 1: Read ALL paragraphs in the Reference Documents carefully, including secondary paragraphs that state preprocessing rules "
        "(e.g., 'ก่อนตรวจรูปแบบ... ระบบจะแปลงรหัส... เป็นตัวพิมพ์ใหญ่ทั้งหมดก่อนเสมอ') or exceptions. Quote all relevant rules and the exact `[RowKey] -> ...` table entry.\n"
        "   - Step 2: For numerical/calculation questions:\n"
        "     * Look up the exact `[RowKey]` entry for the requested code.\n"
        "     * Check minimum threshold first: if `amount < ยอดซื้อขั้นต่ำ`, the result is 0.\n"
        "     * Compute the raw formula (e.g., `raw = amount * rate`).\n"
        "     * Check `ส่วนลดสูงสุดต่อบิล` for that exact `[RowKey]`: if it is a number `M` (e.g., `50 บาท`), apply the cap `final = min(raw, M)` (e.g., `min(100, 50) = 50`).\n"
        "   - Step 3 (CRITICAL): Compare your final conclusion against the Options list and explicitly state the matching option key at the very end of \"reason\" in this exact format:\n"
        "     `สรุป: คำตอบคือ <ข้อความ> ซึ่งตรงกับตัวเลือก <key>`\n"
        f"3. In the \"answer\" field, output ONLY that exact `<key>` from {allowed_keys}."
    )

    if context_str:
        user_msg = (
            f"### เอกสารอ้างอิง (Reference Documents)\n{context_str}\n\n"
            f"### คำถาม (Question)\n{question.strip()}{neg_warning}\n\n"
            f"### ตัวเลือก (Options)\n{options_str}\n\n"
            f"จงคิดวิเคราะห์ทีละขั้นใน \"reason\" โดยบรรทัดสุดท้ายของ \"reason\" ต้องระบุว่า `สรุป: คำตอบคือ ... ซึ่งตรงกับตัวเลือก <key>` แล้วใส่ `<key>` ({', '.join(allowed_keys)}) ลงใน \"answer\":"
        )
    else:
        user_msg = (
            f"### คำถาม (Question)\n{question.strip()}{neg_warning}\n\n"
            f"### ตัวเลือก (Options)\n{options_str}\n\n"
            f"จงคิดวิเคราะห์ทีละขั้นใน \"reason\" โดยบรรทัดสุดท้ายของ \"reason\" ต้องระบุว่า `สรุป: คำตอบคือ ... ซึ่งตรงกับตัวเลือก <key>` แล้วใส่ `<key>` ({', '.join(allowed_keys)}) ลงใน \"answer\":"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_code_prompt(requirement: str, signature: str | None, entry_point: str | None, hits) -> list[dict]:
    """prompt สำหรับโจทย์เขียนโปรแกรม

    รับ:  requirement  โจทย์ (ภาษาไทย)
          signature    บรรทัด def ที่ต้องใช้ เช่น "def is_prime(n: int) -> bool:"
          entry_point  ชื่อฟังก์ชัน เช่น "is_prime"
          hits         chunk ที่ค้นได้ (มีเฉพาะโจทย์ที่อ้างถึงเอกสาร ส่วนใหญ่จะเป็น list ว่าง)
    คืน:  list ของ message
    """
    max_chars = int(getattr(settings, "MAX_CONTEXT_CHARS", 8000) or 8000)
    context_str = _format_context(hits, max_chars)

    sig_line = (signature or f"def {entry_point or 'solution'}(*args, **kwargs):").strip()
    if not sig_line.endswith(":"):
        sig_line += ":"

    system_msg = (
        "You are an expert Python software engineer. Write clean, bug-free Python code that passes 100% of unit tests.\n"
        "Strict rules:\n"
        f"1. Define the exact function signature: `{sig_line}`"
        + (f" (entry point name: `{entry_point}`)" if entry_point else "") + "\n"
        "2. ALWAYS include any required Python standard library imports (e.g., `import re`, `import math`, `from collections import ...`) "
        "at the VERY TOP of the code block before `def` so `NameError` never happens.\n"
        "3. Use ONLY the Python standard library. Do NOT use `input()`, `sys.stdin`, or `print()`. Do NOT run top-level test code.\n"
        "4. Check if the requirement forbids any functions, methods, or loops (e.g., 'ห้ามใช้ max, min, sum, sorted, len, str, for, while, eval'). "
        "If forbidden, NEVER reference those names, attributes, or AST nodes anywhere in the code. Handle empty/zero/negative edge cases safely and return the exact type in the signature.\n"
        "5. When Reference Documents (เอกสารอ้างอิง) are provided:\n"
        "   - If there is a table of items/codes/rules, represent EVERY row and EVERY column in a dictionary (e.g., `'type'`, `'value'`, `'min_spend'`, `'max_discount'` for each code). "
        "NEVER hardcode one row's `min_spend` (like 200) for other rows that have different `min_spend` (like 150 or 500)!\n"
        "   - Follow all normalization rules (e.g., `.upper()`, checking exact format without extra spaces/characters, `total < row['min_spend'] -> 0.0`, "
        "`min(discount, row['max_discount'])`, `min(discount, total)`, and `round(discount, 2)`).\n"
        "6. Reply with a single ```python ... ``` code block only."
    )

    user_parts: list[str] = []
    if context_str:
        user_parts.append(
            f"### เอกสารอ้างอิง (Reference Documents — implement all rules and every column of any table accurately)\n{context_str}"
        )
    user_parts.append(f"### โจทย์ (Requirement)\n{requirement.strip()}")
    user_parts.append(
        f"### รูปแบบฟังก์ชันที่ต้องเขียน (Required Function Signature)\n```python\n{sig_line}\n```\n"
        "Write the complete Python module (including top-level `import` statements) in a single ```python ... ``` block:"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
