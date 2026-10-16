"""✏️ ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

เอกสารที่ได้รับเป็น markdown เช่น lecture note หรือ case study ที่มีหัวข้อ (#, ##), ตาราง และ code block
ชิ้นที่แบ่งแล้วจะถูกนำไปทำ embedding และใช้ค้นหาตอนตอบคำถาม
แบ่งดี = ค้นเจอเนื้อหาที่ตรงคำถามและครบถ้วน

ค่าตั้งต้นตอนยังไม่ทำ: ไม่แบ่งเลย ทั้งเอกสารเป็น chunk เดียว
ค่าที่ปรับได้ (CHUNK_SIZE, CHUNK_OVERLAP, ...) อยู่ใน student/settings.py
"""
import re
from student import settings


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _split_into_sections(text: str) -> list[tuple[list[str], str]]:
    """แยกเอกสารตามหัวข้อ Markdown (# .. ######) โดยไม่แยกถ้า # อยู่ใน code block
    คืนค่าเป็น list ของ (heading_path, section_body)
    """
    lines = text.splitlines()
    sections: list[tuple[list[str], list[str]]] = []
    headings: dict[int, str] = {}
    current_lines: list[str] = []
    in_fence = False
    fence_marker = ""

    def current_path() -> list[str]:
        return [headings[k] for k in sorted(headings.keys())]

    for line in lines:
        m_fence = _FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""

        if not in_fence and getattr(settings, "SPLIT_BY_HEADING", True):
            m_head = _HEADING_RE.match(line)
            if m_head:
                body_str = "\n".join(current_lines).strip()
                if body_str:
                    sections.append((current_path(), current_lines))
                current_lines = []
                level = len(m_head.group(1))
                title = m_head.group(2).strip()
                # ล้างหัวข้อระดับเดียวกันหรือลึกกว่า
                for k in [k for k in headings if k >= level]:
                    del headings[k]
                headings[level] = title
                continue

        current_lines.append(line)

    body_str = "\n".join(current_lines).strip()
    if body_str or not sections:
        sections.append((current_path(), current_lines))

    out: list[tuple[list[str], str]] = []
    for path, sec_lines in sections:
        cleaned = "\n".join(sec_lines).strip()
        if cleaned:
            out.append((path, cleaned))
        elif path:
            out.append((path, ""))
    return out


def _parse_atomic_blocks(section_text: str) -> list[str]:
    """แบ่งเนื้อหาภายในหัวข้อเป็นบล็อกย่อย (ย่อหน้า, ตารางทั้งก้อน, code block ทั้งก้อน)"""
    lines = section_text.splitlines()
    blocks: list[str] = []
    buf: list[str] = []
    mode = "normal"  # "normal" | "fence" | "table"
    fence_char = ""

    def flush() -> None:
        nonlocal buf
        content = "\n".join(buf).strip()
        if content:
            blocks.append(content)
        buf = []

    for line in lines:
        stripped = line.strip()
        m_fence = _FENCE_RE.match(line)

        if mode == "fence":
            buf.append(line)
            if m_fence and m_fence.group(1)[0] == fence_char:
                flush()
                mode = "normal"
            continue

        if m_fence:
            flush()
            mode = "fence"
            fence_char = m_fence.group(1)[0]
            buf.append(line)
            continue

        is_table_line = stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2
        if mode == "table":
            if is_table_line:
                buf.append(line)
                continue
            else:
                flush()
                mode = "normal"

        if is_table_line:
            flush()
            mode = "table"
            buf.append(line)
            continue

        if not stripped:
            flush()
        else:
            buf.append(line)

    flush()
    return blocks


def _split_oversized_block(block: str, max_len: int, overlap: int) -> list[str]:
    """กรณีบล็อกเดียว (เช่น ย่อหน้ายาวมาก หรือตารางขนาดใหญ่) ยาวเกิน max_len"""
    if len(block) <= max_len:
        return [block]

    lines = block.splitlines()
    # ถ้าเป็นตาราง ให้เก็บ header 2 บรรทัดแรกติดไปด้วยทุกชิ้น
    is_table = len(lines) >= 3 and lines[0].strip().startswith("|") and "---" in lines[1]
    table_header = "\n".join(lines[:2]) + "\n" if is_table else ""
    body_lines = lines[2:] if is_table else lines

    pieces: list[str] = []
    cur: list[str] = []
    cur_len = len(table_header)

    for ln in body_lines:
        if len(ln) > max_len and not is_table:
            # บรรทัดเดียวยาวเกิน max_len -> ตัดตามตัวอักษรพร้อม overlap
            if cur:
                pieces.append(table_header + "\n".join(cur))
                cur, cur_len = [], len(table_header)
            step = max(max_len - overlap, 100)
            for start in range(0, len(ln), step):
                sub = ln[start:start + max_len].strip()
                if sub:
                    pieces.append(sub)
            continue

        add_len = len(ln) + (1 if cur else 0)
        if cur and cur_len + add_len > max_len:
            pieces.append((table_header + "\n".join(cur)).strip())
            # เก็บบางบรรทัดท้ายเป็น overlap (ยกเว้นตารางที่มี header ซ้ำอยู่แล้ว)
            if not is_table and overlap > 0:
                ov: list[str] = []
                ov_len = 0
                for prev_ln in reversed(cur):
                    if ov_len + len(prev_ln) > overlap:
                        break
                    ov.insert(0, prev_ln)
                    ov_len += len(prev_ln) + 1
                cur = ov
                cur_len = sum(len(x) for x in cur) + max(len(cur) - 1, 0)
            else:
                cur = []
                cur_len = len(table_header)

        cur.append(ln)
        cur_len += add_len

    if cur:
        pieces.append((table_header + "\n".join(cur)).strip())
    return [p for p in pieces if p.strip()]


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ

    รับ:  doc_id  ชื่อเอกสาร เช่น "session04_01_prompt_fundamentals"
          text    เนื้อหาทั้งหมดของเอกสาร (markdown)
    คืน:  list[str] ข้อความของแต่ละ chunk (ห้ามว่าง)
    """
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return [doc_id or "empty"]

    chunk_size = max(int(getattr(settings, "CHUNK_SIZE", 950) or 950), 200)
    overlap = max(int(getattr(settings, "CHUNK_OVERLAP", 150) or 0), 0)
    include_heading = bool(getattr(settings, "INCLUDE_HEADING", True))

    sections = _split_into_sections(cleaned)
    chunks: list[str] = []

    for path, body in sections:
        header_prefix = ""
        if include_heading and path:
            header_prefix = " > ".join(path) + "\n"

        avail_size = max(chunk_size - len(header_prefix), 200)
        if not body:
            if header_prefix.strip():
                chunks.append(header_prefix.strip())
            continue

        # ถ้าทั้งหัวข้อยาวไม่เกิน chunk_size ให้เก็บไว้เป็นก้อนเดียวกันเลย (เช่น ตาราง + กติกาใต้ตาราง)
        if len(header_prefix) + len(body) <= chunk_size:
            chunks.append((header_prefix + body).strip())
            continue

        raw_blocks = _parse_atomic_blocks(body)
        blocks: list[str] = []
        for b in raw_blocks:
            blocks.extend(_split_oversized_block(b, avail_size, overlap))

        cur_blocks: list[str] = []
        cur_len = 0
        for b in blocks:
            sep_len = 2 if cur_blocks else 0
            if cur_blocks and cur_len + sep_len + len(b) > avail_size:
                chunks.append((header_prefix + "\n\n".join(cur_blocks)).strip())
                # สร้าง overlap จากบล็อกท้ายสุดหากขนาดไม่ใหญ่เกินไป
                ov_blocks: list[str] = []
                ov_len = 0
                if overlap > 0:
                    for prev_b in reversed(cur_blocks):
                        if ov_len + len(prev_b) <= overlap:
                            ov_blocks.insert(0, prev_b)
                            ov_len += len(prev_b) + 2
                        else:
                            # ดึงเฉพาะท้ายบล็อกล่าสุดมาซ้อนกัน (ถ้าไม่ใช่ตาราง)
                            if not ov_blocks and not prev_b.lstrip().startswith("|"):
                                tail = prev_b[-overlap:].strip()
                                if tail:
                                    ov_blocks.append(tail)
                                    ov_len = len(tail)
                            break
                cur_blocks = ov_blocks
                cur_len = sum(len(x) for x in cur_blocks) + max(0, (len(cur_blocks) - 1) * 2)

            cur_blocks.append(b)
            cur_len += (2 if len(cur_blocks) > 1 else 0) + len(b)

        if cur_blocks:
            chunks.append((header_prefix + "\n\n".join(cur_blocks)).strip())

    result = [c.strip() for c in chunks if c and c.strip()]
    return result if result else [cleaned]
