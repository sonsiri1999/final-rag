"""✏️ ค่าที่ปรับจูนได้ — รวมไว้ที่เดียว แก้ที่นี่แล้วรัน selfcheck ดูผล

ใช้ในโค้ดของคุณแบบนี้:
    from student import settings
    size = settings.CHUNK_SIZE

ไฟล์นี้เป็นแค่ที่เก็บค่า ระบบ (core/) ไม่ได้อ่านเอง ค่าจะมีผลก็ต่อเมื่อโค้ดใน student/ นำไปใช้
เพิ่มค่าของตัวเองได้ตามต้องการ
"""

# ---------- Chunking (student/chunking.py) ----------
CHUNK_SIZE = 950            # ความยาวสูงสุดของ chunk (ตัวอักษร) ลองช่วง 300–1500
CHUNK_OVERLAP = 150         # ส่วนที่ซ้อนกันระหว่าง chunk ที่อยู่ติดกัน ลอง 0–20% ของ CHUNK_SIZE
SPLIT_BY_HEADING = True     # แบ่งตามหัวข้อ markdown (#) ก่อน แล้วค่อยตัดตามความยาว
INCLUDE_HEADING = True      # ใส่ชื่อหัวข้อไว้ต้น chunk

# ---------- Embedding (student/embedding.py) ----------
DOC_PREFIX = "search_document: "    # prefix ที่ nomic-embed-text ใช้กับเอกสาร
QUERY_PREFIX = "search_query: "     # prefix ที่ nomic-embed-text ใช้กับคำค้น

# ---------- Retrieval (student/retrieval.py) ----------
TOP_K = 6                   # จำนวน chunk ที่ส่งให้ LLM ลอง 3–10 (ระบบรับได้สูงสุด 20)
QUERY_WITH_OPTIONS = True   # รวมข้อความของตัวเลือก ก–จ เข้าไปในคำค้นด้วย
HYBRID_WEIGHT = 0.45        # น้ำหนักของคะแนน keyword ใน hybrid search (0 = ใช้ vector อย่างเดียว, ลอง 0.3–0.5)
MIN_SCORE = None            # ตัด chunk ที่คะแนนต่ำกว่านี้ทิ้ง (None = ไม่ตัด)

# ---------- Prompt (student/prompts.py) ----------
MAX_CONTEXT_CHARS = 8000    # ความยาวรวมของเอกสารที่ใส่ใน prompt (context ของระบบคือ 8192 token)
