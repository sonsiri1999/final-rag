"""✏️ student/ — ส่วนที่นักศึกษาเขียนเอง

มี 4 ไฟล์ ทำทีละไฟล์ตามลำดับได้เลย (ค้นหาคำว่า "TODO HERE")
  1. chunking.py   แบ่งเอกสารเป็นชิ้น
  2. embedding.py  แปลงข้อความเป็น vector
  3. retrieval.py  ค้นหาชิ้นที่เกี่ยวข้องกับคำถาม
  4. prompts.py    เขียน prompt ให้ LLM ตอบข้อสอบ และเขียนโค้ด

ฟังก์ชันที่ยังเป็น `pass` (คืน None) จะถูกแทนด้วยค่าตั้งต้นใน core/fallbacks.py ซึ่งไม่ได้แก้ปัญหาอะไรเลย
ระบบยังรันได้และตอบถูกรูปแบบ แต่คะแนนเกือบ 0 คะแนนทุกคะแนนจึงมาจากสิ่งที่คุณเขียน
ดูสถานะของแต่ละฟังก์ชันได้ที่ http://localhost:8000/health (หัวข้อ "student_functions")
และดู error ได้จาก log ของ container: docker compose logs -f app

สร้างไฟล์ .py เพิ่มในโฟลเดอร์นี้ได้ และเพิ่ม library ใน requirements.txt ได้
แต่ห้ามแก้ไฟล์ใน core/ และ Dockerfile
"""
