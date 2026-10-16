# 🔒 ห้ามแก้ไฟล์นี้ (ระบบส่งงานตรวจ checksum) — ถ้าต้องใช้ library เพิ่ม ให้เพิ่มใน requirements.txt
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core ./core
COPY student ./student
COPY manifest.json .

RUN useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin appuser
USER 1000

EXPOSE 8000
CMD ["uvicorn", "core.server:app", "--host", "0.0.0.0", "--port", "8000"]
