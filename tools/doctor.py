#!/usr/bin/env python3
"""🩺 ตรวจงานก่อนส่ง — คำสั่งเดียวจบ (Python 3.10+, ใช้แค่ standard library)

    python tools/doctor.py           ตรวจครบทุกขั้น (ต้องเปิด Docker Desktop และ Ollama) แล้วสร้าง zip ให้
    python tools/doctor.py --quick   ตรวจเฉพาะไฟล์ (ไม่ build และไม่รัน)

สิ่งที่ตรวจ:
  1. manifest.json        รหัสนักศึกษา ชื่อ และโมเดล
  2. core/ และ Dockerfile ไม่ถูกแก้ (ถ้าแก้ ระบบส่งงานจะไม่รับ)
  3. student/             compile ผ่าน และยังเหลือ TODO กี่จุด
  4. ไฟล์ต้องห้าม         ไฟล์โมเดล หรือไฟล์ใหญ่เกิน
  5. Ollama               เปิดอยู่ และมีโมเดลที่ระบุใน manifest
  6. Docker               build ผ่าน, /health ตอบภายใน 3 นาที, selfcheck ผ่าน, ไม่มี error จาก student/
  7. สร้าง zip            dist/submission_<รหัส>.zip พร้อมอัปโหลด
"""
import argparse
import hashlib
import json
import py_compile
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_ID = "6612345678"
ALLOWED_LLM = {"qwen2.5-coder:7b", "qwen2.5-coder:14b"}
ALLOWED_EMBED = {"nomic-embed-text"}
OLLAMA = "http://localhost:11434"

results: list[tuple[str, str, str]] = []  # (status, step, message)


def ok(step, msg=""):
    results.append(("✅", step, msg))
    print(f"✅ {step} {msg}".rstrip())


def warn(step, msg):
    results.append(("⚠️", step, msg))
    print(f"⚠️  {step}: {msg}")


def fail(step, msg):
    results.append(("❌", step, msg))
    print(f"❌ {step}: {msg}")


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def http_json(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ------------------------------------------------------------------ static checks
def check_manifest() -> dict | None:
    try:
        m = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail("manifest.json", "ไม่พบไฟล์")
        return None
    except json.JSONDecodeError as exc:
        fail("manifest.json", f"JSON ไม่ถูกต้อง: {exc}")
        return None
    sid = str(m.get("student_id", "")).strip()
    problems = []
    if sid == TEMPLATE_ID:
        problems.append("ยังเป็นรหัสตัวอย่าง ให้แก้ student_id เป็นรหัสของตัวเอง")
    elif not re.fullmatch(r"[0-9]{8,13}", sid):
        problems.append(f"student_id ไม่ถูกรูปแบบ: {sid!r}")
    if not str(m.get("name", "")).strip() or m.get("name") == "ชื่อ นามสกุล":
        problems.append("ยังไม่ได้ใส่ชื่อ-นามสกุล")
    if m.get("llm_model") not in ALLOWED_LLM:
        problems.append(f"llm_model ต้องเป็น {' หรือ '.join(sorted(ALLOWED_LLM))}")
    if m.get("embed_model") not in ALLOWED_EMBED:
        problems.append(f"embed_model ต้องเป็น {', '.join(sorted(ALLOWED_EMBED))}")
    if problems:
        fail("manifest.json", "; ".join(problems))
    else:
        ok("manifest.json", f"({sid} {m['name']}, {m['llm_model']})")
    return m


def check_core() -> None:
    ref_path = ROOT / "core" / "CHECKSUMS.json"
    if not ref_path.exists():
        fail("core/", "ไม่พบ core/CHECKSUMS.json — ให้ดาวน์โหลด starter ใหม่")
        return
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    changed = [name for name, digest in ref.items()
               if not (ROOT / name).exists() or sha256_lf(ROOT / name) != digest]
    extra = [p.relative_to(ROOT).as_posix() for p in (ROOT / "core").glob("*.py")
             if p.relative_to(ROOT).as_posix() not in ref]
    if changed or extra:
        fail("core/ และ Dockerfile", f"ถูกแก้หรือเพิ่มไฟล์: {', '.join(changed + extra)} — "
                                    "ระบบส่งงานจะไม่รับ ให้คัดลอกไฟล์เดิมจาก starter กลับมา")
    else:
        ok("core/ และ Dockerfile", "(ไม่ถูกแก้)")


def check_student() -> list[str]:
    todo, broken = [], []
    for path in sorted((ROOT / "student").glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True, cfile=str(Path(tempfile.gettempdir()) / "doctor_tmp.pyc"))
        except py_compile.PyCompileError as exc:
            broken.append(f"{path.name}: {exc.msg.strip().splitlines()[-1]}")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*#\s*TODO HERE", line):
                todo.append(f"{path.name}:{n} {line.strip().lstrip('#').strip()}")
    (Path(tempfile.gettempdir()) / "doctor_tmp.pyc").unlink(missing_ok=True)
    if broken:
        fail("student/", "compile ไม่ผ่าน — " + "; ".join(broken))
    elif todo:
        warn("student/", f"ยังเหลือ TODO {len(todo)} จุด (ส่งได้ แต่จุดที่ยังไม่ทำจะใช้ค่าตั้งต้นที่คะแนนต่ำ):\n      "
             + "\n      ".join(todo))
    else:
        ok("student/", "(compile ผ่าน และไม่เหลือ TODO)")
    return todo


def check_files() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import pack  # noqa: E402 — same rules as the zip builder
    try:
        files, _ = pack.collect(ROOT)
    except SystemExit as exc:
        fail("ไฟล์ในโปรเจค", str(exc).replace("ERROR: ", ""))
        return
    size = sum(f.stat().st_size for f in files)
    ok("ไฟล์ในโปรเจค", f"({len(files)} ไฟล์, {size / 1024:.0f} KB, ไม่มีไฟล์โมเดล)")


# ------------------------------------------------------------------ runtime checks
def check_ollama(manifest: dict) -> bool:
    try:
        tags = http_json(f"{OLLAMA}/api/tags")
    except OSError:
        fail("Ollama", f"ติดต่อ {OLLAMA} ไม่ได้ — เปิดแอป Ollama ก่อน")
        return False
    have = {m["name"] for m in tags.get("models", [])} | {m["name"].split(":")[0] for m in tags.get("models", [])
                                                         if m["name"].endswith(":latest")}
    missing = [m for m in (manifest.get("llm_model"), manifest.get("embed_model")) if m and m not in have]
    if missing:
        fail("Ollama", "ยังไม่มีโมเดล: " + ", ".join(missing) + " — รัน " + " และ ".join(f"ollama pull {m}" for m in missing))
        return False
    ok("Ollama", "(มีโมเดลครบ)")
    return True


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(cmd: list[str], env_port: int | None = None, timeout: float = 900) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ)
    if env_port:
        env["APP_PORT"] = str(env_port)
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


def check_docker(manifest: dict) -> bool:
    if run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=30).returncode != 0:
        fail("Docker", "ใช้ Docker ไม่ได้ — เปิด Docker Desktop ก่อน")
        return False
    project = f"doctor-{manifest.get('student_id', 'x')}"
    compose = ["docker", "compose", "-p", project]
    print("   ⏳ docker build (ครั้งแรกอาจใช้เวลาหลายนาที)…")
    build = run([*compose, "build"], timeout=1800)
    if build.returncode != 0:
        fail("Docker build", "ไม่ผ่าน:\n" + "\n".join(build.stderr.strip().splitlines()[-12:]))
        return False
    ok("Docker build")

    port = free_port()
    up = run([*compose, "up", "-d", "app"], env_port=port, timeout=300)
    if up.returncode != 0:
        fail("Docker run", up.stderr.strip()[-600:])
        return False
    passed = False
    try:
        print(f"   ⏳ รอ /health ที่พอร์ต {port} (ไม่เกิน 3 นาที)…")
        start, health = time.time(), None
        while time.time() - start < 180:
            try:
                health = http_json(f"http://127.0.0.1:{port}/health")
                break
            except OSError:
                time.sleep(2)
        if health is None:
            logs = run([*compose, "logs", "--tail", "25", "app"]).stdout
            fail("/health", "service ไม่ตอบภายใน 3 นาที — log ล่าสุด:\n" + logs)
            return False
        if str(health.get("student_id")) != str(manifest.get("student_id")):
            fail("/health", f"student_id ไม่ตรงกับ manifest ({health.get('student_id')})")
            return False
        ok("/health", f"(ตอบใน {time.time() - start:.0f} วินาที)")

        print("   ⏳ รัน selfcheck (ประมาณ 1–3 นาที)…")
        check = run([sys.executable, "tools/selfcheck.py", "--url", f"http://127.0.0.1:{port}"], timeout=1800)
        lines = [l for l in check.stdout.splitlines() if l.startswith(("[ask] score", "[code]", "  - "))]
        if check.returncode == 0:
            ok("selfcheck", "\n      " + "\n      ".join(lines))
            passed = True
        else:
            fail("selfcheck", "ไม่ผ่าน:\n      " + "\n      ".join(check.stdout.strip().splitlines()[-15:]))

        functions = http_json(f"http://127.0.0.1:{port}/health").get("student_functions", {})
        errors = {n: s.get("message") for n, s in functions.items() if s.get("state") in ("error", "invalid")}
        if errors:
            warn("ฟังก์ชันใน student/", "มี error ระหว่างทำงาน (ระบบใช้ค่าตั้งต้นแทน):\n      "
                 + "\n      ".join(f"{n}: {m}" for n, m in errors.items())
                 + "\n      ดู traceback ได้ด้วย: docker compose logs app")
        return passed
    finally:
        run([*compose, "down"], timeout=120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="ตรวจเฉพาะไฟล์ ไม่ build และไม่รัน")
    args = ap.parse_args()
    print(f"🩺 doctor — {ROOT}\n")

    manifest = check_manifest()
    check_core()
    check_student()
    check_files()
    if not args.quick and manifest and not any(s == "❌" for s, _, _ in results):
        if check_ollama(manifest):
            check_docker(manifest)

    errors = [r for r in results if r[0] == "❌"]
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ ยังส่งไม่ได้ — แก้ {len(errors)} เรื่องด้านบนก่อน")
        sys.exit(1)
    if args.quick:
        print("✅ ไฟล์พร้อม — รัน `python tools/doctor.py` (ไม่มี --quick) เพื่อตรวจแบบเต็มก่อนส่ง")
        return
    out = run([sys.executable, "tools/pack.py"], timeout=120)
    print(out.stdout.strip() or out.stderr.strip())
    print("✅ พร้อมส่ง — อัปโหลดไฟล์ zip ด้านบนที่เมนู \"ส่งงาน\" หรือ push ขึ้น GitHub แล้วส่ง URL")


if __name__ == "__main__":
    main()
