#!/usr/bin/env python3
"""Pack this project into a submission zip (Python 3.10+, stdlib only).

    python tools/pack.py                 # -> dist/submission_<student_id>.zip

Leaves out virtualenvs, caches, .git, .env and build output; refuses model files and
anything over the portal's limits, and checks manifest.json before you upload.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".idea", ".vscode", "dist",
             ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints", "data", "chroma_db", "index"}
SKIP_FILES = {".env", ".DS_Store", "Thumbs.db"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log"}
FORBIDDEN_SUFFIXES = {".gguf", ".safetensors", ".pt", ".pth", ".onnx", ".ckpt", ".h5", ".tflite"}
MAX_FILE_MB, MAX_ZIP_MB = 20, 50
TEMPLATE_ID = "6612345678"


def fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def collect(root: Path) -> tuple[list[Path], list[str]]:
    files, skipped = [], []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]) or (path.is_dir() and path.name in SKIP_DIRS):
            continue
        if path.is_symlink():
            skipped.append(f"{rel.as_posix()} (symlink)")
            continue
        if not path.is_file():
            continue
        if path.name in SKIP_FILES or path.suffix.lower() in SKIP_SUFFIXES:
            skipped.append(rel.as_posix())
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"{rel.as_posix()} looks like a model file — models must not be submitted")
        if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            fail(f"{rel.as_posix()} is larger than {MAX_FILE_MB}MB")
        files.append(path)
    return files, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output zip path (default dist/submission_<student_id>.zip)")
    args = ap.parse_args()

    if not (ROOT / "Dockerfile").is_file():
        fail("Dockerfile not found at the project root")
    try:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail("manifest.json not found at the project root")
    except json.JSONDecodeError as exc:
        fail(f"manifest.json is not valid JSON: {exc}")
    sid = str(manifest.get("student_id", "")).strip()
    missing = [k for k in ("student_id", "name", "llm_model", "embed_model") if not str(manifest.get(k, "")).strip()]
    if missing:
        fail(f"manifest.json is missing {', '.join(missing)}")
    if sid == TEMPLATE_ID:
        fail("manifest.json still has the example student_id — put your own student ID")

    files, skipped = collect(ROOT)
    out = Path(args.out) if args.out else ROOT / "dist" / f"submission_{sid}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, f"{sid}/{path.relative_to(ROOT).as_posix()}")

    size = out.stat().st_size
    if size > MAX_ZIP_MB * 1024 * 1024:
        out.unlink()
        fail(f"zip is {size / 1024 / 1024:.1f}MB — the portal accepts at most {MAX_ZIP_MB}MB")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"student : {sid} {manifest['name']}")
    print(f"models  : {manifest['llm_model']} / {manifest['embed_model']}")
    print(f"files   : {len(files)}" + (f"  (left out: {', '.join(skipped[:8])}{' …' if len(skipped) > 8 else ''})" if skipped else ""))
    print(f"zip     : {out}  ({size / 1024:.1f} KB)")
    print(f"sha256  : {digest}")
    print("Upload it on the portal: ส่งงาน → อัปโหลด zip")


if __name__ == "__main__":
    main()
