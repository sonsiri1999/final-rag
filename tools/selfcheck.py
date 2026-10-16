#!/usr/bin/env python3
"""Self-check your service against the exam contract before submitting (Python 3.10+, stdlib only).

    python tools/selfcheck.py                                   # http://localhost:8000, sample pack
    python tools/selfcheck.py --url https://xxxx.trycloudflare.com
    python tools/selfcheck.py --no-run-tests                    # skip executing generated code

It checks every endpoint's response shape, scores the sample multiple-choice questions,
and runs the sample code tasks' unit tests locally (on YOUR machine, against code YOUR
service generated). Exit code 0 means the contract checks passed.
"""
import argparse
import ast
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

KEYS = ["ก", "ข", "ค", "ง", "จ", "ฉ", "ช", "ซ"]
TIMEOUTS = {"health": 10, "knowledge": 600, "delete": 30, "ask": 90, "code": 180}
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

TEST_RUNNER = r'''
import json, math, sys, copy
import ast
task = json.load(open(sys.argv[1], encoding="utf-8"))
source = open(sys.argv[2], encoding="utf-8").read()
banned = sorted({n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name) and n.id in task.get("forbidden", [])}
                | {"." + n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute) and n.attr in task.get("forbidden", [])})
if banned:
    print("__RESULT__" + json.dumps([{"ok": False, "args": c.get("args"), "error": "uses forbidden " + ", ".join(banned)} for c in task["tests"]]))
    sys.exit(0)
ns = {"__name__": "solution"}
exec(compile(source, "solution.py", "exec"), ns)
fn = ns[task["entry_point"]]
def same(g, e):
    if isinstance(e, bool) or isinstance(g, bool):
        return isinstance(g, bool) and isinstance(e, bool) and g == e
    if isinstance(e, (int, float)) and isinstance(g, (int, float)):
        return math.isclose(g, e, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(g, tuple): g = list(g)
    if isinstance(e, list) and isinstance(g, list):
        return len(g) == len(e) and all(same(x, y) for x, y in zip(g, e))
    return g == e
out = []
for case in task["tests"]:
    try:
        got = fn(*copy.deepcopy(case.get("args", [])), **copy.deepcopy(case.get("kwargs", {})))
        out.append({"ok": same(got, case["expected"]), "args": case.get("args"), "expected": case["expected"], "got": repr(got)[:120]})
    except Exception as exc:
        out.append({"ok": False, "args": case.get("args"), "error": f"{type(exc).__name__}: {exc}"[:200]})
print("__RESULT__" + json.dumps(out, ensure_ascii=False))
'''


class Fail(Exception):
    pass


def call(base: str, method: str, path: str, kind: str, body: dict | None = None) -> tuple[dict, float]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "User-Agent": "selfcheck/1.0"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUTS[kind]) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise Fail(f"{method} {path}: HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise Fail(f"{method} {path}: {exc}")
    elapsed = time.monotonic() - start
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise Fail(f"{method} {path}: response is not JSON: {raw[:200]}")
    if not isinstance(payload, dict):
        raise Fail(f"{method} {path}: response must be a JSON object")
    return payload, elapsed


def load_pack(pack_dir: Path) -> dict:
    meta = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    docs = [{"doc_id": k["doc_id"], "content": (pack_dir / k["file"]).read_text(encoding="utf-8"),
             "format": k.get("format", "markdown")} for k in meta["knowledge"]]
    questions = []
    for n, item in enumerate(json.loads((pack_dir / meta["mc"]).read_text(encoding="utf-8")), 1):
        texts = [re.sub(rf"^\s*{KEYS[i]}[.)]\s*", "", o["text"], count=1).strip() for i, o in enumerate(item["options"])]
        correct = [o["is_correct"] for o in item["options"]].index(True)
        order = list(range(len(texts)))
        random.Random(f"{meta.get('shuffle_seed')}:{meta['id']}:{n}").shuffle(order)
        questions.append({
            "question_id": f"{meta['id']}-q{n:02d}",
            "question": item["text"],
            "options": [{"key": KEYS[p], "text": texts[o]} for p, o in enumerate(order)],
            "correct_key": KEYS[order.index(correct)],
        })
    tasks = json.loads((pack_dir / meta["code"]).read_text(encoding="utf-8")) if meta.get("code") else []
    return {"meta": meta, "documents": docs, "questions": questions, "tasks": tasks}


def run_unit_tests(task: dict, code: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "task.json").write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
        (tmp / "solution.py").write_text(code, encoding="utf-8")
        (tmp / "runner.py").write_text(TEST_RUNNER, encoding="utf-8")
        limit = float(task.get("timeout_sec", 5)) * len(task["tests"]) + 10
        try:
            proc = subprocess.run([sys.executable, str(tmp / "runner.py"), str(tmp / "task.json"), str(tmp / "solution.py")],
                                  capture_output=True, text=True, encoding="utf-8", timeout=limit, cwd=tmp)
        except subprocess.TimeoutExpired:
            return [{"ok": False, "error": f"timeout after {limit:.0f}s"}]
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("__RESULT__"):
                return json.loads(line[len("__RESULT__"):])
        return [{"ok": False, "error": (proc.stderr.strip().splitlines() or ["no output"])[-1][:300]}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--pack", default=str(Path(__file__).resolve().parent / "sample_pack"))
    ap.add_argument("--no-run-tests", action="store_true")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    pack = load_pack(Path(args.pack))
    ns = pack["meta"]["namespace"]
    problems: list[str] = []

    print(f"== {base}  (pack: {pack['meta']['id']})")
    try:
        health, _ = call(base, "GET", "/health", "health")
        missing = [f for f in ("status", "student_id", "llm_model", "embed_model") if f not in health]
        if missing or health.get("status") != "ok":
            problems.append(f"/health: missing {missing} or status != 'ok'")
        print(f"[health] {health}")

        call(base, "DELETE", f"/knowledge/{ns}", "delete")
        data, sec = call(base, "POST", "/knowledge", "knowledge", {"namespace": ns, "documents": pack["documents"]})
        if not isinstance(data.get("chunks"), int):
            problems.append("/knowledge: 'chunks' must be an integer")
        print(f"[knowledge] {data.get('chunks')} chunks in {sec:.1f}s")
    except Fail as exc:
        print(f"FAIL: {exc}")
        return 1

    correct = 0
    for q in pack["questions"]:
        body = {"namespace": ns, "question_id": q["question_id"], "question": q["question"], "options": q["options"]}
        try:
            data, sec = call(base, "POST", "/ask", "ask", body)
        except Fail as exc:
            problems.append(str(exc))
            print(f"[ask] {q['question_id']}: FAIL {exc}")
            continue
        answer = data.get("answer")
        if answer not in [o["key"] for o in q["options"]]:
            problems.append(f"/ask {q['question_id']}: answer {answer!r} is not an option key")
        if not isinstance(data.get("sources", []), list):
            problems.append(f"/ask {q['question_id']}: 'sources' must be a list")
        ok = answer == q["correct_key"]
        correct += ok
        print(f"[ask] {q['question_id']}: {answer} (key {q['correct_key']}) {'OK' if ok else 'WRONG'}  {sec:.1f}s")
        if not ok:  # show what retrieval found: answer missing from sources = chunking/retrieval, present = prompt
            for src in (data.get("sources") or [])[:3]:
                snippet = " ".join(str(src.get("text", "")).split())[:90]
                print(f"         source {src.get('doc_id')} score={src.get('score')}: {snippet}")
    print(f"[ask] score {correct}/{len(pack['questions'])}")

    for task in pack["tasks"]:
        body = {k: task.get(k) for k in ("task_id", "requirement", "language", "entry_point", "signature")}
        body["namespace"] = ns if task.get("use_knowledge") else None
        try:
            data, sec = call(base, "POST", "/code", "code", body)
        except Fail as exc:
            problems.append(str(exc))
            print(f"[code] {task['task_id']}: FAIL {exc}")
            continue
        raw = data.get("code")
        if not isinstance(raw, str) or not raw.strip():
            problems.append(f"/code {task['task_id']}: 'code' must be a non-empty string")
            continue
        match = CODE_BLOCK.search(raw)
        code = (match.group(1) if match else raw).strip() + "\n"
        try:
            ast.parse(code)
        except SyntaxError as err:
            print(f"[code] {task['task_id']}: SyntaxError {err.msg} (line {err.lineno})  {sec:.1f}s")
            continue
        if args.no_run_tests:
            print(f"[code] {task['task_id']}: syntax OK  {sec:.1f}s")
            continue
        cases = run_unit_tests(task, code)
        passed = sum(1 for c in cases if c.get("ok"))
        print(f"[code] {task['task_id']}: {passed}/{len(task['tests'])} tests  {sec:.1f}s")
        for c in cases:
            if not c.get("ok"):
                print(f"         args={c.get('args')} expected={c.get('expected')} got={c.get('got', c.get('error'))}")

    try:
        call(base, "DELETE", f"/knowledge/{ns}", "delete")
    except Fail as exc:
        problems.append(str(exc))

    if problems:
        print("\nContract problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nContract checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
