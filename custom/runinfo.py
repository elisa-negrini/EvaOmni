"""Files of a run folder: info.txt and output.jsonl (read and write).

info.txt is human-readable AND machine-readable.

The file starts with two sections, one "key : <JSON value>" line each:

  [run]        who/when/where (run name, description, start time, command, GPUs, git)
  [settings]   everything that defines the experiment; --resume refuses to continue
               a run if these differ (idea taken from scripts/eval_example.py,
               run_config.json)

A section ends at the first empty line. Lines appended later (Resumed, Finished,
Scored) are free text outside the sections.

Runs made before this format have a config.json instead; read_run_meta() falls
back to it so old run folders can still be scored and resumed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SECTION_RE = re.compile(r"^\[(\w+)\]")


def format_section(name: str, values: dict) -> str:
    width = max(len(k) for k in values)
    lines = [f"[{name}]"] + [f"{k.ljust(width)} : {json.dumps(v, ensure_ascii=False)}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


def parse_sections(text: str) -> dict[str, dict]:
    sections: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            cur = sections.setdefault(m.group(1), {})
            continue
        if cur is None:
            continue
        if not line.strip():
            cur = None
            continue
        key, sep, value = line.partition(" : ")
        if sep:
            cur[key.strip()] = json.loads(value)
    return sections


def read_run_meta(run_dir: Path) -> dict | None:
    """{"run": {...}, "settings": {...}} or None if the folder is not a run."""
    info = run_dir / "info.txt"
    if info.exists():
        sec = parse_sections(info.read_text(encoding="utf-8"))
        if "settings" in sec and "run" in sec:
            return {"run": sec["run"], "settings": sec["settings"]}
    legacy = run_dir / "config.json"
    if legacy.exists():
        cfg = json.loads(legacy.read_text(encoding="utf-8"))
        return {"run": cfg["runtime"], "settings": cfg["settings"]}
    return None


# ------------------------------------------------------------------ JSON layout
def pretty_json(obj, width: int = 100, indent: int = 1, _level: int = 0) -> str:
    """JSON that breaks lines only where needed: a dict/list that fits in `width`
    characters stays on one line, otherwise one item per line. Strings (e.g. the
    raw model output) are never split."""
    flat = json.dumps(obj, ensure_ascii=False)
    if not isinstance(obj, (dict, list)) or not obj or len(flat) + indent * _level <= width:
        return flat
    pad, end = " " * (indent * (_level + 1)), " " * (indent * _level)
    if isinstance(obj, dict):
        items = [f"{pad}{json.dumps(k, ensure_ascii=False)}: {pretty_json(v, width, indent, _level + 1)}"
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    if not any(isinstance(v, dict) for v in obj):  # e.g. intervals: several per line
        lines, cur = [], ""
        for part in (json.dumps(v, ensure_ascii=False) for v in obj):
            if cur and len(pad) + len(cur) + len(part) + 2 > width:
                lines.append(cur + ",")
                cur = ""
            cur = f"{cur}, {part}" if cur else part
        lines.append(cur)
        return "[\n" + "\n".join(pad + line for line in lines) + "\n" + end + "]"
    items = [pad + pretty_json(v, width, indent, _level + 1) for v in obj]
    return "[\n" + ",\n".join(items) + "\n" + end + "]"


# ------------------------------------------------------------------ output.jsonl
OUTPUT_FILE = "output.jsonl"


def append_record(path: Path, rec: dict) -> None:
    """Append one record (pretty-printed, so it spans several lines) and fsync.
    Several workers may write the same file: an exclusive lock keeps records whole.
    Mechanism of scripts/eval_example.py: append + fsync, the last record per qa_id wins."""
    import fcntl
    import os
    data = (pretty_json(rec) + "\n").encode("utf-8")
    with path.open("ab+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            end = f.seek(0, 2)
            if end:  # after a crash the file may end mid-record: start on a new line
                f.seek(end - 1)
                if f.read(1) != b"\n":
                    data = b"\n" + data
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_records(path: Path) -> list[dict]:
    """All records of a file with one JSON object after the other (one per line or
    pretty-printed). A record cut by a crash is skipped: reading resumes at the
    next line that starts with '{' (only records start at column 0)."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    dec, i, n, out = json.JSONDecoder(), 0, len(text), []
    while True:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, i = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            k = text.find("\n{", i + 1)
            if k < 0:
                break
            i = k + 1
    return out


def flatten(rec: dict) -> dict:
    """{..., "input": {...}, "output": {...}} -> one flat dict (what the scorer reads)."""
    flat = {k: v for k, v in rec.items() if k not in ("input", "output")}
    for part in ("input", "output"):
        flat.update(rec.get(part) or {})
    return flat


def load_outputs(run_dir: Path) -> dict[str, dict]:
    """Latest record per qa_id, flattened. Reads output.jsonl and, for runs made
    before 2026-11-16, responses_w*.jsonl + input_stats_w*.jsonl (output.jsonl wins)."""
    latest: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    for p in sorted(run_dir.glob("input_stats_w*.jsonl")):
        for r in read_records(p):
            stats[str(r["qa_id"])] = r
    for p in sorted(run_dir.glob("responses_w*.jsonl")):
        for r in read_records(p):
            latest[str(r["qa_id"])] = {**stats.get(str(r["qa_id"]), {}), **r}
    for r in read_records(run_dir / OUTPUT_FILE):
        latest[str(r["qa_id"])] = flatten(r)
    return latest
