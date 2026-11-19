#!/usr/bin/env python3
"""Score EvaOmni run outputs with the authors' own metric code.

Input: the records written by our adapter (output.jsonl: one or more records
per qa_id, the LAST one wins; the fields of "input"/"output" are read as if they
were top-level). Required fields per record:
    qa_id         int/str, matching data/test_file.json
    status        "ok" if inference returned text, anything else = failure
    raw_response  the model's raw text output

Two scoring modes, both delegating the metric maths to
scripts/eval_example.py::compute_grounded_scores (imported, not copied):

  strict      raw text is parsed with eval_example.parse_response: the whole
              prediction must be a valid JSON object with a letter A-H and two
              evidence lists whose timestamps are all "[mm:ss,mm:ss]" with
              end > start. Anything malformed -> the question scores zero.

  permissive  raw text is parsed with eval_gemini.parse_model_output (JSON,
              JSON embedded in prose, or a bare letter as last resort); each
              evidence interval is kept only if parse_evidence_interval can
              read it, unreadable intervals are dropped individually.

Modality ablation runs (input_mode v / a / t in the run's [settings]): the evidence
lists that the input's prompt does not ask for are scored as empty, in every mode
(strict: the JSON is re-validated with those lists set to []). So C.G. <= Acc/2 and
B.G. = 0 by construction; evaluation.txt adds the grounding of the given modality alone.

Questions missing from the results file, or with a failed status, score zero
and stay in the denominator (as in the paper, App. D.3).

Also reported: an "always A" baseline, the chance baseline computed per
question as 1/n_options, and Wilson 95% confidence intervals for accuracy.

With --run-dir it reads the run's output.jsonl and writes the folder
<run>/evaluation/:
  results.json     summary per mode and level (all metrics, 95% CI), per-capability
                   metrics, baselines, diagnostics, input statistics, and one entry
                   per question (ground truth, prediction and scores in every mode)
  evaluation.txt   readable summary: every mode vs the paper's Table 2, baselines,
                   diagnostics
It also appends a line to info.txt and rebuilds custom/output/RUNS.md.

Usage:
    python custom/score.py --run-dir custom/output/<run_name>
    python custom/score.py --results <file.jsonl> \
        [--annotation data/test_file.json] [--out <results.json>]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runinfo import flatten, load_outputs, pretty_json, read_records, read_run_meta  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Original, unmodified author code.
eval_example = _load_module("evaomni_eval_example", REPO / "scripts" / "eval_example.py")
eval_gemini = _load_module("evaomni_eval_gemini", REPO / "scripts" / "eval_gemini.py")
compute_grounded_scores = eval_example.compute_grounded_scores
parse_evidence_interval = eval_example.parse_evidence_interval

MAIN_METRICS = ("Acc.", "C.G.", "B.G.")
ALL_METRICS = ("Acc.", "C.G.", "B.G.", "Mean", "Thr@0.3", "Thr@0.5", "Thr@0.7")
_OPTION_RE = re.compile(r"(?m)^\s*([A-Z])\.\s")


# ---------------------------------------------------------------- loading
def load_annotations(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_latest(results_path: Path) -> dict[str, dict]:
    """Last record per qa_id (records may be nested as in output.jsonl or flat)."""
    return {str(r["qa_id"]): flatten(r) for r in read_records(results_path) if r.get("qa_id") is not None}


# ---------------------------------------------------------------- parsing
def parse_strict(raw: str, qa_id) -> tuple[dict | None, str]:
    """eval_example.parse_response: all-or-nothing."""
    try:
        return eval_example.parse_response(raw, qa_id), ""
    except Exception as exc:  # noqa: BLE001 - any failure means "malformed"
        return None, f"{type(exc).__name__}: {exc}"


FREE_TEXT_ERR = "non-JSON output, only letter extracted"  # message emitted by eval_gemini


def _json_object(raw: str):
    """Locate the JSON object exactly as eval_gemini.parse_model_output does."""
    s = eval_gemini._strip_markdown_fence(raw or "")
    for candidate in (s, (eval_gemini._JSON_OBJ_RE.search(s) or [None])[0]):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict):
            return obj
    return None


def answer_source(raw: str) -> str:
    """How the permissive parser obtains the letter (diagnostic only).

    json_letter        JSON object whose "answer" is a single letter A-H (any case)
    json_nonstandard   JSON object whose "answer" is something else; the letter is
                       the first isolated A-H in that field (may be the article "a")
    truncated_json     no complete JSON object (typically cut by max_tokens), but
                       an '"answer": "X"' field is readable and equals the letter
                       the fallback picked; kept in permissive_no_free_text
    free_text          no JSON object; the letter is the first isolated A-H
                       anywhere in the raw text (very likely the article "a")
    none               nothing recoverable
    """
    obj = _json_object(raw)
    if obj is not None:
        ans = str(obj.get("answer", "")).strip()
        if re.fullmatch(r"[A-Ha-h]", ans):
            return "json_letter"
        return "json_nonstandard" if eval_gemini._LETTER_RE.search(ans) or ans else "none"
    parsed, err = eval_gemini.parse_model_output(raw or "")
    if err != FREE_TEXT_ERR:
        return "none"
    # JSON cut off (e.g. max_tokens reached) whose "answer" field is readable and
    # is exactly the letter the fallback picked: not a spurious article "a".
    m = re.search(r'"answer"\s*:\s*"\s*([A-Ha-h])\s*"', raw or "")
    if m and m.group(1).upper() == parsed.get("answer"):
        return "truncated_json"
    return "free_text"


def parse_permissive(raw: str, qa_id) -> tuple[dict | None, str]:
    """eval_gemini.parse_model_output + per-interval salvage."""
    parsed, err = eval_gemini.parse_model_output(raw or "")
    answer = parsed.get("answer", "") if parsed else ""
    if not answer:
        return None, err or "no answer letter found"
    prediction = {"qa_id": qa_id, "answer": answer}
    dropped = 0
    for key in ("visual_evidence", "audio_evidence"):
        intervals = []
        for item in parsed.get(key, []):
            try:
                intervals.append(list(parse_evidence_interval(item.get("time", ""))))
            except Exception:  # noqa: BLE001
                dropped += 1
        prediction[key] = intervals
    note = err or ""
    if dropped:
        note = (note + "; " if note else "") + f"dropped {dropped} unreadable interval(s)"
    return prediction, note


def parse_permissive_no_free_text(raw: str, qa_id) -> tuple[dict | None, str]:
    """Permissive, but a letter recovered from non-JSON free text scores zero."""
    if answer_source(raw) == "free_text":
        return None, "free-text recovery excluded"
    return parse_permissive(raw, qa_id)


_TIME = r"\d+(?::\d{2}){1,2}(?:\.\d+)?"
_BARE_INTERVAL_RE = re.compile(rf"\s*({_TIME})\s*,\s*({_TIME})\s*")
_STRICT_INTERVAL_RE = re.compile(r"\[(\d{2,}):([0-5]\d),(\d{2,}):([0-5]\d)\]")
_SINGLE_POINT_RE = re.compile(rf"\s*\[?\s*{_TIME}\s*\]?\s*")


def add_missing_brackets(t: str) -> str:
    """'mm:ss,mm:ss' -> '[mm:ss,mm:ss]'. Anything else is returned unchanged, so
    single points and zero-length intervals stay invalid."""
    m = _BARE_INTERVAL_RE.fullmatch(t or "")
    return f"[{m.group(1)},{m.group(2)}]" if m else t


def parse_diag_no_brackets(raw: str, qa_id) -> tuple[dict | None, str]:
    """DIAGNOSTIC ONLY: permissive parsing, plus intervals written without the
    square brackets are accepted. Not a main score."""
    parsed, err = eval_gemini.parse_model_output(raw or "")
    answer = parsed.get("answer", "") if parsed else ""
    if not answer:
        return None, err or "no answer letter found"
    prediction, dropped = {"qa_id": qa_id, "answer": answer}, 0
    for key in ("visual_evidence", "audio_evidence"):
        intervals = []
        for item in parsed.get(key, []):
            try:
                intervals.append(list(parse_evidence_interval(add_missing_brackets(item.get("time", "")))))
            except Exception:  # noqa: BLE001
                dropped += 1
        prediction[key] = intervals
    return prediction, (f"dropped {dropped} unreadable interval(s)" if dropped else "")


PARSERS = {"strict": parse_strict, "permissive": parse_permissive,
           "permissive_no_free_text": parse_permissive_no_free_text,
           "diag_no_brackets": parse_diag_no_brackets}
MODES = tuple(PARSERS)
MAIN_MODES = ("strict", "permissive", "permissive_no_free_text")
DIAGNOSTIC_MODES = ("diag_no_brackets",)


# ---------------------------------------------------------------- format errors
def classify_timestamp(t: str, video_seconds: float | None) -> list[str]:
    """Format categories of one evidence timestamp (a list: e.g. a strict interval
    can also lie beyond the end of the video)."""
    t = "" if t is None else str(t)
    if _STRICT_INTERVAL_RE.fullmatch(t):
        cats, candidate = ["strict_ok"], t
    elif _BARE_INTERVAL_RE.fullmatch(t):
        cats, candidate = ["missing_brackets"], add_missing_brackets(t)
    elif _SINGLE_POINT_RE.fullmatch(t):
        return ["single_point"]
    else:
        try:  # readable by the authors' parser although not strict (e.g. decimals, h:mm:ss)
            parse_evidence_interval(t)
            cats, candidate = ["readable_non_strict"], t
        except Exception:  # noqa: BLE001
            return ["other_unparseable"]
    m = re.fullmatch(r"\[\s*([^,]+),\s*([^,]+)\s*\]", candidate)
    try:
        start, end = parse_evidence_interval(candidate)
    except Exception:  # noqa: BLE001
        if m:
            try:
                a, b = (parse_evidence_interval(f"[{m.group(1)},99999]")[0],
                        parse_evidence_interval(f"[{m.group(2)},99999]")[0])
                return cats + ["zero_length" if a == b else "end_before_start"]
            except Exception:  # noqa: BLE001
                pass
        return cats + ["other_unparseable"]
    if video_seconds is not None and end > video_seconds + 1.0:
        cats.append("beyond_video_end")
    return cats


def format_report(records: list[dict], results: dict[str, dict],
                  video_seconds: dict[str, float] | None = None) -> dict:
    """Counts of format problems: per question and per timestamp, with example qa_ids."""
    q_counts: dict[str, int] = {}
    t_counts: dict[str, int] = {}
    examples: dict[str, list] = {}

    def add(d, key, qa):
        d[key] = d.get(key, 0) + 1
        ex = examples.setdefault(key, [])
        if len(ex) < 10 and qa not in ex:
            ex.append(qa)

    for rec in records:
        qa = rec["qa_id"]
        res = results.get(str(qa))
        if not res or res.get("status") != "ok":
            add(q_counts, "no_output_or_infer_error", qa)
            continue
        raw = res.get("raw_response", "")
        qcats = set()
        cause = res.get("stop_cause") or ("length" if res.get("finish_reason") == "length" else "natural_stop")
        if cause != "natural_stop":
            qcats.add("generation_cut_" + cause)
        src = answer_source(raw)
        if src == "truncated_json":
            qcats.add("json_truncated")
        elif src == "free_text":
            qcats.add("no_json_free_text")
        elif src == "json_nonstandard":
            qcats.add("answer_not_single_letter")
        elif src == "none":
            qcats.add("no_answer")
        obj = _json_object(raw)
        if obj is not None:
            for key in ("visual_evidence", "audio_evidence"):
                items = obj.get(key)
                if not isinstance(items, list):
                    qcats.add(f"missing_{key}_list")
                    continue
                if not items:
                    qcats.add(f"empty_{key}_list")
                for it in items:
                    t = it.get("time") if isinstance(it, dict) else it
                    for c in classify_timestamp(t, (video_seconds or {}).get(rec["video_id"])):
                        add(t_counts, c, qa)
                        if c != "strict_ok":
                            qcats.add("has_" + c)
        if not qcats:
            qcats.add("fully_well_formed")
        for c in sorted(qcats):
            add(q_counts, c, qa)
    return {"n_questions": len(records),
            "questions_with": dict(sorted(q_counts.items(), key=lambda kv: -kv[1])),
            "timestamps": dict(sorted(t_counts.items(), key=lambda kv: -kv[1])),
            "example_qa_ids": examples}


# Modality ablation runs (run_qwen_omni.py --input-mode v/a/t): the repository's
# prompt for that input asks only for some evidence lists. The lists it does NOT
# ask for are scored as empty, whatever the model wrote there (coverage and IoU 0
# for that modality, so C.G. <= Acc/2 and B.G. = 0, consistent with the paper's
# Table 4 where audio-only / video-only rows have B.G. 0.0).
UNASKED_BY_INPUT = {"v": ("audio_evidence",), "a": ("visual_evidence",),
                    "t": ("visual_evidence", "audio_evidence")}
GIVEN_MODALITY = {"v": "visual", "a": "audio"}


def fill_unasked(raw: str, unasked) -> str:
    """Strict parsing of a unimodal answer: the JSON (same fence handling as
    eval_example.parse_response) with the unasked lists set to [], re-serialised.
    Anything that is not a JSON object is returned unchanged (it stays unparsable)."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        return raw
    if not isinstance(obj, dict):
        return raw
    return json.dumps({**obj, **{k: [] for k in unasked}})


def build_latest_for_scorer(records: list[dict], results: dict[str, dict], mode: str, unasked=()):
    """Convert our results into the {qa_id: {'status','prediction'}} shape
    expected by compute_grounded_scores. Also return parse diagnostics.
    unasked: evidence lists scored as empty (modality ablation runs)."""
    parser = PARSERS[mode]
    latest, diag = {}, {"ok": 0, "infer_error_or_missing": 0, "parse_fail": 0, "notes": {}}
    for rec in records:
        key = str(rec["qa_id"])
        res = results.get(key)
        if not res or res.get("status") != "ok":
            diag["infer_error_or_missing"] += 1
            continue
        raw = res.get("raw_response", "")
        if unasked and mode == "strict":
            raw = fill_unasked(raw, unasked)
        pred, note = parser(raw, rec["qa_id"])
        if pred is not None and unasked:
            pred.update({k: [] for k in unasked})
        if note:
            diag["notes"][key] = note
        if pred is None:
            diag["parse_fail"] += 1
            continue
        diag["ok"] += 1
        latest[key] = {"status": "ok", "prediction": pred}
    return latest, diag


# ---------------------------------------------------------------- stats
def wilson_ci(successes: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def n_options(rec: dict) -> int:
    return len(_OPTION_RE.findall(rec["options"])) or 8


def summarise(grounded: dict) -> dict:
    """Overall + per-level metrics with Wilson CI on accuracy."""
    rows = grounded["per_question"]

    def block(subset):
        n = len(subset)
        out = {"n": n}
        for m in ALL_METRICS:
            out[m] = 100 * math.fsum(r["scores"][m] for r in subset) / n if n else 0.0
        correct = sum(r["scores"]["Acc."] for r in subset)
        out["Acc. 95% CI"] = wilson_ci(correct, n)
        return out

    levels = sorted({str(r["level"]) for r in rows})
    result = {f"Level {lv}": block([r for r in rows if str(r["level"]) == lv]) for lv in levels}
    result["Overall"] = block(rows)
    return result


def _level_groups(records):
    groups = {"Overall": records}
    for lv in sorted({str(r["level"]) for r in records}):
        groups[f"Level {lv}"] = [r for r in records if str(r["level"]) == lv]
    return groups


def letter_distribution(records, letter_of) -> dict:
    """{group: {letter: count}}; letter_of(rec) -> letter or 'none'."""
    out = {}
    for name, subset in _level_groups(records).items():
        counts: dict[str, int] = {}
        for r in subset:
            k = letter_of(r) or "none"
            counts[k] = counts.get(k, 0) + 1
        out[name] = dict(sorted(counts.items()))
    return out


def answer_source_report(records, results) -> dict:
    """Where permissive answers come from; free-text letters and their accuracy."""
    by_source: dict[str, list] = {}
    for r in records:
        res = results.get(str(r["qa_id"]))
        if not res or res.get("status") != "ok":
            src, letter = "no_output", ""
        else:
            raw = res.get("raw_response", "")
            src = answer_source(raw)
            parsed, _ = eval_gemini.parse_model_output(raw or "")
            letter = (parsed or {}).get("answer", "")
        by_source.setdefault(src, []).append((r, letter))
    report = {}
    for src, items in sorted(by_source.items()):
        letters: dict[str, int] = {}
        for _, l in items:
            letters[l or "none"] = letters.get(l or "none", 0) + 1
        correct = sum(1 for r, l in items if l and l == r["answer"])
        report[src] = {"n": len(items), "correct": correct,
                       "acc_within_group": 100 * correct / len(items) if items else 0.0,
                       "letters": dict(sorted(letters.items())),
                       "qa_ids": [r["qa_id"] for r, _ in items] if src in ("free_text", "json_nonstandard") else None}
    return report


def score_mode(records, results, mode, unasked=()):
    latest, diag = build_latest_for_scorer(records, results, mode, unasked)
    grounded = compute_grounded_scores(records, latest)
    return {"summary": summarise(grounded),
            "modal": {str(r["qa_id"]): {k: r[k] for k in ("audio_coverage", "visual_coverage",
                                                         "audio_iou", "visual_iou")}
                      for r in grounded["per_question"]},
            "parsing": {"parsed": diag["ok"], "unparsable": diag["parse_fail"],
                        "failed_or_missing": diag["infer_error_or_missing"]},
            "by_capability": grounded["by_capability"],
            "scores": {str(r["qa_id"]): r["scores"] for r in grounded["per_question"]},
            "preds": {k: v["prediction"] for k, v in latest.items()}}


def baselines(records):
    always_a = {str(r["qa_id"]): {"status": "ok", "prediction": {
        "qa_id": r["qa_id"], "answer": "A", "audio_evidence": [], "visual_evidence": []}}
        for r in records}
    a_summary = summarise(compute_grounded_scores(records, always_a))

    def chance(subset):
        return {"n": len(subset),
                "Acc.": 100 * math.fsum(1 / n_options(r) for r in subset) / len(subset)}

    levels = sorted({str(r["level"]) for r in records})
    ch = {f"Level {lv}": chance([r for r in records if str(r["level"]) == lv]) for lv in levels}
    ch["Overall"] = chance(records)
    return {"always_A": a_summary, "chance": ch}


# ---------------------------------------------------------------- results
def stop_cause_counts(records: list[dict], results: dict[str, dict]) -> dict:
    out: dict[str, int] = {}
    for r in records:
        res = results.get(str(r["qa_id"]))
        key = "no_output" if not res or res.get("status") != "ok" else res.get(
            "stop_cause", "length" if res.get("finish_reason") == "length" else "natural_stop")
        out[key] = out.get(key, 0) + 1
    return out


def _rounded(obj, nd: int = 3):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_rounded(v, nd) for v in obj]
    return obj


def _intervals(items) -> list:
    out = []
    for it in items or []:
        try:
            out.append(list(parse_evidence_interval(it["time"] if isinstance(it, dict) else it)))
        except Exception:  # noqa: BLE001
            pass
    return out


def given_modality_report(records, modes, modality: str) -> dict:
    """Unimodal runs: grounding of the ONE modality the model was given, per level.
    cov_grounded = mean(correct x coverage), iou_grounded = mean(correct x IoU)
    (C.G. and B.G. computed on that modality alone); also the mean coverage and IoU
    over the correct answers only."""
    out = {}
    for m in MODES:
        sc, md = modes[m]["scores"], modes[m]["modal"]
        per = {}
        for lv, rs in _level_groups(records).items():
            keys = [str(r["qa_id"]) for r in rs]
            ok = [k for k in keys if sc[k]["Acc."]]
            n = len(keys)
            per[lv] = {"n": n, "n_correct": len(ok),
                       "cov_grounded": 100 * math.fsum(sc[k]["Acc."] * md[k][f"{modality}_coverage"]
                                                       for k in keys) / n if n else 0.0,
                       "iou_grounded": 100 * math.fsum(sc[k]["Acc."] * md[k][f"{modality}_iou"]
                                                       for k in keys) / n if n else 0.0,
                       "mean_cov_if_correct": 100 * math.fsum(md[k][f"{modality}_coverage"] for k in ok)
                       / len(ok) if ok else 0.0,
                       "mean_iou_if_correct": 100 * math.fsum(md[k][f"{modality}_iou"] for k in ok)
                       / len(ok) if ok else 0.0}
        out[m] = per
    return {"modality": modality, "by_mode": out}


def score_records(records: list[dict], results: dict[str, dict],
                  video_seconds: dict[str, float] | None = None, input_mode: str = "av") -> dict:
    """Everything that results.json holds except run name and input statistics.
    Metric maths: scripts/eval_example.py::compute_grounded_scores (authors' code).
    input_mode v/a/t: the evidence lists the prompt does not ask for are scored as empty."""
    unasked = UNASKED_BY_INPUT.get(input_mode, ())
    modes = {m: score_mode(records, results, m, unasked) for m in MODES}
    sources = answer_source_report(records, results)
    questions = []
    for rec in records:
        key = str(rec["qa_id"])
        res = results.get(key) or {}
        ok = res.get("status") == "ok"
        q = {"qa_id": rec["qa_id"], "video_id": rec["video_id"], "level": rec["level"],
             "capability": rec["capability"], "status": res.get("status", "missing")}
        if res.get("error_type"):
            q["error_type"] = res["error_type"]
        q.update({
            "answer_source": answer_source(res.get("raw_response", "")) if ok else "no_output",
            "stop_cause": res.get("stop_cause") if ok else None,
            "prompt_tokens": res.get("prompt_tokens"),
            "n_output_tokens": res.get("n_output_tokens"),
            "gt": {"answer": rec["answer"], "visual": _intervals(rec["evidence"]["visual"]),
                   "audio": _intervals(rec["evidence"]["audio"])},
            "pred": {m: ({"answer": p["answer"], "visual": [list(i) for i in p.get("visual_evidence", [])],
                          "audio": [list(i) for i in p.get("audio_evidence", [])]}
                         if (p := modes[m]["preds"].get(key)) else None) for m in MODES},
            "scores": {m: {k: modes[m]["scores"][key][k] for k in MAIN_METRICS} for m in MODES},
        })
        questions.append(q)
    letters = {"gt": letter_distribution(records, lambda r: r["answer"])}
    for m in MODES:
        letters[m] = letter_distribution(records, lambda r, m=m: (modes[m]["preds"].get(str(r["qa_id"]))
                                                                   or {}).get("answer"))
    return _rounded({
        "n_questions": len(records),
        "summary": {m: modes[m]["summary"] for m in MODES},
        "by_capability": {m: modes[m]["by_capability"] for m in MODES},
        "baselines": baselines(records),
        "diagnostics": {
            "parsing": {m: modes[m]["parsing"] for m in MODES},
            "answer_sources": {k: {"n": v["n"], "correct": v["correct"], "letters": v["letters"]}
                               for k, v in sources.items()},
            "stop_causes": stop_cause_counts(records, results),
            "format_errors": format_report(records, results, video_seconds),
            "letters": letters,
        },
        **({"input_mode": input_mode, "unasked_evidence_scored_empty": list(unasked)} if unasked else {}),
        **({"given_modality": given_modality_report(records, modes, GIVEN_MODALITY[input_mode])}
           if input_mode in GIVEN_MODALITY else {}),
        "questions": questions,
    })


def score_file(results_path: Path, annotation: Path) -> dict:
    """Score a single results file against ALL annotation records."""
    return score_records(load_annotations(annotation), load_latest(results_path))


# ---------------------------------------------------------------- run folders
RUNS_MD = Path(__file__).resolve().parent / "output" / "RUNS.md"
EVAL_DIR = "evaluation"


def input_stats_summary(results: dict[str, dict], qa_ids: list[int]) -> dict:
    rows = [results[str(q)] for q in qa_ids if str(q) in results]
    ok = [r for r in rows if r.get("status") == "ok"]

    def rng(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return [min(vals), max(vals)] if vals else None
    errors: dict[str, int] = {}
    for r in rows:
        if r.get("status") != "ok":
            errors[r.get("error_type", "error")] = errors.get(r.get("error_type", "error"), 0) + 1
    return {"n_questions": len(qa_ids), "n_recorded": len(rows), "n_ok": len(ok),
            "missing": len(qa_ids) - len(rows), "errors_by_type": errors,
            "prompt_tokens_min_max": rng("prompt_tokens"), "video_tokens_min_max": rng("video_tokens"),
            "audio_tokens_min_max": rng("audio_tokens"), "frames_min_max": rng("frames"),
            "video_seconds_min_max": rng("video_seconds"), "audio_seconds_min_max": rng("audio_seconds"),
            "context_fraction_max": max((r["context_fraction"] for r in ok if r.get("context_fraction")),
                                        default=None)}


def run_scores(run_dir: Path) -> dict | None:
    """{"summary": {mode: {level: metrics}}, "questions": {mode: {qa_id: scores}}} of a
    scored run (evaluation/results.json, or scores.json for runs before 2026-11-16)."""
    new, old = run_dir / EVAL_DIR / "results.json", run_dir / "scores.json"
    if new.exists():
        res = json.loads(new.read_text(encoding="utf-8"))
        if "summary" not in res:
            return None  # stats-only run
        return {"summary": res["summary"],
                "questions": {m: {q["qa_id"]: q["scores"][m] for q in res["questions"]} for m in res["summary"]}}
    if old.exists():
        sc = json.loads(old.read_text(encoding="utf-8"))
        modes = [m for m in MODES if m in sc]
        return {"summary": {m: sc[m]["summary"] for m in modes},
                "questions": {m: {r["qa_id"]: r["scores"] for r in sc[m]["per_question"]} for m in modes}}
    return None


def rebuild_runs_md() -> None:
    """Rewrite RUNS.md from every run folder in custom/output/ (folders starting
    with '_' are not runs). Rows come from info.txt + the evaluation results, so
    the index never drifts from the run folders."""
    fmt = lambda b: f"{b['Acc.']:.1f} / {b['C.G.']:.1f} / {b['B.G.']:.1f}"
    rows = []
    for d in sorted(RUNS_MD.parent.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        meta = read_run_meta(d)
        if meta is None:
            continue
        rt, st = meta["run"], meta["settings"]
        resp = load_outputs(d)
        n = st["subset_n"]
        n_ok = sum(1 for r in resp.values() if r.get("status") == "ok")
        sc = None if st.get("stats_only") else run_scores(d)
        if sc:
            ov = {m: v["Overall"] for m, v in sc["summary"].items()}
            cells = [fmt(ov["strict"]), fmt(ov["permissive"]),
                     f"{ov['permissive_no_free_text']['Acc.']:.1f}" if "permissive_no_free_text" in ov else "-",
                     fmt(ov["diag_no_brackets"]) if "diag_no_brackets" in ov else "-"]
        else:
            cells = ["(stats only)" if st.get("stats_only") else "(not scored)"] * 2 + ["-", "-"]
        desc = rt["description"].replace("|", "/").replace("\n", " ")
        rows.append((rt["started"], f"| {d.name} | {rt['started'][:16].replace('T', ' ')} | {n} | {desc} | "
                     + " | ".join(cells) + f" | {n_ok} / {n - n_ok} |\n"))
    head = ["# Runs\n", "\n",
            "All scores are percentages over the run's subset (failed questions = 0). The last score column is "
            "DIAGNOSTIC ONLY (permissive + timestamps without square brackets accepted). See each run's "
            "evaluation/ folder for details.\n", "\n",
            "| run | date | n | description | strict Acc / C.G. / B.G. | permissive Acc / C.G. / B.G. "
            "| no-free-text Acc | diag no-brackets Acc / C.G. / B.G. | ok / failed |\n",
            "|---|---|---:|---|---|---|---:|---|---|\n"]
    RUNS_MD.write_text("".join(head + [r for _, r in sorted(rows)]), encoding="utf-8")


# EvaOmni paper (arXiv:2610.28444v2), Table 2: Acc. / C.G. / B.G. in %, all 1027 questions.
PAPER_TABLE2 = {
    "Qwen3-Omni-Instruct-30B-A3B": {
        "Level 1": (28.6, 5.1, 0.6), "Level 2": (22.4, 3.4, 0.4),
        "Level 3": (22.2, 2.4, 0.1), "Overall": (25.2, 4.0, 0.4)},
    "Qwen3-Omni-Thinking-30B-A3B": {
        "Level 1": (26.3, 11.4, 4.8), "Level 2": (24.4, 6.2, 1.2),
        "Level 3": (21.6, 4.1, 0.6), "Overall": (24.8, 8.3, 2.8)},
}
# Table 6(d): Acc. of Qwen3-Omni-Instruct-30B-A3B with one input only; a question
# counts as correct only if it is correct in both of two independent runs.
PAPER_TABLE6D = {"a": 18.2, "v": 12.5}
INPUT_LABELS = {"v": "video frames, no audio", "a": "audio track, no frames",
                "t": "question and options, no video, no audio"}
MODE_LABELS = {"strict": "STRICT", "permissive": "PERMISSIVE",
               "permissive_no_free_text": "PERMISSIVE, NO FREE TEXT",
               "diag_no_brackets": "DIAGNOSTIC ONLY: permissive + timestamps without [] accepted"}
LEVELS = ("Level 1", "Level 2", "Level 3", "Overall")


def format_evaluation(res: dict, title: str, scored: str, meta: dict | None = None) -> str:
    """Plain-text evaluation report: every scoring mode vs the paper's Table 2."""
    st = (meta or {}).get("settings", {})
    paper_model = ("Qwen3-Omni-Thinking-30B-A3B" if "Thinking" in st.get("model", "")
                   else "Qwen3-Omni-Instruct-30B-A3B")
    paper = PAPER_TABLE2[paper_model]
    L = [f"EvaOmni evaluation: {title}"]
    if meta:
        L += [meta["run"]["description"], "",
              f"Model    : {st['model']}",
              f"Subset   : {st['subset']} (n={st['subset_n']})   decoding: {st['decoding']}   "
              f"input: {st['input_mode']}   seed: {st['seed']}"]
    imode = st.get("input_mode", "av")
    unimodal = imode in UNASKED_BY_INPUT
    show_diff = not unimodal
    L += [f"Scored   : {scored}",
          f"Paper    : Table 2, {paper_model} (all 1027 questions)"]
    if unimodal:
        L += [f"INPUT    : {imode} only ({INPUT_LABELS[imode]}). The PAPER rows are Table 2 (video+audio input),",
              "           shown for reference only, no differences printed.",
              f"           Paper Table 6(d), {paper_model}, this input: {PAPER_TABLE6D.get(imode, 'not reported')}"
              + (" (Acc. only; a question counts only if correct in BOTH of two runs:"
                 " custom/tools/both_runs_acc.py RUN1 RUN2)" if imode in PAPER_TABLE6D else
                 " (Table 6(d) text-only uses other models)"),
              "           Evidence lists the prompt does not ask for are scored as empty: "
              "C.G. <= Acc./2, B.G. = 0."]
    L += ["Values in %. Failed or missing answers count 0." + (" In brackets: ours - paper." if show_diff else ""),
          ""]
    if res["n_questions"] != 1027:
        L += [f"WARNING: {res['n_questions']} questions, not the full benchmark; "
              "the paper's numbers are on all 1027 questions.", ""]
    L += [f"{'':12s}{'Acc.':>14s}{'C.G.':>14s}{'B.G.':>14s}{'n':>7s}   Acc. 95% CI", "PAPER"]
    for lv in LEVELS:
        L.append(f"  {lv:10s}" + "".join(f"{v:14.1f}" for v in paper[lv]))
    for mode, summ in res["summary"].items():
        d = res["diagnostics"]["parsing"][mode]
        L += ["", f"{MODE_LABELS.get(mode, mode)}   (parsed {d['parsed']}, unparsable {d['unparsable']}, "
                  f"failed/missing {d['failed_or_missing']})"]
        for lv in LEVELS:
            b = summ.get(lv)
            if b is None:
                continue
            cells = "".join((f"{f'{b[m]:.1f} ({b[m] - q:+.1f})':>14s}" if show_diff else f"{b[m]:14.1f}")
                            for m, q in zip(MAIN_METRICS, paper[lv]))
            lo, hi = b["Acc. 95% CI"]
            L.append(f"  {lv:10s}{cells}{b['n']:7d}   [{lo:.1f}, {hi:.1f}]")
    if "given_modality" in res:
        gm = res["given_modality"]
        L += ["", f"GROUNDING OF THE GIVEN MODALITY ({gm['modality']}) ALONE",
              "  cov-grounded = mean(correct x coverage), IoU-grounded = mean(correct x IoU) on this modality;",
              "  'if correct' = mean coverage / IoU over the correct answers only",
              f"  {'':26s}{'Acc.':>7s}{'cov-grnd':>10s}{'IoU-grnd':>10s}{'cov|ok':>8s}{'IoU|ok':>8s}"]
        for mode in ("strict", "diag_no_brackets"):
            for lv in LEVELS:
                if lv not in gm["by_mode"][mode] or lv not in res["summary"][mode]:
                    continue  # subset without this level (e.g. check10-l1)
                b, s = gm["by_mode"][mode][lv], res["summary"][mode][lv]
                L.append(f"  {mode[:14]:14s} {lv:10s} {s['Acc.']:7.1f}{b['cov_grounded']:10.1f}"
                         f"{b['iou_grounded']:10.1f}{b['mean_cov_if_correct']:8.1f}{b['mean_iou_if_correct']:8.1f}")
    bl = res["baselines"]
    L += ["", "BASELINES (Acc.)"]
    for lv in LEVELS:
        if lv in bl["chance"]:
            L.append(f"  {lv:10s}  always 'A' {bl['always_A'][lv]['Acc.']:5.1f}   "
                     f"chance {bl['chance'][lv]['Acc.']:5.1f}")
    dg = res["diagnostics"]
    L += ["", "DIAGNOSTICS",
          "  answer sources (permissive): " + json.dumps(
              {k: {"n": v["n"], "correct": v["correct"]} for k, v in dg["answer_sources"].items()}),
          "  stop causes: " + json.dumps(dg["stop_causes"]),
          "  format, questions with: " + json.dumps(dg["format_errors"]["questions_with"]),
          "  format, timestamps: " + json.dumps(dg["format_errors"]["timestamps"]),
          "  letters, overall counts:"]
    for name, dist in dg["letters"].items():
        L.append(f"    {name:24s} " + json.dumps(dist["Overall"]))
    return "\n".join(L) + "\n"


def score_run_dir(run_dir: Path, annotation: Path) -> None:
    meta = read_run_meta(run_dir)
    if meta is None:
        sys.exit(f"{run_dir}: no run settings in info.txt (not a run folder?)")
    records_all = load_annotations(annotation)
    subset = meta["settings"]["subset"]
    if subset == "full":
        qa_ids = sorted(r["qa_id"] for r in records_all)
    else:
        subset_file = Path(__file__).resolve().parent / "subsets" / f"{subset}.json"
        qa_ids = json.loads(subset_file.read_text())["qa_ids"]
    keep = set(qa_ids)
    records = [r for r in records_all if r["qa_id"] in keep]
    results = load_outputs(run_dir)
    scored = datetime.now().astimezone().isoformat(timespec="seconds")
    head = {"run": run_dir.name, "scored": scored}
    stats = input_stats_summary(results, qa_ids)
    ev = run_dir / EVAL_DIR
    ev.mkdir(exist_ok=True)
    if meta["settings"].get("stats_only"):
        res = {**head, "input": stats}
        text = None
    else:
        vsec = {r["video_id"]: r["video_seconds"] for r in results.values() if r.get("video_seconds")}
        scored_part = score_records(records, results, vsec, meta["settings"].get("input_mode", "av"))
        questions = scored_part.pop("questions")
        res = {**head, **scored_part, "input": _rounded(stats), "questions": questions}
        text = format_evaluation(res, run_dir.name, scored, meta)
        (ev / "evaluation.txt").write_text(text, encoding="utf-8")
    (ev / "results.json").write_text(pretty_json(res) + "\n", encoding="utf-8")
    with (run_dir / "info.txt").open("a", encoding="utf-8") as f:
        f.write(f"\nScored       : {scored}  -> {EVAL_DIR}/"
                + ("" if text else "  (stats-only run: input statistics only)") + "\n")
    rebuild_runs_md()
    if text:
        print(text)
    print(f"Wrote {ev}/ and updated {RUNS_MD}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="a run folder in custom/output/ (preferred)")
    src.add_argument("--results", type=Path, help="a single results JSONL scored against all 1027 questions")
    ap.add_argument("--annotation", type=Path, default=REPO / "data" / "test_file.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="with --results: where to write the JSON report (default: results.json next to it)")
    args = ap.parse_args()
    if args.run_dir:
        score_run_dir(args.run_dir, args.annotation)
        return
    if not args.results.exists():
        sys.exit(f"results file not found: {args.results}")
    out = score_file(args.results, args.annotation)
    print(format_evaluation(out, str(args.results), datetime.now().astimezone().isoformat(timespec="seconds")))
    dest = args.out or args.results.with_name("results.json")
    dest.write_text(pretty_json(out) + "\n", encoding="utf-8")
    print(f"Full report: {dest}")


if __name__ == "__main__":
    main()
