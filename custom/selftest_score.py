#!/usr/bin/env python3
"""Self-test for custom/score.py using synthetic result files.

  (a) annotations as predictions           -> 100 on every metric, both modes
  (b) wrong letter + perfect GT evidence   -> 0 on every metric, both modes
  (c) half of the questions missing        -> missing rows count zero (denominator = 1027)
  (d) parser edge cases on single questions (strict vs permissive behaviour)

Writes its fixtures to a temporary folder (deleted at exit). Exit code != 0 on failure.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

REPO = score.REPO
ANN = REPO / "data" / "test_file.json"
_TMP = tempfile.TemporaryDirectory(prefix="evaomni_selftest_")
OUT = Path(_TMP.name)
records = score.load_annotations(ANN)
failures: list[str] = []


def check(cond: bool, msg: str):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def gt_raw(rec, answer=None):
    ev = rec["evidence"]
    return json.dumps({
        "answer": answer or rec["answer"],
        "visual_evidence": [{"time": e["time"]} for e in ev["visual"]],
        "audio_evidence": [{"time": e["time"]} for e in ev["audio"]],
    })


def write(name, rows):
    p = OUT / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def wrong_letter(rec):
    letters = score._OPTION_RE.findall(rec["options"])
    return next(l for l in letters if l != rec["answer"] and l <= "H")


# (a) ground truth as prediction
print("\n(a) annotations as predictions")
out_a = score.score_file(write("gt", [{"qa_id": r["qa_id"], "status": "ok", "raw_response": gt_raw(r)}
                                      for r in records]), ANN)
print(score.format_evaluation(out_a, "selftest (a)", "now"))
for mode in ("strict", "permissive"):
    for blk_name, blk in out_a["summary"][mode].items():
        check(all(math.isclose(blk[m], 100.0) for m in score.ALL_METRICS),
              f"{mode} {blk_name}: all metrics = 100")

# (b) wrong letter, perfect evidence
print("\n(b) wrong answers with perfect evidence")
out_b = score.score_file(write("wrong", [{"qa_id": r["qa_id"], "status": "ok",
                                          "raw_response": gt_raw(r, wrong_letter(r))} for r in records]), ANN)
for mode in ("strict", "permissive"):
    for blk_name, blk in out_b["summary"][mode].items():
        check(all(blk[m] == 0.0 for m in score.ALL_METRICS), f"{mode} {blk_name}: all metrics = 0")

# (c) missing rows + failed inference keep the denominator
print("\n(c) half missing, a few infer_error")
half = records[: len(records) // 2]
rows = [{"qa_id": r["qa_id"], "status": "ok", "raw_response": gt_raw(r)} for r in half]
rows += [{"qa_id": r["qa_id"], "status": "infer_error", "raw_response": ""} for r in records[-5:]]
out_c = score.score_file(write("half", rows), ANN)
exp = 100 * len(half) / len(records)
for mode in ("strict", "permissive"):
    check(math.isclose(out_c["summary"][mode]["Overall"]["Acc."], exp, abs_tol=1e-3),
          f"{mode}: overall Acc = {exp:.2f} (n=1027 in denominator)")

# (d) parser edge cases on qa_id of record 0
print("\n(d) parser edge cases")
r0 = records[0]
qa, gt = r0["qa_id"], r0["answer"]
v = r0["evidence"]["visual"][0]["time"]
a = r0["evidence"]["audio"][0]["time"]
cases = {
    # name: (raw, strict_correct, permissive_correct, permissive_BG_should_be_1)
    "fenced json": (f'```json\n{{"answer":"{gt}","visual_evidence":[{{"time":"{v}"}}],'
                    f'"audio_evidence":[{{"time":"{a}"}}]}}\n```', 1, 1, True),
    "bare letter": (f"The answer is {gt}.", 0, 1, False),
    "json in prose": (f'Sure! {{"answer":"{gt}","visual_evidence":[{{"time":"{v}"}}],'
                      f'"audio_evidence":[{{"time":"{a}"}}]}} hope it helps', 0, 1, True),
    "one bad interval": (f'{{"answer":"{gt}","visual_evidence":[{{"time":"{v}"}},{{"time":"[xx]"}}],'
                         f'"audio_evidence":[{{"time":"{a}"}}]}}', 0, 1, True),
    "answer with text": (f'{{"answer":"{gt}. 1 time","visual_evidence":[],"audio_evidence":[]}}', 0, 1, False),
    "missing audio list": (f'{{"answer":"{gt}","visual_evidence":[{{"time":"{v}"}}]}}', 0, 1, False),
}
for name, (raw, s_exp, p_exp, bg1) in cases.items():
    res = {str(qa): {"qa_id": qa, "status": "ok", "raw_response": raw}}
    one = [r0]
    s = score.compute_grounded_scores(one, score.build_latest_for_scorer(one, res, "strict")[0])
    p = score.compute_grounded_scores(one, score.build_latest_for_scorer(one, res, "permissive")[0])
    s_acc = s["per_question"][0]["scores"]["Acc."]
    p_q = p["per_question"][0]["scores"]
    check(s_acc == s_exp and p_q["Acc."] == p_exp and (p_q["B.G."] == 1.0) == bg1,
          f"{name!r}: strict Acc={s_acc} (exp {s_exp}), permissive Acc={p_q['Acc.']} "
          f"(exp {p_exp}), permissive B.G.={p_q['B.G.']:.2f}")

# (e) answer-source classification and the no-free-text variant
print("\n(e) answer sources")
src_cases = {
    # raw: (expected source, permissive letter, permissive_no_free_text keeps it?)
    f'{{"answer":"{gt}","visual_evidence":[],"audio_evidence":[]}}': ("json_letter", gt, True),
    f'{{"answer":"{gt.lower()}","visual_evidence":[],"audio_evidence":[]}}': ("json_letter", gt, True),
    '{"answer":"a man is talking","visual_evidence":[],"audio_evidence":[]}': ("json_nonstandard", "A", True),
    "I think it is a man talking, so the answer is E.": ("free_text", "A", False),
    "No idea.": ("none", "", False),
    '```json\n{"answer": "E", "visual_evidence": [{"time": "00:14"}, {"time": "00:2':
        ("truncated_json", "E", True),
}
for raw, (exp_src, exp_letter, kept) in src_cases.items():
    src = score.answer_source(raw)
    pred, _ = score.parse_permissive(raw, qa)
    pred_nf, _ = score.parse_permissive_no_free_text(raw, qa)
    letter = pred["answer"] if pred else ""
    check(src == exp_src and letter == exp_letter and (pred_nf is not None) == kept,
          f"{raw[:45]!r}: source={src} (exp {exp_src}), letter={letter!r} (exp {exp_letter!r}), "
          f"kept without free text={pred_nf is not None}")

mixed = [{"qa_id": r["qa_id"], "status": "ok", "raw_response": gt_raw(r)} for r in records[:10]]
mixed += [{"qa_id": r["qa_id"], "status": "ok", "raw_response": "Looking at a scene, answer " + r["answer"]}
          for r in records[10:20]]
out_e = score.score_file(write("mixed", mixed), ANN)
fs = out_e["diagnostics"]["answer_sources"]["free_text"]
check(fs["n"] == 10 and fs["letters"] == {"A": 10}, f"free_text n={fs['n']}, letters={fs['letters']}")
exp_nf = 100 * 10 / len(records)
check(math.isclose(out_e["summary"]["permissive_no_free_text"]["Overall"]["Acc."], exp_nf, abs_tol=1e-3),
      f"permissive_no_free_text overall Acc = {exp_nf:.2f} (only the 10 JSON answers)")
dist = out_e["diagnostics"]["letters"]["permissive"]["Overall"]
check(sum(dist.values()) == len(records) and dist.get("none") == len(records) - 20,
      f"pred letter distribution sums to 1027, none={dist.get('none')}")

print("\n" + ("ALL TESTS PASSED" if not failures else f"{len(failures)} FAILURE(S)"))
sys.exit(1 if failures else 0)
