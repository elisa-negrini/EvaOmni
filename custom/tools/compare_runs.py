#!/usr/bin/env python3
"""Compare two runs question by question (custom/output/<run>).

Reports, on the qa_ids present in both runs:
  * identical raw responses, identical predicted letter (permissive parser);
  * correctness of each run (strict / permissive) and the 2x2 agreement table;
  * per-metric overall difference (from each run's evaluation results, if scored);
  * input-token differences (from the "input" part of output.jsonl), e.g. to
    confirm that a muted run sees the same number of audio tokens.

Usage:
  python custom/tools/compare_runs.py RUN_A RUN_B [--show N]
The comparison is printed (nothing is written to disk).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CUSTOM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUSTOM))
import score  # noqa: E402

OUT = CUSTOM / "output"


def load(run: str):
    d = OUT / run
    outputs = score.load_outputs(d)  # flat records: response and input fields together
    return outputs, outputs, score.run_scores(d)


def letter(rec, mode):
    if not rec or rec.get("status") != "ok":
        return None
    pred, _ = score.PARSERS[mode](rec.get("raw_response", ""), rec["qa_id"])
    return pred["answer"] if pred else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--show", type=int, default=10, help="print up to N differing questions")
    args = ap.parse_args()
    ann = {str(r["qa_id"]): r for r in score.load_annotations(score.REPO / "data" / "test_file.json")}
    ra, sa, sca = load(args.run_a)
    rb, sb, scb = load(args.run_b)
    common = sorted(set(ra) & set(rb), key=int)
    rows = []
    for q in common:
        a, b = ra[q], rb[q]
        la, lb = letter(a, "permissive"), letter(b, "permissive")
        gt = ann[q]["answer"]
        xa, xb = sa.get(q, {}), sb.get(q, {})
        rows.append({
            "qa_id": int(q), "gt": gt, "letter_a": la, "letter_b": lb,
            "strict_a": letter(a, "strict"), "strict_b": letter(b, "strict"),
            "raw_identical": a.get("raw_response") == b.get("raw_response"),
            "same_letter": la == lb,
            "correct_a": la == gt, "correct_b": lb == gt,
            "prompt_tokens": [xa.get("prompt_tokens"), xb.get("prompt_tokens")],
            "audio_tokens": [xa.get("audio_tokens"), xb.get("audio_tokens")],
            "video_tokens": [xa.get("video_tokens"), xb.get("video_tokens")],
            "audio_peak": [xa.get("audio_peak"), xb.get("audio_peak")],
            "raw_a": a.get("raw_response", ""), "raw_b": b.get("raw_response", ""),
        })
    n = len(rows)
    pct = lambda k: 100 * sum(r[k] for r in rows) / n if n else 0.0
    both = sum(r["correct_a"] and r["correct_b"] for r in rows)
    only_a = sum(r["correct_a"] and not r["correct_b"] for r in rows)
    only_b = sum(r["correct_b"] and not r["correct_a"] for r in rows)
    summary = {
        "run_a": args.run_a, "run_b": args.run_b, "n_common": n,
        "only_in_a": len(set(ra) - set(rb)), "only_in_b": len(set(rb) - set(ra)),
        "raw_identical_pct": pct("raw_identical"), "same_letter_pct": pct("same_letter"),
        "acc_permissive_a": pct("correct_a"), "acc_permissive_b": pct("correct_b"),
        "correct_both": both, "correct_only_a": only_a, "correct_only_b": only_b,
        "correct_neither": n - both - only_a - only_b,
        "same_audio_tokens_pct": 100 * sum(r["audio_tokens"][0] == r["audio_tokens"][1] for r in rows) / n if n else 0,
        "same_video_tokens_pct": 100 * sum(r["video_tokens"][0] == r["video_tokens"][1] for r in rows) / n if n else 0,
    }
    if sca and scb and n:
        # Metrics recomputed on the COMMON questions only (the two runs may cover
        # different subsets, e.g. sub200 vs full), per scoring mode, from the
        # per-question scores in each run's evaluation results.
        common_ids = {int(q) for q in common}
        per_mode = {}
        for mode in score.MODES:
            if mode not in sca["questions"] or mode not in scb["questions"]:
                continue
            qa_a, qa_b = sca["questions"][mode], scb["questions"][mode]
            block = {}
            for m in ("Acc.", "C.G.", "B.G."):
                va = [qa_a[q][m] for q in sorted(common_ids)]
                vb = [qa_b[q][m] for q in sorted(common_ids)]
                diffs = [y - x for x, y in zip(va, vb)]
                block[m] = {"a": round(100 * sum(va) / n, 2), "b": round(100 * sum(vb) / n, 2),
                            "b_minus_a": round(100 * sum(diffs) / n, 2),
                            "n_questions_changed": sum(1 for d in diffs if abs(d) > 1e-9),
                            "mean_abs_change_per_question": round(100 * sum(abs(d) for d in diffs) / n, 2),
                            "max_abs_change": round(100 * max((abs(d) for d in diffs), default=0), 1)}
            per_mode[mode] = block
        summary["metrics_on_common_questions"] = per_mode
        by_level = {}
        for r in rows:
            lv = ann[str(r["qa_id"])]["level"]
            b_ = by_level.setdefault(f"Level {lv}", {"n": 0, "same_letter": 0})
            b_["n"] += 1
            b_["same_letter"] += r["same_letter"]
        summary["same_letter_by_level"] = by_level
    print(json.dumps(summary, indent=1))
    shown = 0
    for r in rows:
        if r["raw_identical"] or shown >= args.show:
            continue
        shown += 1
        print(f"\n--- qa {r['qa_id']} gt={r['gt']}  A={r['letter_a']}  B={r['letter_b']}  "
              f"audio tok {r['audio_tokens']} peak {r['audio_peak']}")
        print(f"  A: {r['raw_a'][:300]!r}")
        print(f"  B: {r['raw_b'][:300]!r}")


if __name__ == "__main__":
    main()
