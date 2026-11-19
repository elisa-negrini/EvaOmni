#!/usr/bin/env python3
"""Accuracy as in the paper's Table 6(d): a question counts as correct only if it
is correct in BOTH of two independent runs.

Reads the per-question scores of two scored runs (evaluation/results.json) and
prints, per scoring mode and level: accuracy of each run, "correct in both"
(the paper's number), "correct in at least one", and the paper's Table 6(d) value
for Qwen3-Omni-Instruct when the runs' input is audio-only or video-only.

Usage:
  python custom/tools/both_runs_acc.py RUN_1 RUN_2
Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CUSTOM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUSTOM))
import score  # noqa: E402
from runinfo import read_run_meta  # noqa: E402

OUT = CUSTOM / "output"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_1")
    ap.add_argument("run_2")
    args = ap.parse_args()
    runs = [args.run_1, args.run_2]
    metas = [read_run_meta(OUT / r) for r in runs]
    scs = [score.run_scores(OUT / r) for r in runs]
    for r, m, s in zip(runs, metas, scs):
        if m is None or s is None:
            sys.exit(f"{r}: not a scored run folder in {OUT}")
    st = [m["settings"] for m in metas]
    diff = {k: (st[0].get(k), st[1].get(k)) for k in set(st[0]) | set(st[1])
            if k not in ("seed",) and st[0].get(k) != st[1].get(k)}
    if diff:
        print(f"WARNING: settings differ between the two runs: {sorted(diff)}\n")
    mode_in = st[0].get("input_mode", "av")
    ann = {r["qa_id"]: r for r in score.load_annotations(score.REPO / "data" / "test_file.json")}
    common = sorted(set(scs[0]["questions"]["strict"]) & set(scs[1]["questions"]["strict"]))
    print(f"{runs[0]}  vs  {runs[1]}   input: {mode_in}   common questions: {len(common)}")
    if mode_in in score.PAPER_TABLE6D:
        print(f"Paper Table 6(d), Qwen3-Omni-Instruct, this input, correct in both runs: "
              f"{score.PAPER_TABLE6D[mode_in]} (Overall)")
    print("Values in % of the questions; 'both' is the paper's Table 6(d) rule.\n")
    print(f"{'mode':26s} {'level':9s} {'n':>5s} {'run 1':>7s} {'run 2':>7s} {'both':>7s} {'either':>7s}")
    for mode in ("strict", "permissive", "diag_no_brackets"):
        if mode not in scs[0]["questions"] or mode not in scs[1]["questions"]:
            continue
        q1, q2 = scs[0]["questions"][mode], scs[1]["questions"][mode]
        groups = {"Overall": common}
        for lv in sorted({ann[q]["level"] for q in common}):
            groups[f"Level {lv}"] = [q for q in common if ann[q]["level"] == lv]
        for lv in ("Level 1", "Level 2", "Level 3", "Overall"):
            qs = groups.get(lv, [])
            if not qs:
                continue
            n = len(qs)
            a = [bool(q1[q]["Acc."]) for q in qs]
            b = [bool(q2[q]["Acc."]) for q in qs]
            pct = lambda xs: 100 * sum(xs) / n
            print(f"{mode:26s} {lv:9s} {n:5d} {pct(a):7.1f} {pct(b):7.1f} "
                  f"{pct([x and y for x, y in zip(a, b)]):7.1f} {pct([x or y for x, y in zip(a, b)]):7.1f}")
        print()


if __name__ == "__main__":
    main()
