#!/usr/bin/env python3
"""How long did the runs take, how long will the running ones take, how long would a new one take.

For every run folder in custom/output/:
  duration   from info.txt ("Finished ... since first start N min"); for a run still
             going, the time from "started" to the last record of output.jsonl
  done       answered questions (status ok), distinct videos, generated tokens,
             answers cut by the token limit (loops)

Time model, fitted on the finished runs (least squares):
  minutes = startup + a * videos + b * questions + c * generated_tokens
Reading a video (prefill) is paid once per video; the other questions of the same
video reuse it (prefix caching). Generation costs c per token; loops dominate it.

For a running run the ETA is a simple proportion (time so far / questions so far);
it is rough because loops are not spread evenly over the videos.

Usage:
  python custom/tools/run_times.py                     # table + model + ETAs
  python custom/tools/run_times.py --predict full 70   # new run on 'full' (or a subset name),
                                                       # assuming 70 generated tokens per question
Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

CUSTOM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUSTOM))
import score  # noqa: E402
from runinfo import read_run_meta  # noqa: E402

OUT = CUSTOM / "output"
ANN = CUSTOM.parent / "data" / "test_file.json"


def collect():
    rows = []
    for d in sorted(OUT.iterdir()):
        meta = read_run_meta(d) if d.is_dir() else None
        if not meta:
            continue
        info = (d / "info.txt").read_text(encoding="utf-8") if (d / "info.txt").exists() else ""
        fin = re.findall(r"since first start ([\d.]+) min", info)
        out = score.load_outputs(d)
        ok = [r for r in out.values() if r.get("status") == "ok"]
        started = dt.datetime.fromisoformat(meta["run"]["started"])
        times = [dt.datetime.fromisoformat(r["time"]) for r in out.values() if r.get("time")]
        minutes = float(fin[-1]) if fin else ((max(times) - started).total_seconds() / 60 if times else 0.0)
        rows.append(dict(
            run=d.name, finished=bool(fin), minutes=minutes, total=meta["settings"].get("subset_n"),
            done=len(ok), videos=len({r["video_id"] for r in ok}),
            tokens=sum(r.get("n_output_tokens") or 0 for r in ok),
            cut=sum(1 for r in ok if str(r.get("stop_cause", "")).startswith("max")),
            stats_only=meta["settings"].get("stats_only", False),
            max_tokens=meta["settings"].get("sampling_params", {}).get("max_tokens")))
    return rows


def fit(rows):
    use = [r for r in rows if r["finished"] and not r["stats_only"] and r["done"]]
    X = np.array([[1.0, r["videos"], r["done"], r["tokens"]] for r in use])
    y = np.array([r["minutes"] for r in use])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, use, X @ coef


def subset_size(name: str) -> tuple[int, int]:
    recs = json.loads(ANN.read_text(encoding="utf-8"))
    if name != "full":
        ids = set(json.loads((CUSTOM / "subsets" / f"{name}.json").read_text())["qa_ids"])
        recs = [r for r in recs if r["qa_id"] in ids]
    return len(recs), len({r["video_id"] for r in recs})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", nargs=2, metavar=("SUBSET", "TOKENS_PER_QUESTION"),
                    help="estimate a new run on SUBSET ('full' or a file in custom/subsets/)")
    args = ap.parse_args()
    rows = collect()

    print(f"{'run':45s} {'state':9s} {'min':>6s} {'done':>9s} {'videos':>6s} {'out tokens':>10s} "
          f"{'cut':>4s} {'max_tok':>7s} {'min/100q':>8s}")
    for r in rows:
        state = "finished" if r["finished"] else "RUNNING?"
        per100 = 100 * r["minutes"] / r["done"] if r["done"] else 0
        print(f"{r['run']:45s} {state:9s} {r['minutes']:6.1f} {r['done']:>4d}/{r['total'] or '?':<4} "
              f"{r['videos']:6d} {r['tokens']:10d} {r['cut']:4d} {str(r['max_tokens']):>7s} {per100:8.1f}")

    coef, use, pred = fit(rows)
    print("\nTime model (fitted on the finished runs, stats-only excluded):")
    print(f"  minutes = {coef[0]:.1f} + {coef[1] * 60:.2f} s x videos + {coef[2] * 60:.2f} s x questions "
          f"+ {coef[3] * 60 * 1000:.1f} s x 1000 generated tokens  (~{1 / (coef[3] * 60):.0f} tokens/s)")
    for r, p in zip(use, pred):
        print(f"  {r['run']:45s} real {r['minutes']:6.1f}  model {p:6.1f}")

    running = [r for r in rows if not r["finished"] and r["done"] and r["total"]]
    if running:
        print("\nRuns without a 'Finished' line (running, or interrupted):")
        for r in running:
            left = r["minutes"] / r["done"] * (r["total"] - r["done"])
            print(f"  {r['run']}: {r['done']}/{r['total']} in {r['minutes']:.0f} min, "
                  f"about {left:.0f} min left at the same pace")

    if args.predict:
        n_q, n_v = subset_size(args.predict[0])
        tpq = float(args.predict[1])
        est = coef[0] + coef[1] * n_v + coef[2] * n_q + coef[3] * n_q * tpq
        print(f"\nEstimate for {args.predict[0]} ({n_q} questions, {n_v} videos, {tpq:g} tokens/question): "
              f"{est:.0f} min ({est / 60:.1f} h)")
        print("  tokens/question seen so far: normal answers ~70; greedy 16384 ~1630; sampling ~550; "
              "constrained json ~490 (all including loops)")


if __name__ == "__main__":
    main()
