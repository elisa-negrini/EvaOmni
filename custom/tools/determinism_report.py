#!/usr/bin/env python3
"""Determinism report for the pairs of runs made by custom/run_determinism.sh.

For every pair of run folders <name>-a / <name>-b in custom/output/ (run with identical
settings in two separate processes) it prints how many answers are identical
character by character, how many have the same letter (permissive parser), and,
for the differing ones, after how many characters they diverge.
Also checks that the two runs had the same settings and the same inputs
(prompt tokens; audio hash for masking runs).

Usage:
  python custom/tools/determinism_report.py [--filter det48]
Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CUSTOM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUSTOM))
import score  # noqa: E402
from runinfo import read_run_meta  # noqa: E402

OUT = CUSTOM / "output"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", default="det48", help="only run names containing this text")
    args = ap.parse_args()
    names = sorted(d.name for d in OUT.iterdir() if d.is_dir() and args.filter in d.name)
    pairs = [(n, n[:-2] + "-b") for n in names if n.endswith("-a") and n[:-2] + "-b" in names]
    if not pairs:
        sys.exit(f"no -a/-b pairs containing {args.filter!r} in {OUT}")
    print(f"{'pair':50s} {'n':>4s} {'identical':>10s} {'same letter':>12s} {'diverge at (median chars)':>26s}  checks")
    for a_name, b_name in pairs:
        ma, mb = read_run_meta(OUT / a_name), read_run_meta(OUT / b_name)
        same_settings = ma and mb and ma["settings"] == mb["settings"]
        a, b = score.load_outputs(OUT / a_name), score.load_outputs(OUT / b_name)
        ks = [k for k in a if k in b and a[k].get("status") == "ok" and b[k].get("status") == "ok"]
        if not ks:
            print(f"{a_name[:-2]:50s} (no answered questions in common)")
            continue
        ident = [a[k]["raw_response"] == b[k]["raw_response"] for k in ks]
        letter = lambda r, k: (score.parse_permissive(r[k]["raw_response"], k)[0] or {}).get("answer")  # noqa: E731
        same_letter = sum(letter(a, k) == letter(b, k) for k in ks)
        div = sorted(len(os.path.commonprefix([a[k]["raw_response"], b[k]["raw_response"]]))
                     for k, i in zip(ks, ident) if not i)
        same_in = all(a[k].get("prompt_tokens") == b[k].get("prompt_tokens")
                      and (a[k].get("audio_mask") or {}).get("audio_sha256")
                      == (b[k].get("audio_mask") or {}).get("audio_sha256") for k in ks)
        checks = ("settings ok" if same_settings else "SETTINGS DIFFER") + ", " + (
            "inputs ok" if same_in else "INPUTS DIFFER")
        print(f"{a_name[:-2]:50s} {len(ks):4d} {sum(ident):5d} ({100 * sum(ident) / len(ks):3.0f}%) "
              f"{same_letter:5d} ({100 * same_letter / len(ks):3.0f}%) "
              f"{(str(div[len(div) // 2]) if div else '-'):>26s}  {checks}")


if __name__ == "__main__":
    main()
