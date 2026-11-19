#!/usr/bin/env python3
"""Create the reusable qa_id subsets in custom/subsets/ (run once).

Subsets (all drawn with fixed seeds, saved with their recipe):
  sub200       200 questions stratified by level, proportional allocation
               (85 / 85 / 30), seed 20261113. Used for run-to-run variability
               and future experiments.
  sub100       100 questions stratified by level, proportional allocation
               (42 / 43 / 15), seed 100. Used for test runs.
  check10-l1   10 Level-1 questions from 10 distinct videos, spread across the
               three Level-1 capabilities, seed 1. Used for the audio check.
  check50      50 questions stratified by level (21 / 21 / 8), distinct videos,
               seed 2. Used for the response-format check.
  onepervideo  the lowest qa_id of each of the 392 videos. Used for the input
               statistics / truncation check.

Refuses to overwrite an existing subset file, so subsets stay frozen.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANN = REPO / "data" / "test_file.json"
OUT = REPO / "custom" / "subsets"


def stratified(records, per_level: dict[int, int], seed: int, distinct_videos: bool = False,
               spread_key: str | None = None):
    rng = random.Random(seed)
    chosen = []
    for level, k in sorted(per_level.items()):
        pool = sorted((r for r in records if r["level"] == level), key=lambda r: r["qa_id"])
        rng.shuffle(pool)
        if spread_key:  # round-robin over the values of spread_key (e.g. capability)
            buckets = defaultdict(list)
            for r in pool:
                buckets[r[spread_key]].append(r)
            order = sorted(buckets)
            pool = []
            while any(buckets.values()):
                for key in order:
                    if buckets[key]:
                        pool.append(buckets[key].pop(0))
        seen_videos, picked = set(), []
        for r in pool:
            if distinct_videos and r["video_id"] in seen_videos:
                continue
            picked.append(r)
            seen_videos.add(r["video_id"])
            if len(picked) == k:
                break
        assert len(picked) == k, (level, len(picked), k)
        chosen += picked
    return sorted(r["qa_id"] for r in chosen)


def main():
    records = json.loads(ANN.read_text(encoding="utf-8"))
    ann_sha = hashlib.sha256(ANN.read_bytes()).hexdigest()
    first_per_video = {}
    for r in sorted(records, key=lambda r: r["qa_id"]):
        first_per_video.setdefault(r["video_id"], r["qa_id"])

    subsets = {
        "sub200": dict(qa_ids=stratified(records, {1: 85, 2: 85, 3: 30}, seed=20261113),
                       recipe="stratified by level, proportional (85/85/30), random.Random(20261113)"),
        "sub100": dict(qa_ids=stratified(records, {1: 42, 2: 43, 3: 15}, seed=100),
                       recipe="stratified by level, proportional (42/43/15), random.Random(100)"),
        "check10-l1": dict(qa_ids=stratified(records, {1: 10}, seed=1, distinct_videos=True,
                                             spread_key="capability"),
                           recipe="Level 1 only, round-robin over capabilities, distinct videos, random.Random(1)"),
        "check50": dict(qa_ids=stratified(records, {1: 21, 2: 21, 3: 8}, seed=2, distinct_videos=True),
                        recipe="stratified by level (21/21/8), distinct videos, random.Random(2)"),
        "onepervideo": dict(qa_ids=sorted(first_per_video.values()),
                            recipe="lowest qa_id of each video (392 videos)"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    by_id = {r["qa_id"]: r for r in records}
    for name, info in subsets.items():
        path = OUT / f"{name}.json"
        if path.exists():
            print(f"[keep] {path} already exists, not overwritten")
            continue
        ids = info["qa_ids"]
        counts = defaultdict(int)
        for q in ids:
            counts[f"level{by_id[q]['level']}"] += 1
        payload = {"name": name, "recipe": info["recipe"], "annotation_sha256": ann_sha,
                   "n": len(ids), "counts_by_level": dict(sorted(counts.items())),
                   "n_videos": len({by_id[q]["video_id"] for q in ids}), "qa_ids": ids}
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"[new]  {path}: n={len(ids)} {dict(sorted(counts.items()))} videos={payload['n_videos']}")


if __name__ == "__main__":
    sys.exit(main())
