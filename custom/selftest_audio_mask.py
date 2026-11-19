#!/usr/bin/env python3
"""Self-test for custom/audio_mask.py on synthetic waveforms (no GPU, no video).

  (a) evidence segments: padding, clipping at the track ends, merging
  (b) silence: zeros inside, untouched outside, same length/dtype, input not modified
  (c) noise: RMS of the whole track inside, deterministic
  (d) random control: same lengths as the evidence, no overlap, inside the track,
      same positions for silence and noise, different across qa_ids
  (e) random control that does not fit: nothing masked, control_infeasible
  (f) all 1027 annotations: evidence parsed, random control placed (5-minute track)

Exit code != 0 on failure.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_mask as am  # noqa: E402

SR = am.SR
F = round(am.FADE_S * SR)
ANN = Path(__file__).resolve().parents[1] / "data" / "test_file.json"
failures: list[str] = []


def check(cond: bool, msg: str):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def rec(qa_id, *times):
    return {"qa_id": qa_id, "evidence": {"audio": [{"time": t} for t in times]}}


def track(seconds, seed=0):
    return (np.random.default_rng(seed).standard_normal(seconds * SR) * 0.1).astype(np.float32)


def overlaps(x, y):
    return any(a < d and c < b for a, b in x for c, d in y)


print("\n(a) evidence segments")
r = rec(1, "[00:10,00:12]", "[00:11,00:20]", "[00:40,00:41]")
check(am.evidence_segments(r, 60 * SR, 1.0) == [(9 * SR, 21 * SR), (39 * SR, 42 * SR)],
      "pad 1 s, overlapping intervals merged: [9,21] [39,42]")
check(am.evidence_segments(rec(1, "[00:00,00:02]", "[00:58,01:05]"), 60 * SR, 1.0)
      == [(0, 3 * SR), (57 * SR, 60 * SR)], "clipped at 0 and at the end of the track")

print("\n(b) silence on the evidence")
wav = track(60)
orig = wav.copy()
out, info = am.mask_for_question(r, wav, "evidence", "silence", 1.0)
inside = np.zeros(len(wav), bool)
for a, b in [(9 * SR, 21 * SR), (39 * SR, 42 * SR)]:
    inside[a:b] = True
    check(np.all(out[a + F:b - F] == 0), f"zeros in [{a / SR:.0f},{b / SR:.0f}] (fades excluded)")
check(np.array_equal(out[~inside], wav[~inside]), "outside the segments: identical samples")
check(out.shape == wav.shape and out.dtype == np.float32, "same length and dtype")
check(np.array_equal(wav, orig), "input waveform not modified")
check(info["masked_seconds"] == 15.0 and info["segments_s"] == [[9.0, 21.0], [39.0, 42.0]],
      f"info: masked_seconds={info['masked_seconds']}, segments={info['segments_s']}")
check(info["audio_sha256"] == am.waveform_sha256(out), "hash of the masked waveform")

print("\n(c) noise on the evidence")
out_n, info_n = am.mask_for_question(r, wav, "evidence", "noise", 1.0)
rms_track = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
rms_in = float(np.sqrt(np.mean(out_n[9 * SR + F:21 * SR - F].astype(np.float64) ** 2)))
check(abs(rms_in / rms_track - 1) < 0.02, f"RMS inside {rms_in:.4f} ~ track RMS {rms_track:.4f}")
check(np.array_equal(out_n[~inside], wav[~inside]), "outside the segments: identical samples")
check(am.mask_for_question(r, wav, "evidence", "noise", 1.0)[1]["audio_sha256"] == info_n["audio_sha256"],
      "deterministic (same hash twice)")

print("\n(d) random control")
_, ri_s = am.mask_for_question(r, wav, "random", "silence", 1.0)
out_rn, ri_n = am.mask_for_question(r, wav, "random", "noise", 1.0)
segs = [(round(a * SR), round(b * SR)) for a, b in ri_s["segments_s"]]
ev = [(9 * SR, 21 * SR), (39 * SR, 42 * SR)]
check(sorted(b - a for a, b in segs) == sorted(b - a for a, b in ev), f"same lengths as evidence: {segs}")
check(not overlaps(segs, ev) and not any(overlaps([s], segs[i + 1:]) for i, s in enumerate(segs)),
      "no overlap with the evidence or with each other")
check(all(0 <= a < b <= len(wav) for a, b in segs), "inside the track")
check(ri_s["segments_s"] == ri_n["segments_s"], "same positions for silence and noise")
check(not ri_s["control_infeasible"] and ri_s["masked_seconds"] == 15.0, "feasible, 15 s masked")
check(np.array_equal(out_rn[inside], wav[inside]), "evidence left untouched")
check(am.mask_for_question(rec(2, *[e["time"] for e in r["evidence"]["audio"]]), wav, "random", "silence",
                           1.0)[1]["segments_s"] != ri_s["segments_s"], "different qa_id -> different positions")

print("\n(e) random control that does not fit")
r_big = rec(3, "[00:05,00:25]")
out_i, info_i = am.mask_for_question(r_big, track(30), "random", "silence", 1.0)
check(info_i["control_infeasible"] and info_i["masked_seconds"] == 0.0 and np.array_equal(out_i, track(30)),
      "22 s evidence in a 30 s track: control_infeasible, audio unchanged")

print("\n(f) all annotations (5-minute track)")
records = json.loads(ANN.read_text(encoding="utf-8"))
n_seg, infeasible = 0, []
for q in records:
    end = max(am.parse_time(e["time"])[1] for e in q["evidence"]["audio"])
    n = max(300, end + 5) * SR
    ev = am.evidence_segments(q, n, 1.0)
    n_seg += len(ev)
    rs = am.random_segments(ev, n, random.Random(am._seed(q["qa_id"], "positions")))
    if rs is None:
        infeasible.append(q["qa_id"])
    elif (overlaps(rs, ev) or any(overlaps([s], rs[i + 1:]) for i, s in enumerate(rs))
          or sorted(b - a for a, b in rs) != sorted(b - a for a, b in ev)):
        failures.append(f"qa {q['qa_id']}: bad random segments")
check(True, f"{len(records)} questions parsed, {n_seg} evidence segments, "
            f"infeasible on a 5-minute track: {infeasible}")

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED")
sys.exit(1 if failures else 0)
