#!/usr/bin/env python3
"""Self-test for custom/video_mask.py on synthetic frame tensors (no GPU, no video file).

  (a) frame timestamps: same indices as qwen_omni_utils (1 FPS, odd count, 768 cap)
  (b) evidence segments: padding, pair alignment, clipping, merging
  (c) black: zeros inside, untouched outside, same shape/dtype, input not modified
  (d) noise: integers in [0, 255], per-channel mean/std of the video, deterministic
  (e) random control: same lengths, pair-aligned, no overlap, inside the video,
      same positions for black and noise, different across qa_ids
  (f) random control that does not fit: nothing masked, control_infeasible
  (g) all 1027 annotations: evidence parsed, random control placed (5-minute video)

Exit code != 0 on failure.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_mask  # noqa: E402
import video_mask as vm  # noqa: E402

ANN = Path(__file__).resolve().parents[1] / "data" / "test_file.json"
failures: list[str] = []


def check(cond: bool, msg: str):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def rec(qa_id, *times):
    return {"qa_id": qa_id, "evidence": {"visual": [{"time": t} for t in times]}}


def video(n, seed=0, h=28, w=48):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn((n, 3, h, w), generator=g) * torch.tensor([20.0, 40.0, 30.0]).view(1, 3, 1, 1) \
        + torch.tensor([60.0, 120.0, 150.0]).view(1, 3, 1, 1)  # channels with different statistics
    return v.round().clamp(0, 255)


def overlaps(x, y):
    return any(a < d and c < b for a, b in x for c, d in y)


print("\n(a) frame timestamps")
idx = vm.frame_indices(60, 1.0, 1.0)
check(idx == list(range(60)), "60 frames at 1 fps -> indices 0..59")
idx = vm.frame_indices(293, 1.0, 1.0)
check(len(idx) == 292 and idx[0] == 0 and idx[-1] == 292 and max(abs(i - k) for k, i in enumerate(idx)) == 1,
      "293 frames -> 292 kept, indices drift by up to 1 frame (0 ... 292)")
idx = vm.frame_indices(779, 1.0, 1.0)
check(len(idx) == 768 and idx[-1] == 778, "779 frames -> capped at 768 (0 ... 778)")

print("\n(b) evidence segments (pair units)")
T = [float(k) for k in range(60)]
r = rec(1, "[00:10,00:12]", "[00:11,00:20]", "[00:40,00:41]")
# hits: 9..20 and 39..41 -> pairs [4,11) (frames 8..21) and [19,21) (frames 38..41)
check(vm.evidence_pairs(r, T, 1.0) == [(4, 11), (19, 21)], f"pad 1 s, pairs, merged: {vm.evidence_pairs(r, T, 1.0)}")
check(vm.evidence_pairs(rec(1, "[00:00,00:02]", "[00:58,01:05]"), T, 1.0) == [(0, 2), (28, 30)],
      "clipped at 0 and at the end of the video")
check(vm.evidence_pairs(rec(1, "[01:10,01:20]"), T, 1.0) == [], "interval after the end -> no frames")
check(vm.evidence_pairs(rec(1, "[00:11,00:12]"), T, 0.0) == [(5, 6)],
      "pad 0, [11,12]: frame 11 only -> widened to the pair (10,11)")
T_odd = [i / 1.0 for i in vm.frame_indices(293, 1.0, 1.0)]
ev = vm.evidence_pairs(rec(1, "[04:00,04:05]"), T_odd, 0.0)
hit = [k for k, t in enumerate(T_odd) if 240 <= t < 245]
check(ev == [(hit[0] // 2, hit[-1] // 2 + 1)], f"real timestamps used (drifting indices): frames {hit}")

print("\n(c) black on the evidence")
v = video(60)
orig = v.clone()
out, info = vm.mask_for_question(r, v, T, "evidence", "black", 1.0)
inside = torch.zeros(60, dtype=torch.bool)
for a, b in [(8, 22), (38, 42)]:
    inside[a:b] = True
check(bool((out[inside] == 0).all()), "zeros on frames 8..21 and 38..41")
check(torch.equal(out[~inside], v[~inside]), "outside the segments: identical frames")
check(out.shape == v.shape and out.dtype == v.dtype, "same shape and dtype")
check(torch.equal(v, orig), "input tensor not modified")
check(info["segments_frames"] == [[8, 22], [38, 42]] and info["masked_frames"] == 18
      and info["segments_s"] == [[8.0, 21.0], [38.0, 41.0]], f"info: {info}")
check(info["frames_sha256"] == vm.frames_sha256(out), "hash of the masked tensor")

print("\n(d) noise on the evidence")
out_n, info_n = vm.mask_for_question(r, v, T, "evidence", "noise", 1.0)
x = out_n[inside]
check(bool((x == x.round()).all() and (x >= 0).all() and (x <= 255).all()), "integers in [0, 255]")
m_vid = v.double().mean(dim=(0, 2, 3)); s_vid = v.double().std(dim=(0, 2, 3))
m_in = x.double().mean(dim=(0, 2, 3)); s_in = x.double().std(dim=(0, 2, 3))
check(bool(((m_in - m_vid).abs() < 1.5).all() and ((s_in / s_vid - 1).abs() < 0.03).all()),
      f"per-channel mean {m_in.numpy().round(1)} ~ {m_vid.numpy().round(1)}, "
      f"std {s_in.numpy().round(1)} ~ {s_vid.numpy().round(1)}")
check(np.allclose(info_n["noise_mean"], m_vid.numpy(), atol=1e-3), "noise_mean = video mean")
check(torch.equal(out_n[~inside], v[~inside]), "outside the segments: identical frames")
check(not torch.equal(out_n[8], out_n[9]), "noise differs from frame to frame")
check(vm.mask_for_question(r, v, T, "evidence", "noise", 1.0)[1]["frames_sha256"] == info_n["frames_sha256"],
      "deterministic (same hash twice)")

print("\n(e) random control")
_, ri_b = vm.mask_for_question(r, v, T, "random", "black", 1.0)
out_rn, ri_n = vm.mask_for_question(r, v, T, "random", "noise", 1.0)
segs = [tuple(s) for s in ri_b["segments_frames"]]
evf = [(8, 22), (38, 42)]
check(sorted(b - a for a, b in segs) == sorted(b - a for a, b in evf), f"same lengths as evidence: {segs}")
check(all(a % 2 == 0 and b % 2 == 0 for a, b in segs), "aligned to frame pairs")
check(not overlaps(segs, evf) and not any(overlaps([s], segs[i + 1:]) for i, s in enumerate(segs)),
      "no overlap with the evidence or with each other")
check(all(0 <= a < b <= 60 for a, b in segs), "inside the video")
check(ri_b["segments_frames"] == ri_n["segments_frames"], "same positions for black and noise")
check(not ri_b["control_infeasible"] and ri_b["masked_frames"] == 18, "feasible, 18 frames masked")
check(torch.equal(out_rn[inside], v[inside]), "evidence left untouched")
r2 = rec(2, *[e["time"] for e in r["evidence"]["visual"]])
check(vm.mask_for_question(r2, v, T, "random", "black", 1.0)[1]["segments_frames"] != ri_b["segments_frames"],
      "different qa_id -> different positions")

print("\n(f) random control that does not fit")
v30 = video(30)
out_i, info_i = vm.mask_for_question(rec(3, "[00:05,00:25]"), v30, T[:30], "random", "black", 1.0)
check(info_i["control_infeasible"] and info_i["masked_frames"] == 0 and torch.equal(out_i, v30),
      "22 s evidence in a 30 s video: control_infeasible, frames unchanged")

print("\n(g) all annotations (5-minute video, 1 fps)")
records = json.loads(ANN.read_text(encoding="utf-8"))
n_seg, infeasible = 0, []
for q in records:
    end = max(audio_mask.parse_time(e["time"])[1] for e in q["evidence"]["visual"])
    n = max(300, end + 5) // 2 * 2
    times = [float(k) for k in range(n)]
    ev = vm.evidence_pairs(q, times, 1.0)
    n_seg += len(ev)
    if not ev:
        failures.append(f"qa {q['qa_id']}: no evidence frames")
        continue
    rs = audio_mask.random_segments(ev, n // 2, random.Random(vm._seed(q["qa_id"], "positions")))
    if rs is None:
        infeasible.append(q["qa_id"])
    elif (overlaps(rs, ev) or any(overlaps([s], rs[i + 1:]) for i, s in enumerate(rs))
          or sorted(b - a for a, b in rs) != sorted(b - a for a, b in ev)):
        failures.append(f"qa {q['qa_id']}: bad random segments")
check(True, f"{len(records)} questions parsed, {n_seg} evidence segments, "
            f"infeasible on a 5-minute video: {infeasible}")

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED")
sys.exit(1 if failures else 0)
