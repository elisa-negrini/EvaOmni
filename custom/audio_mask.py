"""Audio masking for the audio-evidence experiment (custom/run_audiomask.sh).

For one question, part of the video's audio waveform (16 kHz mono, as returned by
qwen_omni_utils) is replaced, without cutting anything, so length and timing stay
the same as in av:

  target "evidence"  the question's audio evidence intervals, +/- pad seconds,
                     clipped to the track and merged where they overlap;
  target "random"    control: the same number of segments with the same lengths,
                     at random positions that touch neither the (padded) evidence
                     nor each other. Positions depend only on qa_id (fixed seed),
                     so they are the same for silence and noise and across repeats.
                     If they do not fit, nothing is masked and the question is
                     flagged control_infeasible;

  fill "silence"     zeros;
  fill "noise"       white Gaussian noise with the RMS of the whole original track
                     (seeded by qa_id), clipped to [-1, 1].

Each segment fades from the original into the fill over its first FADE_S seconds
and back over its last FADE_S seconds; samples outside the segments are untouched.

The same function (mask_for_question) is used by the run and by the viewer, which
recomputes the masked audio and checks it against the hash saved in output.jsonl.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np

SR = 16000          # qwen_omni_utils sample rate
FADE_S = 0.010      # fade at each end of a segment, inside the segment
SEED = 20261118     # fixed: random positions and noise depend only on qa_id
TARGETS = ("evidence", "random")
FILLS = ("silence", "noise")


def _seed(qa_id, what: str) -> int:
    return int(hashlib.sha256(f"{SEED}-{qa_id}-{what}".encode()).hexdigest()[:8], 16)


def parse_time(t: str) -> tuple[int, int]:
    """'[mm:ss,mm:ss]' -> (start, end) in seconds."""
    a, b = t.strip().strip("[]").split(",")
    sec = lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])  # noqa: E731
    return sec(a), sec(b)


def _merge(segs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(segs):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def evidence_segments(rec: dict, n_samples: int, pad: float) -> list[tuple[int, int]]:
    """Audio evidence of the question as sample ranges [a, b): padded, clipped, merged."""
    segs = []
    for e in rec["evidence"]["audio"]:
        s, t = parse_time(e["time"])
        a = max(0, round((s - pad) * SR))
        b = min(n_samples, round((t + pad) * SR))
        if b > a:
            segs.append((a, b))
    return _merge(segs)


def random_segments(evidence: list[tuple[int, int]], n_samples: int,
                    rng: random.Random) -> list[tuple[int, int]] | None:
    """Segments with the evidence lengths, uniformly placed in the free space
    (longest first). None if one does not fit."""
    placed: list[tuple[int, int]] = []
    for length in sorted((b - a for a, b in evidence), reverse=True):
        free, pos = [], 0
        for a, b in _merge(evidence + placed):
            if a > pos:
                free.append((pos, a))
            pos = max(pos, b)
        if pos < n_samples:
            free.append((pos, n_samples))
        starts = [(a, b - length) for a, b in free if b - a >= length]  # inclusive start ranges
        total = sum(e - s + 1 for s, e in starts)
        if total == 0:
            return None
        u = rng.randrange(total)
        for s, e in starts:
            if u <= e - s:
                placed.append((s + u, s + u + length))
                break
            u -= e - s + 1
    return sorted(placed)


def apply_mask(wav: np.ndarray, segments: list[tuple[int, int]], fill: str,
               noise_rms: float, noise_seed: int) -> np.ndarray:
    """Copy of wav with each segment replaced by the fill (fades inside the segment)."""
    out = wav.copy()
    rng = np.random.default_rng(noise_seed)
    fade = round(FADE_S * SR)
    for a, b in segments:
        n = b - a
        new = np.zeros(n) if fill == "silence" else rng.standard_normal(n) * noise_rms
        w = np.ones(n)
        f = min(fade, n // 2)
        if f:
            ramp = np.arange(1, f + 1) / (f + 1)
            w[:f], w[n - f:] = ramp, ramp[::-1]
        mixed = wav[a:b].astype(np.float64) * (1 - w) + new * w
        out[a:b] = np.clip(mixed, -1.0, 1.0)
    return out


def waveform_sha256(wav: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(wav, dtype=np.float32).tobytes()).hexdigest()


def mask_for_question(rec: dict, wav: np.ndarray, target: str, fill: str,
                      pad: float) -> tuple[np.ndarray, dict]:
    """(masked copy of wav, info for output.jsonl). wav itself is never modified."""
    s = lambda x: round(x / SR, 3)  # noqa: E731
    evidence = evidence_segments(rec, len(wav), pad)
    info: dict = {"target": target, "fill": fill}
    if target == "evidence":
        segments = evidence
    else:
        segments = random_segments(evidence, len(wav), random.Random(_seed(rec["qa_id"], "positions")))
        info["evidence_segments_s"] = [[s(a), s(b)] for a, b in evidence]
        info["control_infeasible"] = segments is None
        if segments is None:
            info.update(segments_s=[], masked_seconds=0.0, audio_sha256=waveform_sha256(wav))
            return wav, info
    noise_rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    out = apply_mask(wav, segments, fill, noise_rms, _seed(rec["qa_id"], "noise"))
    info["segments_s"] = [[s(a), s(b)] for a, b in segments]
    info["masked_seconds"] = s(sum(b - a for a, b in segments))
    if fill == "noise":
        info["noise_rms"] = round(noise_rms, 6)
    info["audio_sha256"] = waveform_sha256(out)
    return out, info


def settings_entry(target: str, fill: str, pad: float) -> dict:
    """What goes into [settings] of info.txt."""
    return {
        "target": target,
        "fill": fill,
        "pad_s": pad,
        "fade_s": FADE_S,
        "seed": SEED,
        "where": ("custom/audio_mask.py, on a per-question copy of the 16 kHz waveform from "
                  "qwen_omni_utils, just before the request; same length, frames untouched"),
        "segments": ("evidence: the question's audio evidence +/- pad, clipped, merged" if target == "evidence"
                     else "random: same number and lengths as the padded evidence, uniform positions "
                          "outside it and non-overlapping, seed from qa_id; if they do not fit, no mask "
                          "and control_infeasible=true"),
        "fill_how": ("zeros" if fill == "silence"
                     else "white Gaussian noise, RMS of the whole original track, seed from qa_id, clip [-1, 1]"),
        "fade": "linear, inside each segment (outside samples untouched)",
    }
