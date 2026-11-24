"""Video masking for the visual-evidence experiment (custom/run_videomask.sh).

For one question, some frames of the frame tensor returned by qwen_omni_utils
(T x 3 x H x W, float32 with integer values 0..255, ~1 FPS) are replaced, without
cutting anything: number of frames, video tokens, timing, audio and prompt stay
as in av.

  target "evidence"  the frames of the question's visual evidence intervals:
                     frame k is hit if  start - pad <= t_k < end + pad,
                     with t_k its real timestamp (see frame_times);
  target "random"    control: the same number of segments with the same lengths,
                     at random positions that touch neither the (padded) evidence
                     nor each other (same placement code as custom/audio_mask.py).
                     Positions depend only on qa_id (fixed seed), so they are the
                     same for black and noise and across repeats. If they do not
                     fit, nothing is masked and the question is flagged
                     control_infeasible;

  fill "black"       zeros (all channels);
  fill "noise"       white Gaussian noise, independent per pixel and per frame,
                     with the per-channel mean and std of the whole original video
                     (seeded by qa_id), rounded to integers and clipped to [0, 255].

Frame pairs. The model groups frames in pairs (temporal_patch_size = 2: frames
2g and 2g+1 make one temporal token). Segments are widened outwards to whole
pairs, so that every temporal token is either fully masked or fully intact; the
random control is placed in pair units too.

No fade: frames are replaced as a whole. Frames outside the segments are untouched.
The same function (mask_for_question) can be used by a viewer, which can recompute
the masked tensor and check it against the hash saved in output.jsonl.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

import audio_mask  # parse_time, _merge, random_segments (same placement as the audio control)

SEED = 20261123     # fixed: random positions and noise depend only on qa_id
PAIR = 2            # frames per temporal token (temporal_patch_size)
TARGETS = ("evidence", "random")
FILLS = ("black", "noise")


def _seed(qa_id, what: str) -> int:
    return int(hashlib.sha256(f"{SEED}-video-{qa_id}-{what}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ frame timestamps
def frame_indices(total_frames: int, video_fps: float, fps: float) -> list[int]:
    """Indices of the source frames that qwen_omni_utils (decord backend, no
    video_start/video_end) keeps for {"fps": fps}; same functions, same formula."""
    from qwen_omni_utils.v2_5.vision_process import calculate_video_frame_range, smart_nframes
    ele = {"fps": fps}
    start, end, total = calculate_video_frame_range(ele, total_frames, video_fps)
    nframes = smart_nframes(ele, total_frames=total, video_fps=video_fps)
    return torch.linspace(start, end, nframes).round().long().tolist()


def frame_times(path: str, fps: float) -> list[float]:
    """Real timestamp (s) of each frame in the tensor qwen_omni_utils returns for path."""
    import decord
    vr = decord.VideoReader(str(path))
    total, vfps = len(vr), vr.get_avg_fps()
    return [i / vfps for i in frame_indices(total, vfps, fps)]


# ------------------------------------------------------------------ segments (in pair units)
def evidence_pairs(rec: dict, times: list[float], pad: float) -> list[tuple[int, int]]:
    """Visual evidence of the question as pair ranges [ga, gb): padded, widened to
    whole pairs, clipped to the video, merged. Intervals with no frame are dropped."""
    n_pairs = len(times) // PAIR
    segs = []
    for e in rec["evidence"]["visual"]:
        s, t = audio_mask.parse_time(e["time"])
        hit = [k for k, tk in enumerate(times[:n_pairs * PAIR]) if s - pad <= tk < t + pad]
        if hit:
            segs.append((hit[0] // PAIR, hit[-1] // PAIR + 1))
    return audio_mask._merge(segs)


def _to_frames(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(a * PAIR, b * PAIR) for a, b in pairs]


# ------------------------------------------------------------------ fill
def video_stats(frames: torch.Tensor) -> tuple[list[float], list[float]]:
    """Per-channel mean and std of all pixels of the video, float64, frame by frame
    (deterministic), rounded to 3 decimals so they do not depend on summation order."""
    a = frames.numpy() if isinstance(frames, torch.Tensor) else frames
    n = a.shape[0] * a.shape[2] * a.shape[3]
    s = np.zeros(a.shape[1]); s2 = np.zeros(a.shape[1])  # noqa: E702
    for f in a:
        f64 = f.astype(np.float64)
        s += f64.sum(axis=(1, 2))
        s2 += (f64 * f64).sum(axis=(1, 2))
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
    return [round(float(x), 3) for x in mean], [round(float(x), 3) for x in std]


def apply_mask(frames: torch.Tensor, segments: list[tuple[int, int]], fill: str,
               mean: list[float] | None, std: list[float] | None, noise_seed: int) -> torch.Tensor:
    """Copy of frames with frames [a, b) of each segment replaced by the fill."""
    out = frames.clone()
    rng = np.random.default_rng(noise_seed)
    _, c, h, w = frames.shape
    for a, b in segments:
        if fill == "black":
            out[a:b] = 0.0
        else:
            m = np.asarray(mean, np.float64).reshape(1, c, 1, 1)
            sd = np.asarray(std, np.float64).reshape(1, c, 1, 1)
            noise = rng.standard_normal((b - a, c, h, w)) * sd + m
            out[a:b] = torch.from_numpy(np.clip(np.rint(noise), 0, 255).astype(np.float32))
    return out


def frames_sha256(frames: torch.Tensor) -> str:
    a = np.ascontiguousarray(frames.numpy() if isinstance(frames, torch.Tensor) else frames, dtype=np.float32)
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()


# ------------------------------------------------------------------ per question
def mask_for_question(rec: dict, frames: torch.Tensor, times: list[float], target: str, fill: str,
                      pad: float) -> tuple[torch.Tensor, dict]:
    """(masked copy of frames, info for output.jsonl). frames itself is never modified."""
    if len(times) != frames.shape[0]:
        raise ValueError(f"{len(times)} frame timestamps for {frames.shape[0]} frames")
    span = lambda segs: [[round(times[a], 3), round(times[b - 1], 3)] for a, b in segs]  # noqa: E731
    n_pairs = frames.shape[0] // PAIR
    evidence = evidence_pairs(rec, times, pad)
    info: dict = {"target": target, "fill": fill}
    if not evidence:
        info["no_evidence_frames"] = True  # every visual interval is outside the frames
    if target == "evidence":
        pairs = evidence
    else:
        pairs = audio_mask.random_segments(evidence, n_pairs, random.Random(_seed(rec["qa_id"], "positions")))
        info["evidence_frames"] = [list(s) for s in _to_frames(evidence)]
        info["control_infeasible"] = pairs is None
        if pairs is None:
            info.update(segments_frames=[], segments_s=[], masked_frames=0,
                        frames_sha256=frames_sha256(frames))
            return frames, info
    segments = _to_frames(pairs)
    mean = std = None
    if fill == "noise" and segments:
        mean, std = video_stats(frames)
    out = apply_mask(frames, segments, fill, mean, std, _seed(rec["qa_id"], "noise"))
    info["segments_frames"] = [list(s) for s in segments]          # [first, last + 1)
    info["segments_s"] = span(segments)                            # timestamps of first and last masked frame
    info["masked_frames"] = sum(b - a for a, b in segments)
    info["masked_fraction"] = round(info["masked_frames"] / frames.shape[0], 4)
    if fill == "noise":
        info.update(noise_mean=mean, noise_std=std)
    info["frames_sha256"] = frames_sha256(out)
    return out, info


def settings_entry(target: str, fill: str, pad: float) -> dict:
    """What goes into [settings] of info.txt."""
    return {
        "target": target,
        "fill": fill,
        "pad_s": pad,
        "seed": SEED,
        "where": ("custom/video_mask.py, on a per-question copy of the frame tensor from "
                  "qwen_omni_utils (after its resize), just before the request; same number of "
                  "frames, audio untouched"),
        "frame_time": ("real timestamp of each kept frame: indices from qwen_omni_utils' own "
                       "calculate_video_frame_range/smart_nframes + linspace(...).round(), / source fps"),
        "segments": ("evidence: frames with start - pad <= t < end + pad for each visual interval"
                     if target == "evidence"
                     else "random: same number and lengths as the padded evidence, uniform positions "
                          "outside it and non-overlapping, seed from qa_id; if they do not fit, no mask "
                          "and control_infeasible=true"),
        "pair_alignment": "segments widened outwards to whole frame pairs (temporal_patch_size 2)",
        "fill_how": ("zeros" if fill == "black"
                     else "white Gaussian noise per pixel and frame, per-channel mean/std of the whole "
                          "original video, seed from qa_id, rounded, clip [0, 255]"),
    }
