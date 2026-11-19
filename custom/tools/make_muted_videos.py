#!/usr/bin/env python3
"""Make copies of videos whose audio track is kept but replaced by digital silence.

For each source video:
  * the video stream is copied bit-for-bit (-c:v copy), so frames are identical;
  * the audio stream is replaced by `anullsrc` with the SAME sample rate,
    channel layout and DURATION as the original audio track (some originals
    have audio a few hundred ms longer than the video), encoded with AAC;
  * the result is verified: identical decoded video frames, audio track present,
    every decoded audio sample == 0, audio duration within 0.05 s of the source.

Requires the `ffmpeg`/`ffprobe` binaries (av_env provides them) and PyAV.

Usage:
  python custom/tools/make_muted_videos.py --subset check10-l1
  python custom/tools/make_muted_videos.py --video-ids 029,053
Output: custom/media/muted/<video_id>.mp4 (+ manifest.json). Existing files are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "resized_video"
DST = REPO / "custom" / "media" / "muted"


def probe_audio(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=sample_rate,channels,channel_layout,codec_name", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


def frame_hashes(path: Path) -> list[str]:
    with av.open(str(path)) as c:
        return [hashlib.md5(f.to_ndarray(format="rgb24").tobytes()).hexdigest() for f in c.decode(video=0)]


def audio_stats(path: Path) -> tuple[float, float]:
    """(max |sample|, decoded audio seconds)."""
    peak, n, sr = 0.0, 0, None
    with av.open(str(path)) as c:
        for f in c.decode(audio=0):
            a = f.to_ndarray()
            sr = f.sample_rate
            peak = max(peak, float(np.abs(a).max()) if a.size else 0.0)
            n += f.samples
    return peak, n / sr if sr else 0.0


def duration(path: Path) -> float:
    with av.open(str(path)) as c:
        return c.duration / 1e6


def mute(video_id: str) -> dict:
    src, dst = SRC / f"{video_id}.mp4", DST / f"{video_id}.mp4"
    a = probe_audio(src)
    layout = a.get("channel_layout") or ("stereo" if a["channels"] == 2 else "mono")
    _, src_audio_sec = audio_stats(src)
    if not dst.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-f", "lavfi", "-t", f"{src_audio_sec:.4f}",
                        "-i", f"anullsrc=r={a['sample_rate']}:cl={layout}",
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                        "-movflags", "+faststart", str(dst)], check=True)
    same_frames = frame_hashes(src) == frame_hashes(dst)
    peak, audio_sec = audio_stats(dst)
    d_src, d_dst = duration(src), duration(dst)
    info = {"video_id": video_id, "source": str(src.relative_to(REPO)), "muted": str(dst.relative_to(REPO)),
            "audio_params": {"sample_rate": a["sample_rate"], "channel_layout": layout},
            "identical_video_frames": same_frames, "muted_audio_peak": peak,
            "source_audio_seconds": round(src_audio_sec, 3), "muted_audio_seconds": round(audio_sec, 3),
            "duration_source": round(d_src, 2), "duration_muted": round(d_dst, 2)}
    info["ok"] = same_frames and peak == 0.0 and abs(src_audio_sec - audio_sec) < 0.05 and audio_sec > 0
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subset", help="name of a file in custom/subsets/ (without .json)")
    g.add_argument("--video-ids", help="comma-separated video ids")
    args = ap.parse_args()
    if args.subset:
        ids = json.loads((REPO / "custom" / "subsets" / f"{args.subset}.json").read_text())["qa_ids"]
        ann = {r["qa_id"]: r for r in json.loads((REPO / "data" / "test_file.json").read_text())}
        video_ids = sorted({ann[q]["video_id"] for q in ids})
    else:
        video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
    DST.mkdir(parents=True, exist_ok=True)
    manifest_path = DST / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for vid in video_ids:
        info = mute(vid)
        manifest[vid] = info
        print(("OK  " if info["ok"] else "BAD ") + json.dumps(info))
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
    if not all(manifest[v]["ok"] for v in video_ids):
        raise SystemExit("some muted videos failed verification")


if __name__ == "__main__":
    main()
