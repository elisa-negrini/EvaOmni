#!/usr/bin/env python3
"""Qualitative viewer for runs in custom/output/ (streamlit).

For one run and one question: the video, the question and options, ground truth
(answer and evidence with descriptions), the model's answer, scores and raw
response, and a timeline with GT evidence, predicted intervals and masked segments.

The video plays the audio the model received:
  * audio-mask runs: the 16 kHz waveform rebuilt with custom/audio_mask.py and the
    run's [settings], muxed onto the frames with ffmpeg. Its hash is checked against
    the one saved in output.jsonl (match = exactly what the model got; what you hear
    is that waveform after AAC encoding for the browser);
  * other runs: the file in the run's video_dir (e.g. the muted copies for av-mute).

Usage (from the repository root; open the port forwarded by VSCode):
  ~/miniconda3/envs/av_env/bin/streamlit run custom/tools/viewer.py --server.port 8501 --server.headless true
Nothing is written to custom/output/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import altair as alt
import pandas as pd
import soundfile as sf
import streamlit as st

CUSTOM = Path(__file__).resolve().parents[1]
REPO = CUSTOM.parent
sys.path.insert(0, str(CUSTOM))
import audio_mask  # noqa: E402
from run_qwen_omni import INPUT_MODES  # noqa: E402  (also puts av_env's ffmpeg on PATH)
from runinfo import load_outputs, read_run_meta  # noqa: E402

OUT = CUSTOM / "output"
ANN = REPO / "data" / "test_file.json"
ROWS = ["GT visual", "GT audio", "pred visual", "pred audio", "masked"]
warnings.filterwarnings("ignore", message=".*audioread.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


def mmss(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


@st.cache_data
def annotations() -> dict[int, dict]:
    return {r["qa_id"]: r for r in json.loads(ANN.read_text(encoding="utf-8"))}


def run_names() -> list[str]:
    return sorted(d.name for d in OUT.iterdir()
                  if d.is_dir() and not d.name.startswith("_") and read_run_meta(d) is not None)


@st.cache_data
def load_run(name: str):
    d = OUT / name
    res = d / "evaluation" / "results.json"
    questions = {q["qa_id"]: q for q in json.loads(res.read_text())["questions"]} if res.exists() else {}
    return read_run_meta(d), load_outputs(d), questions


@st.cache_data(max_entries=8, show_spinner="Rebuilding the audio the model received...")
def masked_video(video_path: str, qa_id: int, target: str, fill: str, pad: float) -> tuple[bytes, str]:
    """(mp4 bytes: original frames + masked 16 kHz audio, sha256 of the masked waveform)."""
    from qwen_omni_utils.v2_5.audio_process import process_audio_info
    # same call as the run (process_mm_info -> process_audio_info, use_audio_in_video=True)
    wav = process_audio_info([{"role": "user", "content": [{"type": "video", "video": video_path}]}],
                             use_audio_in_video=True)[0]
    masked, info = audio_mask.mask_for_question(annotations()[qa_id], wav, target, fill, pad)
    with tempfile.TemporaryDirectory(prefix="evaomni_viewer_") as tmp:
        sf.write(f"{tmp}/a.wav", masked, audio_mask.SR, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_path, "-i", f"{tmp}/a.wav",
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        f"{tmp}/out.mp4"], check=True)
        return Path(f"{tmp}/out.mp4").read_bytes(), info["audio_sha256"]


def timeline(rows: list[dict], duration: float):
    df = pd.DataFrame(rows, columns=["row", "start", "end"])
    df["from"], df["to"] = df["start"].map(mmss), df["end"].map(mmss)
    return (alt.Chart(df).mark_bar(height=14).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[0, duration]), title=None,
                axis=alt.Axis(labelExpr="timeFormat(datum.value * 1000, '%M:%S')")),
        x2="end:Q",
        y=alt.Y("row:N", sort=ROWS, scale=alt.Scale(domain=ROWS), title=None),
        color=alt.Color("row:N", sort=ROWS, legend=None),
        tooltip=["row", "from", "to"]).properties(height=170))


# ------------------------------------------------------------------ page
st.set_page_config(page_title="EvaOmni viewer", layout="wide")
ann = annotations()

with st.sidebar:
    run = st.selectbox("Run", run_names())
    if st.button("Reload run files"):
        st.cache_data.clear()
    meta, outputs, questions = load_run(run)
    settings = meta["settings"]
    levels = st.multiselect("Level", [1, 2, 3])
    caps = st.multiselect("Capability", sorted({r["capability"] for r in ann.values()}))
    outcome = st.radio("Answer (permissive)", ["any", "correct", "wrong"], horizontal=True)
    parser = st.radio("Predicted intervals from", ["permissive", "diag_no_brackets"],
                      help="diag_no_brackets also reads mm:ss,mm:ss written without [ ]")

    def keep(q: int) -> bool:
        r = ann[q]
        if str(q) not in outputs or (levels and r["level"] not in levels) or (caps and r["capability"] not in caps):
            return False
        if outcome != "any":
            acc = (questions.get(q) or {}).get("scores", {}).get("permissive", {}).get("Acc.")
            return acc is not None and (acc == 1) == (outcome == "correct")
        return True

    qa_ids = [q for q in sorted(ann) if keep(q)]
    st.caption(f"{len(qa_ids)} questions")
    if not qa_ids:
        st.stop()
    if st.session_state.get("qa") not in qa_ids:
        st.session_state["qa"] = qa_ids[0]

    def step(d: int):
        st.session_state["qa"] = qa_ids[(qa_ids.index(st.session_state["qa"]) + d) % len(qa_ids)]

    c1, c2 = st.columns(2)
    c1.button("◀ prev", on_click=step, args=(-1,), width="stretch")
    c2.button("next ▶", on_click=step, args=(1,), width="stretch")
    qa = st.selectbox("qa_id", qa_ids, key="qa",
                      format_func=lambda q: f"qa {q} · video {ann[q]['video_id']} · L{ann[q]['level']}")

rec, out, q = ann[qa], outputs[str(qa)], questions.get(qa, {})
mask = settings.get("audio_mask")
st.markdown(f"**{run}** — {meta['run'].get('description', '')}")
st.caption(f"input mode `{settings.get('input_mode', 'av')}`: {INPUT_MODES.get(settings.get('input_mode', 'av'), ('', ''))[1]}"
           + (f" · audio mask: **{mask['target']} / {mask['fill']}**, pad {mask['pad_s']} s" if mask else ""))

video_path = str(Path(settings.get("video_dir", REPO / "data" / "resized_video")) / f"{rec['video_id']}.mp4")
duration = out.get("video_seconds") or out.get("audio_seconds") or 0.0

# timeline rows
rows = [{"row": f"GT {m}", "start": a, "end": b}
        for m in ("visual", "audio") for a, b in (audio_mask.parse_time(e["time"]) for e in rec["evidence"][m])]
pred = (q.get("pred") or {}).get(parser) or {}
rows += [{"row": f"pred {m}", "start": a, "end": b} for m in ("visual", "audio") for a, b in pred.get(m, [])]
info = out.get("audio_mask") or {}
rows += [{"row": "masked", "start": a, "end": b} for a, b in info.get("segments_s", [])]

left, right = st.columns([3, 2])
with left:
    jumps = {"start": 0.0} | {f"{r['row']} {mmss(r['start'])}–{mmss(r['end'])}": r["start"] for r in rows}
    jump = st.selectbox("Play from", list(jumps))
    if mask:
        source = st.radio("Audio", ["model input (masked)", "original track"], horizontal=True)
        if source.startswith("model"):
            data, sha = masked_video(video_path, qa, mask["target"], mask["fill"], mask["pad_s"])
            same = sha == info.get("audio_sha256")
            st.video(data, start_time=int(jumps[jump]))
            st.caption(("✅ same waveform as the run (hash matches)" if same
                        else "⚠️ hash differs from the run: this is NOT exactly what the model received")
                       + (" · control_infeasible: nothing masked" if info.get("control_infeasible") else ""))
        else:
            st.video(video_path, start_time=int(jumps[jump]))
    else:
        st.video(video_path, start_time=int(jumps[jump]))
    if duration:
        st.altair_chart(timeline(rows, duration), width="stretch")

with right:
    st.markdown(f"**qa {qa}** · L{rec['level']} · {rec['capability']} · video {rec['video_id']} "
                f"({mmss(duration)})")
    st.markdown(rec["question"])
    st.text(rec["options"])
    scores = q.get("scores", {})
    table = [{"parser": p, "answer": ((q.get("pred") or {}).get(p) or {}).get("answer"), **scores.get(p, {})}
             for p in ("strict", "permissive", "diag_no_brackets") if p in scores]
    st.markdown(f"GT answer: **{rec['answer']}** · status `{out.get('status')}`")
    if table:
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    else:
        st.caption("run not scored yet (custom/score.py)")
    with st.expander("Raw response", expanded=True):
        st.code(out.get("raw_response") or out.get("error", ""), language="json", wrap_lines=True)
    with st.expander("GT evidence"):
        for m in ("visual", "audio"):
            for e in rec["evidence"][m]:
                st.markdown(f"**{m}** `{e['time']}` — {e['description']}")
    with st.expander("Model input (output.jsonl)"):
        st.json({k: v for k, v in out.items() if k not in ("raw_response", "error")}, expanded=False)
