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
import re
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


COLORS = ["#2563eb", "#ea580c", "#60a5fa", "#fb923c", "#6b7280"]  # visual blue, audio orange, pred lighter, masked grey
TIMELINE_PLOT_H = 175  # px, plot area; with the axis the chart is ~205 px
# Run page fills the window without scrolling: fixed header line + top row (video | question box)
# + timeline. TOP_ROW = what is left of the viewport for the video and the scrollable question box.
TOP_ROW = "calc(100vh - 330px)"
RUN_PAGE_CSS = f"""<style>
[data-testid="stMain"] {{ overflow: hidden !important; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 3.2rem !important; padding-bottom: 0 !important; }}
div.st-key-videobox video {{ max-height: {TOP_ROW} !important; width: 100%; object-fit: contain; background: #000; }}
div.st-key-qbox {{ height: {TOP_ROW} !important; max-height: {TOP_ROW} !important; flex: none !important;
                   overflow-y: auto !important; flex-wrap: nowrap !important;
                   border: 1px solid rgba(128,128,128,0.3); border-radius: 0.5rem; padding: 0.6rem 0.9rem; }}
</style>"""


def timeline(rows: list[dict], domain: tuple[float, float]):
    """Bars at their real times; bars shorter than 0.5% of the visible span are drawn
    at that width so 1-2 s intervals stay visible (tooltip shows the real times).
    Rows with no interval say "none". The x axis is fixed to the domain (no zoom/pan:
    scrolling over the chart used to zoom it out far beyond the video's length)."""
    lo, hi = domain
    df = pd.DataFrame(rows, columns=["row", "start", "end"])
    df["from"], df["to"] = df["start"].map(mmss), df["end"].map(mmss)
    df["seconds"] = (df["end"] - df["start"]).round(1)
    df["draw_end"] = df[["end"]].max(axis=1).clip(lower=df["start"] + 0.005 * (hi - lo))
    empty = pd.DataFrame({"row": [r for r in ROWS if r not in set(df["row"])], "x": lo})
    x = alt.X("start:Q", scale=alt.Scale(domain=[lo, hi], nice=False, zero=False), title=None,
              axis=alt.Axis(labelExpr="timeFormat(datum.value * 1000, '%M:%S')", grid=True, tickCount=20))
    y = alt.Y("row:N", sort=ROWS, scale=alt.Scale(domain=ROWS), title=None,
              axis=alt.Axis(labelFontSize=13, labelLimit=200))
    bars = alt.Chart(df).mark_bar(height=20, cornerRadius=2, clip=True).encode(
        x=x, x2="draw_end:Q", y=y,
        color=alt.Color("row:N", scale=alt.Scale(domain=ROWS, range=COLORS), legend=None),
        tooltip=["row", "from", "to", "seconds"])
    none = alt.Chart(empty).mark_text(align="left", dx=4, color="#9ca3af", fontStyle="italic").encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[lo, hi], nice=False, zero=False)), y=y, text=alt.value("none"))
    return (bars + none).properties(height=TIMELINE_PLOT_H)


def option_lines(options: str, gt: str, pred: str | None) -> str:
    """Options as markdown, with the GT and the predicted option marked."""
    out = []
    for line in options.splitlines():
        letter, tags = line[:1], []
        if letter == gt:
            tags.append(":green-background[**GT**]")
        if letter == pred:
            tags.append(":green-background[**pred ✓**]" if pred == gt else ":red-background[**pred ✗**]")
        text = line.replace("*", "\\*")
        out.append(f"- {' '.join(tags)} **{text}**" if tags else f"- {text}")
    return "\n".join(out)


# ------------------------------------------------------------------ page
st.set_page_config(page_title="EvaOmni viewer", layout="wide")
ann = annotations()

PARSERS = ("strict", "permissive", "diag_no_brackets")
BASELINE = "qwen3omni_av_greedy_maxtok-2048"  # the masking runs say "otherwise as" this run
MASK_ORDER = {("evidence", "silence"): 1, ("evidence", "noise"): 2, ("random", "silence"): 3, ("random", "noise"): 4}


def go(page: str, run: str | None = None):
    st.session_state["page"] = page
    if run:
        st.session_state["run"] = run
    st.session_state.pop("masked_table", None)  # forget the row clicked in the table


@st.cache_data
def masked_runs() -> list[dict]:
    """Baseline + full-benchmark audio-mask runs, in table order, with their overall scores."""
    rows = []
    for name in run_names():
        meta = read_run_meta(OUT / name)
        s, mask = meta["settings"], meta["settings"].get("audio_mask")
        if not (name == BASELINE or (mask and s.get("subset") == "full")):
            continue
        res = OUT / name / "evaluation" / "results.json"
        summary = json.loads(res.read_text())["summary"] if res.exists() else {}
        row = {"run": name,
               "audio masked": {"evidence": "evidence", "random": "random (control)"}[mask["target"]] if mask else "— (baseline)",
               "filled with": mask["fill"] if mask else "—",
               "repeat": name.rsplit("_", 1)[-1] if mask else "—",
               "n": s.get("subset_n"),
               "_order": (MASK_ORDER.get((mask["target"], mask["fill"]), 9) if mask else 0, name)}
        for p in PARSERS:
            overall = (summary.get(p) or {}).get("Overall") or {}
            for k in ("Acc.", "C.G.", "B.G."):
                row[f"{p} {k}"] = overall.get(k)
        rows.append(row)
    return sorted(rows, key=lambda r: r["_order"])


def page_home():
    st.title("EvaOmni viewer")
    st.caption("Qwen3-Omni runs in custom/output/")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("🔇 Audio masked runs")
        st.markdown("Runs where part of the audio track was replaced by silence or noise "
                    "(the audio evidence, or random control segments), with their scores.")
        st.button("Open", key="home_masked", on_click=go, args=("masked",), type="primary")
    with c2, st.container(border=True):
        st.subheader("📂 All runs")
        st.markdown("Every run in custom/output/, chosen from the sidebar menu.")
        st.button("Open", key="home_all", on_click=go, args=("all",))


def page_masked():
    st.button("← Home", on_click=go, args=("home",))
    st.title("🔇 Audio masked runs")
    st.markdown(
        "All 1027 questions; greedy decoding, max_tokens 2048, otherwise identical to the baseline. "
        "Only the audio the model receives changes (frames untouched):\n"
        "- **evidence**: the question's annotated audio evidence (± 1 s pad, merged) is masked;\n"
        "- **random (control)**: the same number of segments with the same lengths, placed at random "
        "outside the evidence (7 questions cannot fit them: nothing masked there);\n"
        "- **silence** = zeros, **noise** = white noise; 10 ms fade; **r1 / r2** = two repeats of the same run.\n\n"
        "Scores in %, all questions, per parser (strict / permissive / diag_no_brackets): "
        "**Acc.** accuracy, **C.G.** coverage-grounded, **B.G.** IoU-grounded. "
        "**Click a row** to open the run.")
    rows = masked_runs()
    df = pd.DataFrame(rows).drop(columns="_order")
    metric_cols = [f"{p} {k}" for p in PARSERS for k in ("Acc.", "C.G.", "B.G.")]
    st.markdown(" ".join(f'<span style="background:{PARSER_TINT[p]};padding:2px 8px;border-radius:4px;'
                         f'margin-right:6px">{p}</span>' for p in PARSERS), unsafe_allow_html=True)

    def tint(col: pd.Series) -> list[str]:
        return [f"background-color: {PARSER_TINT[col.name.split()[0]]}"] * len(col)

    styled = (df.style.format("{:.1f}", subset=metric_cols, na_rep="—")
              .apply(tint, subset=metric_cols)
              .apply(lambda r: ["background-color: rgba(128,128,128,0.16)" if r["repeat"] == "—" else ""] * len(r),
                     axis=1, subset=["run", "audio masked", "filled with", "repeat", "n"]))
    short = {"strict": "strict", "permissive": "perm.", "diag_no_brackets": "diag"}
    config = {"run": st.column_config.TextColumn(width="large"), "n": st.column_config.NumberColumn(width="small"),
              **{c: st.column_config.NumberColumn(f"{short[c.split()[0]]} {c.split()[1]}", width="small")
                 for c in metric_cols}}
    ev = st.dataframe(styled, hide_index=True, width="stretch", height=38 * (len(df) + 1) + 3,
                      on_select="rerun", selection_mode="single-row", key="masked_table", column_config=config)
    if ev.selection.rows:
        go("run", df.iloc[ev.selection.rows[0]]["run"])
        st.rerun()

    effects_section(rows)


# ------------------------------------------------------------------ masking effects (masked page)
PARSER_TINT = {"strict": "rgba(42,120,214,0.13)", "permissive": "rgba(235,104,52,0.13)",
               "diag_no_brackets": "rgba(27,175,122,0.15)"}
GOOD_TEXT, BAD_TEXT = "#16a34a", "#e5484d"  # readable on light and dark backgrounds
IOU_BINS = ["disjoint (IoU 0)", "IoU < 0.5", "IoU ≥ 0.5", "identical (IoU 1)"]
IOU_COLORS = ["#b7d3f6", "#6da7ec", "#2a78d6", "#104281"]  # one-hue ordinal ramp, light = less overlap


def signed_color(v) -> str:
    if v is None or pd.isna(v) or abs(v) < 0.05:
        return ""
    return f"color: {GOOD_TEXT if v > 0 else BAD_TEXT}; font-weight: 600"


def _merge(iv: list) -> list[tuple[float, float]]:
    out = []
    for a, b in sorted((float(a), float(b)) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _length(m: list) -> float:
    return sum(b - a for a, b in m)


def _inter(x: list, y: list) -> float:
    i = j = 0
    tot = 0.0
    while i < len(x) and j < len(y):
        tot += max(0.0, min(x[i][1], y[j][1]) - max(x[i][0], y[j][0]))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return tot


def av_pair(visual: list, audio: list) -> dict:
    """Agreement between one question's visual and audio intervals (each list merged into a union)."""
    v, a = _merge(visual or []), _merge(audio or [])
    both = bool(v) and bool(a)
    inter = _inter(v, a) if both else 0.0
    union = _length(v) + _length(a) - inter
    iou = inter / union if both and union > 0 else None
    return {"both": both, "iou": iou, "identical": both and v == a,
            "visual_s": _length(v) if v else None, "audio_s": _length(a) if a else None}


@st.cache_data
def av_agreement(name: str, source: str) -> pd.DataFrame:
    """One row per question: audio-visual agreement of the intervals; source = a parser or 'gt'."""
    qs = json.loads((OUT / name / "evaluation" / "results.json").read_text())["questions"]
    rows = []
    for q in qs:
        iv = q.get("gt") if source == "gt" else ((q.get("pred") or {}).get(source) or {})
        rows.append({"qa_id": q["qa_id"], "level": q["level"], **av_pair(iv.get("visual"), iv.get("audio"))})
    return pd.DataFrame(rows)


def iou_bin(r) -> str | None:
    if not r["both"]:
        return None
    return (IOU_BINS[3] if r["identical"] or r["iou"] >= 0.999 else IOU_BINS[0] if r["iou"] == 0
            else IOU_BINS[1] if r["iou"] < 0.5 else IOU_BINS[2])


@st.cache_data
def question_scores(name: str) -> dict[str, dict[int, dict]]:
    """parser -> qa_id -> {Acc., C.G., B.G.} (0-1 per question), plus 'level' and 'infeasible'."""
    qs = json.loads((OUT / name / "evaluation" / "results.json").read_text())["questions"]
    out = {p: {q["qa_id"]: q["scores"].get(p) or {} for q in qs} for p in PARSERS}
    out["level"] = {q["qa_id"]: q["level"] for q in qs}
    out["infeasible"] = {int(k) for k, v in load_outputs(OUT / name).items()
                         if (v.get("audio_mask") or {}).get("control_infeasible")}
    return out


def effects_section(rows: list[dict]):
    st.divider()
    st.header("Effect of the masking")
    parser = st.radio("Parser", PARSERS, index=1, horizontal=True, key="effects_parser")
    base = next(r for r in rows if r["repeat"] == "—")
    runs = [r for r in rows if r["repeat"] != "—"]
    cond = lambda r: f"{r['audio masked'].split()[0]} · {r['filled with']}"
    conds = list(dict.fromkeys(cond(r) for r in runs))

    # 1. overall scores and deltas vs baseline, one row per run
    d = pd.DataFrame([{"condition": cond(r), "target": r["audio masked"], "repeat": r["repeat"], "metric": k,
                       "delta": r[f"{parser} {k}"] - base[f"{parser} {k}"], "value": r[f"{parser} {k}"]}
                      for r in runs for k in ("Acc.", "C.G.", "B.G.")])
    mean = d.groupby(["condition", "target", "metric"], as_index=False)[["delta", "value"]].mean()
    ev_acc = mean[(mean.metric == "Acc.") & (mean.target == "evidence")].set_index("condition")["delta"]
    rd_acc = mean[(mean.metric == "Acc.") & (mean.target != "evidence")].set_index("condition")["delta"]
    spread = d.groupby(["condition", "metric"])["delta"].agg(lambda s: s.max() - s.min()).max()
    st.markdown(
        f"**In short** ({parser}, baseline Acc. {base[f'{parser} Acc.']:.1f} / C.G. {base[f'{parser} C.G.']:.1f} / "
        f"B.G. {base[f'{parser} B.G.']:.1f}): masking the **audio evidence** changes accuracy by "
        + ", ".join(f"**{v:+.1f}** ({c.split(' · ')[1]})" for c, v in ev_acc.items())
        + " points; masking **random segments of the same length** (control) by "
        + ", ".join(f"**{v:+.1f}** ({c.split(' · ')[1]})" for c, v in rd_acc.items())
        + f". Run-to-run noise (r1 vs r2, greedy) is at most {spread:.1f} points.")

    st.subheader("1 · Change vs baseline")
    st.caption(f"{parser}, all 1027 questions, in %. Δ = run − baseline, in points: "
               ":green[**green**] = better than the baseline, :red[**red**] = worse.")
    tab = [{"run": "baseline (no masking)", **{k: base[f"{parser} {k}"] for k in ("Acc.", "C.G.", "B.G.")}}]
    for r in runs:
        tab.append({"run": f"{cond(r)} · {r['repeat']}",
                    **{c: v for k in ("Acc.", "C.G.", "B.G.")
                       for c, v in ((k, r[f"{parser} {k}"]), (f"Δ {k}", r[f"{parser} {k}"] - base[f"{parser} {k}"]))}})
    t1 = pd.DataFrame(tab).set_index("run")[["Acc.", "Δ Acc.", "C.G.", "Δ C.G.", "B.G.", "Δ B.G."]]
    dcols = ["Δ Acc.", "Δ C.G.", "Δ B.G."]
    num = t1[dcols].copy()
    for c in dcols:  # as text: streamlit shows "None" for a missing number (baseline row)
        t1[c] = ["—" if pd.isna(v) else f"{v:+.1f}" for v in num[c]]
    st.dataframe(t1.style.format("{:.1f}", subset=["Acc.", "C.G.", "B.G."])
                 .apply(lambda col: [signed_color(v) for v in num[col.name]], subset=dcols)
                 .set_properties(subset=dcols, **{"text-align": "right"}),
                 width="stretch", height=38 * (len(t1) + 1) + 3)

    # 3. by level
    st.subheader("2 · Accuracy by level")
    st.caption(f"{parser} Acc. in %; Δ = condition (mean of r1, r2) − baseline. "
               "Level 1 has 436 questions, level 2 438, level 3 153.")
    lv = question_scores(base["run"])["level"]

    def acc(name: str, level: int | None) -> float:
        s = question_scores(name)[parser]
        qs = [q for q in s if level is None or lv[q] == level]
        return 100 * sum(s[q].get("Acc.") == 1 for q in qs) / len(qs)

    levels = [("Level 1", 1), ("Level 2", 2), ("Level 3", 3), ("Overall", None)]
    tab = [{"condition": "baseline (no masking)", **{n: acc(base["run"], l) for n, l in levels}}]
    for c in conds:
        names = [r["run"] for r in runs if cond(r) == c]
        tab.append({"condition": c, **{n: sum(acc(x, l) for x in names) / len(names) for n, l in levels}})
    t = pd.DataFrame(tab).set_index("condition")
    delta = t - t.iloc[0]
    show = t.copy().astype(object)
    for c in show.columns:
        show[c] = [f"{v:.1f}" if i == 0 else f"{v:.1f}  ({dv:+.1f})" for i, (v, dv) in enumerate(zip(t[c], delta[c]))]

    def shade(col: pd.Series) -> list[str]:
        out = []
        for i, dv in enumerate(delta[col.name]):
            a = min(0.45, abs(dv) / 8)
            out.append("" if i == 0 else f"background-color: rgba({'227,73,72' if dv < 0 else '42,120,214'},{a:.2f})")
        return out

    st.dataframe(show.style.apply(shade), width="stretch")

    # 3. audio-visual agreement of the intervals
    st.subheader("3 · Do the visual and audio intervals coincide?")
    st.caption(
        "Per question, the visual intervals (merged into one set of seconds) are compared with the audio intervals: "
        "**IoU** = seconds in both ÷ seconds in either; *identical* = exactly the same intervals. "
        "Only questions with at least one visual **and** one audio interval are compared (column *both given*). "
        f"First row: the dataset annotations; the other rows: the model's predicted intervals ({parser}).")
    sources = [("dataset annotations (GT)", base["run"], "gt"), ("baseline (no masking)", base["run"], parser)]
    sources += [(f"{cond(r)} · {r['repeat']}", r["run"], parser) for r in runs]
    stats, bins = [], []
    for label, name, src in sources:
        a = av_agreement(name, src)
        b = a[a["both"]]
        stats.append({"intervals of": label,
                      "both given %": 100 * len(b) / len(a),
                      "identical %": 100 * b["identical"].mean() if len(b) else None,
                      "mean IoU": b["iou"].mean() if len(b) else None,
                      "median IoU": b["iou"].median() if len(b) else None,
                      "mean visual s": a["visual_s"].mean(), "mean audio s": a["audio_s"].mean()})
        counts = b.apply(iou_bin, axis=1).value_counts() if len(b) else pd.Series(dtype=int)
        bins += [{"intervals of": label, "bin": k, "order": i, "questions": int(counts.get(k, 0)),
                  "share": counts.get(k, 0) / len(b) if len(b) else 0} for i, k in enumerate(IOU_BINS)]
    labels = [s[0] for s in sources]
    left, right = st.columns([1, 1])
    with left:
        bdf = pd.DataFrame(bins)
        chart = alt.Chart(bdf).mark_bar(height=22, stroke="white", strokeWidth=1).encode(
            x=alt.X("share:Q", stack="normalize", title="share of the questions with both given",
                    axis=alt.Axis(format="%", orient="top")),
            y=alt.Y("intervals of:N", sort=labels, title=None,
                    axis=alt.Axis(labelFontSize=12, labelLimit=260, labelOverlap=False)),
            color=alt.Color("bin:N", sort=IOU_BINS, scale=alt.Scale(domain=IOU_BINS, range=IOU_COLORS),
                            legend=alt.Legend(title=None, orient="bottom", columns=2)),
            order=alt.Order("order:Q"),
            tooltip=["intervals of", "bin", "questions", alt.Tooltip("share:Q", format=".1%")])
        st.altair_chart(chart.properties(height=35 * len(labels)), width="stretch",
                        height=35 * len(labels) + 110)  # rows = table row height; + axis and legend
    with right:
        sdf = pd.DataFrame(stats).set_index("intervals of")
        st.dataframe(sdf.style.format({"both given %": "{:.1f}", "identical %": "{:.1f}", "mean IoU": "{:.2f}",
                                       "median IoU": "{:.2f}", "mean visual s": "{:.1f}", "mean audio s": "{:.1f}"},
                                      na_rep="—")
                     .apply(lambda r: ["background-color: rgba(128,128,128,0.16)" if r.name.startswith("dataset")
                                       else ""] * len(r), axis=1),
                     width="stretch", height=38 * (len(sdf) + 1) + 3)


page = st.session_state.setdefault("page", "home")
if page == "home":
    page_home()
    st.stop()
if page == "masked":
    page_masked()
    st.stop()

with st.sidebar:
    if page == "run":  # opened from the masked-runs table: menu limited to those runs
        st.button("← Audio masked runs", on_click=go, args=("masked",), width="stretch")
        choices = [r["run"] for r in masked_runs()]
    else:
        st.button("← Home", on_click=go, args=("home",), width="stretch")
        choices = run_names()
    if st.session_state.get("run") not in choices:
        st.session_state["run"] = choices[0]
    run = st.selectbox("Run", choices, key="run")
    if st.button("Reload run files"):
        st.cache_data.clear()
    meta, outputs, questions = load_run(run)
    settings = meta["settings"]
    mask = settings.get("audio_mask")
    source = (st.radio("Audio played", ["model input (masked)", "original track"], horizontal=True,
                       help="model input = the masked waveform rebuilt with custom/audio_mask.py")
              if mask else "original track")
    hash_note = st.empty()
    zoom = st.radio("Timeline", ["whole video", "around the intervals"], horizontal=True)
    st.divider()
    levels = st.multiselect("Level", [1, 2, 3])
    caps = st.multiselect("Capability", sorted({r["capability"] for r in ann.values()}))
    no_brackets = st.toggle("Accept intervals without [ ]", key="no_brackets",
                            help="Off: permissive parser. On: diag_no_brackets, which also reads intervals "
                                 "written as mm:ss,mm:ss. Changes the predicted intervals (timeline and filter "
                                 "below); the answer letter is the same with both.")
    parser = "diag_no_brackets" if no_brackets else "permissive"
    st.markdown("**Show only questions with**")
    f_correct = st.checkbox("correct answer", key="f_correct")
    f_wrong = st.checkbox("wrong answer", key="f_wrong",
                          help="Ticking both correct and wrong (or neither) shows every answer.")
    f_valid = st.checkbox("valid predicted interval", key="f_valid",
                          help=f"At least one predicted interval (visual or audio) readable by the {parser} parser.")

    def keep(q: int) -> bool:
        r = ann[q]
        if str(q) not in outputs or (levels and r["level"] not in levels) or (caps and r["capability"] not in caps):
            return False
        qq = questions.get(q) or {}
        if f_correct != f_wrong:  # exactly one ticked
            acc = qq.get("scores", {}).get(parser, {}).get("Acc.")
            if acc is None or (acc == 1) != f_correct:
                return False
        if f_valid:
            p = (qq.get("pred") or {}).get(parser) or {}
            if not (p.get("visual") or p.get("audio")):
                return False
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

st.markdown(RUN_PAGE_CSS, unsafe_allow_html=True)
rec, out, q = ann[qa], outputs[str(qa)], questions.get(qa, {})
description = re.sub(r";?\s*otherwise as \S+\s*$", "", meta["run"].get("description", ""))
st.markdown(f"**{run}** — {description}")

video_dir = Path(settings.get("video_dir", REPO / "data" / "resized_video"))
if not video_dir.is_dir():  # run made on another machine: same folder under this repo
    video_dir = next((d for d in (REPO / "data" / video_dir.name, CUSTOM / "media" / video_dir.name)
                      if d.is_dir()), video_dir)
video_path = str(video_dir / f"{rec['video_id']}.mp4")
duration = out.get("video_seconds") or out.get("audio_seconds") or 0.0

# timeline rows
rows = [{"row": f"GT {m}", "start": a, "end": b}
        for m in ("visual", "audio") for a, b in (audio_mask.parse_time(e["time"]) for e in rec["evidence"][m])]
pred = (q.get("pred") or {}).get(parser) or {}
rows += [{"row": f"pred {m}", "start": a, "end": b} for m in ("visual", "audio") for a, b in pred.get(m, [])]
info = out.get("audio_mask") or {}
rows += [{"row": "masked", "start": a, "end": b} for a, b in info.get("segments_s", [])]

left, right = st.columns([1, 1])
with left, st.container(key="videobox"):
    if source.startswith("model"):
        data, sha = masked_video(video_path, qa, mask["target"], mask["fill"], mask["pad_s"])
        st.video(data)
        hash_note.caption(("✅ same waveform as the run (hash matches)" if sha == info.get("audio_sha256")
                           else "⚠️ hash differs from the run: this is NOT exactly what the model received")
                          + (" · control_infeasible: nothing masked" if info.get("control_infeasible") else ""))
    else:
        st.video(video_path)

with right, st.container(key="qbox"):
    st.markdown(f"**qa {qa}** · L{rec['level']} · {rec['capability']} · video {rec['video_id']} "
                f"({mmss(duration)})")
    st.markdown(rec["question"])
    pred_answer = pred.get("answer")
    verdict = ("" if pred_answer is None else " ✓" if pred_answer == rec["answer"] else " ✗")
    st.markdown(f"GT: **{rec['answer']}** · pred ({parser}): **{pred_answer or '—'}**{verdict}"
                f" · status `{out.get('status')}`")
    st.markdown(option_lines(rec["options"], rec["answer"], pred_answer))
    scores = q.get("scores", {})
    table = [{"parser": p, "answer": ((q.get("pred") or {}).get(p) or {}).get("answer"), **scores.get(p, {})}
             for p in ("strict", "permissive", "diag_no_brackets") if p in scores]
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

# full-width timeline under video and question
if duration:
    span = [(r["start"], r["end"]) for r in rows]
    if zoom == "whole video" or not span:
        domain = (0.0, duration)
    else:
        a, b = min(s for s, _ in span), max(e for _, e in span)
        m = max(10.0, 0.15 * (b - a))
        domain = (max(0.0, a - m), min(duration, b + m))
    st.altair_chart(timeline(rows, domain), width="stretch")
