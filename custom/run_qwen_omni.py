#!/usr/bin/env python3
"""EvaOmni adapter for Qwen3-Omni-30B-A3B-Instruct with vLLM (offline, thinker only).

Follows the model card (Qwen/Qwen3-Omni-30B-A3B-Instruct, "vLLM Usage" and
"Setting for Evaluation"):
  * vLLM `LLM` class, bf16, thinker only (vLLM skips talker/code2wav: text output only);
  * `qwen_omni_utils.process_mm_info(messages, use_audio_in_video=True)` to read the
    video frames AND the video's own audio track (16 kHz);
  * `mm_processor_kwargs={"use_audio_in_video": True}` so audio and video are
    interleaved in time by the processor;
  * no system prompt; the multimodal input comes before the question text;
  * greedy decoding (evaluation setting) or the snippet's sampling
    (temperature=0.6, top_p=0.95, top_k=20), max_tokens=16384, LLM(seed=...).

Optional, NOT part of the replication: --constrained json forces the output format
with vLLM structured outputs (see evidence_schema); the prompt stays the same.

The prompt is imported from the repository's prompts.py (key "video_audio").

Modality ablations (--input-mode v / a / t, paper Table 4 and Table 6(d)): video
frames only / the video's audio track only / question and options only. The missing
input is omitted (no placeholder) and the repository's prompt for that input is used
(video_only / audio_only / text_only). The [settings] of av runs are unchanged.

Audio masking (--audio-mask evidence / random, av only; custom/audio_mask.py): for
each question, the audio evidence intervals (or, as a control, random segments of
the same lengths) are replaced by silence or noise on a copy of the waveform, just
before the request. Frames, prompt and the rest of the audio are as in av.

Run layout (custom/output/<run_name>/):
  info.txt           [run] + [settings] (settings compared on --resume), then
                     resume / finish / scoring lines appended
  output.jsonl       one record per answered question (pretty-printed JSON, several
                     lines each): what went in (input) and what came out (output);
                     appended + fsync'd, the last record per qa_id wins
  evaluation/        written by custom/score.py (results.json, evaluation.txt)
Worker logs are printed to the terminal (nothing else is written).

Examples:
  python custom/run_qwen_omni.py --run-name qwen3omni-instr_av_greedy_check50 \
      --description "response-format check" --subset check50 --decoding greedy
  python custom/run_qwen_omni.py --resume qwen3omni-instr_av_greedy_full <same options>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import audio_mask
from runinfo import OUTPUT_FILE, append_record, format_section, load_outputs, read_run_meta

# qwen_omni_utils reads the video's audio through audioread, which calls the
# `ffmpeg` binary: make the binaries of this interpreter's env (av_env) visible
# even when the env is not activated.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

CUSTOM = Path(__file__).resolve().parent
REPO = CUSTOM.parent
OUTPUT_ROOT = CUSTOM / "output"
SUBSETS = CUSTOM / "subsets"
ANNOTATION = REPO / "data" / "test_file.json"

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
PROMPT_KEY = "video_audio"
FPS = 1.0  # data/resized_video is already sampled at 1 FPS
# Input modes: default video folder and description. Both use the video's own
# audio track with use_audio_in_video=True; av-mute points to copies whose audio
# track is kept but silent (custom/tools/make_muted_videos.py).
INPUT_MODES = {
    "av": (REPO / "data" / "resized_video", "video frames + the video's own audio track"),
    "av-mute": (CUSTOM / "media" / "muted",
                "same frames, audio track replaced by digital silence (track kept)"),
    "v": (REPO / "data" / "resized_video", "video frames only, no audio; prompt video_only"),
    "a": (REPO / "data" / "resized_video", "the video's audio track only, no frames; prompt audio_only"),
    "t": (REPO / "data" / "resized_video", "question and options only, no video, no audio; prompt text_only"),
}
# Modality ablations (paper: Table 4 and Table 6(d)). The missing input is OMITTED
# (no placeholder: no black frames, no silent audio) and the repository's prompt for
# that input is used; it tells the model what it has.
UNIMODAL = {
    "v": {"prompt_key": "video_only", "video": True, "audio": False,
          "message_order": "video first, then question text"},
    "a": {"prompt_key": "audio_only", "video": False, "audio": True,
          "message_order": "audio first, then question text"},
    "t": {"prompt_key": "text_only", "video": False, "audio": False,
          "message_order": "question text only"},
}


def mode_prompt_key(mode: str) -> str:
    return UNIMODAL[mode]["prompt_key"] if mode in UNIMODAL else PROMPT_KEY


def mode_inputs(mode: str) -> tuple[bool, bool]:
    """(video given, audio given)"""
    return (UNIMODAL[mode]["video"], UNIMODAL[mode]["audio"]) if mode in UNIMODAL else (True, True)

# --constrained json: structured outputs of vLLM. The prompt is unchanged; at each
# step vLLM only allows the tokens that keep the output valid against this schema
# (JSON with the prompt's three fields, a letter among the question's options, and
# timestamps "[mm:ss,mm:ss]" exactly as the prompt asks). Not enforced (left to the
# model): how many intervals (0 to MAX_EVIDENCE_ITEMS per list), end > start, times
# within the video.
MAX_EVIDENCE_ITEMS = 100
TIME_PATTERN = r"^\[\d{2}:[0-5]\d,\d{2}:[0-5]\d\]$"
_OPTION_LETTER_RE = re.compile(r"(?m)^\s*([A-Z])\.\s")


def option_letters(options: str) -> list[str]:
    """Letters of a question's options ('A. ...\\nB. ...' -> ['A', 'B', ...])."""
    return _OPTION_LETTER_RE.findall(options)


def evidence_schema(letters: list[str]) -> dict:
    """JSON schema of the answer; property order = the prompt's order."""
    items = {"type": "array", "maxItems": MAX_EVIDENCE_ITEMS,
             "items": {"type": "object",
                       "properties": {"time": {"type": "string", "pattern": TIME_PATTERN}},
                       "required": ["time"], "additionalProperties": False}}
    return {"type": "object",
            "properties": {"answer": {"type": "string", "enum": letters},
                           "visual_evidence": items, "audio_evidence": items},
            "required": ["answer", "visual_evidence", "audio_evidence"],
            "additionalProperties": False}


# ------------------------------------------------------------------ helpers
def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_prompt_template(key: str = PROMPT_KEY) -> str:
    sys.path.insert(0, str(REPO))
    from prompts import PROMPTS  # original, unmodified repository file
    return PROMPTS[key]


def pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"


def git_info() -> dict:
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True).stdout.strip()
    originals = run("status", "--porcelain", "--", ".", ":(exclude)custom")
    return {"commit": run("rev-parse", "HEAD"), "status_full": run("status", "--porcelain"),
            "status_outside_custom": originals,
            "originals_unmodified": originals == ""}


def gpu_names(indices: list[int]) -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    rows = {int(l.split(",")[0]): l.strip() for l in out if l.strip()}
    return [rows.get(i, f"{i}, unknown") for i in indices]


def load_subset(name: str, records: list[dict]) -> list[int]:
    if name == "full":
        return sorted(r["qa_id"] for r in records)
    path = SUBSETS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"subset not found: {path}")
    ids = json.loads(path.read_text())["qa_ids"]
    known = {r["qa_id"] for r in records}
    missing = [q for q in ids if q not in known]
    if missing:
        raise SystemExit(f"subset {name} has unknown qa_ids: {missing[:10]}")
    return sorted(ids)


def request_seed(base_seed: int, qa_id) -> int:
    """Per-request seed: makes sampling independent of batching and sharding."""
    return int(sha256_bytes(f"{base_seed}-{qa_id}".encode())[:8], 16) % (2**31 - 1)


def sampling_settings(args) -> dict:
    if args.decoding == "greedy":
        return {"temperature": 0.0, "max_tokens": args.max_tokens}
    return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": args.max_tokens}


# ------------------------------------------------------------------ run folder / settings
def build_settings(args, records, qa_ids) -> dict:
    """Everything that defines the experiment. Must be identical to resume."""
    key = mode_prompt_key(args.input_mode)
    template = load_prompt_template(key)
    settings = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "engine": "vllm.LLM (offline, V1 engine, thinker only)",
        "dtype": "bfloat16",
        "library_versions": {p: pkg_version(p) for p in
                             ("vllm", "torch", "transformers", "numpy", "qwen-omni-utils", "librosa",
                              "av", "decord")},
        "annotation": str(ANNOTATION.relative_to(REPO)),
        "annotation_sha256": sha256_bytes(ANNOTATION.read_bytes()),
        "subset": args.subset,
        "subset_n": len(qa_ids),
        "subset_qa_ids_sha256": sha256_bytes(json.dumps(qa_ids).encode()),
        "video_dir": str(Path(args.video_dir).resolve()),
        "input_mode": args.input_mode,
        "use_audio_in_video": True,
        "fps": FPS,
        "video_frame_processing": ("qwen_omni_utils.process_mm_info defaults (image_patch_size=14, "
                                   "FPS clamp to all frames, bicubic resize to multiples of 28), then the "
                                   "HF Qwen3-Omni video processor inside vLLM (patch 16, merge 2)"),
        "time_encoding": ("vLLM 0.17.1 default second_per_grid_ts=2.0 s per temporal grid "
                          "(temporal_patch_size=2 at 1 FPS) x position_id_per_seconds=13"),
        "system_prompt": None,
        "message_order": "video first, then question text",
        "prompt_key": key,
        "prompt_sha256": sha256_bytes(template.encode()),
        "prompt_fill": f"PROMPTS['{key}'].format(question=rec['question'], options=rec['options'])",
        "decoding": args.decoding,
        "sampling_params": sampling_settings(args),
        "seed": args.seed,
        "seed_scheme": ("LLM(seed=seed) as in the model card, plus per-request "
                        "SamplingParams(seed=sha256(f'{seed}-{qa_id}')) when sampling"),
        "max_model_len": args.max_model_len,
        "limit_mm_per_prompt": {"image": 0, "video": 1, "audio": 1},
        "mm_processor_cache_gb": 0,
        "mm_processor_cache_reason": ("vLLM 0.17.1 bug: on a processor-cache hit the video item is None and "
                                      "Qwen3-Omni's _maybe_apply_prompt_updates then treats "
                                      "use_audio_in_video as False (assertion on the audio placeholder). "
                                      "Cache disabled; KV prefix caching stays on."),
        "enable_prefix_caching": True,
        "max_num_seqs": args.max_num_seqs,
        "max_num_seqs_reason": ("vLLM 0.17.1 bug: merge_interleaved_embeddings (use_audio_in_video) separates "
                                "video/audio embeddings by token counts only, which breaks when one engine "
                                "step holds multimodal embeddings of >1 request. max_num_seqs=1 = one "
                                "request per step (also makes greedy independent of batch composition)."),
        "long_video_policy": ("no truncation, no frame dropping: prompts longer than max_model_len "
                              "are recorded as status=error (error_type=prompt_too_long) and score zero"),
        "stats_only": args.stats_only,
    }
    # Added only when used, so the [settings] of av runs stay exactly as
    # before (--resume of older runs keeps working).
    if args.input_mode in UNIMODAL:
        um = UNIMODAL[args.input_mode]
        settings["use_audio_in_video"] = False
        settings["message_order"] = um["message_order"]
        if not um["video"]:
            for k in ("fps", "video_frame_processing", "time_encoding"):
                settings[k] = "n/a (no video input)"
        settings["modality_ablation"] = {
            "inputs_given": {"video": um["video"], "audio": um["audio"], "question_and_options": True},
            "missing_inputs": "omitted: no placeholder (no black frames, no silent audio)",
            "prompt": f"repository prompts.py PROMPTS['{key}'] (says which input the model has)",
            "video_only": ("frames read as in av (qwen_omni_utils.process_mm_info), "
                           "use_audio_in_video=False, no audio item") if um["video"] else None,
            "audio_only": ("waveform extracted exactly as in av (qwen_omni_utils process_audio_info on the "
                           "video element with use_audio_in_video=True, 16 kHz), passed as an audio item; "
                           "no frames") if um["audio"] and not um["video"] else None,
            "scoring": ("evidence lists the prompt does not ask for are scored as empty "
                        "(custom/score.py): C.G. <= Acc/2 and B.G. = 0 by construction"),
        }
        settings["modality_ablation"] = {k: v for k, v in settings["modality_ablation"].items() if v is not None}
    if args.audio_mask != "none":
        settings["audio_mask"] = audio_mask.settings_entry(args.audio_mask, args.mask_fill, args.mask_pad)
    # Determinism options (custom/run_determinism.sh); added only when used.
    if args.no_prefix_caching:
        settings["enable_prefix_caching"] = False
    if args.batch_invariant:
        settings["batch_invariant"] = {
            "how": "VLLM_BATCH_INVARIANT=1 set in the worker before importing vLLM, attention_backend=FLASH_ATTN",
            "vllm_doc": "vLLM batch-invariant mode: fixed kernel configs, sorted MoE top-k, no split-KV, no TF32",
        }
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") is not None:
        settings["VLLM_ENABLE_V1_MULTIPROCESSING"] = os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]
    if args.constrained == "json":
        template_schema = evidence_schema(["<option letters of the question>"])
        settings["structured_output"] = {
            "mode": "json",
            "how": ("SamplingParams(structured_outputs=StructuredOutputsParams(json=schema)), "
                    "vLLM structured-output backend 'auto' (= xgrammar for this schema); prompt unchanged"),
            "xgrammar_version": pkg_version("xgrammar"),
            "schema": template_schema,
            "schema_sha256": sha256_bytes(json.dumps(template_schema, sort_keys=True).encode()),
            "answer_enum": "letters parsed from rec['options'] (A-H; qa 209: A-I; qa 483: A-F)",
            "max_evidence_items": MAX_EVIDENCE_ITEMS,
            "not_enforced": "number of intervals (0..max), end > start, times within the video",
        }
    return settings


def unique_run_dir(name: str) -> Path:
    base = OUTPUT_ROOT / name
    if not base.exists():
        return base
    k = 2
    while (OUTPUT_ROOT / f"{name}_v{k}").exists():
        k += 1
    return OUTPUT_ROOT / f"{name}_v{k}"


def write_info_header(run_dir: Path, run: dict, settings: dict) -> None:
    """info.txt = [run] + [settings] sections (see custom/runinfo.py); later lines are appended."""
    text = (f"EvaOmni run: {run['run_name']}\n{run['description']}\n\n"
            + format_section("run", run) + "\n" + format_section("settings", settings) + "\n")
    (run_dir / "info.txt").write_text(text, encoding="utf-8")


def append_info(run_dir: Path, text: str) -> None:
    with (run_dir / "info.txt").open("a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")


def shard_by_video(pending: list[dict], n: int, video_dir: Path) -> list[list[int]]:
    """Keep all questions of a video on one worker; balance by duration x questions."""
    groups: dict[str, list[dict]] = {}
    for r in pending:
        groups.setdefault(r["video_id"], []).append(r)
    costs = {vid: len(rs) * video_seconds(video_dir / f"{vid}.mp4") for vid, rs in groups.items()}
    shards: list[list[int]] = [[] for _ in range(n)]
    load = [0.0] * n
    for vid in sorted(groups, key=lambda v: -costs[v]):
        k = load.index(min(load))
        shards[k] += [r["qa_id"] for r in groups[vid]]
        load[k] += costs[vid]
    return shards


def video_seconds(path: Path) -> float:
    import av
    with av.open(str(path)) as c:
        return c.duration / 1e6


# ------------------------------------------------------------------ parent
def parent(args) -> None:
    records = json.loads(ANNOTATION.read_text(encoding="utf-8"))
    by_id = {r["qa_id"]: r for r in records}
    qa_ids = load_subset(args.subset, records)
    settings = build_settings(args, records, qa_ids)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]

    if args.resume:
        run_dir = OUTPUT_ROOT / args.resume
        meta = read_run_meta(run_dir)
        if meta is None:
            raise SystemExit(f"--resume: no run settings found in {run_dir}/info.txt")
        if meta["settings"] != settings:
            old = meta["settings"]
            diff = {k: (old.get(k), settings.get(k)) for k in set(old) | set(settings)
                    if old.get(k) != settings.get(k)}
            raise SystemExit("--resume refused: settings differ from the run's [settings]:\n"
                             + json.dumps(diff, indent=1, default=str))
        run = meta["run"]
        append_info(run_dir, f"\nResumed      : {now()}  GPUs {gpu_names(gpus)}  command: {shlex.join(sys.argv)}")
    else:
        if not args.run_name or not args.description:
            raise SystemExit("--run-name and --description are required for a new run")
        run_dir = unique_run_dir(args.run_name)
        run_dir.mkdir(parents=True)
        git = git_info()
        run = {"run_name": run_dir.name, "description": args.description, "started": now(),
               "command": shlex.join([sys.executable] + sys.argv), "host": socket.gethostname(),
               "python": sys.version.split()[0], "gpus": gpu_names(gpus), "num_workers": len(gpus),
               "gpu_memory_utilization": args.gpu_memory_utilization, "batch_size": args.batch_size,
               "git_commit": git["commit"], "originals_unmodified": git["originals_unmodified"],
               "git_status_outside_custom": git["status_outside_custom"]}
        write_info_header(run_dir, run, settings)
    print(f"Run dir: {run_dir}", flush=True)

    done = load_outputs(run_dir)
    pending = [by_id[q] for q in qa_ids
               if done.get(str(q), {}).get("status") != "ok"
               and done.get(str(q), {}).get("error_type") != "prompt_too_long"]
    print(f"Questions: {len(qa_ids)}  already done: {len(qa_ids) - len(pending)}  pending: {len(pending)}",
          flush=True)
    t0 = time.time()
    rcs: list[int] = []
    if pending:
        n = min(len(gpus), len(pending))
        shards = shard_by_video(pending, n, Path(args.video_dir))
        procs = []
        for k in range(n):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[k]))
            cmd = [sys.executable, __file__, "--_worker", str(k), "--_run-dir", str(run_dir),
                   "--_shard", ",".join(map(str, shards[k]))] + _forward_args(args)
            procs.append(subprocess.Popen(cmd, env=env))  # worker output goes to this terminal
            print(f"  worker {k} on GPU {gpus[k]}: {len(shards[k])} questions", flush=True)
        _monitor(run_dir, procs, len(qa_ids))
        rcs = [p.wait() for p in procs]
        if any(rcs):
            print(f"WARNING: worker exit codes {rcs}; see the worker output above", flush=True)

    final = load_outputs(run_dir)
    ok = sum(1 for q in qa_ids if final.get(str(q), {}).get("status") == "ok")
    errs = {}
    for q in qa_ids:
        rec = final.get(str(q))
        if rec is None:
            errs["missing"] = errs.get("missing", 0) + 1
        elif rec.get("status") != "ok":
            errs[rec.get("error_type", "error")] = errs.get(rec.get("error_type", "error"), 0) + 1
    ended = now()
    started = dt.datetime.fromisoformat(run["started"])
    append_info(run_dir, "\n".join([
        f"Finished     : {ended}  (this session {round((time.time() - t0) / 60, 1)} min; "
        f"since first start {round((dt.datetime.fromisoformat(ended) - started).total_seconds() / 60, 1)} min)",
        f"Questions    : {len(qa_ids)}  succeeded {ok}  failed {len(qa_ids) - ok} {errs or ''}",
    ]))
    if not args.no_score:
        subprocess.run([sys.executable, str(CUSTOM / "score.py"), "--run-dir", str(run_dir)], check=False)

    # Exit status for scripts (custom/run_weekend.sh): non-zero if a worker died
    # or any question failed for a reason other than prompt_too_long (video 293,
    # an accepted, documented failure).
    worker_rcs = rcs if pending else []
    real_failures = {k: v for k, v in errs.items() if k != "prompt_too_long"}
    if any(worker_rcs) or real_failures:
        print(f"RUN FAILED: worker exit codes {worker_rcs}, failures {real_failures}. "
              f"Resume with --resume {run_dir.name} and the same options.", flush=True)
        sys.exit(1)
    print(f"RUN OK: {ok}/{len(qa_ids)} answered" + (f", {errs}" if errs else ""), flush=True)


def _forward_args(args) -> list[str]:
    out = ["--subset", args.subset, "--video-dir", args.video_dir, "--input-mode", args.input_mode,
           "--decoding", args.decoding, "--seed", str(args.seed), "--max-tokens", str(args.max_tokens),
           "--max-model-len", str(args.max_model_len), "--gpu-memory-utilization",
           str(args.gpu_memory_utilization), "--max-num-seqs", str(args.max_num_seqs),
           "--batch-size", str(args.batch_size), "--constrained", args.constrained,
           "--audio-mask", args.audio_mask, "--mask-fill", args.mask_fill, "--mask-pad", str(args.mask_pad)]
    if args.stats_only:
        out.append("--stats-only")
    if args.no_prefix_caching:
        out.append("--no-prefix-caching")
    if args.batch_invariant:
        out.append("--batch-invariant")
    return out


def _monitor(run_dir: Path, procs, total: int) -> None:
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        done = load_outputs(run_dir)
        n_ok = sum(1 for r in done.values() if r.get("status") == "ok")
        print(f"  [{now()}] recorded {len(done)}/{total} (ok {n_ok})", flush=True)


# ------------------------------------------------------------------ worker
def worker(args) -> None:
    # CUDA gets initialised in this process before vLLM starts its engine-core
    # subprocess, so that subprocess must be spawned, not forked.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if args.batch_invariant:
        # read by vLLM at import time: must be set before the first vllm import
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    # Long videos: large ViT activations; reduce fragmentation.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from qwen_omni_utils import process_mm_info
    from qwen_omni_utils.v2_5.audio_process import process_audio_info
    from qwen_omni_utils.v2_5.vision_process import get_video_reader_backend
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    k = args._worker
    run_dir = Path(args._run_dir)
    out_path = run_dir / OUTPUT_FILE
    import warnings
    warnings.filterwarnings("ignore", message=".*audioread.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
    records = {r["qa_id"]: r for r in json.loads(ANNOTATION.read_text(encoding="utf-8"))}
    shard = [records[int(q)] for q in args._shard.split(",") if q]
    shard.sort(key=lambda r: (r["video_id"], r["qa_id"]))
    template = load_prompt_template(mode_prompt_key(args.input_mode))
    want_video, want_audio = mode_inputs(args.input_mode)
    av_mode = args.input_mode not in UNIMODAL
    video_dir = Path(args.video_dir)
    print(f"[w{k}] {len(shard)} questions, video reader backend: {get_video_reader_backend()}", flush=True)

    llm = LLM(model=MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization, tensor_parallel_size=1,
              limit_mm_per_prompt={"image": 0, "video": 1, "audio": 1},
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len, seed=args.seed,
              mm_processor_cache_gb=0, enable_prefix_caching=not args.no_prefix_caching,
              **({"attention_backend": "FLASH_ATTN"} if args.batch_invariant else {}))
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok = processor.tokenizer
    video_pad = tok.convert_tokens_to_ids("<|video_pad|>")
    audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>")
    base_sp = sampling_settings(args)
    if args.stats_only:
        base_sp = {**base_sp, "max_tokens": 1}

    def preprocess(video_id: str):
        """Read frames + the video's audio exactly as in the model card. Cached per video."""
        path = video_dir / f"{video_id}.mp4"
        t = time.time()
        content = [{"type": "video", "video": str(path), "fps": FPS}]
        messages = [{"role": "user", "content": content}]
        meta = {"video_seconds": round(video_seconds(path), 2)}
        audios = videos = None
        if want_video:
            # av: frames + audio track; v: the same frames, no audio
            audios, images, videos, vkw = process_mm_info(messages, use_audio_in_video=av_mode,
                                                          return_video_kwargs=True)
            v = videos[0]
            meta.update({"frames": int(v.shape[0]),
                         "frame_hw_after_qwen_omni_utils": [int(v.shape[2]), int(v.shape[3])],
                         "sample_fps": vkw.get("fps", [None])[0]})
        elif want_audio:
            # a: the same waveform as av (same function, same call), without reading frames
            audios = process_audio_info(messages, use_audio_in_video=True)
        if want_audio:
            meta.update({"audio_seconds": round(len(audios[0]) / 16000, 2) if audios else 0.0,
                         "audio_peak": float(abs(audios[0]).max()) if audios and len(audios[0]) else 0.0})
        meta["preprocess_s"] = round(time.time() - t, 2)
        return audios, videos, meta

    def build_request(rec, audios, videos):
        prompt = template.format(question=rec["question"], options=rec["options"])
        path = str(video_dir / f"{rec['video_id']}.mp4")
        content = []
        if want_video:
            content.append({"type": "video", "video": path, "fps": FPS})
        elif want_audio:
            content.append({"type": "audio", "audio": path})  # chat template only; data passed below
        content.append({"type": "text", "text": prompt})
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        if av_mode:
            inputs = {"prompt": text,
                      "multi_modal_data": {"video": videos, "audio": audios},
                      "mm_processor_kwargs": {"use_audio_in_video": True}}
        elif want_video:
            inputs = {"prompt": text, "multi_modal_data": {"video": videos},
                      "mm_processor_kwargs": {"use_audio_in_video": False}}
        elif want_audio:
            inputs = {"prompt": text, "multi_modal_data": {"audio": audios}}
        else:
            inputs = {"prompt": text}
        sp = dict(base_sp)
        if args.decoding == "sample":
            sp["seed"] = request_seed(args.seed, rec["qa_id"])
        if args.constrained == "json":
            from vllm.sampling_params import StructuredOutputsParams
            schema = evidence_schema(option_letters(rec["options"]))
            return inputs, SamplingParams(**sp, structured_outputs=StructuredOutputsParams(json=schema)), sp
        return inputs, SamplingParams(**sp), sp

    def record(rec, meta, out=None, error=None):
        """One record per attempt in output.jsonl: identity and status, then what went
        into the model (input) and what came out (output). Settings shared by all
        questions (sampling params, max_model_len, fps, ...) are in info.txt."""
        r = {"qa_id": rec["qa_id"], "video_id": rec["video_id"], "level": rec["level"],
             "capability": rec["capability"], "time": now()}
        inp = dict(meta or {})
        if error is not None:
            etype = "prompt_too_long" if "longer than the maximum model length" in error else "error"
            r.update(status="error", error_type=etype, error=error[-2000:], input=inp)
            append_record(out_path, r)
            return
        ids = list(out.prompt_token_ids or [])
        o = out.outputs[0]
        n_out = len(o.token_ids)
        # vLLM reports finish_reason="length" both when max_tokens is reached and when
        # prompt + output reaches max_model_len (generation is cut, no error).
        if o.finish_reason != "length":
            stop_cause = "natural_stop"
        elif len(ids) + n_out >= args.max_model_len:
            stop_cause = "max_model_len_reached"
        else:
            stop_cause = "max_tokens_reached"
        n_video, n_audio = ids.count(video_pad), ids.count(audio_pad)
        frames = inp.get("frames")
        inp.update(prompt_tokens=len(ids), video_tokens=n_video, audio_tokens=n_audio,
                   text_and_special_tokens=len(ids) - n_video - n_audio,
                   video_tokens_per_temporal_grid=n_video / (frames / 2) if frames else None,
                   context_fraction=round(len(ids) / args.max_model_len, 3))
        r["status"] = "ok"
        if (n_video > 0) != want_video or (n_audio > 0) != want_audio:
            r.update(status="error", error_type="mm_tokens_missing",
                     error=f"prompt has {n_video} video and {n_audio} audio tokens "
                           f"(expected video {want_video}, audio {want_audio})")
        r["input"] = inp
        r["output"] = {"stop_cause": stop_cause, "n_output_tokens": n_out, "raw_response": o.text}
        append_record(out_path, r)

    # Group questions by video; preprocess the next videos in the background.
    videos_order = []
    for r in shard:
        if not videos_order or videos_order[-1] != r["video_id"]:
            videos_order.append(r["video_id"])
    by_video = {}
    for r in shard:
        by_video.setdefault(r["video_id"], []).append(r)
    pool = ThreadPoolExecutor(max_workers=3)
    futures = {vid: pool.submit(preprocess, vid) for vid in videos_order[:4]}
    next_i = 4

    batch: list[tuple] = []

    def abort_if_engine_dead(tb: str) -> None:
        """A dead engine (e.g. CUDA OOM) is not a property of the question: stop the
        worker WITHOUT recording, so that --resume retries these questions."""
        if "EngineDeadError" in tb or "OutOfMemoryError" in tb or "Engine core" in tb:
            print(f"[w{k}] ENGINE DEAD, aborting worker without recording:\n{tb[-3000:]}", flush=True)
            os._exit(3)

    def flush(batch):
        if not batch:
            return
        t = time.time()
        try:
            outs = llm.generate([b[1] for b in batch], [b[2] for b in batch], use_tqdm=False)
            for (rec, _, _, meta, _), out in zip(batch, outs):
                record(rec, meta, out=out)
        except Exception:  # noqa: BLE001 - retry one by one to isolate the failing request
            tb = traceback.format_exc()
            abort_if_engine_dead(tb)
            print(f"[w{k}] batch failed, retrying one by one:\n{tb[-1500:]}", flush=True)
            for rec, inputs, sparams, meta, _ in batch:
                try:
                    out = llm.generate([inputs], [sparams], use_tqdm=False)[0]
                    record(rec, meta, out=out)
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    abort_if_engine_dead(tb)
                    record(rec, meta, error=tb)
        print(f"[w{k}] {now()} wrote {len(batch)} ({round(time.time() - t, 1)} s)", flush=True)

    for vid in videos_order:
        try:
            audios, videos, meta = futures.pop(vid).result()
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
            for rec in by_video[vid]:
                record(rec, None, error="preprocess failed: " + err)
            continue
        finally:
            if next_i < len(videos_order):
                futures[videos_order[next_i]] = pool.submit(preprocess, videos_order[next_i])
                next_i += 1
        for rec in by_video[vid]:
            q_audios, q_meta = audios, meta
            if args.audio_mask != "none":
                # per-question copy; `audios` is shared by all questions of this video
                masked, info = audio_mask.mask_for_question(rec, audios[0], args.audio_mask,
                                                            args.mask_fill, args.mask_pad)
                q_audios, q_meta = [masked], {**meta, "audio_mask": info}
            inputs, sparams, sp = build_request(rec, q_audios, videos)
            batch.append((rec, inputs, sparams, q_meta, sp))
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
    flush(batch)
    pool.shutdown()
    print(f"[w{k}] done", flush=True)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", help="modello_input_decodifica_sottoinsieme[_extra]")
    ap.add_argument("--resume", help="exact name of an existing folder in custom/output/")
    ap.add_argument("--description", help="required for a new run; goes to info.txt")
    ap.add_argument("--subset", default="full", help="'full' or a file name in custom/subsets/ (no .json)")
    ap.add_argument("--input-mode", default="av", choices=list(INPUT_MODES),
                    help="; ".join(f"{k} = {v[1]}" for k, v in INPUT_MODES.items()))
    ap.add_argument("--video-dir", default=None, help="default depends on --input-mode")
    ap.add_argument("--decoding", default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--constrained", default="none", choices=["none", "json"],
                    help="json = vLLM structured outputs: answer JSON with '[mm:ss,mm:ss]' timestamps "
                         f"(max {MAX_EVIDENCE_ITEMS} intervals per list); prompt unchanged")
    ap.add_argument("--audio-mask", default="none", choices=["none", *audio_mask.TARGETS],
                    help="av only: mask the question's audio evidence, or random segments of the same "
                         "lengths outside it (control); see custom/audio_mask.py")
    ap.add_argument("--mask-fill", default="silence", choices=list(audio_mask.FILLS))
    ap.add_argument("--mask-pad", type=float, default=1.0, help="seconds added on each side of the evidence")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--stats-only", action="store_true",
                    help="generate 1 token only: measures input sizes/truncation, no scoring")
    ap.add_argument("--gpus", default=None,
                    help="physical GPU indices, e.g. 2 or 0,1 (one worker per GPU). Default: the "
                         "CUDA_VISIBLE_DEVICES of the command line (CUDA_VISIBLE_DEVICES=2 python ...)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                    help="model card uses 0.95; 0.95 OOMs in the vision encoder on long videos")
    ap.add_argument("--max-num-seqs", type=int, default=1,
                    help="keep 1: see max_num_seqs_reason in info.txt (model card uses 8)")
    ap.add_argument("--batch-size", type=int, default=4, help="requests per llm.generate call")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="determinism test: disable vLLM prefix caching (slower)")
    ap.add_argument("--batch-invariant", action="store_true",
                    help="determinism test: vLLM batch-invariant mode (VLLM_BATCH_INVARIANT=1, FLASH_ATTN)")
    ap.add_argument("--_worker", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--_run-dir", help=argparse.SUPPRESS)
    ap.add_argument("--_shard", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.resume and args.run_name:
        ap.error("use either --run-name (new run) or --resume (existing folder), not both")
    if args.video_dir is None:
        args.video_dir = str(INPUT_MODES[args.input_mode][0])
    if args._worker is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if args.gpus is None:
            if not cvd:
                ap.error("say which GPU: CUDA_VISIBLE_DEVICES=N python ... (or --gpus N)")
            args.gpus = cvd
        elif cvd and cvd != args.gpus:
            ap.error(f"CUDA_VISIBLE_DEVICES={cvd} and --gpus {args.gpus} disagree: give only one of them")
        if not re.fullmatch(r"\d+(,\d+)*", args.gpus):
            ap.error(f"GPUs must be physical indices like 2 or 0,1 (got {args.gpus!r})")
    if args.constrained == "json" and args.input_mode in UNIMODAL:
        ap.error("--constrained json has a video+audio schema; not available with --input-mode v/a/t")
    if args.audio_mask != "none" and args.input_mode != "av":
        ap.error("--audio-mask is only available with --input-mode av")
    if args._worker is not None:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
