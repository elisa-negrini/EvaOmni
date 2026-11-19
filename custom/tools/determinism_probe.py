#!/usr/bin/env python3
"""Where does greedy nondeterminism start? Stage-by-stage fingerprints of one process.

Runs the first question of each det48 video (12 questions) through vLLM exactly as
custom/run_qwen_omni.py does (av, same preprocessing, prompt and LLM settings), with
max_tokens=1, and records a sha256 fingerprint of the data at every stage:

  cpu_video / cpu_audio   frames and waveform from qwen_omni_utils (CPU)
  vit_in / vit_out        input (pixel values) and output of the vision encoder (GPU)
  aud_in / aud_out        input (mel features) and output of the audio encoder (GPU)
  lm_in / lm_out          input embeddings / positions and output hidden states of the
                          language model (GPU; one entry per prefill chunk)
  logprobs                top-20 log-probabilities of the first generated token

The 12 questions are run twice in the same process (pass 1, pass 2), prefix caching off.
Run it twice (two separate processes) and compare:

  CUDA_VISIBLE_DEVICES=0 python custom/tools/determinism_probe.py --name probe-a
  CUDA_VISIBLE_DEVICES=0 python custom/tools/determinism_probe.py --name probe-b
  python custom/tools/determinism_probe.py --compare probe-a probe-b

The first stage whose fingerprint differs is where the nondeterminism comes from.
Output: custom/probe_output/<name>.json (nothing else is written). About 3-4 minutes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CUSTOM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUSTOM))
OUT = CUSTOM / "probe_output"
STAGES = ["cpu_video", "cpu_audio", "vit_in", "vit_out", "aud_in", "aud_out", "lm_in", "lm_out", "logprobs"]


def fp(t) -> dict:
    """Fingerprint of a tensor or ndarray: sha256 of the raw bytes, shape, dtype, float64 sum."""
    import numpy as np
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach()
        s = float(t.double().sum())
        t = t.contiguous().cpu()
        b = (t.view(torch.uint16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
        return {"sha": hashlib.sha256(b).hexdigest()[:16], "shape": list(t.shape), "dtype": str(t.dtype), "sum": s}
    a = np.ascontiguousarray(t)
    return {"sha": hashlib.sha256(a.tobytes()).hexdigest()[:16], "shape": list(a.shape), "dtype": str(a.dtype),
            "sum": float(a.astype("float64").sum())}


def probe(args):
    # same environment as the adapter's worker, plus: engine in this process, so that
    # the hooks below live in the same process as this script
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # apply_model with a local function
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import run_qwen_omni as R
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"{args.name}.json"
    if out_path.exists():
        sys.exit(f"{out_path} exists: choose another --name")
    records = {r["qa_id"]: r for r in json.loads(R.ANNOTATION.read_text(encoding="utf-8"))}
    qa_ids = R.load_subset("det48", list(records.values()))
    first = {}
    for q in qa_ids:
        first.setdefault(records[q]["video_id"], q)
    questions = [records[q] for q in sorted(first.values(), key=lambda q: (records[q]["video_id"], q))]
    video_dir = R.INPUT_MODES["av"][0]
    template = R.load_prompt_template(R.PROMPT_KEY)

    llm = LLM(model=R.MODEL_ID, revision=R.MODEL_REVISION, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=0.80, tensor_parallel_size=1,
              limit_mm_per_prompt={"image": 0, "video": 1, "audio": 1},
              max_num_seqs=1, max_model_len=65536, seed=0,
              mm_processor_cache_gb=0, enable_prefix_caching=False,
              enforce_eager=args.eager,
              **({"attention_backend": "FLASH_ATTN"} if args.batch_invariant else {}))
    processor = Qwen3OmniMoeProcessor.from_pretrained(R.MODEL_ID, revision=R.MODEL_REVISION)

    log: list = []  # (stage, fingerprint), filled by the hooks

    def install(model):
        def pre(stage, key, pos):
            def h(mod, a, kw):
                x = kw.get(key) if key in kw else (a[pos] if len(a) > pos else None)
                if x is not None:
                    log.append((stage, fp(x)))
                if stage == "lm_in" and kw.get("positions", a[1] if len(a) > 1 else None) is not None:
                    log.append(("lm_pos", fp(kw.get("positions", a[1] if len(a) > 1 else None))))
            return h

        def post(stage):
            def h(mod, a, kw, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(o, torch.Tensor):
                    log.append((stage, fp(o)))
            return h
        model.visual.register_forward_pre_hook(pre("vit_in", "x", 0), with_kwargs=True)
        model.visual.register_forward_hook(post("vit_out"), with_kwargs=True)
        model.audio_tower.register_forward_pre_hook(pre("aud_in", "input_features", 0), with_kwargs=True)
        model.audio_tower.register_forward_hook(post("aud_out"), with_kwargs=True)
        model.register_forward_pre_hook(pre("lm_in", "inputs_embeds", 3), with_kwargs=True)
        model.register_forward_hook(post("lm_out"), with_kwargs=True)
        return type(model).__name__
    print("hooks on", llm.apply_model(install), flush=True)

    cache = {}
    results = []
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    for p in range(1, args.passes + 1):
        for rec in questions:
            vid = rec["video_id"]
            if vid not in cache:
                path = video_dir / f"{vid}.mp4"
                msgs = [{"role": "user", "content": [{"type": "video", "video": str(path), "fps": R.FPS}]}]
                audios, _, videos, _ = process_mm_info(msgs, use_audio_in_video=True, return_video_kwargs=True)
                cache[vid] = (audios, videos)
            audios, videos = cache[vid]
            prompt = template.format(question=rec["question"], options=rec["options"])
            content = [{"type": "video", "video": str(video_dir / f"{vid}.mp4"), "fps": R.FPS},
                       {"type": "text", "text": prompt}]
            text = processor.apply_chat_template([{"role": "user", "content": content}],
                                                 tokenize=False, add_generation_prompt=True)
            inputs = {"prompt": text, "multi_modal_data": {"video": videos, "audio": audios},
                      "mm_processor_kwargs": {"use_audio_in_video": True}}
            log.clear()
            out = llm.generate([inputs], [sp], use_tqdm=False)[0]
            lp = out.outputs[0].logprobs[0]
            entry = {"pass": p, "qa_id": rec["qa_id"], "video_id": vid,
                     "prompt_tokens": len(out.prompt_token_ids),
                     "cpu_video": fp(videos[0]), "cpu_audio": fp(audios[0]),
                     "gpu": [[s, f] for s, f in log],
                     "logprobs": {str(t): v.logprob for t, v in sorted(lp.items())},
                     "token": out.outputs[0].token_ids[0]}
            results.append(entry)
            print(f"pass {p} qa {rec['qa_id']}: {len(log)} GPU fingerprints, token {entry['token']}", flush=True)
    meta = {"name": args.name, "eager": args.eager, "batch_invariant": args.batch_invariant,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "time": time.strftime("%F %T")}
    out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print("written", out_path)


def stages_of(e):
    """Ordered list of (stage label, sha, sum) for one question."""
    st = [("cpu_video", e["cpu_video"]["sha"], e["cpu_video"]["sum"]),
          ("cpu_audio", e["cpu_audio"]["sha"], e["cpu_audio"]["sum"])]
    seen = {}
    for s, f in e["gpu"]:
        seen[s] = seen.get(s, 0) + 1
        st.append((f"{s}#{seen[s]}", f["sha"], f["sum"]))
    lp = e["logprobs"]
    st.append(("logprobs", hashlib.sha256(json.dumps(lp, sort_keys=True).encode()).hexdigest()[:16],
               max(lp.values())))
    return st


def first_diff(e1, e2):
    """First stage (in the order of e1) whose fingerprint differs. A stage present on one
    side only (e.g. encoder output reused from vLLM's encoder cache) is noted and skipped."""
    s2 = {l: (h, v) for l, h, v in stages_of(e2)}
    skipped = []
    for l, h, v in stages_of(e1):
        if l not in s2:
            skipped.append(l)
            continue
        if s2[l][0] != h:
            return f"{l} (sum {v:.6g} vs {s2[l][1]:.6g})" + (f" [not run on one side: {skipped}]" if skipped else "")
    return None if not skipped else f"identical where both ran [not run on one side: {skipped}]"


def compare(a_name, b_name):
    A = json.loads((OUT / f"{a_name}.json").read_text())
    B = json.loads((OUT / f"{b_name}.json").read_text())
    for label, X in (("A", A), ("B", B)):
        print(f"{label} = {X['meta']}")
    key = lambda e: (e["pass"], e["qa_id"])  # noqa: E731
    a, b = {key(e): e for e in A["results"]}, {key(e): e for e in B["results"]}
    rows = [("A vs B, same pass", [(a[k], b[k]) for k in sorted(a) if k in b])]
    for label, X in (("A", a), ("B", b)):
        rows.append((f"{label}: pass 1 vs pass 2 (same process)",
                     [(X[k], X[(2, k[1])]) for k in sorted(X) if k[0] == 1 and (2, k[1]) in X]))
    for title, pairs in rows:
        print(f"\n== {title}: first stage that differs")
        for e1, e2 in pairs:
            d = first_diff(e1, e2)
            same_tok = "same token" if e1["token"] == e2["token"] else "DIFFERENT TOKEN"
            print(f"  pass {e1['pass']} qa {e1['qa_id']:5d} video {e1['video_id']:>4}: "
                  f"{d or 'identical at every stage'}  ({same_tok})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="output name, e.g. probe-a")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--eager", action="store_true", help="enforce_eager=True (no torch.compile, no CUDA graphs)")
    ap.add_argument("--batch-invariant", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.name:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            ap.error("say which GPU: CUDA_VISIBLE_DEVICES=N python ...")
        probe(args)
    else:
        ap.error("give --name (probe) or --compare A B")


if __name__ == "__main__":
    main()
