#!/usr/bin/env python3
"""Empirical check: what happens when prompt + output exceed max_model_len?

Uses the longest-fitting video (036, ~62.3k prompt tokens) with the same
pipeline as run_qwen_omni.py (qwen_omni_utils, use_audio_in_video=True,
max_model_len=65536, max_num_seqs=1, mm_processor_cache_gb=0) and greedy
decoding with max_tokens=16384 and ignore_eos=True, so the model cannot stop
naturally before the context is full.

Expected from vLLM 0.17.1 source (v1/core/sched/utils.py::check_stop): no error;
generation stops when prompt+output == max_model_len, finish_reason="length".
The result is printed (nothing is written to disk).

Run on one GPU:  CUDA_VISIBLE_DEVICES=0 python custom/tools/test_maxlen.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

CUSTOM = Path(__file__).resolve().parents[1]
REPO = CUSTOM.parent
sys.path.insert(0, str(CUSTOM))
import run_qwen_omni as R  # noqa: E402


def main():
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    ann = {r["qa_id"]: r for r in json.loads(R.ANNOTATION.read_text())}
    rec = next(r for r in ann.values() if r["video_id"] == "036")
    video = str(REPO / "data" / "resized_video" / "036.mp4")
    msgs = [{"role": "user", "content": [{"type": "video", "video": video, "fps": R.FPS},
                                         {"type": "text", "text": R.load_prompt_template().format(
                                             question=rec["question"], options=rec["options"])}]}]
    audios, _, videos = process_mm_info(msgs, use_audio_in_video=True)
    processor = Qwen3OmniMoeProcessor.from_pretrained(R.MODEL_ID, revision=R.MODEL_REVISION)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    llm = LLM(model=R.MODEL_ID, revision=R.MODEL_REVISION, dtype="bfloat16", gpu_memory_utilization=0.80,
              limit_mm_per_prompt={"image": 0, "video": 1, "audio": 1}, max_num_seqs=1,
              max_model_len=65536, seed=1234, mm_processor_cache_gb=0)
    sp = SamplingParams(temperature=0.0, max_tokens=16384, ignore_eos=True)
    try:
        out = llm.generate([{"prompt": text, "multi_modal_data": {"video": videos, "audio": audios},
                             "mm_processor_kwargs": {"use_audio_in_video": True}}], [sp], use_tqdm=False)[0]
        o = out.outputs[0]
        res = {"qa_id": rec["qa_id"], "video_id": "036", "error": None,
               "prompt_tokens": len(out.prompt_token_ids), "output_tokens": len(o.token_ids),
               "prompt_plus_output": len(out.prompt_token_ids) + len(o.token_ids),
               "max_model_len": 65536, "max_tokens": 16384, "finish_reason": o.finish_reason,
               "output_head": o.text[:300], "output_tail": o.text[-200:]}
    except Exception as e:  # noqa: BLE001
        res = {"qa_id": rec["qa_id"], "video_id": "036", "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
