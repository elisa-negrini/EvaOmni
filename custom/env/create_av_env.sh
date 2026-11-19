#!/usr/bin/env bash
# Recreate av_env: same Python/vLLM/torch/transformers/numpy as vllm_env,
# plus qwen-omni-utils, librosa and an ffmpeg binary.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
conda create -y -n av_env -c conda-forge python=3.12.13 ffmpeg
PY="$(conda run -n av_env which python)"
"$PY" -m pip install -r "$HERE/av_env_base_requirements.txt"
"$PY" -m pip install -c "$HERE/av_env_base_requirements.txt" -r "$HERE/av_env_extra_requirements.txt"
"$PY" -m pip check
