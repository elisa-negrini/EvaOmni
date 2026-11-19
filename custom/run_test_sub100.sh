#!/bin/bash
# Test run: Qwen3-Omni Instruct, greedy, 100 questions (custom/subsets/sub100.json).
# Usage: bash custom/run_test_sub100.sh <gpu>      e.g. bash custom/run_test_sub100.sh 0

set -e
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=~/miniconda3/envs/av_env/bin/python

$PY custom/run_qwen_omni.py \
    --run-name qwen3omni_av_greedy_sub100 \
    --description "Test run with the new output layout: greedy, 100 questions" \
    --subset sub100 \
    --input-mode av \
    --decoding greedy \
    --gpus $GPU
