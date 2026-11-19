#!/bin/bash
# Qwen3-Omni Instruct, greedy, 100 questions (custom/subsets/sub100.json), with the
# output format forced by vLLM structured outputs (--constrained json): same prompt,
# but the answer must be a JSON with "[mm:ss,mm:ss]" timestamps (max 100 per list).
# Everything else is identical to custom/run_test_sub100.sh, to compare the two:
#   ~/miniconda3/envs/av_env/bin/python custom/tools/compare_runs.py \
#       qwen3omni_av_greedy_sub100 qwen3omni_av_greedy-json_sub100
# Usage: bash custom/run_test_sub100_json.sh <gpu>      e.g. bash custom/run_test_sub100_json.sh 1

set -e
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=~/miniconda3/envs/av_env/bin/python

$PY custom/run_qwen_omni.py \
    --run-name qwen3omni_av_greedy-json_sub100 \
    --description "Greedy, 100 questions, output format forced with vLLM structured outputs (JSON schema, [mm:ss,mm:ss], max 100 intervals per list); same prompt. Compare with qwen3omni_av_greedy_sub100" \
    --subset sub100 \
    --input-mode av \
    --decoding greedy \
    --constrained json \
    --gpus $GPU
