#!/bin/bash
# Qwen3-Omni Instruct, greedy, ALL 1027 questions, with the output format forced by
# vLLM structured outputs (--constrained json): same prompt, but the answer must be
# a JSON with "[mm:ss,mm:ss]" timestamps (max 100 per list). Same settings as
# custom/run_test_sub100_json.sh, on the full benchmark.
# Compare with the free greedy run on the full benchmark:
#   ~/miniconda3/envs/av_env/bin/python custom/tools/compare_runs.py \
#       qwen3omni-instr_av_greedy_full qwen3omni_av_greedy-json_full
# Usage: bash custom/run_full_json.sh <gpu>      e.g. bash custom/run_full_json.sh 1
# If interrupted: same command with "--resume qwen3omni_av_greedy-json_full" in place
# of "--run-name ..." (questions already answered are skipped).

set -e
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=~/miniconda3/envs/av_env/bin/python

$PY custom/run_qwen_omni.py \
    --run-name qwen3omni_av_greedy-json_full \
    --description "Greedy, all 1027 questions, output format forced with vLLM structured outputs (JSON schema, [mm:ss,mm:ss], max 100 intervals per list); same prompt. Compare with qwen3omni-instr_av_greedy_full" \
    --subset full \
    --input-mode av \
    --decoding greedy \
    --constrained json \
    --gpus $GPU
