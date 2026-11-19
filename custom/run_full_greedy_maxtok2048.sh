#!/bin/bash
# Qwen3-Omni Instruct, greedy, ALL 1027 questions, free output (no constraint),
# max_tokens 2048 instead of the model card's 16384, to save time.
# Why it does not change the scores: in qwen3omni-instr_av_greedy_full the longest
# answer that stopped by itself has 338 tokens, and all 102 answers cut by the
# limit are loops (lost evidence, letter recovered from the start of the answer).
# Greedy generation does not depend on the limit, it only stops earlier.
# Compare with the 16384 run:
#   ~/miniconda3/envs/av_env/bin/python custom/tools/compare_runs.py \
#       qwen3omni-instr_av_greedy_full qwen3omni_av_greedy_full_maxtok-2048
# Usage: bash custom/run_full_greedy_maxtok2048.sh <gpu>      e.g. bash custom/run_full_greedy_maxtok2048.sh 2
# If interrupted: same command with "--resume qwen3omni_av_greedy_full_maxtok-2048"
# in place of "--run-name ..." (questions already answered are skipped).

set -e
cd "$(dirname "$0")/.."

GPU=${1:-0}
PY=~/miniconda3/envs/av_env/bin/python

$PY custom/run_qwen_omni.py \
    --run-name qwen3omni_av_greedy_maxtok-2048 \
    --description "Greedy, all 1027 questions, free output, max_tokens 2048 (model card: 16384) to stop loops earlier; otherwise identical to qwen3omni-instr_av_greedy_full. Measures run time and, against that run, greedy run-to-run variability on the full benchmark" \
    --subset full \
    --input-mode av \
    --decoding greedy \
    --max-tokens 2048 \
    --gpus $GPU
