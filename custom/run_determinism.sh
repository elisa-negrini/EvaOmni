#!/bin/bash
# Determinism tests, Qwen3-Omni Instruct, greedy, 48 questions (custom/subsets/det48.json:
# 12 videos of at most 5 min, 4 questions each). Every configuration is run TWICE
# (-a, -b) in two separate processes with identical settings; then
#   python custom/tools/determinism_report.py
# prints, for each pair, how many answers are identical character by character.
#
# Why: greedy runs with identical settings differ (av: about 25-35% identical answers),
# but the two full runs with the audio evidence replaced by noise were 100% identical,
# with bit-identical inputs. Causes ruled out so far: different inputs, cache reuse
# between questions of the same video (first questions differ as much), one GPU/process
# being "unlucky" (in the silence runs both GPUs differ), CPU audio features
# (bit-identical in 3 processes).
#
#   base       = exactly our run settings: does the nondeterminism show on 48 questions?
#   noise      = audio evidence -> noise: is it deterministic here too?
#   noprefix   = vLLM prefix caching off
#   nomp       = VLLM_ENABLE_V1_MULTIPROCESSING=0 (vLLM's reproducibility advice, offline)
#   batchinv   = vLLM batch-invariant mode (VLLM_BATCH_INVARIANT=1, FLASH_ATTN backend)
# About 5 min per line (batchinv: the first one recompiles; noprefix a bit slower).
#
# One line per run, executed one after the other. Edit CUDA_VISIBLE_DEVICES on each
# line; comment out (#) the lines you do not want. Keep each pair (-a, -b) on the same GPU.
# Usage, from tmux:   bash custom/run_determinism.sh

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate av_env

COMMON="--subset det48 --decoding greedy --max-tokens 2048"
D="determinism test, greedy, max_tokens 2048, 48 questions; run twice with identical settings"

CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av_greedy_det48_base-a --input-mode av $COMMON --description "Baseline (our run settings), copy a; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av_greedy_det48_base-b --input-mode av $COMMON --description "Baseline (our run settings), copy b; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskev-noise_greedy_det48_base-a --input-mode av --audio-mask evidence --mask-fill noise --mask-pad 1.0 $COMMON --description "Audio evidence -> noise (as the full masking runs), copy a; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskev-noise_greedy_det48_base-b --input-mode av --audio-mask evidence --mask-fill noise --mask-pad 1.0 $COMMON --description "Audio evidence -> noise (as the full masking runs), copy b; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av_greedy_det48_noprefix-a --input-mode av --no-prefix-caching $COMMON --description "Prefix caching off, copy a; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av_greedy_det48_noprefix-b --input-mode av --no-prefix-caching $COMMON --description "Prefix caching off, copy b; $D"
CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 python custom/run_qwen_omni.py --run-name qwen3omni_av_greedy_det48_nomp-a --input-mode av $COMMON --description "VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine in the same process), copy a; $D"
CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 python custom/run_qwen_omni.py --run-name qwen3omni_av_greedy_det48_nomp-b --input-mode av $COMMON --description "VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine in the same process), copy b; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av_greedy_det48_batchinv-a --input-mode av --batch-invariant $COMMON --description "vLLM batch-invariant mode (FLASH_ATTN), copy a; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av_greedy_det48_batchinv-b --input-mode av --batch-invariant $COMMON --description "vLLM batch-invariant mode (FLASH_ATTN), copy b; $D"

python custom/tools/determinism_report.py
