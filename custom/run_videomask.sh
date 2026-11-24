#!/bin/bash
# Visual-evidence masking, Qwen3-Omni Instruct, video + audio (av).
# For each question the frames of the visual evidence (+/- 1 s, widened to whole frame
# pairs) are replaced by black or by Gaussian noise (per-channel mean/std of the whole
# video); audio, prompt, number of frames and the other frames are as in av.
# Control: random segments with the same number and lengths, outside the evidence
# (same positions for black and noise). custom/video_mask.py.
# Greedy, free output, max_tokens 2048, as the av reference qwen3omni_av_greedy_maxtok-2048.
# Each condition twice (_r1, _r2), as in the audio masking.
#
# One line per run, executed one after the other. Edit CUDA_VISIBLE_DEVICES on each
# line (one GPU = one worker; "0,1" = two GPUs); comment out (#) the lines you do not want.
# The first two lines are 10-question tests (a few minutes each): if one of them
# fails, stop the script (Ctrl-C) before the full runs.
#
# Usage, from tmux:   bash custom/run_videomask.sh
# Interrupted run: same line with "--resume <run name>" in place of "--run-name ...".
# Before: python custom/selftest_video_mask.py
# Afterwards, per question: python custom/tools/compare_runs.py qwen3omni_av-vmaskev-black_greedy_r1 qwen3omni_av-vmaskrand-black_greedy_r1

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate av_env

COMMON="--input-mode av --decoding greedy --max-tokens 2048 --video-mask-pad 1.0"
D="greedy, max_tokens 2048, pad 1 s, frame pairs; otherwise as qwen3omni_av_greedy_maxtok-2048"


# --- tests, 10 questions --------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskev-black_greedy_check10 --video-mask evidence --video-fill black --subset check10-l1 $COMMON --description "Test: visual evidence frames replaced by black, 10 questions; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskrand-noise_greedy_check10 --video-mask random --video-fill noise --subset check10-l1 $COMMON --description "Test: random control frames replaced by noise, 10 questions; $D"

# --- full benchmark, two runs per condition -------------------------------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskev-black_greedy_r1 --video-mask evidence --video-fill black --subset full $COMMON --description "Video masking run 1 of 2: visual evidence frames replaced by black, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskev-black_greedy_r2 --video-mask evidence --video-fill black --subset full $COMMON --description "Video masking run 2 of 2: visual evidence frames replaced by black, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskev-noise_greedy_r1 --video-mask evidence --video-fill noise --subset full $COMMON --description "Video masking run 1 of 2: visual evidence frames replaced by noise, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskev-noise_greedy_r2 --video-mask evidence --video-fill noise --subset full $COMMON --description "Video masking run 2 of 2: visual evidence frames replaced by noise, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskrand-black_greedy_r1 --video-mask random --video-fill black --subset full $COMMON --description "Video masking control run 1 of 2: random segments (evidence lengths) replaced by black, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskrand-black_greedy_r2 --video-mask random --video-fill black --subset full $COMMON --description "Video masking control run 2 of 2: random segments (evidence lengths) replaced by black, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskrand-noise_greedy_r1 --video-mask random --video-fill noise --subset full $COMMON --description "Video masking control run 1 of 2: random segments (evidence lengths) replaced by noise, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom/run_qwen_omni.py --run-name qwen3omni_av-vmaskrand-noise_greedy_r2 --video-mask random --video-fill noise --subset full $COMMON --description "Video masking control run 2 of 2: random segments (evidence lengths) replaced by noise, all 1027 questions; $D"
