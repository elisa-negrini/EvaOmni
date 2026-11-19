#!/bin/bash
# Modality ablation, Qwen3-Omni Instruct: video only (v), audio only (a), text only (t).
# The missing input is omitted (no placeholder); prompt = the repository's prompt
# for that input (video_only / audio_only / text_only). Greedy, free output,
# max_tokens 2048, as the video+audio reference qwen3omni_av_greedy_maxtok-2048.
# Each input twice (_r1, _r2): the paper's Table 6(d) counts a question as correct
# only if it is correct in both runs (Qwen3-Omni: audio 18.2, video 12.5, no text value).
#
# One line per run, executed one after the other. Edit CUDA_VISIBLE_DEVICES on each
# line (one GPU = one worker; "0,1" = two GPUs); comment out (#) the lines you do not want.
# The first three lines are 10-question tests (a few minutes each): if one of them
# fails, stop the script (Ctrl-C) before the full runs.
#
# Usage, from tmux:   bash custom/run_unimodal.sh
# Interrupted run: same line with "--resume <run name>" in place of "--run-name ...".
# Afterwards, the paper's "correct in both runs":
#   python custom/tools/both_runs_acc.py qwen3omni_a_greedy_maxtok-2048_r1 qwen3omni_a_greedy_maxtok-2048_r2

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate av_env

COMMON="--decoding greedy --max-tokens 2048"
D="greedy, max_tokens 2048, missing inputs omitted, repository prompt for this input"


# --- full benchmark, two runs per input -----------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_t_greedy_r1 --input-mode t --subset full $COMMON --description "Modality ablation run 1 of 2: text only (question and options), all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_t_greedy_r2 --input-mode t --subset full $COMMON --description "Modality ablation run 2 of 2: text only (question and options), all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_a_greedy_r1 --input-mode a --subset full $COMMON --description "Modality ablation run 1 of 2: audio only (no frames), all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_a_greedy_r2 --input-mode a --subset full $COMMON --description "Modality ablation run 2 of 2: audio only (no frames), all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_v_greedy_r1 --input-mode v --subset full $COMMON --description "Modality ablation run 1 of 2: video only (no audio), all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_v_greedy_r2 --input-mode v --subset full $COMMON --description "Modality ablation run 2 of 2: video only (no audio), all 1027 questions; $D"
