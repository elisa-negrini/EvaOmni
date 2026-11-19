#!/bin/bash
# Audio-evidence masking, Qwen3-Omni Instruct, video + audio (av).
# For each question the audio evidence (+/- 1 s) is replaced by silence or by noise
# (RMS of the whole track), with a 10 ms fade; frames, prompt and the rest of the
# audio are as in av. Control: random segments with the same number and lengths,
# outside the evidence (same positions for silence and noise). custom/audio_mask.py.
# Greedy, free output, max_tokens 2048, as the av reference qwen3omni_av_greedy_maxtok-2048.
# Each condition twice (_r1, _r2), as in the modality ablation.
#
# One line per run, executed one after the other. Edit CUDA_VISIBLE_DEVICES on each
# line (one GPU = one worker; "0,1" = two GPUs); comment out (#) the lines you do not want.
# The first two lines are 10-question tests (a few minutes each): if one of them
# fails, stop the script (Ctrl-C) before the full runs.
#
# Usage, from tmux:   bash custom/run_audiomask.sh
# Interrupted run: same line with "--resume <run name>" in place of "--run-name ...".
# Before: python custom/selftest_audio_mask.py
# Afterwards, per question: python custom/tools/compare_runs.py qwen3omni_av-maskev-silence_greedy_r1 qwen3omni_av-maskrand-silence_greedy_r1
# Listening: streamlit run custom/tools/viewer.py --server.port 8501 --server.headless true

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate av_env

COMMON="--input-mode av --decoding greedy --max-tokens 2048 --mask-pad 1.0"
D="greedy, max_tokens 2048, pad 1 s, fade 10 ms; otherwise as qwen3omni_av_greedy_maxtok-2048"


# --- tests, 10 questions --------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskev-silence_greedy_check10 --audio-mask evidence --mask-fill silence --subset check10-l1 $COMMON --description "Test: audio evidence replaced by silence, 10 questions; $D"
CUDA_VISIBLE_DEVICES=0 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskrand-noise_greedy_check10 --audio-mask random --mask-fill noise --subset check10-l1 $COMMON --description "Test: random control segments replaced by noise, 10 questions; $D"

# --- full benchmark, two runs per condition -------------------------------------------------
CUDA_VISIBLE_DEVICES=0,1 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskev-silence_greedy_r1 --audio-mask evidence --mask-fill silence --subset full $COMMON --description "Audio masking run 1 of 2: audio evidence replaced by silence, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskev-silence_greedy_r2 --audio-mask evidence --mask-fill silence --subset full $COMMON --description "Audio masking run 2 of 2: audio evidence replaced by silence, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskev-noise_greedy_r1 --audio-mask evidence --mask-fill noise --subset full $COMMON --description "Audio masking run 1 of 2: audio evidence replaced by noise, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskev-noise_greedy_r2 --audio-mask evidence --mask-fill noise --subset full $COMMON --description "Audio masking run 2 of 2: audio evidence replaced by noise, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskrand-silence_greedy_r1 --audio-mask random --mask-fill silence --subset full $COMMON --description "Audio masking control run 1 of 2: random segments (evidence lengths) replaced by silence, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskrand-silence_greedy_r2 --audio-mask random --mask-fill silence --subset full $COMMON --description "Audio masking control run 2 of 2: random segments (evidence lengths) replaced by silence, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskrand-noise_greedy_r1 --audio-mask random --mask-fill noise --subset full $COMMON --description "Audio masking control run 1 of 2: random segments (evidence lengths) replaced by noise, all 1027 questions; $D"
CUDA_VISIBLE_DEVICES=0,1 python custom/run_qwen_omni.py --run-name qwen3omni_av-maskrand-noise_greedy_r2 --audio-mask random --mask-fill noise --subset full $COMMON --description "Audio masking control run 2 of 2: random segments (evidence lengths) replaced by noise, all 1027 questions; $D"
