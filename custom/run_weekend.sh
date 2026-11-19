#!/usr/bin/env bash
# =============================================================================
# EvaOmni / Qwen3-Omni-30B-A3B-Instruct — runs to launch BY HAND on ONE GPU.
#
#   bash custom/run_weekend.sh --gpu 2            # run everything on GPU 2
#   bash custom/run_weekend.sh --gpu 2 --dry-run  # only print the commands
#   GPU=2 bash custom/run_weekend.sh              # same, GPU from the environment
#
# Must be launched from the repository root (/home/elisa.negrini/EvaOmni).
# Order: (1) max_model_len test, (2) greedy full, (3) greedy repeat on sub200,
#        (4) OPTIONAL sampling full seed 1234, (5) OPTIONAL sampling seed 4321
#        on sub200. To skip 4 and/or 5, comment out their block at the bottom.
#
# Every step prints its start/end time. Nothing is logged to custom/output/; to
# keep a log, redirect it yourself, e.g.  bash custom/run_weekend.sh --gpu 2 2>&1 | tee ~/weekend.log
# The script stops at the first failing step (set -e). Questions of video 293
# that do not fit in the context (prompt_too_long) are NOT a failure: the
# runner exits with 0 in that case.
#
# ---- If a run is interrupted (crash, reboot, Ctrl-C) ------------------------
# Do NOT relaunch this whole script: a new run with an existing name would go
# to a new folder <name>_v2, and this script refuses to start a run whose
# folder already exists. Instead resume THAT run only: same command as below,
# with "--resume <exact folder name>" in place of "--run-name ...". Remove
# "--description" or keep it (it is ignored on resume). The resume is accepted only if all
# settings are identical to the [settings] in the folder's info.txt; questions already
# answered are skipped. Example for the greedy full run on GPU 2:
#
#   ~/miniconda3/envs/av_env/bin/python custom/run_qwen_omni.py \
#       --resume qwen3omni-instr_av_greedy_full \
#       --subset full --input-mode av --decoding greedy --seed 1234 \
#       --max-tokens 16384 --max-model-len 65536 --gpus 2 --no-score
#   ~/miniconda3/envs/av_env/bin/python custom/score.py --run-dir custom/output/qwen3omni-instr_av_greedy_full
#   (score.py also writes evaluation.txt, the comparison with the paper)
#
# Then comment out the steps already completed in this file and relaunch it
# for the remaining ones.
# =============================================================================
set -euo pipefail

# ---- settings you may change -------------------------------------------------
GPU="${GPU:-0}"                 # physical GPU index (as in nvidia-smi); or use --gpu N
MIN_FREE_MIB=80000              # refuse to start if the GPU has less free memory than this
PY="$HOME/miniconda3/envs/av_env/bin/python"   # interpreter of the av_env environment
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (use --gpu N, --dry-run)"; exit 2 ;;
    esac
done

if [[ ! -f custom/run_qwen_omni.py ]]; then
    echo "ERROR: launch from the repository root (cd /home/elisa.negrini/EvaOmni)."; exit 2
fi

# ---- helpers (only printing, timing and checks) ------------------------------
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

run() {                                  # print the command (copy-pasteable), then run it unless --dry-run
    echo "+ $(printf '%q ' "$@")"
    if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}

check_gpu_free() {                       # stop (never wait or share) if the GPU is busy
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    echo "GPU $GPU: ${free} MiB free (need >= ${MIN_FREE_MIB} MiB)"
    if [[ $DRY_RUN -eq 0 && "$free" -lt "$MIN_FREE_MIB" ]]; then
        echo "ERROR: GPU $GPU does not have enough free memory. Someone else is probably using it."
        echo "       Nothing was started. Check nvidia-smi and choose a free GPU with --gpu N."
        nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv -i "$GPU" || true
        exit 1
    fi
}

check_new_run() {                        # refuse to create <name>_v2 by accident
    if [[ -d "custom/output/$1" ]]; then
        echo "ERROR: custom/output/$1 already exists. To continue it use --resume $1"
        echo "       (see the comment at the top of this file), then comment out this step."
        [[ $DRY_RUN -eq 1 ]] || exit 1
    fi
}

begin() { echo; echo "=================================================================="; \
          echo "[$(stamp)] START: $1"; echo "=================================================================="; \
          check_gpu_free; }
end()   { echo "[$(stamp)] END:   $1"; }

echo "[$(stamp)] run_weekend.sh on GPU $GPU  (dry-run=$DRY_RUN)"
echo "git commit: $(git rev-parse --short HEAD)   modified originals (should be empty):"
git status --porcelain -- . ':(exclude)custom' || true


# =============================================================================
# 1) Empirical test: what happens when prompt + output exceed max_model_len?
#    Video 036 (62.3k prompt tokens) with ignore_eos=True, so the model cannot
#    stop before the context is full. Expected (from the vLLM source): no
#    error, output cut at 65536 - 62265 = 3271 tokens, finish_reason="length".
#    The result is printed.   (~5 min)
# =============================================================================
begin "1/5 max_model_len test"
run env CUDA_VISIBLE_DEVICES="$GPU" "$PY" custom/tools/test_maxlen.py
end "1/5 max_model_len test"


# =============================================================================
# 2) Greedy, full benchmark (1027 questions) — the main replication run.
#    The model card prescribes greedy decoding for evaluating Instruct models,
#    so this is our best guess of the authors' setting. Then: scoring in every
#    version (strict, permissive, permissive_no_free_text, diagnostic) and
#    comparison with the paper's Table 2 (evaluation.txt).   (estimate on 1 GPU: 4–8 h)
# =============================================================================
check_new_run qwen3omni-instr_av_greedy_full
begin "2/5 greedy full"
run "$PY" custom/run_qwen_omni.py \
    --run-name qwen3omni-instr_av_greedy_full \
    --description "Main replication run: greedy decoding (model card evaluation setting), all 1027 questions, video+audio (use_audio_in_video), 1 FPS resized_video, max_tokens 16384, max_model_len 65536" \
    --subset full \
    --input-mode av \
    --decoding greedy \
    --seed 1234 \
    --max-tokens 16384 \
    --max-model-len 65536 \
    --gpus "$GPU" \
    --no-score
run "$PY" custom/score.py --run-dir custom/output/qwen3omni-instr_av_greedy_full
end "2/5 greedy full"


# =============================================================================
# 3) Greedy REPEATED on the 200 frozen questions (custom/subsets/sub200.json).
#    Identical settings to run 2: the differences measure vLLM's own
#    non-determinism, i.e. the noise floor for judging the gap to the paper.
#    Then: question-by-question comparison with the same 200 questions of run 2
#    (letters changed, C.G./B.G. changes).   (estimate on 1 GPU: 1–2 h)
# =============================================================================
check_new_run qwen3omni-instr_av_greedy_sub200-r2
begin "3/5 greedy repeat sub200"
run "$PY" custom/run_qwen_omni.py \
    --run-name qwen3omni-instr_av_greedy_sub200-r2 \
    --description "Greedy repeat (identical settings to qwen3omni-instr_av_greedy_full) on the frozen sub200 subset: measures vLLM non-determinism" \
    --subset sub200 \
    --input-mode av \
    --decoding greedy \
    --seed 1234 \
    --max-tokens 16384 \
    --max-model-len 65536 \
    --gpus "$GPU" \
    --no-score
run "$PY" custom/score.py --run-dir custom/output/qwen3omni-instr_av_greedy_sub200-r2
run "$PY" custom/tools/compare_runs.py qwen3omni-instr_av_greedy_full qwen3omni-instr_av_greedy_sub200-r2 --show 10
end "3/5 greedy repeat sub200"


# =============================================================================
# 4) OPTIONAL — Sampling, full benchmark, seed 1234.
#    The settings of the model card's vLLM code snippet: temperature 0.6,
#    top_p 0.95, top_k 20 (fixed by --decoding sample), seed 1234. This is the
#    other setting the authors may have used. Comment out this block to skip.
#    (estimate on 1 GPU: 4–8 h)
# =============================================================================
check_new_run qwen3omni-instr_av_sample-s1234_full
begin "4/5 sampling seed 1234 full"
run "$PY" custom/run_qwen_omni.py \
    --run-name qwen3omni-instr_av_sample-s1234_full \
    --description "Sampling as in the model card vLLM snippet (temperature 0.6, top_p 0.95, top_k 20), seed 1234, all 1027 questions, otherwise identical to qwen3omni-instr_av_greedy_full" \
    --subset full \
    --input-mode av \
    --decoding sample \
    --seed 1234 \
    --max-tokens 16384 \
    --max-model-len 65536 \
    --gpus "$GPU" \
    --no-score
run "$PY" custom/score.py --run-dir custom/output/qwen3omni-instr_av_sample-s1234_full
run "$PY" custom/tools/compare_runs.py qwen3omni-instr_av_greedy_full qwen3omni-instr_av_sample-s1234_full --show 5
end "4/5 sampling seed 1234 full"


# =============================================================================
# 5) OPTIONAL — Sampling with a DIFFERENT seed (4321) on the 200 frozen
#    questions. Compared with the same 200 questions of run 4, this measures
#    the variability due to sampling (with the same seed we would only measure
#    non-determinism). Needs run 4. Comment out this block to skip.
#    (estimate on 1 GPU: 1–2 h)
# =============================================================================
check_new_run qwen3omni-instr_av_sample-s4321_sub200
begin "5/5 sampling seed 4321 sub200"
run "$PY" custom/run_qwen_omni.py \
    --run-name qwen3omni-instr_av_sample-s4321_sub200 \
    --description "Sampling (temperature 0.6, top_p 0.95, top_k 20) with seed 4321 on the frozen sub200 subset: measures sampling variability against qwen3omni-instr_av_sample-s1234_full" \
    --subset sub200 \
    --input-mode av \
    --decoding sample \
    --seed 4321 \
    --max-tokens 16384 \
    --max-model-len 65536 \
    --gpus "$GPU" \
    --no-score
run "$PY" custom/score.py --run-dir custom/output/qwen3omni-instr_av_sample-s4321_sub200
run "$PY" custom/tools/compare_runs.py qwen3omni-instr_av_sample-s1234_full qwen3omni-instr_av_sample-s4321_sub200 --show 10
end "5/5 sampling seed 4321 sub200"


echo
echo "[$(stamp)] ALL STEPS DONE. Index of runs: custom/output/RUNS.md"
