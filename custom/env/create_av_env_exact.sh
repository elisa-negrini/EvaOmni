#!/usr/bin/env bash
# Recreate av_env EXACTLY as it was on coruscant on 2026-11-20 (the env of every run so far).
#   av_env_conda.txt  conda list --explicit --md5: Python 3.12.13 and ffmpeg 9.0.2
#                     (conda-forge), with every build pinned; no solver involved
#   av_env_pip.txt    pip freeze: all 230 Python packages at their exact versions
#                     (vllm 0.17.1, torch 2.10.0+cu128, transformers 4.57.6, numpy 2.2.6,
#                     qwen-omni-utils 0.0.9, librosa 0.11.0, flash_attn wheel by URL, ...)
# The older create_av_env.sh gives the same main versions, but leaves the dependencies
# of librosa/audioread and the ffmpeg version free.
#
# Usage:   bash custom/env/create_av_env_exact.sh
# Needs: miniconda in $HOME/miniconda3, internet, NVIDIA driver >= 570 (CUDA 12.8 wheels).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx av_env; then
    echo "av_env already exists: remove it first (conda env remove -n av_env)"; exit 1
fi
# Every package comes from conda-forge: --override-channels keeps conda away from the
# Anaconda "defaults" channels (otherwise: CondaToSNonInteractiveError on a fresh miniconda).
conda create -y -n av_env --override-channels -c conda-forge --file "$HERE/av_env_conda.txt"
conda activate av_env
python -m pip install -r "$HERE/av_env_pip.txt"
# pip check reports "decord 0.6.0 is not supported on this platform" on the original
# env too: expected, not an error.
python -m pip check || true
python -c "import vllm, torch, transformers, qwen_omni_utils, librosa; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda ok:', torch.cuda.is_available())"
which ffmpeg
