#!/bin/bash
# Where does the greedy nondeterminism start? custom/tools/determinism_probe.py runs the
# first question of each det48 video (12 questions, max_tokens=1, twice in the same
# process) and records a fingerprint of the data at every stage: CPU frames/audio ->
# vision/audio encoder input and output -> language model input and output -> logprobs
# of the first token. Two processes with identical settings, then compare: the first
# stage that differs is where the nondeterminism comes from.
#   probe        = our run settings (prefix caching off, engine in the same process)
#   probe-eager  = same, enforce_eager=True (no torch.compile, no CUDA graphs)
# About 4 min per line. Output: custom/probe_output/<name>.json
#
# One line per run, executed one after the other. Edit CUDA_VISIBLE_DEVICES on each
# line; comment out (#) the lines you do not want. Keep each pair (-a, -b) on the same GPU.
# Usage, from tmux:   bash custom/run_probe.sh

cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate av_env

CUDA_VISIBLE_DEVICES=0 python custom/tools/determinism_probe.py --name probe-a
CUDA_VISIBLE_DEVICES=0 python custom/tools/determinism_probe.py --name probe-b
CUDA_VISIBLE_DEVICES=0 python custom/tools/determinism_probe.py --name probe-eager-a --eager
CUDA_VISIBLE_DEVICES=0 python custom/tools/determinism_probe.py --name probe-eager-b --eager

python custom/tools/determinism_probe.py --compare probe-a probe-b
python custom/tools/determinism_probe.py --compare probe-eager-a probe-eager-b
