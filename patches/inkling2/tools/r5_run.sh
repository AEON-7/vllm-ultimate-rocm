#!/bin/bash
# usage: r5_run.sh GPU [prep|run] [extra args]   e.g. r5_run.sh 1 run
# Runs the aeon-r5 worktree code via PYTHONPATH (no venv install; the
# serving vllm/ tree and its servers are untouched). "run" needs that GPU free.
set -e
G=$1; shift
cd ~/inkling/engines/aeon-cleanroom && source env.sh
WT=$HOME/inkling/engines/aeon-cleanroom/wt-r5
# compiled artefacts the worktree lacks (copied, never modified, from the main checkout)
for f in vllm/vllm/*.so vllm/vllm/_version.py; do [ -e "$WT/${f#vllm/}" ] || cp -p "$f" "$WT/${f#vllm/}"; done
[ -e "$WT/vllm/third_party/triton_kernels" ] || { [ -e vllm/vllm/third_party/triton_kernels ] && cp -rp vllm/vllm/third_party/triton_kernels "$WT/vllm/third_party/"; } || true
export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 PYTHONPATH=$WT
unset AEON_ABLATE AEON_WEIGHT_OVERLAY AEON_VARIANT_DIR AEON_CAPTURE_DIR
mkdir -p runs/r5 && cd /tmp
timeout 7200 python ~/inkling/engines/aeon-cleanroom/aeon/r5_validate.py "$@" 2>&1 | tee ~/inkling/engines/aeon-cleanroom/runs/r5/validate_$1.log
