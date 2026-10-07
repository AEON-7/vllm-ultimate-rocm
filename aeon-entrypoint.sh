#!/bin/bash
# AEON vLLM Ultimate ROCm entrypoint. The official image's entrypoint is
# `vllm serve`; this wrapper keeps that and refuses a CUDA-only host.
set -euo pipefail

if [[ ! -e /dev/kfd ]]; then
  echo "aeon-vllm-ultimate-rocm: /dev/kfd is missing, so this is not a ROCm host." >&2
  echo "NVIDIA x86: ghcr.io/aeon-7/aeon-vllm-ultimate-rtx" >&2
  echo "DGX Spark:  ghcr.io/aeon-7/aeon-vllm-ultimate" >&2
  exit 1
fi

export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"
export AEON_MXFP6_DEQUANT_AT_LOAD="${AEON_MXFP6_DEQUANT_AT_LOAD:-1}"

echo "aeon-vllm-ultimate-rocm early-access (vLLM v0.31.0). Pending community testing." >&2
echo "AEON_MXFP6_DEQUANT_AT_LOAD=${AEON_MXFP6_DEQUANT_AT_LOAD} VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER}" >&2

exec vllm serve "$@"
