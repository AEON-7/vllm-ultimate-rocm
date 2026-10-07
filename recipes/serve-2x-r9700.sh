#!/usr/bin/env bash
# Early-access serve for AEON Ultimate MXFP4/MXFP6 on 2x R9700.
# See README.md. This does not load the NVFP4-MIXED checkpoint.
set -euo pipefail

IMAGE="${AEON_IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate-rocm:2026-10-06-v0.31.0-x86-early-access}"
MODEL="${AEON_DIR:-$HOME/models/aeon-mxfp4-rocm}"
DRAFT="${AEON_DFLASH:-$MODEL/dflash2}"
CACHE="${AEON_CACHE:-$HOME/.cache/aeon-vllm-rocm}"
NAME="${AEON_NAME:-aeon-rocm}"
PORT="${AEON_PORT:-8000}"
GPUS="${HIP_VISIBLE_DEVICES:-0,1}"

mkdir -p "$CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

exec docker run -d --name "$NAME" \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc=host \
  --shm-size 16g \
  -p "$PORT:8000" \
  -e HIP_VISIBLE_DEVICES="$GPUS" \
  -e ROCR_VISIBLE_DEVICES="$GPUS" \
  -e NCCL_PROTO=Simple \
  -e NCCL_P2P_DISABLE=1 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e AEON_MXFP6_DEQUANT_AT_LOAD="${AEON_MXFP6_DEQUANT_AT_LOAD:-1}" \
  -v "$MODEL":/model:ro \
  -v "$DRAFT":/draft:ro \
  -v "$CACHE":/root/.cache/vllm \
  "$IMAGE" \
  /model \
  --served-model-name aeon \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size "${AEON_TP:-2}" \
  --max-model-len "${AEON_MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${AEON_MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${AEON_MAX_BATCHED:-8192}" \
  --gpu-memory-utilization "${AEON_GPU_UTIL:-0.90}" \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":9,"num_speculative_tokens_per_batch_size":[[1,2,9],[3,8,7],[9,12,6],[13,16,4]],"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
