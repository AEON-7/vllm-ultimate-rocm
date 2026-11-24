# AEON clean-room vLLM (ROCm, gfx1201) environment. Source me.
source ~/inkling/engines/aeon-cleanroom/venv/bin/activate
export ROCM_PATH=$(rocm-sdk path --root)
export ROCM_HOME=$ROCM_PATH HIP_PATH=$ROCM_PATH
export PATH=$(rocm-sdk path --bin):$PATH
export PYTORCH_ROCM_ARCH=gfx1201
export VLLM_TARGET_DEVICE=rocm
export VLLM_NO_USAGE_STATS=1
export VLLM_ROCM_USE_AITER=0
export HSA_ENABLE_SDMA=0
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF PYTORCH_ALLOC_CONF
export TRITON_CACHE_DIR=~/inkling/engines/aeon-cleanroom/.triton-cache
export GPU_MAX_HW_QUEUES=1  # AEON M2: one HW queue per device; cross-queue barrier polling stalls graph dispatch on gfx12
