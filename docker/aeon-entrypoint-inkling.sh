#!/bin/bash
# AEON Inkling serve wrapper. Unset allocator conf so expandable_segments cannot be on.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF PYTORCH_ALLOC_CONF
exec vllm serve "$@"
