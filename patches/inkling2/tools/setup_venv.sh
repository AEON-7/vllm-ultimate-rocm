#!/bin/bash
# AEON clean-room venv: TheRock ROCm 10.0 SDK + torch 2.13 (pins from upstream vLLM docker/Dockerfile.rocm_base)
set -euo pipefail
cd ~/inkling/engines/aeon-cleanroom
export UV_CACHE_DIR=~/inkling/engines/aeon-cleanroom/.uv-cache
UV=~/.local/bin/uv
IDX=https://stable.repo.amd.com/rocm/whl-next/
NIDX=https://nightly.repo.amd.com/rocm/whl-next/
test -d venv || $UV venv --python /usr/bin/python3.14 venv
source venv/bin/activate
$UV pip install --index-url $IDX --extra-index-url $NIDX --index-strategy unsafe-best-match \
  "rocm[libraries,devel,device-gfx1201]==10.0.0" \
  "torch[device-gfx1201]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1201]==0.28.0+rocm10.0.0"
rocm-sdk init || true
python -c "import torch;print(torch.__version__, torch.version.hip)"
$UV pip list | grep -i -E "triton|torch|rocm"
echo SETUP_DONE
