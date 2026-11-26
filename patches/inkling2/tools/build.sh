#!/bin/bash
set -euo pipefail
cd ~/inkling/engines/aeon-cleanroom
source env.sh
export UV_CACHE_DIR=~/inkling/engines/aeon-cleanroom/.uv-cache
UV=~/.local/bin/uv
$UV pip install "cmake<4" ninja packaging "setuptools>=78.1.1,<80" "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel jinja2 pybind11 Cython
cd vllm
$UV pip install -r requirements/common.txt "transformers>=5.16.1,<5.24.0" numba timm datasets
export MAX_JOBS=6 CMAKE_BUILD_PARALLEL_LEVEL=6
python -c "import torch;print(torch.__version__, torch.version.hip)"
which hipcc; hipcc --version | head -2
pip --version || true
$UV pip install --no-build-isolation -v -e . 2>&1
cp -r $(rocm-sdk path --root)/share/amd_smi /tmp/aeon_amd_smi_src && (cd /tmp/aeon_amd_smi_src && $UV pip install .)
echo BUILD_DONE
