# Inkling-2-Mini from-source ROCm image (gfx1201)

This branch builds a from-source vLLM ROCm image for Inkling-2-Mini on
Radeon AI PRO R9700 (gfx1201). It is not the older Qwen early-access
image on main (`aeon-vllm-ultimate-rocm:2026-10-06-v0.31.0-x86-early-access`,
official vLLM v0.31.0 + ROCm 7.2.3 + two Python patches).

The image matches the cleanroom venv:

- Python 3.14
- TheRock ROCm 10.0.0
- torch 2.13.0+rocm10.0.0 with device-gfx1201
- torchvision 0.28.0+rocm10.0.0
- AITER off, GPU_MAX_HW_QUEUES=1, HSA_ENABLE_SDMA=0
- allocator conf unset (never set expandable_segments)

The full source tree used on center is `~/inkling/engines/aeon-docker`
at `0714f860c` plus the Docker integration commit. This repository stays
a patch-and-docs tree; it does not vendor the upstream vLLM sources.

## Build

1. Start from vLLM `1fd30e4f6` and apply `patches/inkling2/0001` through
   `0018` in order (or check out the cleanroom tip `0714f860c`).
2. Copy `docker/Dockerfile.rocm`, `docker/Dockerfile.rocm_base`, and
   `docker/docker-bake-rocm.hcl` from this branch into that tree's
   `docker/` directory if they are not already there.
3. Copy `docs/inkling2-rocm/MANIFEST` to `aeon/MANIFEST`, this repo's
   `docker/aeon-entrypoint-inkling.sh` to `aeon/entrypoint.sh`, the
   tools under `patches/inkling2/tools/` that the Dockerfile expects
   at `tools/inkling2/` and `aeon/scripts/`, and the patch files to
   `aeon/patches/`.
4. With Docker Engine installed (no amdgpu-dkms, firmware, or kernel
   changes; no reboot):

       docker buildx bake -f docker/docker-bake-rocm.hcl aeon

   Default tag: `ghcr.io/aeon-7/aeon-vllm-ultimate-rocm:inkling2-r6`

## Serve

Use `--network host`, `--device /dev/kfd`, `--device /dev/dri`,
`--group-add video`, `--group-add render`. Keep
`--gpu-memory-utilization` at or below 0.80 so each GPU holds at least
15% VRAM free. Do not set expandable_segments. Cap output tokens.

DSpark stays opt-in at serve time (`--speculative-config`). Residual
and readers-bake features are in the tree; baked checkpoints load as
stock-format weights via AEON_WEIGHT_OVERLAY or a full checkpoint dir.

## Manifest

See `docs/inkling2-rocm/MANIFEST` for the commit list, patch map, and
the env flags baked into the image. After a successful build, the same
file is at `/opt/aeon/MANIFEST` in the image.
