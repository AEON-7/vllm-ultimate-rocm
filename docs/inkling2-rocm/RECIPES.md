# Recipes: AEON vLLM Ultimate ROCm (Inkling-2-Mini, gfx1201)

ASCII only. Serve recipes for the published from-source gfx1201 image.

The image is published on GHCR as `ghcr.io/aeon-7/aeon-vllm-ultimate-rocm`
with both of these tags (same image):

- `inkling2-r6`
- `inkling2-r6-gfx1201-2026-11-25`

Digest (both tags): sha256:f5679cd80380bacbfa8137490a4b729182d0d805bba589b132cd2e3a603afcec

Pull:

```
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate-rocm:inkling2-r6
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate-rocm:inkling2-r6-gfx1201-2026-11-25
```

Stack: Python 3.14, TheRock ROCm 10.0.0, torch 2.13.0+rocm10.0.0, gfx1201.
Entrypoint is `/opt/aeon/entrypoint.sh`, which unsets
`PYTORCH_CUDA_ALLOC_CONF`, `PYTORCH_HIP_ALLOC_CONF`, and `PYTORCH_ALLOC_CONF`
so `expandable_segments` cannot be turned on. Do not set those variables.
Keep at least 15% VRAM free on each GPU that is in use. Cap output tokens.
Do not put an iGPU in `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`
(on some hosts the iGPU is a later index than the discrete cards).
Mount the stock checkpoint and any overlay read-only.

On some hosts, `--group-add render` can fail with "no matching entries in
group file". Use numeric GIDs instead (example below uses video=44,
render=991). Confirm with `getent group video render` and
`stat -c %g /dev/kfd`.

Replace `$IMG` with one of the tags above. Replace `$STOCK` with the
Inkling-2-Mini checkpoint directory.

Common flags (TP=1, GPU 0):

```
docker run --rm --name aeon-inkling-serve \
  --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 991 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc=host \
  --shm-size 16g \
  -e HIP_VISIBLE_DEVICES=0 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e HSA_ENABLE_SDMA=0 \
  -e GPU_MAX_HW_QUEUES=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -v "$STOCK":/model:ro \
  -v "$PWD/.triton-cache":/root/.cache/triton \
  "$IMG" \
  /model \
  --served-model-name Inkling-2-Mini \
  --host 127.0.0.1 --port 8010 \
  --tensor-parallel-size 1 --language-model-only \
  --max-model-len 16384 --max-num-seqs 16 \
  --gpu-memory-utilization 0.80 \
  --max-logprobs 1000 --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0}'
```

## 1) Stock, TP=1, GPU 0, port 8010

Use the common flags as written. No overlay.

Check `GET http://127.0.0.1:8010/v1/models` and `rocm-smi --showmeminfo vram`.
If free VRAM on GPU 0 is under 15%, stop the container. Do not raise
`--gpu-memory-utilization`.

A full stock-format baked checkpoint directory can be mounted as `/model`
the same way, with no overlay.

## 2) Weight overlay, TP=1, GPU 0, port 8010

Same as (1), plus `AEON_WEIGHT_OVERLAY` and a read-only overlay mount.
Example for the osv1 overlay:

```
  -e AEON_WEIGHT_OVERLAY=/overlay/overlay_osv1_l0-23_s1_c1__r414547_s05.safetensors \
  -v /path/to/overlay_osv1_l0-23_s1_c1__r414547_s05.safetensors:/overlay/overlay_osv1_l0-23_s1_c1__r414547_s05.safetensors:ro \
```

The log should contain `AEON overlay: replaced 55 tensors` before the API
is ready. On a TP=1 gfx1201 run at util 0.80, this serve used 73.3% of
GPU 0 VRAM (26.7% free) after graph capture.

## 3) Stock, TP=2, GPUs 0 and 1, port 8010

Both discrete GPUs must stay at least 15% free. Leave any iGPU out.

Change only:

```
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  ...
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.80 \
```

If either GPU drops under 15% free, stop the container. Do not raise util.

## 4) Weight overlay, TP=2, GPUs 0 and 1, port 8010

(3) plus the overlay env and `:ro` mount from (2). Confirm the overlay
replacement line before scoring.

## 5) DSpark (opt-in)

Speculative decoding is off unless you pass `--speculative-config`. The
drafter is the checkpoint's bundled `mtp.safetensors` (3-layer bf16,
block size 8), so `num_speculative_tokens` must be 7. Drop
`--language-model-only` from the common flags so that file can load.
`AEON_DSPARK_GEMV` defaults to 1 on this path; set it to 0 to restore
the torch draft GEMM.

Recommended for code/math-heavy single-user serving:

```
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "num_speculative_tokens_per_batch_size": [[1, 2, 7], [3, 256, 0]]}'
```

This drafts for 1-2 concurrent requests and turns speculation off at 3
or more. Leave it off for prose-heavy use, for multi-variant ablation
servers (those must be eager), and for anything that needs exact
plain-greedy output. Details: `docs/inkling2-rocm/M3_RESULTS.md`.

## 6) Multi-variant ablation (opt-in, eager)

Off unless `AEON_VARIANT_DIR` is set. Add:

```
  -e AEON_VARIANT_DIR=/variants \
  -v /path/to/variant_dir:/variants:ro \
```

and `--enforce-eager` (required; CUDA graphs are incompatible with this
path). `AEON_BATCH_INVARIANT` defaults to 1 when the variant dir is set.
The worker rescans the directory (see `AEON_VARIANT_RESCAN_S`, default
1s). Select a named variant per request with extra_args `aeon_variant`.
Do not combine this path with DSpark. Variant file format and residual
edits: `docs/inkling2-rocm/R5_RESIDUAL.md`.

## Validation

On gfx1201, stock capscreen through this image was a bitwise item-level
match to the from-source venv build (200/200 answers and token counts,
seed 0). Loading an overlay versus serving the equivalent baked
checkpoint produced bitwise-identical weights (743/743 parameters and
buffers). Edited-weight runs at temperature 1.0 showed run-to-run
variance even with a fixed seed, so those scores are not bit-identical
across repeats.

## Notes

- Image ENV already sets `PYTORCH_ROCM_ARCH=gfx1201`, `VLLM_ROCM_USE_AITER=0`,
  `HSA_ENABLE_SDMA=0`, `GPU_MAX_HW_QUEUES=1`, `VLLM_NO_USAGE_STATS=1`.
- AITER is installed as a wheel without JIT prebuild; leave it off at runtime.
- CPU tools in the image: `/opt/aeon/tools/r5_bake.py`, `r6_validate.py`,
  `test_r5_bake.py`, `test_r6_bake.py`. `/opt/aeon/MANIFEST` lists commits
  and stack pins. `test_r5_bake.py` parses argv at import, so run it as a
  script, not via an unadorned `pytest` collection of that file.
- Build notes: `docs/inkling2-rocm/DOCKER.md`.
