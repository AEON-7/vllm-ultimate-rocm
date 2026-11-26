# Inkling-2-Mini on ROCm (gfx1201 / RDNA4): patch series

Apply on vLLM main `1fd30e4f6e8f` (Nov 23 2026; includes the merged NVIDIA
Inkling-2-Mini support, vllm-project/vllm#69028):

```
git clone https://github.com/vllm-project/vllm && cd vllm
git checkout 1fd30e4f6e8f
git am /path/to/patches/inkling2/*.patch
```

| # | Patch | Source | License |
|---|-------|--------|---------|
| 0001 | Add Inkling 2 Small support on ROCm (the `vllm/models/inkling2/amd/` package) | upstream vllm-project/vllm#67685 by Indira Vats (AMD), head `800f72e235ac`, cherry-picked verbatim | Apache-2.0 |
| 0002 | Loop ops for the AMD package (blend norms, fp32 AttnRes mixes) | `pair_rms_norm` / `blend_rms_norm` ported from upstream `nvidia/ops/loop.py` (#69028), PTX rsqrt replaced with `tl.rsqrt`; AttnRes no-norm mode is original AEON | Apache-2.0 |
| 0003 | NVFP4 and MXFP8 routed experts on RDNA4 | original AEON work (clean room, from `docs/inkling2-rocm/SPEC.md`); parameter names follow upstream `nvidia/moe.py` | Apache-2.0 |
| 0004 | Per-pass KV caches and conv states for looped layers | port of the upstream NVIDIA design (`nvidia/attention.py`, `nvidia/loop.py`, #69028) to the AMD package | Apache-2.0 |
| 0005 | Inkling-2-Mini (`inkling2_loop`) on RDNA4 and CDNA | loop model ported from upstream `nvidia/loop.py` (#69028); gfx12 wiring, arch gate, offloader hook are AEON | Apache-2.0 |
| 0006 | AEON research hooks: weight overlay, activation capture, runtime o_proj ablation | original AEON work | Apache-2.0 |
| 0007 | M2: decode MoE (fused router + per-expert GEMV) for T<=16 | original AEON work | Apache-2.0 |
| 0008 | M2: fused decode loop-controller blend; single HW queue on ROCm | original AEON work | Apache-2.0 |
| 0009 | M2: small-M bf16 GEMV for qkvr/o_proj/dynamic taps; multi-row decode MoE | original AEON work | Apache-2.0 |
| 0010 | M2: fused router + sort top-k, dot-path MoE, MXFP8 hybrid decode, fused combine/norm, split-KV merge in attention | original AEON work | Apache-2.0 |
| 0011 | M2: static short conv fused with the following AttnRes mix (T<=8); MoE T=1 retune | original AEON work | Apache-2.0 |
| 0012 | M2: K/V tap projections in the qkvr GEMV launch; one message mix per loop pass | original AEON work | Apache-2.0 |
| 0013 | AEON per-request multi-variant o_proj ablation (same change as PR #3's 0007, renumbered to apply on 0012) | original AEON work | Apache-2.0 |
| 0014 | M2: batch-invariant MoE mode, on automatically for ablation variant servers | original AEON work | Apache-2.0 |
| 0015 | M3 (opt-in): DSpark draft KV write via the cache op on ROCm | original AEON work | Apache-2.0 |
| 0016 | M3 (opt-in): DSpark draft dense layers on the AEON small-M GEMV | original AEON work | Apache-2.0 |
| 0017 | R5-1: per-request residual-stream variant edits (`resid_dirs` / `resid_scales`) | original AEON work | Apache-2.0 |

Clean room: no code, binaries or images from vllm-redshift were read or used.
Sources were upstream vLLM (Apache-2.0), the model's HF config and
transformers code, and AMD/Triton documentation.

## Build (native, no Docker)

`tools/setup_venv.sh` (TheRock ROCm 10.0 pip SDK + torch 2.13.0+rocm10.0.0,
pins from upstream `docker/Dockerfile.rocm_base`), `tools/env.sh`, then
`tools/build.sh` (`MAX_JOBS=6 uv pip install --no-build-isolation -e .`, plus
`amdsmi` from the SDK so vLLM detects the ROCm platform).
Never set `PYTORCH_HIP_ALLOC_CONF=expandable_segments`. `tools/env.sh` sets
`GPU_MAX_HW_QUEUES=1` (M2: one HW queue per device; graph dispatch stalls on
cross-queue barriers on gfx12 otherwise).

## M3 DSpark (opt-in, off by default)

Speculative decoding only runs with an explicit `--speculative-config`;
see `docs/inkling2-rocm/M3_RESULTS.md` for numbers and the recommended
config. `AEON_DSPARK_GEMV=0` puts the draft back on torch linear.

## M2 decode switches (all default on; set to 0 to fall back)

`AEON_GEMV`, `AEON_FUSE_TAPS` (+ `AEON_FUSE_TAPS_MAX_M`, default 4),
`AEON_BATCH_MESSAGES`, `AEON_FUSE_SCONV_MIX`, `AEON_FUSE_SPLIT_COMBINE`,
`AEON_MOE_HYBRID_FMTS`. `AEON_BATCH_INVARIANT` (default on only when
`AEON_VARIANT_DIR` is set): per-row expert GEMV so each request's numerics
are independent of batch composition. See `docs/inkling2-rocm/M2_RESULTS.md`.

## AEON hooks (patch 0006), all off by default

- `AEON_WEIGHT_OVERLAY=a.safetensors[,b...]`: tensors replace the checkpoint
  tensors with the same checkpoint name at load, e.g.
  `model.llm.layers.12.attn.wo_ud.weight` (bf16 [2304, 3072]). Stock weights
  are only read; no checkpoint copy or symlink tree is needed.
- `AEON_CAPTURE_DIR`, `AEON_CAPTURE_APPS` (default `0-47`),
  `AEON_CAPTURE_WHAT` (`resid,oproj_in,oproj_out`): per layer application
  (looped layers 6..17 counted per pass: app = 6 + 12 * pass + layer - 6)
  dump at the last prefill row of each request. Files carry positions and
  the last 8 prompt ids for matching. Warmup steps also write files; filter
  by `last_ids`. Keep prompts within one prefill chunk.
- `AEON_ABLATE=edit.safetensors` with `dirs` [48, 2304] and `scales` [48]:
  o_proj output y -> y - s_a r_a (r_a . y), the runtime form of
  W' = (I - s r r^T) W with a separate scale per loop pass.
- Full logprobs: `max_logprobs=1000`, `SamplingParams(prompt_logprobs=1000)`
  gives teacher-forced top-1000 scoring (see `tools/kl_eval.py`).

## Tools

- `tools/kl_eval.py`: KL(stock transformers ref || vLLM) and top-1 on the
  AEON stock reference set (full-vocab fp16 log-softmax memmap).
- `tools/gen_bench.py`: multi-turn chats (greedy, capped, loop check) and
  bs1 / bs8 decode tok/s.
- `tools/test_moe_dequant.py`, `tools/moe_live.py`: expert kernel checks.
- `tools/ref_dump.py`, `tools/vllm_dump.py`, `tools/cmp_dump.py`:
  per-layer, per-pass activation diff vs transformers.
- `tools/hooks_test.py`: functional test of the AEON hooks.
- `tools/prof_decode.py`: rough per-module decode time split.
- M2: `tools/graph_gap.py` (graph-replay ms/step), `tools/gap_analyze.py`
  (rocprofv3 kernel-trace breakdown), `tools/det_test.py` / `tools/det2.py`
  (row-divergence / race checks), `tools/test_gemv2.py`, `tools/test_gemv3x.py`,
  `tools/test_moe_*.py`, `tools/test_auto_mx.py`, `tools/test_cfg_t1.py`
  (kernel unit tests), `tools/gen_bench_m2.json`, `tools/kl_m2_b8.json`.

## R5-1 residual-stream variants (patch 0017, research tool)

Variant files and `POST /aeon/variants` take optional `resid_dirs` / `resid_scales` (same shapes as
`dirs` / `scales`). After layer application a, the attention-input mix of application a+1 gets the
per-row rank-K removal (o_proj-only, residual-only or both). Stock rows are bitwise untouched and mixed
batches are bitwise equal to running alone. The bakeable subset (apps 0-4, 41-47) folds offline into
stock-format weights with `tools/r5_bake.py`. See `docs/inkling2-rocm/R5_RESIDUAL.md`. Tools:
`tools/r5_bake.py`, `tools/test_r5_bake.py`, `tools/r5_validate.py` + `tools/r5_run.sh`,
`tools/r5_unit*.py`, `tools/r5_bake_analysis.py`; results in `tools/r5_results/`.
