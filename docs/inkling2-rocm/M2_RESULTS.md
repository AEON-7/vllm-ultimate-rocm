# Inkling-2-Mini on ROCm: M2 results (speed)

Hardware: 1x AMD Radeon AI PRO R9700 (gfx1201, 32 GB), TP1, ROCm 10.0 pip SDK,
torch 2.13.0+rocm10.0.0, vLLM main `1fd30e4f6e8f` plus patches 0001-0012.
Graph mode (full HIP graphs), async scheduling, `GPU_MAX_HW_QUEUES=1`.

## Gates

| Gate | Target | M2 |
|------|--------|----|
| Single-stream decode, 256 new tokens | >= 79 tok/s | 79.2 tok/s (runs 79.2 / 79.2 / 79.2) |
| 8 concurrent, 256 new tokens each | >= 255 tok/s | 373.1 tok/s |
| Mean KL vs transformers (stock) | <= 0.0035 | 0.00219 |
| Top-1 agreement | >= 97% | 98.12% |
| Sample chats | no loops | 10 chats, 0 loops |

M1 (eager) was 18.8 / 140.7 tok/s. KL was run with `--max-batched 8` so the
T <= 8 decode kernels are the ones being scored (60 prompts, 1645 positions,
top-1000 logprobs against a full-vocab fp16 reference). With
`--max-batched 16` only the T = 16 prefill path runs.

## bs1 per-token profile (rocprofv3 kernel trace, graph replay, 59 steps)

Wall 13.39 ms/step, GPU busy 11.30 ms/step, 571 kernels/step.


| Kernel | ms/step | per step |
|--------|---------|----------|
| `_gemv_kernel` (qkvr + K/V taps, o_proj) | 4.39 | 96 |
| `_expert_gemv_kernel` (routed + shared experts, combine fused) | 3.39 | 96 |
| lm_head (hipBLASLt) | 1.46 | 1 |
| `_sconv_mix_kernel` (short conv + AttnRes mix) | 0.61 | 96 |
| `_rel_attention_kernel` (split-KV merge fused) | 0.45 | 48 |
| `_up_blend_kernel` | 0.32 | 48 |
| `_router_logits_kernel` (router + sort top-k) | 0.22 | 48 |
| `_down_kernel` | 0.18 | 48 |
| `_attn_prologue_kernel` | 0.17 | 48 |
| everything else | ~0.1 | ~40 |

Remaining gap to wall is launch gaps between graph nodes (~3.3 us each on
gfx1201 with one HW queue). Dense GEMV runs near 580 GB/s, MoE near
490 GB/s, lm_head near 640 GB/s.

## What changed (patches 0007-0012, original AEON work, Apache-2.0)

- 0007 decode MoE for T <= 16: fused router and per-expert GEMV.
- 0008 fused loop-controller blend; one HW queue on ROCm.
- 0009 small-M bf16 GEMV for qkvr / o_proj / dynamic taps; multi-row MoE.
- 0010 router + sort top-k in one kernel; tensor-core (`tl.dot`) path for
  experts with 3+ rows; MXFP8 hybrid decode; combine and blend norm fused
  into the last-arriving program; split-KV merge inside the attention kernel;
  `tl.debug_barrier()` before every last-arriver atomic (without it, rows of
  identical requests could diverge).
- 0011 static short conv fused with the following AttnRes mix (T <= 8);
  NVFP4 T = 1 tile retune.
- 0012 K/V dynamic tap projections in the qkvr GEMV launch (M <= 4); one
  message mix per loop pass, read from the stack without a bank copy.

Every fusion has an `AEON_*` env switch (default on) and was checked
bit-exact or with the KL harness. The research hook point is unchanged:
`HOOKS.oproj` right after o_proj in `Inkling2Attention.forward`.

## Notes

- Results are deterministic within one process. Across processes and between
  GPUs some near-tie tokens can differ, so exactness checks (M3 spec decode vs
  greedy) are done in-process.
- At M = 8 the fused taps launch measured slower in the identical-prompt bs8
  benchmark (306 vs 372 tok/s), so it is limited to M <= 4
  (`AEON_FUSE_TAPS_MAX_M`).

## Next (M3)

DSpark drafter from the bundled `mtp.safetensors` in the loop model, gates
>= 86 tok/s single, >= 316 tok/s at 8, output identical to non-spec greedy.
