# Inkling-2-Mini on ROCm: M3 results (DSpark speculative decoding, opt-in draft)

Status: opt-in draft, parked. Speculative decoding is OFF by default and
only runs when you pass `--speculative-config`. It helps on predictable text
(math, structured code) for a single user and hurts on novel prose.

Hardware and base: 1x Radeon AI PRO R9700 (gfx1201), TP1, graph mode,
patches 0001-0016. DSpark drafter = the checkpoint's bundled
`mtp.safetensors` (3-layer bf16, block size 8, so `num_speculative_tokens`
must be 7). 256 new tokens, greedy, `ignore_eos`. Plain decode for
comparison: 79-80 tok/s bs1.

## What changed (original AEON work, Apache-2.0)

- 0015: the draft's KV-cache write calls `reshape_and_cache_flash` from
  `vllm._custom_ops` directly. On ROCm, `flash_attn.py` only imports it when
  flash-attn varlen is available (it is not), so the draft hit a NameError.
- 0016: the draft's dense layers use the AEON small-M GEMV for up to 8 rows
  (`AEON_DSPARK_GEMV=0` restores torch). hipBLASLt small-M tiles read the
  draft weights at about half bandwidth; this saves about 2.2 ms per spec
  step at bs1 (33.9 -> 31.7 ms/step on the math prompt).

## Results

AL = mean acceptance length (tokens emitted per verify step).

| Prompt | Plain bs1 | Spec bs1 (k=7) | AL |
|---|---|---|---|
| math (word problem, step by step) | 80.4 | 105.2 (103.5 before 0016) | 3.3-3.5 |
| code (Python module + tests) | 79.1 | 75.6-80.4 | 2.5-2.8 |
| story (novel prose) | 79.2 | 55.7-59.5 | 1.9 |

Code moves with AL: the spec and plain trajectories diverge (see below), so
each run drafts slightly different text.

8 concurrent requests (tok/s):

| Prompt | Plain | Spec at all batch sizes | Spec off at 3+ requests |
|---|---|---|---|
| story | 373.1 | 205.2 | 368.1 |
| code | 284.6 | 220.9 | 291.4 |
| math | 383.3 | 456.0 | (not measured) |

## Why bs1 is limited: the ~31 ms spec step (rocprofv3, code prompt, bs1)

GPU busy about 31 ms per verify + draft step (before 0016):

| Part | ms/step |
|---|---|
| Routed MoE at T=8 (NVFP4 multi-row kernel; up to 64 distinct experts, near bandwidth) | 12.6 |
| Other MoE kernels (MXFP8, dot path, grouped GEMM) | ~2.8 |
| qkvr GEMV (target) | 3.6 |
| hipBLASLt: draft dense layers + target o_proj at M=8 | ~5.5 (~3.3 after 0016) |
| Attention | 1.6 |
| lm_head (target and draft) | 1.5 |
| loop blend, sconv, router, prologue, misc | ~3 |

A plain decode step is about 12.6 ms. Checking 8 tokens can touch up to 64
experts per layer instead of 8, so a verify step costs about as much as a
bs8 decode step. Speculation only wins when AL is above about 2.5.

## Tried and rejected

- MoE decode kernels up to T=64 (`AEON_MOE_DECODE_MAX=64`): correct (KL
  0.00193, top-1 97.9% with 64-token chunks) but bs8 spec fell from 205 to
  152 tok/s; the grouped GEMM is faster at T=64.
- Hybrid MoE path for NVFP4 at T=8 (`AEON_MOE_HYBRID_FMTS=0,1`): 73.3 vs 80.4.
- Batch-invariant MoE (`AEON_BATCH_INVARIANT=1`) under spec: 71.0.
- Adaptive verification: upstream asserts it off for Inkling2.
- All M2 fusions off (eager A/B): AL 1.92 vs 1.89, so the fusions do not
  hurt the draft.

## Output is not token-identical to plain greedy

Same GPU, bs1: spec and plain greedy match for the first 54 (code) and 134
(math) of 256 tokens, then a near-tie token flips. The verify step runs 8
tokens per request through kernels whose reduction order depends on the
token count (GEMV tile by M, MoE multi-row paths, attention split plan,
fused short conv only up to 8 tokens, router tile, hipBLASLt lm_head tile),
so logits differ slightly from the 1-token decode. Upstream on NVIDIA is not
token-identical either. An exact mode (fixed per-row reduction everywhere,
own lm_head GEMV) was scoped at several days with a speed cost and was not
built.

## How to use (opt-in)

Recommended for code/math-heavy single-user serving:

```
--speculative-config '{"method": "dspark", "num_speculative_tokens": 7,
  "num_speculative_tokens_per_batch_size": [[1, 2, 7], [3, 256, 0]]}'
```

This drafts for 1-2 concurrent requests and turns speculation off at 3 or
more, so batched throughput stays at plain-decode level. Leave it off for
prose-heavy use, ablation variant servers (variants are eager only), and
anything that needs exact plain-greedy output.

Tools: `patches/inkling2/tools/spec_bench.py` (`--spec 7`, `--prompt
story|code|math`, `--spec-extra JSON`, `--bs-list`; saves token ids and spec
metrics). Raw results: `patches/inkling2/tools/m3_results/`.
