# Inkling-2-Mini on ROCm: Milestone 1 results (correct, eager)

Status: **M1 PASS** on 1x AMD Radeon AI PRO R9700 (gfx1201, 32 GB), TP1.

## Build

- vLLM main `1fd30e4f6e8f` + upstream #67685 (head `800f72e235ac`, verbatim)
  + 5 AEON patches (`patches/inkling2/`). Native editable build, no Docker.
- TheRock ROCm 10.0.0 pip SDK, torch 2.13.0+rocm10.0.0, triton
  3.8.0+rocm10.0.0, Python 3.14, `PYTORCH_ROCM_ARCH=gfx1201`, `MAX_JOBS=6`.
- Model: thinkingmachines/Inkling-2-Mini stock checkpoint (read-only),
  NVFP4 experts in layers 0-19, MXFP8 in 20-23, loop core 6..17 x 3 passes.

Reused verbatim from upstream (Apache-2.0): the whole AMD Inkling package
from #67685 (attention prologue and relative attention kernels, AttnRes,
short conv, router/MoE for gfx950, DSpark, vision), and from #69028 the loop
model logic, controller blend kernels and per-pass cache design (ported to
the AMD package). Written new (AEON): NVFP4/MXFP8 expert decode + grouped
GEMM for RDNA4, fp32 no-norm AttnRes mode, per-pass wiring on AMD, gfx12
arch enablement, offloader hook, research hooks, all test harnesses.

## Correctness gate (vs transformers reference, stock model)

Reference: AEON stock KL set, transformers bf16 (eager attention, eager
experts), full-vocab log-softmax over the first <=32 answer tokens per
prompt, chat template with `reasoning_effort="none"`. vLLM scores the same
token sequences teacher-forced with `prompt_logprobs=1000`; KL(ref || vLLM)
over vLLM's top-1000 plus one tail bucket (ref mass outside the top-1000:
2.2e-5 mean).

| metric | gate | result |
|---|---|---|
| prompts | >= 32 | 185 (the 15 reference rows with 0 answer tokens are skipped) |
| positions | | 5029 |
| mean KL (per prompt, then averaged) | <= 0.0035 | **0.00158** |
| mean KL per position | | 0.00176 |
| p95 KL per position | | 0.0081 |
| top-1 agreement | >= 97% | **98.27%** |

First 32 prompts: KL 0.00192, top-1 98.54%. An earlier 200-row run gave
0.00173 / 98.59% (batch-order noise only).

Layer-level check (`tools/cmp_dump.py`, one 45-token prompt, CPU bf16
transformers vs GPU vLLM): layer 0 attention rel 0.006, MoE rel 0.011;
drift grows slowly through the 3 loop passes as expected from bf16
reduction order. MoE kernel vs torch dequant reference on real weights: rel
2e-4 (NVFP4) / 3e-4 (MXFP8).

Bug found and fixed during M1: the checkpoint interleaves gate/up rows in
`w13` (row 2i = gate i, 2i+1 = up i; the transformers converter applies
`Interleave(dim=1)`). The gfx12 MoE de-interleaves weight and block scales
once at load. Before the fix: KL 5.55, top-1 9%.

## Chat sanity (greedy, max 400 tokens, `reasoning_effort=none`)

5 two-turn chats, no loops detected (zlib ratio check), 9/10 replies end on
stop; the haiku reply hits the 400 cap while counting syllables (not a
loop). Examples (final answer part):

- "Say hello ... what 2+2 equals" -> "Hello! 2+2 equals 4."
- follow-up "multiply that result by 7" -> "Multiplying 4 by 7 gives 28
  because 4 groups of 7 total 28."
- "capital of Australia?" -> "The capital of Australia is Canberra."
- "hash map average lookup time and why" -> "Average lookup time is O(1): a
  good hash function spreads keys across buckets ... Worst case can degrade
  to O(n) if many keys collide ..."
- winter haiku -> "White snow falls softly / blanketing the quiet town /
  softly, deeply still"

Full samples: `patches/inkling2/tools/gen_bench_eager.json`.

## Speed (TP1, 256 output tokens, ignore_eos, median of 3)

| mode | bs1 tok/s | bs8 aggregate tok/s |
|---|---|---|
| eager | 18.8 | 140.7 |
| FULL_DECODE_ONLY graphs | 18.4 | 129.0 |

Decode time split at bs1 (synchronizing hooks, ratios only): MoE ~40%
(0.54 ms per visit x 48 visits), attention ~23%, short convs ~11%, rest
(AttnRes mixes, loop blends, logits, launches) ~26%.

## Abliteration features

- (A) Overlay loading: done (`AEON_WEIGHT_OVERLAY`), tested: zeroed
  `layers.12.attn.wo_ud` overlay gives exactly 0 o_proj output at
  applications 12/24/36. No checkpoint copy.
- (B) Logprobs: done, upstream vLLM `max_logprobs=1000`,
  `prompt_logprobs=1000` (teacher-forced), used by `tools/kl_eval.py`.
- (C) Activation capture: done (`AEON_CAPTURE_*`), resid / o_proj in / o_proj
  out at apps 0-47 with passes counted separately, last prefill row.
- (D) Runtime rank-1 edits: global per-server form done (`AEON_ABLATE`,
  per-application scale, so per-pass loop scales work). Per-request
  variants (many edits in one server) are design only, see below.

### D: per-request rank-1 ablation design

y' = y - s_a r_a (r_a . y) costs one [T, D] x [D] dot and one axpy per
application. Per-request needs a token -> variant map: pass a variant id in
`SamplingParams.extra_args`, have the model runner build an int32 [T]
`variant_of_token` (like LoRA's token mapping), and keep a bank
`dirs[V, 48, D]`, `scales[V, 48]` on GPU. The hook gathers r per token and
applies it in one small Triton kernel fused after wo_ud. Effort: about 1
day for the eager path, plus care for CUDA graphs (fixed-size mapping
buffer). Plain multi-LoRA on o_proj cannot express per-pass scales because
the loop shares weights across passes; the hook can.

## Next milestones

- M2 (>= 79 tok/s bs1, >= 255 bs8, TP1): skinny-M split-K expert GEMV for
  decode (ideal weight traffic is ~1.5 GB/token, ~2.4 ms at R9700 bandwidth
  vs ~26 ms now), fuse router + plan + both expert GEMMs, fuse
  AttnRes/blend/sconv chains, make the attention path graph-capturable and
  run full CUDA graphs for decode, trim per-pass KV (about 1 MB/token
  today). Estimate: 2-4 agent days with GPU access.
- M3 (DSpark >= 86 / 316): wire the #67685 DSpark drafter (mtp.safetensors)
  into the loop model's aux hidden states, verify acceptance, tune block
  size. Estimate: 1-2 days after M2.
