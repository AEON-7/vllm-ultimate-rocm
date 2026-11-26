# Inkling-2-Mini on ROCm gfx1201: provenance and licensing

Written 2026-11-23 by the AEON stage-1 (spec) agent. Purpose: for every component in SPEC.md, record where
the idea or the code comes from, under what license, and whether we may build on the code directly or must
reimplement the idea from scratch.

## Legend

- UPSTREAM-CODE: permissively licensed code we may copy, modify and ship, keeping its license header and
  NOTICE entries.
- PUBLIC-IDEA: a published technique (paper, tutorial, vendor doc) whose ideas anyone may implement; we
  write our own code and cite it.
- REDSHIFT-IDEA: appears to be original to vllm-redshift. That repo has NO license (GitHub `license: null`,
  no LICENSE file, checked 2026-11-23), so no code may be copied. We reimplement the idea from SPEC.md alone
  and credit it in ACKNOWLEDGEMENTS. Ideas are not copyrightable; expression is.
- OURS: decisions specific to the AEON build.

## 1. Sources

| ID | Source | License | Link | Notes |
|---|---|---|---|---|
| S1 | vLLM main (vllm-project/vllm) | Apache-2.0 | https://github.com/vllm-project/vllm | Base of the build. Pin a commit after 2026-11-21 16:19 UTC. Box clone used for this spec: `1fd30e4f6e8f` (2026-11-23) |
| S2 | vLLM PR #69028 "[Model] Add Inkling 2 Mini" (Woosuk Kwon), merged 2026-11-21 16:19 UTC | Apache-2.0 | https://github.com/vllm-project/vllm/pull/69028 | Loop model (`nvidia/loop.py`), configs (`Inkling2ModelConfig.loop_config`, `layer_visit_ids`), DSpark config for loops, Mini widths. NVIDIA kernels only. Builds on Inkling 2 Small support #67325, #67880, #68761 |
| S3 | vLLM PR #67685 "[ROCm][Model] Add Inkling 2 Small support on ROCm" (indivats), open draft, head `800f72e235ac` (updated 2026-11-23) | Apache-2.0 (submitted to an Apache-2.0 project, SPDX headers on every file) | https://github.com/vllm-project/vllm/pull/67685 | `vllm/models/inkling2/amd/`: Triton relative-bias attention with split-KV decode, AttnRes mix, routing, MXFP8 grouped GEMM via `tl.dot_scaled` (gfx950), short conv, DSpark draft, vision, quant config. Targets MI355X with the all-MXFP8 Small checkpoint. Fetch from vllm-project only, never from redshift's copy |
| S4 | vLLM Inkling v1 AMD backend (`vllm/models/inkling/amd/`, on main) | Apache-2.0 | https://github.com/vllm-project/vllm/tree/main/vllm/models/inkling/amd | Relative-attention decode, sconv + SWA attention for the first Inkling family on AMD; useful patterns for ROCm attention metadata |
| S5 | vLLM generic spec decode: DSpark speculator, rejection sampler, `draft_sample_method`, `enable_adaptive_verification`, draft `kv_cache_dtype` (`vllm/v1/worker/gpu/spec_decode/`, `vllm/config/speculative.py`) | Apache-2.0 | https://github.com/vllm-project/vllm/tree/main/vllm/v1/worker/gpu/spec_decode | Already upstream; we only add ROCm model pieces and two policy changes |
| S6 | vLLM Triton fused_moe (`vllm/model_executor/layers/fused_moe/`, `moe_align_block_size`) | Apache-2.0 | https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe | Expert-sorted row layout, block alignment, Triton grouped GEMM structure |
| S7 | vLLM Triton attention backends (`vllm/v1/attention/backends/triton_attn.py`, `rocm_attn.py`, unified Triton attention, split-KV "paged attention v2" style) | Apache-2.0 | https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends | Paged-KV Triton attention on ROCm incl. gfx12; ViT varlen attention for vision |
| S8 | vLLM KV cache managers (`vllm/v1/core/single_type_kv_cache_manager.py`: `SlidingWindowManager`, `DFlashSlidingWindowManager`) | Apache-2.0 | https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py | Base classes for our tail-only conv-state manager |
| S9 | Hugging Face transformers `inkling2` and `inkling2_loop` modeling | Apache-2.0 | https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2_loop | Numerics oracle (SPEC R1). Local copies `modeling_inkling2.py`, `modeling_loop.py` |
| S10 | Thinking Machines Inkling-2-Mini model card, config, `hf_quant_config.json`, `mtp.safetensors` | Apache-2.0 (HF license tag `apache-2.0`) | https://huggingface.co/thinkingmachines/Inkling-2-Mini | Architecture facts, packed formats, DSpark draft. Our weights are commit `b85c3cfa` (same tensors as `ed40e7d`) |
| S11 | Triton language and tutorials (fused attention, grouped GEMM, matmul, layer norm) | MIT | https://github.com/triton-lang/triton , https://triton-lang.org/main/getting-started/tutorials/ | Tiling, online softmax, persistent kernels |
| S12 | ROCm Triton fork and AMD Triton docs (gfx12 / RDNA4 support, `waves_per_eu`, WMMA lowering) | MIT | https://github.com/ROCm/triton , https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-triton-kernel.html | Tuning knobs and backend behavior on gfx12 |
| S13 | AMD AITER | MIT | https://github.com/ROCm/aiter | Reference CDNA kernels (MoE, attention, FP4/FP8 GEMM). Mostly CDNA-only; consult for ideas, no gfx12 paths needed |
| S14 | AMD composable_kernel | MIT (LICENSE text; GitHub shows NOASSERTION because of the multi-holder header) | https://github.com/ROCm/composable_kernel | WMMA GEMM and FP8 conversion references for RDNA; ideas only, we do not link CK |
| S15 | AMD RDNA4 ISA guide and ROCm docs (WMMA fp8/bf8 on gfx12, wave32, LDS) | AMD documentation, public | https://www.amd.com/en/support/tech-docs (RDNA4 ISA), https://rocm.docs.amd.com | Hardware facts in SPEC section 7 |
| S16 | AMD pip ROCm ("TheRock") wheels: `rocm==10.0.0`, `torch==2.13.0+rocm10.0.0`, `triton==3.8.0+rocm10.0.0` | TheRock repo MIT; ROCm components MIT / NCSA / BSD per component; PyTorch BSD-3; Triton MIT | https://github.com/ROCm/TheRock | Binary dependencies, installed not vendored. Keep their license files in the image |
| S17 | FlashAttention / FlashAttention-2 papers and code (online softmax tiling, split-KV "flash-decoding") | Papers public; code BSD-3-Clause | https://arxiv.org/abs/2205.14135 , https://arxiv.org/abs/2307.08691 , https://github.com/Dao-AILab/flash-attention , https://crfm.stanford.edu/2023/10/12/flashdecoding.html | Attention tiling and split-KV combine |
| S18 | Attention sinks: Xiao et al. 2023 "Efficient Streaming Language Models with Attention Sinks"; learned per-head sink logits as in OpenAI gpt-oss and FlashAttention/vLLM `sinks` support | Paper public; vLLM code Apache-2.0 | https://arxiv.org/abs/2309.17453 , https://github.com/vllm-project/vllm | The "extra softmax logit with zero value" formulation |
| S19 | Online softmax: Milakov and Gimelshein 2018 | Paper | https://arxiv.org/abs/1805.02867 | |
| S20 | Speculative sampling: Leviathan et al. 2023; Chen et al. 2023 | Papers | https://arxiv.org/abs/2211.17192 , https://arxiv.org/abs/2302.01318 | Rejection sampling for sampled drafts |
| S21 | Block-diffusion style drafting (DFlash and successors), DSpark checkpoint format | Papers / model card | https://huggingface.co/thinkingmachines/Inkling-2-Mini , upstream vLLM `dspark` docs | Draft block of anchor + masks |
| S22 | OCP Microscaling (MX) spec v1.0 (MXFP8, e8m0 scales) and NVIDIA NVFP4 format (e2m1 + e4m3 per-16 scale + fp32 global) | OCP spec public; NVIDIA docs public | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf , https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | Dequant math in SPEC section 3 |
| S23 | vllm-redshift (Abhinand B) | NONE (no license) | https://github.com/abhinand5/vllm-redshift | Learning source only. Stage-1 agent read it; implementer must not |

## 2. Per-component provenance

| SPEC section | Component | Idea from | Code we may build on | Classification |
|---|---|---|---|---|
| 1 | Loop model math (passes, per-pass caches, message controller, layer visits) | S2, S9, S10 | S2 `nvidia/loop.py` (model logic is mostly device-neutral PyTorch; port it to `amd/`), S9 | UPSTREAM-CODE |
| 1 | Loop on ROCm: `amd/loop.py`, per-pass caches in the AMD attention, fp32 AttnRes for message reads | S2 (semantics) | S2, S3 | REDSHIFT-IDEA for the act of porting it to AMD (redshift is the only public ROCm Mini port); the design itself is fixed by S2. Reimplement from S2 and credit |
| 1 | Fused pair-norm / blend-norm kernels for the controller | S2 (`ops/loop.py` names `pair_rms_norm`, `blend_rms_norm`) | S2 kernel logic is Triton in places; reuse where it is plain Triton, rewrite CUDA-specific paths | UPSTREAM-CODE / OURS |
| 1 | Override-shard-safe loader rule | OURS (from our overlay trial) | - | OURS |
| 2 | Relative-bias attention, sinks, gate, L2 clip, log scaling (math) | S9, S10, S18 | S3 `amd/ops/attention.py` (Triton, gfx950-tuned) | UPSTREAM-CODE |
| 2 | Split-KV decode with combine | S17, S3 | S3, S7 | UPSTREAM-CODE |
| 2 | Dedicated tiled prefill kernel for relative attention on gfx12 | S11 tutorial, S17 | S3 kernel as starting point | PUBLIC-IDEA / OURS (redshift also has one; do not consult it) |
| 2 | FP8 KV read into fp16 operands; GQA-group packing without padding in split-KV on gfx12 | S17 (GQA packing is common, e.g. FlashInfer/FA decode), S15 | - | REDSHIFT-IDEA (the specific gfx12 combination and its reported gains); generic idea is public. Reimplement and credit |
| 3 | Router (sigmoid, bias for selection, renorm, route_scale, low-id ties) | S9, S2 | S3 `amd/ops/moe.py` routing | UPSTREAM-CODE |
| 3 | MXFP8 expert GEMM | S22, S3 | S3 `amd/ops/triton_bmm.py` (uses `tl.dot_scaled`, gfx950 only) | UPSTREAM-CODE for structure; gfx12 path is new |
| 3 | MXFP8 on gfx12 via fp8 WMMA or in-register dequant | S15, S22 | - | REDSHIFT-IDEA (first public gfx12 port); also obvious from S15. Reimplement |
| 3 | NVFP4 expert GEMM and GEMV on a GPU with no FP4 math, weights kept as shipped, decoded in registers | S22; general weight-only-quant GEMM idea (Marlin, Machete, GPTQ Triton kernels, S6 int4 paths) | S6 (Triton fused_moe int4 w4a16 path as a structural model) | PUBLIC-IDEA, with REDSHIFT-IDEA credit for applying it to Mini on gfx1201. Reimplement |
| 3 | BF16-activation multi-row GEMV for 1-8 token decode/verify with SwiGLU fused | general (S6 has small-M paths; GEMV fusion is standard) | S6 | PUBLIC-IDEA / REDSHIFT-IDEA credit |
| 3 | Mixed NVFP4/MXFP8 per-layer detection from `hf_quant_config.json` | S2 `quant_config.py`, S10 | S2 | UPSTREAM-CODE |
| 4 | 200058 vs 201024 vocab, padded head, logits slice, fp32 head output, softcap | S2, S9, S10 | S2 (`ParallelLMHead(padded, org_vocab_size=...)`) | UPSTREAM-CODE |
| 4 | Optional FP8 per-row LM head | general (per-channel FP8 weight-only) | S1 FP8 linear utils | PUBLIC-IDEA / REDSHIFT-IDEA credit (opt-in flag there) |
| 5 | DSpark draft model math, Markov head, feature capture at layer visits | S2, S10, S21 | S2 `nvidia/dspark.py`, `speculator.py`; S3 `amd/dspark.py` | UPSTREAM-CODE |
| 5 | Draft `kv_cache_dtype=fp8`, `draft_sample_method=probabilistic`, `enable_adaptive_verification` | S5 | S5 | UPSTREAM-CODE (config knobs exist upstream; we make the ROCm draft honor them) |
| 5 | Single-launch Markov recurrence; FP8 (e4m3, per-row) copies of draft layers and draft head | general fusion / weight-only FP8 | - | REDSHIFT-IDEA. Reimplement and credit |
| 5 | Adaptive verify: profiling with random routed ids; online cost refit with GPU-event timing and a context-length slope | S5 for the policy | S5 | REDSHIFT-IDEA for both changes. Reimplement from SPEC 5.2 item 5 and credit |
| 6 | Short conv math, conv-state cache layout (`InklingConvState`) | S9, S2, S4 | S2 `nvidia/sconv.py` model side, S3 `amd/ops/sconv.py` (Triton) | UPSTREAM-CODE |
| 6 | Fused conv + AttnRes mix + norm kernel | S2 (`csrc/inkling2/conv_mix.cu`, CUDA) | S2 as a spec; S3 Triton sconv | UPSTREAM-CODE semantics; Triton version is OURS / REDSHIFT-IDEA credit |
| 6 | Tail-only conv-state cache manager, with a tail long enough to cover prefix-hit resend boundaries | S8 (`DFlashSlidingWindowManager` already backs only tails for DFlash) | S8 | REDSHIFT-IDEA for applying it to conv state and for the resend-boundary fix. Reimplement and credit |
| 7 | Per-arch config tables, offline sweep, no runtime autotune in serving | S11, S12 | - | OURS |
| 7 | `GPU_MAX_HW_QUEUES=2` default on RDNA4 | HIP runtime env var (AMD docs); the RDNA4 decode finding | - | REDSHIFT-IDEA (the measurement and the default). One-line platform change; reimplement and credit |
| 7 | `ROCPROFILER_REGISTER_ENABLED=0` to stop idle CPU spin | ROCm docs | - | REDSHIFT-IDEA credit for noticing it; env var is AMD's |
| 7 | gfx12 bring-up guards (keep SM12x-only config checks CUDA-only so ROCm gfx12 is not mistaken for SM120) | S1 | S1 | OURS (trivial; redshift has the same fix, credit) |
| 8 | pip ROCm 10.0 + torch 2.13 + Triton 3.8 stack on python:3.12-slim, from-source vLLM for gfx1201 | S16, S1 build docs | S1 `docker/Dockerfile.rocm*`, `requirements/rocm*.txt` | UPSTREAM-CODE + OURS. REDSHIFT-IDEA credit for proving this stack on gfx1201 |
| 9 | Vision via vLLM Triton varlen attention on ROCm | S7 | S7, S3 `amd/vision.py` | UPSTREAM-CODE |

## 3. What needs original work vs upstream reuse

Upstream reuse (copy and adapt, keep Apache headers):
- Everything model-semantic: configs, loop logic, DSpark model, routing, quant config, logits, conv state
  layout (S2).
- The AMD package skeleton and its Triton kernels for attention, AttnRes, routing, sconv, DSpark ops,
  vision, MXFP8 GEMM (S3), as a starting point that we retune for gfx12.
- Spec-decode policy, rejection sampler, draft KV dtype, adaptive verification framework (S5).
- KV cache manager base classes (S8), Triton attention backends for vision (S7), fused_moe layout (S6).

Original work for us (no usable permissive code exists for gfx1201):
- NVFP4 expert GEMM and GEMV on gfx12 (no FP4 or block-scaled math in hardware).
- MXFP8 expert path on gfx12 (S3's `tl.dot_scaled` path does not lower on gfx12).
- Porting the loop core and per-pass caches into the AMD backend (S3 predates Mini).
- gfx12 tuning tables, the dedicated prefill kernel, split-KV FP8 decode with GQA packing.
- Decode GEMV and launch fusion for HIP-graph decode at batch 1.
- Fused conv + AttnRes in Triton.
- Tail-only conv-state cache manager.
- Adaptive verify profiling fix and online refit.
- DSpark draft: FP8 draft KV on ROCm, single-launch Markov step, FP8 draft weights.

## 4. License concerns

1. vllm-redshift has no license. All rights reserved by default. We copy nothing: no code, no comments, no
   kernel bodies, no tuning tables, no test files, no patch series files, no scripts, no docs text, no
   images. Its patch `upstream/0001` and `upstream/0002` files are copies of Apache-2.0 vLLM PRs; get those
   from vllm-project directly so there is no question of where our copy came from.
2. Upstream PR #67685 is unmerged. Its code is Apache-2.0 under vLLM's contribution terms and carries SPDX
   headers, so we can use it. Risk: it can still change or be withdrawn. Pin the PR head sha we used and
   record it in our patch header. Drop our carried copy when it merges.
3. The NVIDIA-only CUDA sources in S2 (`csrc/inkling2/*.cu`) are Apache-2.0 but do not build for ROCm; we
   only need to exclude them from the HIP build.
4. Binary dependencies (S16) are installed from AMD and PyTorch indexes, not vendored. Their licenses (MIT,
   BSD, NCSA, Apache) allow redistribution in an image with notices; include license files and list them in
   NOTICE. AMD's device wheels may include proprietary-but-redistributable components; check the wheel
   METADATA license fields at build time and record them.
5. Model weights (S10) are Apache-2.0. We do not ship weights in the image.
6. Other repos already excluded by our NOTICE (GGZ14/vllm-mxfp4, vllm-radiance, libr4d) stay excluded.
7. The third-party "Strata" engine mentioned in redshift's README is closed-source; nothing from it is used.

## ACKNOWLEDGEMENTS

We thank Abhinand B for vllm-redshift (https://github.com/abhinand5/vllm-redshift), the first public port of
Inkling-2-Mini to a single Radeon AI PRO R9700 (gfx1201). Studying it and measuring it on our hardware showed
us what a consumer RDNA4 card needs for this model: decoding NVFP4 and MXFP8 experts in registers instead of
requantizing, Triton relative-bias attention with sinks and an FP8 cache, a tail-only conv-state cache that
lets two full-length contexts fit in 32 GB, DSpark with FP8 draft state and cost-aware adaptive verification,
and RDNA4 runtime settings such as limiting HIP hardware queues. Our implementation is written independently
from our own specification and from Apache-2.0 upstream vLLM code; it contains none of vllm-redshift's code.
Any mistakes are ours.
