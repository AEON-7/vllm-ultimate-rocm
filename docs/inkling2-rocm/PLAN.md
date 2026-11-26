# Inkling-2-Mini on ROCm gfx1201: implementation and image plan

Written 2026-11-23 by the AEON stage-1 (spec) agent. Companion to SPEC.md and PROVENANCE.md.

## 1. Clean-room rules for the implementer (read first)

1. Never open, clone, fetch, pull or search `abhinand5/vllm-redshift` or anything derived from it: its code,
   patch files, issues, commit messages, README, CHANGELOG, user guide, CI, Docker images
   (`ghcr.io/abhinand5/vllm-redshift`), or the trial tree on center (`~/inkling/engines/redshift/`,
   including `repo/` and `build/`). Do not run its images. Do not ask anyone to paste from it.
2. Your inputs are: SPEC.md, PROVENANCE.md, this PLAN.md; upstream vLLM (vllm-project/vllm main and PRs
   #69028 and #67685, fetched from vllm-project); Hugging Face transformers `inkling2`/`inkling2_loop`; the
   model checkpoint, its config files and model card; public docs and papers listed in PROVENANCE.md; and
   AEON's own eval scripts and measurements (KL and greedy harnesses, benchmark JSONs). The trial report
   `/workspace/inkling/engines/redshift_REPORT.md` contains measured numbers only and may be read for targets.
3. If the spec is ambiguous, resolve it from upstream code or the transformers reference, or ask the spec
   author (stage 1) for a prose clarification. Never resolve it by looking at redshift.
4. Every new file carries `SPDX-License-Identifier: Apache-2.0` and, if derived from upstream vLLM, keeps
   the upstream copyright line plus "Modified by AEON-7". Files we write from scratch: "Copyright 2026 AEON-7".
5. Keep a dev log (`docs/inkling2-rocm/DEVLOG.md`) with date, what you built and which sources you used for
   each kernel. This is our evidence of independent creation.
6. Plain ASCII punctuation in code comments, docs, commit messages and PR text.
7. Do not use center's GPUs when another job holds them; check with the owner first. Keep >= 15% VRAM free on
   each GPU in every test. Never OOM.

## 2. Repository layout (in AEON-7/vllm-ultimate-rocm)

Current state: a Python-only overlay on `vllm/vllm-openai-rocm:v0.31.0` (digest-pinned) with two hash-pinned
file replacements applied by `install_patches.py`:
- C1 `mamba_hybrid.py` (DFlash2 prefix-cache fix, vLLM #55601): upstream since 2026-10-07. Drop it.
- C2 `mxfp6_emulation.py` (MXFP6 dequant-once): still needed for Qwen3.8 MXFP6. The upstream
  `emulation.py` changed after v0.31.0 (Parameter import, qdq lambda refactor, `weight_scale` handling), so
  it needs a rebase and new stock/patched sha256 pins.

A Python overlay cannot carry this work (new compiled ops, a different torch/Triton/ROCm stack, and a vLLM
commit newer than any ROCm release image). The new image is a from-source build. Proposed layout, added next
to the existing files (the existing Dockerfile stays until the new one is validated):

```
vllm-src.env                      VLLM_COMMIT=<upstream main sha after 2026-11-21 16:19 UTC>, VERSION
series/                           patch series applied in order on VLLM_COMMIT
  series.conf                     ordered list + sha256 of each patch
  upstream/0001-pr67685-rocm-inkling2-small.patch     (from vllm-project PR head, until merged)
  core/0101-rocm-rdna4-hw-queues.patch
  core/0102-gfx12-config-guards.patch
  core/0103-tail-conv-state-manager.patch
  core/0104-adaptive-verify-realistic-profile.patch
  core/0105-adaptive-verify-online-refit.patch
  core/0106-loader-index-authoritative.patch
  model/0201-amd-quant-config-mixed-experts.patch
  model/0202-amd-loop-core.patch
  model/0203-amd-attention-gfx12.patch
  model/0204-amd-nvfp4-moe.patch
  model/0205-amd-mxfp8-moe-gfx12.patch
  model/0206-amd-decode-gemv-fusion.patch
  model/0207-amd-conv-mix.patch
  model/0208-amd-dspark-gfx12.patch
  model/0209-amd-vision-gfx12.patch
  model/0210-amd-tests.patch
  aeon/0301-mxfp6-dequant-once.patch      (C2 rebased)
docker/Dockerfile.gfx12           new from-source image
docker/constraints-rocm10.txt     hash-pinned torch/triton/rocm wheels
scripts/apply-series.sh           clone VLLM_COMMIT, verify hashes, apply series, refuse on any fuzz
scripts/serve.sh, scripts/bench.sh
benchmarks/inkling2_rocm/         sweep scripts + checked-in gfx12 config tables
docs/inkling2-rocm/               SPEC, PROVENANCE, PLAN, DEVLOG
NOTICE                            updated (section 5)
```

Development happens on a branch of a full vLLM clone on center; the series is cut from it with
`git format-patch` per path group and must rebuild the branch tree byte for byte (`scripts/apply-series.sh`
plus `git diff --exit-code` in CI).

## 3. Patch series (what each patch contains)

Base: upstream vLLM main at a commit after #69028 merged (2026-11-21 16:19 UTC). Pick the newest commit that
builds for ROCm on the day work starts, then hold it fixed until the series is green.

| Patch | Contents | SPEC | Depends on |
|---|---|---|---|
| upstream/0001 | PR #67685 at its current head (`800f72e235ac` or newer), rebased onto VLLM_COMMIT. Gives `vllm/models/inkling2/amd/` and ROCm dispatch | - | base |
| core/0101 | RDNA4: default `GPU_MAX_HW_QUEUES=2` in `vllm/platforms/rocm.py` before HIP init unless set | 7 | base |
| core/0102 | Guard SM12x-specific config checks with `is_cuda()` so gfx12 is not treated as SM120 | 7 | base |
| core/0103 | Tail-only conv-state cache manager (subclass of the DFlash sliding manager) with prefix-resend reach; registered for Inkling conv-state groups | 6.4 | base |
| core/0104 | Adaptive verify startup profile with random routed token ids (not padding) | 5.2 | base |
| core/0105 | Adaptive verify online refit (GPU-event timing, context slope, outlier rejection, env off switch) | 5.2 | 0104 |
| core/0106 | Loader honors the index: a key mapped to shard S is ignored in other shards (overlay safety) | 1.2 | base |
| model/0201 | AMD quant config: parse NVFP4 and MXFP8 groups per layer from `hf_quant_config.json`; load NVFP4 experts as stored | 3.4 | 0001 |
| model/0202 | `amd/loop.py`: loop model, per-pass caches, message controllers, pass-aware routed-expert capture; `amd/ops/loop.py` pair/blend norms | 1 | 0001, 0201 |
| model/0203 | Attention for gfx12: prologue, tiled prefill kernel, split-KV decode/verify with GQA packing, FP8 KV reads, gfx12 config tables | 2, 7 | 0202 |
| model/0204 | NVFP4 MoE: in-register dequant grouped GEMM (prefill) and BF16-activation GEMV with fused SwiGLU (decode/verify), reference dequant helper for tests | 3.2 | 0201 |
| model/0205 | MXFP8 MoE on gfx12 (dequant-to-fp16 dot by default; fp8 WMMA path opt-in) | 3.3 | 0201 |
| model/0206 | Decode GEMVs for bf16 linears, multi-row verify GEMV, fused q/k/v/r/gate/sink projection, softcap epilogue, launch-count reductions for HIP graphs | 7 | 0203 |
| model/0207 | Triton conv + AttnRes mix + norm fusion | 6.2 | 0202 |
| model/0208 | DSpark on gfx12: draft attention with FP8 draft KV, Markov step kernel, FP8 draft weight copies, fold refresh after weight reload | 5 | 0203 |
| model/0209 | Vision via vLLM Triton varlen attention on ROCm, encoder graphs | 9 | 0001 |
| model/0210 | Tests under `tests/models/inkling2/` (ROCm, gfx12 markers) and `tests/v1/core/` | all | all |
| aeon/0301 | C2 MXFP6 dequant-once rebased onto the new `emulation.py` | - | base |

Rule: each patch must apply with zero fuzz, build, and pass its own tests at the point it lands (bisectable).

## 4. Image changes

`docker/Dockerfile.gfx12` (new; the old overlay Dockerfile is kept as a separate tag until retired):
1. `FROM python:3.12-slim@sha256:<pinned>`; apt only build essentials, git, ninja, cmake, ccache, libnuma.
2. `pip install -c docker/constraints-rocm10.txt` the pip ROCm 10.0 SDK, `torch==2.13.0+rocm10.0.0`,
   `triton==3.8.0+rocm10.0.0`, `torchvision==0.28.0+rocm10.0.0`, the gfx1201 device wheel; all hash-pinned.
3. Fetch vLLM at `VLLM_COMMIT` (tarball by sha, checksum verified), run `scripts/apply-series.sh`.
4. Build with `PYTORCH_ROCM_ARCH=gfx1201` (optionally `gfx1200;gfx1201`), `MAX_JOBS` from build arg,
   `VLLM_TARGET_DEVICE=rocm`; exclude `csrc/inkling2/*.cu`. Rust frontend optional via build arg, toolchain
   from a pinned rust image stage (no piped installer).
5. Runtime env: `VLLM_NO_USAGE_STATS=1`, `ROCPROFILER_REGISTER_ENABLED=0`, `VLLM_ROCM_USE_AITER=0`,
   `AEON_MXFP6_DEQUANT_AT_LOAD=1`; do not set allocator env vars.
6. Labels: `ai.aeon.vllm.commit`, `ai.aeon.series.sha256` (hash of series.conf), `ai.aeon.target=gfx1201`,
   `ai.aeon.patches=<list>`. Keep the refuse-if-unpatched idea: at build end, verify each patched file's sha256
   against a manifest generated from the series, so an image can never ship labeled but unpatched.
7. Entrypoint: reuse `aeon-entrypoint.sh` (updated for the new paths).
8. CI: GitHub Actions builds on tag push only, pushes to GHCR only after the owner (AEON-7) approves publishing (manual
   dispatch). No publishing from this PR.

Native path on center (no Docker; Docker/Podman is not installed there and needs sudo plus the owner's yes):
the same constraints in a uv venv under `~/inkling/engines/aeon-rocm/`, separate from the redshift trial
venv, with `env.sh`, `serve.sh` (GPUS/TP/PORT/KV/SPEC knobs) and `ROCR_VISIBLE_DEVICES` pinning only.

## 5. NOTICE update (when code lands)

Add: Apache-2.0 vLLM (base and PRs #69028, #67685 with PR numbers and shas), transformers (reference only,
not shipped), Triton (MIT), PyTorch (BSD-3), AMD ROCm pip packages (per-wheel licenses), and the
ACKNOWLEDGEMENTS paragraph from PROVENANCE.md stating that vllm-redshift was a learning source and that no
code from it is included.

## 6. Test plan

### 6.1 Kernel unit tests (pytest, gfx1201, vs PyTorch fp32 references)

| Area | Reference | Shapes | Pass criteria |
|---|---|---|---|
| NVFP4 dequant | LUT dequant in torch | real layer 0 tensors + random | exact (fp32) |
| NVFP4 GEMM / GEMV | `x @ dequant(W)^T` fp32 | rows/expert 1,2,3,6,8,16,64,1024; K 2304/1152 | max rel 1e-2, mean abs <= 1e-3 * mean abs ref |
| MXFP8 GEMM / GEMV | same with e8m0 dequant | same | same |
| Router | torch topk with low-id ties | T 1..4096 | identical ids; weights max abs 1e-6 |
| Full MoE layer | R1 `Inkling2Experts` on real weights, layer 0 and 20 | 512 captured hidden states | cosine >= 0.9995 per token |
| Attention prefill | eager fp32 with explicit bias + sink column | T 64, 513; ctx up to 70000; local/global | max abs 2e-2, mean abs 2e-3 |
| Attention decode/verify | same | q 1..8; ctx 1..196608; splits 1/4/32 | same; split invariance 1e-3 |
| FP8 KV | reference on dequantized cache | same | max abs 6e-2, mean abs 6e-3 |
| Paging | contiguous vs random block tables | block 16 and 128 | bit-identical |
| L2 clip / log scale / prologue | R1 modules | positions 0, 8191, 8192, 8193, 150000 | max abs 1e-3 (fp32) |
| Short conv (static, dynamic) | R1 conv modules with state | T 1, 3, 4, 5, 1000; chunked | max abs 1e-3 |
| Conv + AttnRes fusion | torch composition | Q 1 and 3 | max rel 1e-3 |
| Loop controller blend | torch fp32 | S 4, T 1..257 | max rel 1e-3 |
| DSpark draft layer, Markov step | torch port of S2 math | 1..8 requests | base logits 5e-2; Markov bit-exact in fp32 mode |
| Tail conv-state manager | allocation counts, hits at offsets 0..127 past alignment | - | exact |
| Graph replay | eager vs HIP graph at padded batch | 1, 3, 8 | identical |

### 6.2 Model-level vs transformers

Reference: transformers bf16 eager on center (`~/inkling/phase3/stock_kl/logprobs.f16`, full-vocab logprobs,
50 stock KL prompts, 1,375 answer positions). Harness: our `redshift_eval/kl.py` and `greedy.py` pointed at the
new server (they only talk to the OpenAI API, so they carry over unchanged).

| Check | Pass |
|---|---|
| Teacher-forced KL(vLLM, ref) over top-1000 tokens, TP=1, HIP graphs, bf16 KV | mean <= 0.0035 (target ~0.003), p95 <= 0.02 |
| Top-1 agreement | >= 97% (target 98%) |
| Greedy, 40 benign prompts x 256 tokens | first token 40/40; median first divergence >= 24 tokens |
| Overlay candidate (wo_ud zeroed at L5, L11) | KL >= 0.2 (proves the override shard is applied) |
| Long context: 2 x 191K needle prompts concurrently at FP8 KV | both answered |
| Tiny looped random model vs R1 | max abs logit 2e-2, top-1 >= 99% |
| Refusal/loop sanity: stock harmful-none 100 prompts via vLLM | within 2 of the transformers stock baseline (99/100 refusals by the local judge), 0 loops |

### 6.3 Throughput (one R9700, TP=1, HIP graphs, bf16 KV, T=0, ignore_eos, 256 new tokens, warm)

Harness: `redshift_eval/bench.py` (ours), 3 runs, median.

| Config | bs1 tok/s | bs8 aggregate tok/s |
|---|---|---|
| no spec | >= 79 | >= 255 |
| DSpark k=7, fp8 draft KV, adaptive verify, probabilistic drafts | >= 86 | >= 316 |

Also record: prefill tok/s at 4K/16K/64K, ms per decode step, launches per step, VRAM per GPU (>= 15% free
at max-model-len 16384, max-num-seqs 8, gpu util 0.80), and a 2-hour soak at 8 concurrent with no HIP faults.
TP=2 must work (correctness), but the deployment target is one TP=1 server per GPU.

## 7. Effort estimate

Estimates are focused engineer-days for one experienced implementer (or one agent with a human reviewer),
with center available for testing. They assume the upstream PRs stay as they are today.

| # | Component | Original or reuse | Effort (days) |
|---|---|---|---|
| 0 | Base pick, series tooling, apply-series and hash manifest, native venv on center | reuse + ours | 1.0 |
| 1 | Bring in #67685, rebase onto base, gfx12 guards (0102), HW queues (0101), loader rule (0106) | reuse | 1.0 |
| 2 | Quant config for mixed NVFP4/MXFP8 layers, as-stored loading | reuse | 0.5 |
| 3 | Loop core on AMD: model, per-pass caches, controllers, capture | port of upstream | 2.0-3.0 |
| 4 | Attention gfx12: prologue, prefill kernel, split-KV decode/verify, FP8 KV, tables | original on upstream base | 3.0-4.0 |
| 5 | NVFP4 MoE GEMM + GEMV with fused SwiGLU | original | 3.0-4.0 |
| 6 | MXFP8 MoE on gfx12 | original | 1.5-2.0 |
| 7 | Decode GEMVs, projection fusion, launch reduction, full HIP-graph decode | original | 2.0-3.0 |
| 8 | Conv + AttnRes fusion in Triton | original from upstream CUDA semantics | 1.5-2.0 |
| 9 | Tail-only conv-state manager + tests | original on upstream base | 1.5-2.0 |
| 10 | DSpark on gfx12 (draft attention, FP8 draft KV, Markov kernel, FP8 draft copies) | port + original | 2.5-3.0 |
| 11 | Adaptive verify profile + online refit | original on upstream base | 1.0-1.5 |
| 12 | Vocab/logits path, optional FP8 head | reuse + small original | 0.5-1.0 |
| 13 | Vision on ROCm | reuse | 0.5-1.0 |
| 14 | Dockerfile.gfx12, constraints, CI (build only), NOTICE | ours | 1.0-1.5 |
| 15 | C2 MXFP6 rebase + Qwen3.8 MXFP6 recheck | ours | 0.25-0.5 |
| 16 | Validation: unit, model-level, throughput, soak, tuning sweeps | ours | 2.0-3.0 |
| | Total | | about 25-35 engineer-days |

Milestones:
- M1 correct and slow (items 0-3, 5-6 with simple kernels, 12): text serving with KL <= 0.0035. About 7-9 days.
  This is already useful for abliteration scoring if it beats transformers' 5-6 tok/s.
- M2 fast (4, 7, 8, 9 tuned): >= 79 / >= 255. About +8-10 days.
- M3 DSpark and extras (10, 11, 13, 14): >= 86 / >= 316, image builds. About +6-8 days.
- Calendar: about 4-6 weeks for one implementer; 2-3 weeks with 2-3 parallel implementers split by
  kernel area (attention / MoE / loop+DSpark), since the kernels have clean interfaces.

For comparison: vendoring redshift as-is was estimated at 2-3 days, but it cannot be published without a
license from its author. This plan costs more and produces code we own.

## 8. Out of scope for this PR

This PR adds docs only. It does not change the Dockerfile, patches, install script, recipes or README, and
nothing is built or published. Publishing an image, a model or quants needs the owner (AEON-7)'s explicit yes.
