# Inkling-2-Mini on ROCm gfx1201: functional spec (clean room, stage 1)

Status: draft for implementation. Written 2026-11-23 by the AEON stage-1 (spec) agent.

Target: `thinkingmachines/Inkling-2-Mini` (architecture `Inkling2LoopForConditionalGeneration`,
`model_type: inkling2_loop`, weights served exactly as shipped) on one AMD Radeon AI PRO R9700
(RDNA4, gfx1201, 32 GB GDDR6, wave32), inside our own vLLM build in `AEON-7/vllm-ultimate-rocm`.

This document says WHAT each component must do and how to test it. It does not say how any third-party
implementation does it. Read the clean-room rules in PLAN.md before you start. In short: never open
`abhinand5/vllm-redshift` (code, patches, issues, commits, images). Work only from this spec, PROVENANCE.md,
PLAN.md, upstream vLLM (Apache-2.0), the Hugging Face transformers reference modeling code (Apache-2.0), the
model checkpoint and its config, and the public docs and papers listed in PROVENANCE.md.

Interface names below (`Inkling2Attention`, `Inkling2LoopModel`, `attn_res`, ...) are the names used in
upstream vLLM `vllm/models/inkling2/` (Apache-2.0). Use them so our backend plugs into the upstream package
dispatch the same way the NVIDIA backend does.

## 0. Ground truth and conventions

### 0.1 Reference implementations (the numerics oracle)

- R1, model math: Hugging Face transformers `models/inkling2_loop` (`modeling_inkling2_loop.py`, Apache-2.0).
  Our local copy is `/workspace/inkling/modeling_loop.py` (box) and `~/inkling/modeling_loop.py` (center).
  This is the model-level oracle for every test in section 9 of PLAN.md.
- R2, serving semantics: upstream vLLM `vllm/models/inkling2/` after PR #69028 (merged 2026-11-21):
  `configs.py`, `quant_config.py`, `speculator.py`, `nvidia/{model,loop,attention,moe,sconv,dspark}.py` and
  `nvidia/ops/*`. Their docstrings define cache layout, loop pass order, routing capture, DSpark feature
  capture and the logits path. Our AMD backend must expose the same module and method surface.
- R3, ROCm skeleton: upstream vLLM PR #67685 ("[ROCm][Model] Add Inkling 2 Small support on ROCm", draft,
  Apache-2.0, head `800f72e235ac` on 2026-11-23). It adds `vllm/models/inkling2/amd/` with Triton kernels for
  gfx950 (MI355X): relative-bias attention with split-KV decode, AttnRes mix, routing, MXFP8 grouped GEMM via
  `tl.dot_scaled`, short conv, DSpark draft and vision. Fetch it from vllm-project/vllm only.

### 0.2 Checkpoint facts (from config.json, hf_quant_config.json and the safetensors header)

| Item | Value |
|---|---|
| hidden_size D | 2304 |
| decoder layers | 24 (ids 0..23) |
| loop core | layers 6..17 (`core_start=6`, `core_end=18`), `num_passes=3`, `gate_rank=64`, `is_varloop=false` |
| layer visits per token | 24 + 2*12 = 48 (`Inkling2ModelConfig.num_layer_visits`) |
| global (full) attention layers | 5, 11, 17, 23 (all ids not in `local_layer_ids`) |
| local (sliding) layers | the other 20, window 1024 |
| heads | 24 query heads, 8 KV heads (GQA group 3), head_dim 128, both layer kinds |
| relative bias | `d_rel=16`, `rel_extent=1024`, `rel_l2_clip_threshold=8.0` |
| q/k L2 clip | `qk_l2_clip_threshold = 22.6274...` (= 16*sqrt(2)) |
| log scaling (global layers) | `log_scaling_n_floor=8192`, `log_scaling_alpha=0.1` |
| AttnRes | `n_layers_per_attn_res_block=6`, values normalized |
| MoE | 128 routed experts, top-8, no shared expert, `intermediate_size=1152`, `route_scale=8.0`, gate bias on, `mlp_gate_softcap=4.0`, `mlp_up_softcap=25.0` |
| experts layers 0..19 | NVFP4: `w13_weight` u8 [128, 2304, 1152], `w13_weight_scale` e4m3 [128, 2304, 144], `w2_weight` u8 [128, 2304, 576], `w2_weight_scale` e4m3 [128, 2304, 72], plus `*_weight_amax` f32 [128] and `*_weight_outer_scale_denominator` f32 [1] |
| experts layers 20..23 | MXFP8: `w13_weight` e4m3 [128, 2304, 2304], `w13_weight_scale` u8 (e8m0) [128, 2304, 72], `w2_weight` e4m3 [128, 2304, 1152], `w2_weight_scale` u8 [128, 2304, 36] |
| everything else | bf16 (AttnRes queries and router bias in f32) |
| vocab | embedding and unembedding tables are [201024, 2304]; real vocab (`unpadded_vocab_size`) is 200058; highest tokenizer id is 200057 |
| final logit softcap | 30.0 |
| DSpark draft | `mtp.safetensors`, 0.59 GB, 3 layers (local ids 0 and 2 sliding, window 1024), block_size 8, `markov_rank=256`, `markov_head_type=rnn_hidden`, `target_layer_ids=[5, 23, 41, 48]` (layer visits; 48 = final-norm output) |

### 0.3 Notation

- T tokens in a step, H=24 query heads, Hk=8 KV heads, d=128 head dim, E=128 experts, k=8 top-k, I=1152.
- "fp32 accumulate" means every reduction (dot, sum, softmax normalizer) is carried in fp32.
- "bf16 out" means rounded once to bf16 at the end, round-to-nearest-even.
- All kernels must be deterministic for a fixed batch composition (no atomics in reductions that change
  summation order between runs), except where a section explicitly allows split reductions with a fixed
  split count.

## 1. inkling2_loop model wiring on ROCm

### 1.1 Required behavior

The forward pass for one step is (R1 and R2 are authoritative where this summary is short):

1. Embed: `x0 = RMSNorm_embed(embed[ids])` (bf16 table, fp32 norm).
2. Prelude: layers 0..5 run once.
3. Loop core: layers 6..17 run `P = 3` passes. Every pass starts from the AttnRes state left by the prelude
   (not from the previous pass output). Each pass of each core layer has its own KV cache and its own
   short-conv state cache.
4. Coda: layers 18..23 run once, starting from the AttnRes state after the last pass.
5. Final norm, unembed (section 4), softcap.

Residual stream: there is no plain residual add. Each sublayer (attention, MLP) reads its input as a
block attention-residual (AttnRes) mix: the stored outputs of finished blocks of 6 sublayer-layers plus the
current partial block are scored by a learned fp32 pseudo-query,

    score_n = (q . x_n) * rsqrt(mean(x_n^2) + eps)            (q pre-scaled by D^-0.5)
    mix     = sum_n softmax_n(score) * RMSNormUnweighted(x_n)  (normalized values)

and then goes through the sublayer's RMSNorm. Global layers have separate q, k and v input norms
(`q_input_norm`, `k_input_norm`, `v_input_norm`) and so take three mixes with three queries
(`attn_queries.i` shaped [3, D]); local layers take one. Block outputs are stored in bf16, mixes are fp32.

Loop messages (passes 2 and 3 of every core layer): each of the four core streams
`LOOP_STREAMS = (attn, k, v, mlp)` is blended with a message read from the previous pass:

- Message read: an AttnRes-style mix over the previous pass's two block states, with the per-layer,
  per-stream learned queries `middle_cycle_cross_pass_queries` and `middle_cycle_cross_pass_kv_queries`
  ([24, D] each; see R2 `loop.py` for which rows map to which layer and stream), values RMS-normalized.
- Controller (rank r = 64), per stream s, in fp32 except the two small GEMMs:

      pair    = [RMSNormUnweighted(dest), RMSNormUnweighted(message)]        # (T, 2D)
      h       = pair_bf16 @ down_s^T                                          # (T, r), down_s: [r, 2D] bf16
      gain    = h @ gain_s^T,  forget = h @ forget_s^T                        # (T, D), fp32 out
      blended = (1 - tanh(forget)) * dest + tanh(gain) * message              # fp32

  then the stream's own RMSNorm (weight row of that stream) gives the bf16 sublayer input.
- Pass 1 has no message (dest passes through).

Per-pass caches: for core layer L and pass p, the KV cache and the conv-state cache are distinct
allocations. Pass 0 uses the layer's own cache; passes 1 and 2 use extra pass caches registered as extra
attention layers (upstream name `Inkling2AttentionPassCache`). The KV cache manager must see 24 + 2*12 = 48
attention-layer cache specs, of which the sliding ones are window 1024 and the global ones are full.
`loop_share_cache` and `loop_num_passes` overrides exist upstream for variable-loop research checkpoints only;
Mini has `is_varloop=false`, so reject them with the same error upstream raises.

Layer-visit numbering (needed by DSpark and routed-expert capture): visit ids follow
`Inkling2ModelConfig.layer_visit_ids` (prelude ids 0..5, then core visits 6..17, 18..29, 30..41, then the coda
at 42..47). DSpark taps visits 5, 23, 41 and 48 (48 = final-norm output).

### 1.2 Inputs and outputs

- Module surface: `Inkling2LoopForConditionalGeneration` (registered for `model_type inkling2_loop` on ROCm),
  `Inkling2LoopModel`, `Inkling2LoopDecoderLayer`, `Inkling2MessageController`, in
  `vllm/models/inkling2/amd/loop.py`, mirroring the NVIDIA module's public methods (`forward`,
  `load_weights`, `process_weights_after_loading`, `set_aux_hidden_state_layers` for DSpark,
  `embed_input_ids`, `embed_multimodal`).
- Weight names: `model.llm.layers.{i}.controllers.{attn,k,v,mlp}.{down,gain,forget}_weight`
  (bf16 [64, 4608], [2304, 64], [2304, 64]); stack them per layer into `controller_down` [4, 64, 4608] and
  `controller_gain_forget` [4, 4608, 64] as upstream does, so one batched GEMM serves all four streams.
- Loader must be last-write-wins and must accept extra override shards (we serve abliteration candidates as a
  symlinked tree plus a `zz-*.safetensors` override shard and a rewritten index). Add an explicit, tested
  rule: if the index maps a key to shard S, skip that key in any other shard. Do not rely on file order.

### 1.3 Numerics

- AttnRes mixes and controller blends in fp32; block storage bf16; `down` and `gain/forget` GEMMs in bf16
  operands with fp32 accumulate (gain/forget output kept fp32).
- RMSNorm eps 1e-6 everywhere in the loop path.
- The blended fp32 stream must not be rounded to bf16 before its RMSNorm.

### 1.4 Edge cases

- Mixed prefill/decode batches: every pass must see the same token ordering and the same cache slot mapping
  as pass 0, only with a different cache tensor.
- Chunked prefill: conv state and KV of every pass must be continued across chunks.
- Prefix-cache hits: a hit restores KV and conv state for all three passes of every core layer (they are
  separate cache groups; the hit length must be the minimum over all groups).
- Padding tokens in captured graphs: they must not route to experts and must not write real cache slots.
- HIP graph capture: the per-pass cache selection must be static per captured graph (index by Python int,
  not by a device tensor).

### 1.5 How to test

- Unit: tiny random looped config (D=256, 6 layers, core 2..4, 3 passes, E=8, k=2) built from the
  transformers config class; compare our model's logits to R1 on random ids (prefill of 64 tokens, then 16
  single-token decodes). Pass: max abs logit diff <= 2e-2 and top-1 equal on >= 99% of positions.
- Controller op: `blend` vs a torch fp32 reference on random (S=4, T in {1, 3, 8, 257}, D=2304). Pass:
  max rel err <= 1e-3 on the fp32 output before the norm.
- Cache isolation: run pass 2 with pass 1's cache tensor swapped in and confirm the logits change (proves
  passes are not sharing).
- Real model: section 9 of PLAN.md (KL and top-1 vs transformers).

## 2. Relative-bias attention with sinks and output gate (Triton, ROCm)

### 2.1 Required behavior

Per layer and per head h, for query token i at position pos_i and key token j at pos_j (j <= i, and for local
layers pos_i - pos_j < 1024):

    q_i   = clip_l2(q_scale_h * W_q x_i, 22.6274)  (* tau_i on global layers)
    k_j   = clip_l2(Kconv(W_k x)_j, 22.6274)       (no learned scale on keys; conv first, see 2.1.1)
    v_j   = RMSNormUnweighted(Vconv(W_v x)_j)      (values: unweighted RMS norm, eps 1e-6, no gamma)
    r_i   = clip_l2(rel_scale_h * W_r x_i, 8.0)    (* tau_i on global layers), r_i in R^16
    dist  = pos_i - pos_j
    bias  = (r_i . P[:, dist]) if 0 <= dist < 1024 else 0      (P = rel_logits_proj.proj, [16, 1024])
    logit = (q_i . k_j) / sqrt(128) + bias
    sink  = s_{i,h} = (W_sink x_i)[h]               (one extra logit per query, with value vector 0)
    p     = softmax over {logit_j for visible j} U {sink}
    o_i   = sum_j p_j v_j                           (sink contributes probability mass only)
    out_i = W_o ( sigmoid(W_gate x_i) * o_i )       (gate in fp32, product rounded to bf16 before W_o)

where `clip_l2(v, t) = v * min(1, t / (||v||_2 + 1e-6))` applied per head vector after the learned per-head
scale, and `tau_i = 1 + 0.1 * ln(max((pos_i + 1) / 8192, 1))` (global layers only, for both q and r). Local
layers use `rel_extent = sliding_window_size = 1024`, global layers use `rel_extent = 1024` too; the bias is
zero beyond it but global keys stay visible.

#### 2.1.1 K/V short convolution

Keys and values pass through a causal depthwise conv of width 4 before the cache write:

- Local layers: static per-channel taps (`k_sconv.weight`, `v_sconv.weight` [1024, 1, 4]).
- Global layers: dynamic taps, `static_w` [4, 1024] plus a per-token, per-KV-head delta
  `weight_proj(x)` [32 = 8 heads * 4 taps].
- Tap 0 weights the current token; tap t weights the token t steps back. Residual form (R1):
  `out = (x + conv(x)) / sqrt(W + 1)` with W = 4, so the divisor is sqrt(5), accumulated in fp32 and rounded
  to bf16 once. The same convention applies to the sublayer output convs of section 6.1; confirm each against
  R1 in a unit test rather than assuming.
- Padding tokens are zeroed before the conv (R1 `apply_mask_to_padding_states`).
- The previous 3 tokens per sequence come from the conv-state cache (section 6.4).

Attention inputs per layer kind: local layers project q, k, v, r, gate and sink from one input (pass 1) or
from separate stream inputs (passes 2 and 3 of core layers, where k and v have their own blended streams).
Global layers always take separate q, k, v inputs.

### 2.2 Interfaces

- `Inkling2Attention` (amd) with `forward(x_q, x_k=None, x_v=None, tau=None, pass_idx=0, out=None)`
  returning this rank's bf16 (T, D) output, like upstream.
- Kernels (names are ours; suggested):
  - `attn_prologue(...)`: fused per-token projection epilogue: L2 clips, log scaling, K/V conv, FP8/bf16 cache
    write. One launch per layer per step.
  - `rel_prefill_attention(q, r, k_cache, v_cache, block_table, seq_meta, sinks, ...) -> o` for query
    lengths > a threshold (tiled over queries and keys).
  - `rel_decode_attention(...)`: split-KV kernel for query lengths 1..8 (decode and DSpark verify), followed
    by a split-combine kernel that also folds the sink and applies the gate.
  - `gate_sink(o_partial, lse, sinks, gate) -> bf16`: combine step.
- Attention metadata builder must order decode/verify requests first (as upstream) and record the split.

### 2.3 Numerics and precision

- QK and PV dots: bf16 or fp16 operands, fp32 accumulate. Softmax in fp32 with running max (online softmax,
  Milakov and Gimelshein 2018; FlashAttention-2 tiling). The sink joins the softmax as one extra logit:
  initialize the running max and normalizer so the sink counts exactly once per query row, then
  `o = (sum_j e^{l_j - m} v_j) / (sum_j e^{l_j - m} + e^{s - m})`.
- Bias: compute `r_i . P[:, dist]` in fp32 from bf16 r and P. Precompute per query block the [16, window]
  slice product once if it helps; never materialize a full [T, T] bias.
- KV cache: bf16 default (our scoring setting). FP8 e4m3 (OCP `float8_e4m3fn`) optional with a per-tensor
  scale of 1.0 as upstream FP8 KV; dequantize to fp16 in registers before the dot (fp16 keeps more mantissa
  than bf16 for the e4m3 range). Values that overflow e4m3 must saturate, not become NaN.
- Gate: `sigmoid` in fp32, product rounded to bf16, then `W_o` GEMM (bf16, fp32 accumulate).

### 2.4 Edge cases

- Query length 1 with context length 1 (first decode after a 1-token prompt).
- Sequences whose context crosses the 1024 window boundary inside a key tile (local layers): mask per key.
- dist exactly 0 and exactly 1023 (bias on), 1024 (bias off; for local layers the key is also masked).
- Paged KV with block size 16 and 128 (both must work), non-contiguous block tables, prefix-cache hits that
  start mid-block.
- Long context: 196,608 tokens. Split-KV decode must keep enough splits to fill 64 CUs at batch 1.
- Positions beyond 8192 switch on log scaling on global layers; test 8191, 8192, 8193 and 150000.
- All-masked rows are impossible (a query always sees itself), but the sink must still make a fully
  underflowed row finite.

### 2.5 How to test

- Torch reference: eager fp32 attention with an explicit [T, S] bias and an appended sink column, from R1's
  `eager_attention_forward` semantics. Compare kernel output (bf16) at T in {1, 2, 7, 8, 64, 513}, context in
  {1, 15, 1023, 1024, 1025, 4097, 70000}, both layer kinds, bf16 and FP8 KV. Pass: max abs err <= 2e-2,
  mean abs err <= 2e-3 vs fp32 reference (bf16 KV); <= 6e-2 / 6e-3 for FP8 KV against a reference that
  uses the same dequantized FP8 keys and values.
- Paging: same inputs written through random block tables must give bit-identical output to contiguous.
- Split invariance: decode output with 1, 4 and 32 splits must agree within 1e-3.
- Graph replay: capture at batch 8, replay at batch 3 with padding; outputs of the 3 real rows equal eager.

## 3. Expert MoE on gfx1201 using the shipped packed weights

### 3.1 Router (all 24 layers)

    logits = x_fp32 @ gate.weight^T                       # [T, 128], fp32
    s      = sigmoid(logits)
    idx    = top8(s + gate.bias)                          # bias only for selection
    w      = s[idx] / (sum(s[idx]) + 1e-20) * 8.0         # route_scale
    y      = sum_k w_k * Expert_{idx_k}(x)                 # fp32 accumulate, bf16 out

Ties break toward the lower expert id (upstream test `test_routing_matches_torch_and_breaks_ties_low`).
Routed-expert capture (`enable_return_routed_experts`) records ids at the row of the layer visit.

Expert function (softcapped SwiGLU, R1 `Inkling2SoftcappedSwiGLU`):

    [g, u] = x @ W13^T              # W13: [2*I, D], g first then u
    a      = 4.0*tanh(g/4.0) * sigmoid(g) * 25.0*tanh(u/25.0)
    out    = a @ W2^T               # W2: [D, I]

(The routing weight may be applied inside the activation before W2, as upstream does; the result must match
applying it after W2 within tolerance.)

### 3.2 NVFP4 experts (layers 0..19)

Format (as shipped; do not requantize or repack on disk):
- Values: e2m1, two per byte along K, low nibble first. Code -> value table
  {0, 0.5, 1, 1.5, 2, 3, 4, 6} with bit 3 as sign.
- Block scale: one e4m3 per 16 consecutive K elements, row-major [E, N, K/16].
- Global scale per expert and projection: `g_e = amax[e] / outer_scale_denominator`.
- Dequantized weight: `W[e, n, k] = e2m1(n, k) * e4m3_scale(n, k // 16) * g_e`. This is exact in fp32, and
  every product of an e2m1 value and an e4m3 scale is exactly representable in fp16 except for scale
  magnitudes outside fp16 range; check the actual checkpoint scale range at load and fall back to keeping the
  global scale outside the dot (multiply the fp32 accumulator) so the in-register fp16 product stays exact.

RDNA4 has no FP4 matrix instruction and no block-scaled dot (`tl.dot_scaled` lowers to MFMA scaled ops on
gfx950 only). Required approach, our own design:

- Prefill and large batches (rows per expert >= ~16): grouped GEMM over expert-sorted rows. Each program
  loads a [BLOCK_N, BLOCK_K] packed tile, expands nibbles to fp16 via a 16-entry lookup or bit arithmetic,
  multiplies by the broadcast block scale, and feeds `tl.dot` with fp16 operands (activations cast
  bf16 -> fp16 with a per-row scale guard, or kept bf16 with bf16 dequant if range requires; pick one and
  test). fp32 accumulate. Apply g_e on the accumulator.
- Decode and verify (1..8 tokens): a GEMV path. Activations stay bf16 (no activation quantization); each
  program owns a slice of N for one (token, expert) pair or a small group of pairs for the same expert, and
  streams the packed weights once. Fuse the SwiGLU into the W13 epilogue so only `a` (bf16, [rows, I]) is
  written. Fuse the routing weight and the top-k sum into the W2 path or into a follow-up combine kernel.
- Expert-sorted row layout (`route` -> `(sorted_rows, expert_offsets, src2dst)`) follows upstream's
  `ExpertPlan` contract so the combine kernel can be shared.

### 3.3 MXFP8 experts (layers 20..23)

Format: e4m3 (OCP) values [E, N, K], one e8m0 scale byte per 32 K elements, value = fp8 * 2^(s - 127).

Required approach:
- gfx1201 supports FP8 (OCP e4m3/e5m2) WMMA, but not per-32-element scaled dot. Two acceptable designs:
  (a) dequantize the weight tile to fp16 in registers (`fp8 -> fp16` exact, times `2^(s-127)` exact unless the
  exponent leaves fp16 range; guard it) and `tl.dot` in fp16; or
  (b) quantize activations to e4m3 per 32-block, run fp8 x fp8 WMMA per K-block of 32 with fp32 accumulate,
  and apply both scales to each partial sum before adding. (b) changes numerics relative to R1 (activations
  are not quantized there) and must stay behind an opt-in flag.
- Default: (a) for decode (BF16 activations, like the NVFP4 GEMV) and (a) for prefill unless (b) is both
  faster and within the model-level tolerance.

### 3.4 Interfaces

- `Inkling2QuantConfig` (amd) parses `hf_quant_config.json` parameter groups and answers, per layer,
  `expert_format(layer_id) -> {"nvfp4", "mxfp8"}`. Layers 0..19 and 20..23 must be detected from the file,
  not hard-coded.
- `Inkling2Experts.forward(x, routing) -> bf16 [T, D]` with the per-format kernels behind it.
- Weight loading keeps the checkpoint layout byte-for-byte (no shuffle, no swizzle). Any derived layout
  (for example a transposed scale copy for coalescing) is built in `process_weights_after_loading` and the
  original freed, with total VRAM for experts <= checkpoint bytes + 5%.

### 3.5 Edge cases

- An expert with zero rows in a step (skip, no NaN).
- All 8 picks of a token on one expert is impossible (top-k distinct), but many tokens on one expert is
  common in prefill: the GEMM tiling must handle 0..T rows per expert.
- Padding tokens: routed to no expert, output 0.
- K and N of w2 (K=1152, N=2304) and w13 (K=2304, N=2304) are not powers of two: masks on every edge tile.

### 3.6 How to test

- Format unit tests: random packed values and scales -> our in-kernel dequant written out vs a torch LUT
  dequant (exact equality in fp32).
- GEMM/GEMV vs torch: `x @ dequant(W)^T` in fp32 for rows in {1, 2, 3, 6, 8, 16, 64, 1024} per expert.
  Pass: max rel err <= 1e-2 (bf16 out), mean abs err <= 1e-3 * mean |ref|.
- Full MoE layer (router + experts + combine) vs R1's `Inkling2Experts` loaded from the real layer 0 and
  layer 20 weights on 512 real hidden states captured from the transformers reference. Pass: cosine >= 0.9995
  per token.
- Routing determinism: identical selections over 100 repeated runs; tie order matches the torch reference.

## 4. 200058-logit vs 201024-vocab handling

### 4.1 Required behavior

- The embedding table and the unembedding (`model.llm.unembed.weight`) are stored padded to 201024 rows.
  Rows 200058..201023 are padding.
- The served vocabulary is 200058 (`config.text_config.vocab_size` after `Inkling2ModelConfig` maps
  `unpadded_vocab_size`). Logits returned to the sampler, logprobs and `prompt_logprobs` are exactly
  200058 wide. Our transformers reference also emits 200058 logits, so KL comparisons are column-aligned.
- Use upstream `ParallelLMHead(padded_vocab_size, org_vocab_size=vocab_size)` plus the logits slice, as in
  upstream `Inkling2ForConditionalGeneration`; do not compute the padding rows at all if the GEMM can be
  bounded to 200058 rows (saves 0.5% of the head).
- Logit path: final RMSNorm -> bf16 head GEMM with fp32 output (`head_dtype=float32` in the MM config) ->
  softcap `30 * tanh(z / 30)` in fp32.
- The DSpark drafter shares the target's embedding and head (aliased, not copied).

### 4.2 Optional FP8 head (opt-in, off by default)

- Env `AEON_INKLING2_FP8_LM_HEAD=1`: at load, quantize the head rows to e4m3 with one fp32 scale per row
  (amax / 448), read it in a dedicated GEMV/GEMM, free the bf16 head. The drafter's folded projection must be
  computed from the bf16 head before it is freed.
- Must stay off for all abliteration scoring.

### 4.3 Edge cases and tests

- Sampling never returns an id >= 200058 (test: set logits of padding rows to +inf in a debug build and
  confirm they are never computed or are sliced off).
- `max_logprobs` up to 1000 works (our KL harness requests 1000).
- Test: compare our full-vocab logprobs on 20 prompts to the transformers reference: same width, KL
  computed over all 200058 columns.

## 5. DSpark speculative decoding on ROCm

### 5.1 What DSpark is (from R2 and the checkpoint config)

A 3-layer dense draft transformer in `mtp.safetensors`. For each request it takes an anchor (the last
committed token) plus 7 mask tokens as one block of 8 positions, attends bidirectionally inside the block and
to context K/V built from target features captured at layer visits 5, 23, 41 and the final-norm output, and
produces 7 draft tokens. A recurrent (`rnn_hidden`, rank 256) Markov head corrects each draft step's base
logits using the previous draft token and a running state. Upstream vLLM already has the generic DSpark
speculator, rejection sampler, `draft_sample_method` ("greedy" | "probabilistic"), adaptive verification
(`enable_adaptive_verification`) and a separate draft `kv_cache_dtype`; the Inkling-specific draft model is
`vllm/models/inkling2/nvidia/dspark.py` and `vllm/models/inkling2/speculator.py`.

Serve flag (target form, same as upstream):

    --speculative-config '{"method":"dspark","num_speculative_tokens":7,
                           "kv_cache_dtype":"fp8","enable_adaptive_verification":true,
                           "draft_sample_method":"probabilistic"}'

Only `num_speculative_tokens=7` is valid (block size 8 = anchor + 7).

### 5.2 Required ROCm pieces

1. Draft model on ROCm (`vllm/models/inkling2/amd/dspark.py`): RMSNorms, q/k norm, RoPE (theta 1e6), the
   bidirectional block attention over [context K/V, block], sliding window 1024 on draft layers 0 and 2, MLP
   (intermediate 9216), and the Markov head. Context K/V are written once per committed token into the draft
   KV cache (`precompute_and_store_context_kv`).
2. FP8 draft KV cache: when the speculative config sets `kv_cache_dtype: "fp8"`, the draft attention reads
   e4m3 K/V (dequant to fp16 in registers) while the target cache stays at its own dtype. Draft-side only:
   it changes proposals, never the accepted output distribution.
3. Markov head: per step, `state, bias = step(state, prev_token)`; logits = softcap(base_logits + bias). The
   7 steps are sequential by nature; implement them as one persistent kernel (loop over steps inside the
   kernel, one program per request or per vocab tile with a grid-wide barrier-free design) or as 7 small
   launches captured in one HIP graph. The `(vocab, 256)` folded projection (`hidden_out` folded into the head,
   computed in fp64 then stored bf16, as upstream) is recomputed whenever target weights change (including
   our overlay shards).
4. Sampled drafts (`draft_sample_method: probabilistic`): draft token d_t sampled from the draft
   distribution q_t at the request's temperature/top-p/top-k; verification uses standard speculative
   rejection sampling (Leviathan et al. 2023; Chen et al. 2023): accept with prob min(1, p(d)/q(d)), on reject
   sample from norm(max(p - q, 0)). Greedy (T=0) must reduce to exact argmax matching. The verifier must have
   q available (store the draft probabilities, or their needed entries, per step).
5. Adaptive verify (`enable_adaptive_verification`): per step choose how many of the 7 drafts to verify from a
   cost model `cost(batch, width, context)` and the acceptance estimate. Upstream provides the policy and a
   startup profile. Our additions (in a separate core patch):
   - Profile with realistic MoE routing: dummy profiling batches must use random token ids routed to experts
     (not padding, not all-zero ids), otherwise MoE verify cost is underestimated.
   - Online refit: time real decode steps with GPU events recorded on the stream (no host sync), fit a per
     (batch, width) offset plus a shared slope over context length in thousands of tokens, reject outliers
     outside [0.5x, 2x] of the prediction, and fall back to the profile for unseen cells. Env switch to disable
     (`AEON_ADAPTIVE_VERIFY_ONLINE=0`) and to log stats every N steps.
6. FP8 draft weights (opt-in or default-on after validation): e4m3 copies with per-row scales of the 3 draft
   layers' linears and of the draft head; draft-only, output-exact for the target.

### 5.3 Numerics

- Target verification always runs the full-precision target path (bf16 weights, as-shipped experts).
- Draft base logits: bf16 GEMM as upstream; Markov recurrence fp32 inside a step, rounded at the boundaries
  upstream rounds.
- Greedy speculative output must equal non-speculative greedy output token-for-token for the same batch
  composition (allowing only the known bf16 batch-shape drift, which we measure separately).

### 5.4 Edge cases

- Requests finishing inside a draft block (EOS at draft position 3): remaining drafts discarded.
- Prefix-cache hit on a resent prompt: draft context K/V must be rebuilt or restored for the hit range.
- Mixed batches of speculative and prefill requests.
- Long context (>= 160K): draft attention must use split-KV too, or draft cost dominates.

### 5.5 How to test

- Draft model vs a torch port of R2's draft math on random features: base logits max abs err <= 5e-2.
- Markov kernel vs 7 torch steps: bit-identical state and logits in fp32 mode.
- End to end at T=0: 40 prompts x 256 tokens, DSpark on vs off, outputs identical (same batch shape).
- At T=1: acceptance length on our benchmark set >= 2.0; rejection sampler distribution test (chi-square on a
  toy vocab) as in upstream's rejection-sampler tests.
- FP8 draft KV and FP8 draft weights: acceptance length drop <= 5% vs bf16 draft.

## 6. Short convolutions, conv-state cache and launch fusion

### 6.1 Sublayer output convs

Every decoder layer applies a causal depthwise width-4 conv to its attention output stream
(`attn_sconv.weight` [2304, 1, 4]) and MLP output stream (`mlp_sconv.weight`) before the output joins the
residual block, plus the K and V convs of section 2.1.1. The four streams of a layer share one conv-state
cache entry packed head-major (upstream `InklingConvState`).

### 6.2 Fused conv + AttnRes mix

Required: one kernel per sublayer boundary that (a) applies the output conv to the sublayer's partial output,
(b) adds it into the partial block, (c) computes the next sublayer's AttnRes mix (1 or 3 queries) and its
RMSNorm, and (d) writes the conv-state tail. Upstream does this in CUDA (`csrc/inkling2/conv_mix.cu`,
`allreduce_conv_mix.cu`); ours is Triton and TP=1 first (no all-reduce fusion needed at TP=1; at TP=2 an
unfused all-reduce then this kernel is acceptable).

### 6.3 Numerics

fp32 accumulate for conv and mix, bf16 block storage, conv state stored bf16.

### 6.4 Tail-only conv-state cache (core vLLM change)

Conv state is a per-position cache that a step only reads for the W-1 = 3 positions before its first token.
Upstream stores it in a sliding-window KV group. Required change: a cache manager variant that backs only the
blocks needed for (a) the 3-token window before each in-flight step and (b) any prefix-cache hit boundary that
can be reached on a resend: hits land on the last block-aligned position (alignment up to 128 tokens) minus
the hash lookahead, so the backed tail must reach back that far from each step end. Interior blocks of a long
prefill chunk map to the null block. Goal: per 196K request, conv state memory drops to a small constant and
two 196K requests fit beside the weights in 32 GB at FP8 KV.

Tests: unit test in `tests/v1/core/` for allocation counts and for a prefix hit at every offset in
[aligned, aligned + 127]; end to end, resending a 40K prompt must hit the cache for all but the last <= 128
tokens.

## 7. gfx1201 tuning choices

Facts to design for (verify each on center before relying on it):
- RDNA4, 64 CUs on the R9700, 32 GB GDDR6 (~640 GB/s), wave32 native (Triton `num_warps` counts 32-lane
  waves here), WMMA 16x16x16 for fp16/bf16 and OCP fp8/bf8 with fp32 accumulate, no MFMA, no scaled-MFMA, no
  FP4 matrix math, 64 KB LDS available per workgroup in CU mode.
- Decode is memory-bound: about 3.5B active params per token, with NVFP4 experts at ~0.56 bytes/param.
  Kernel launch count matters as much as bandwidth at batch 1.

Required tuning process (not fixed constants):
1. Every Triton kernel exposes a small config dataclass (BLOCK_M/N/K, num_warps, num_stages, waves_per_eu,
   split count) chosen by a pure function of (shape, arch). Arch key `gfx12` vs `gfx950` vs CUDA.
2. Populate gfx12 tables from an offline sweep script (`benchmarks/inkling2_rocm/sweep_*.py`) run on center,
   checked into the repo as data. No runtime autotune in the serving path (it breaks HIP graph capture and
   first-request latency); allow `AEON_INKLING2_AUTOTUNE=1` for development only.
3. Starting ranges for the sweep: GEMV kernels num_warps 4..8, 1 stage, N-slice 16..128 rows per program;
   grouped GEMM BLOCK_M 16..64, BLOCK_N 64..128, BLOCK_K 32..128, num_warps 4..8, num_stages 1..2;
   attention BLOCK_N (keys) 32..128, split count chosen so splits * KV heads * batch >= 2 * CU count.
4. Split-KV decode on gfx12: pack all 3 query heads of a GQA group (and all verify tokens of a request) into
   one program's M dimension so a K/V tile is loaded once per group; pad M only to the WMMA multiple.
5. Launch count: target <= ~800 kernel launches per decode step at batch 1 and run decode under full HIP
   graphs. Fuse: q/k/v/r/gate/sink projections into one GEMV on verify steps; SwiGLU into the W13 epilogue;
   softcap into the head epilogue; L2 clips and log scaling into the attention prologue.
6. Runtime knob: on RDNA4 set `GPU_MAX_HW_QUEUES=2` by default before HIP initializes, unless the user set it.
   (Measured large effect on this card; see PROVENANCE.md for credit. Re-measure on our build.)
7. Idle CPU: set `ROCPROFILER_REGISTER_ENABLED=0` in the image environment (stops a ROCr event thread from
   spinning when idle). Re-measure.
8. Keep `PYTORCH_HIP_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` unset by default. `expandable_segments` was
   unstable on our ROCm stack (it corrupted MoE routing in our transformers runs on center).

## 8. Build and packaging

- Base: `python:3.12-slim` (Debian) plus AMD's pip-distributed ROCm 10.0 runtime/SDK ("TheRock" wheels):
  `torch==2.13.0+rocm10.0.0`, `triton==3.8.0+rocm10.0.0` (ROCm fork build), `torchvision==0.28.0+rocm10.0.0`,
  the `rocm==10.0.0` SDK wheel and the gfx1201 device wheel. Pin all of these in a constraints file with
  hashes. No system ROCm in the image. Our center host runs kernel driver ROCm 7.1 packages and the pip 10.0
  user space works on it; do not mix a system ROCm 10.1 HIP runtime into the venv.
- vLLM: from source at a pinned upstream commit after #69028 plus our patch series (PLAN.md), with
  `PYTORCH_ROCM_ARCH=gfx1201` (add `gfx1200` if we want RX 9070 owners; others optional). Compiled extension:
  vLLM's `_C` / `_rocm_C` ops build for gfx1201. The Inkling CUDA sources (`csrc/inkling2/*.cu`) are
  NVIDIA-only: guard them out of the ROCm build or provide a HIP-compatible stub so the build succeeds; our
  backend must not import them.
- Rust frontend: optional (upstream added Inkling renderer bits to `rust/src/chat`). Build it if the toolchain
  is present, otherwise use the Python frontend. Install rustup from the distro or a pinned toolchain image,
  not a piped shell script.
- Env defaults in the image: `VLLM_NO_USAGE_STATS=1`, `GPU_MAX_HW_QUEUES=2` (RDNA4 only, via platform code),
  `ROCPROFILER_REGISTER_ENABLED=0`, `VLLM_ROCM_USE_AITER=0` (AITER has no gfx12 path for these ops).
- Triton cache: a named volume for `~/.triton/cache` so first-start compile cost is paid once.
- Native (no Docker) path: the same constraints installed into a uv venv on center, with an `env.sh` that sets
  `ROCM_PATH` and `LD_LIBRARY_PATH` to the pip SDK and pins a single GPU with `ROCR_VISIBLE_DEVICES=N` only
  (setting `HIP_VISIBLE_DEVICES` as well double-remaps and hides the device).

Tests: image smoke test runs `python -c "import vllm, torch; print(torch.cuda.get_device_properties(0))"`,
the kernel unit tests on gfx1201 and a 64-token chat completion.

## 9. Vision tower (lower priority)

Image input is part of the model (`vision_config`, 23-layer ViT, 1024 wide, axial RoPE, merger to 2304).
Upstream's NVIDIA path uses flash-attn varlen. Required on ROCm: use vLLM's existing Triton varlen attention
backend for the ViT (it supports gfx12), keep encoder HIP graphs, cap images per prompt via
`--limit-mm-per-prompt`. Test: one image prompt, compare projected image embeddings to R1 (cosine >= 0.999).
Our abliteration work is text-only (`--language-model-only`), so this ships after the text path.

## 10. Global acceptance criteria

See PLAN.md section 6 for the full test plan. Summary:
- Teacher-forced on our 50 stock KL prompts (1,375 answer positions) vs transformers bf16: mean KL <= 0.0035,
  top-1 agreement >= 97%, at bf16 KV, T=0.
- Throughput on one R9700, TP=1, HIP graphs, bf16 KV, 256 new tokens, T=0: >= 79 tok/s single stream,
  >= 255 tok/s aggregate at 8 concurrent; with DSpark (fp8 draft KV, adaptive verify, probabilistic drafts):
  >= 86 and >= 316.
- VRAM: >= 15% free per GPU at our scoring settings (max len 16384, 8 seqs, gpu util 0.80).
- No HIP faults in a 2-hour soak at 8 concurrent.
