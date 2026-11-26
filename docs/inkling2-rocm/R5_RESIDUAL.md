# R5-1: per-request residual-stream edits for Inkling-2-Mini (AMD) + bakeability

Patch 0017 (applies on 0016). Off unless AEON_VARIANT_DIR is set and a variant file carries
resid_dirs/resid_scales. Research tool; the bakeable subset is folded offline by
`tools/r5_bake.py` (below).

## What it does
- Variant files and `POST /aeon/variants` accept optional `resid_dirs` / `resid_scales`, with the same
  shapes as `dirs` / `scales` (`[D] | [24,D] | [48,D] | [48,K,D]`, and `[48] | [48,K]`). A variant can be
  o_proj-only, residual-only or both, and the two parts have separate scales and ranks. Omitted keys mean
  no edit of that kind. If only one key of a pair is given, the request is rejected (400).
- After application a (0..47), the stream entering application a+1 is the tensor that
  `AEON_CAPTURE_WHAT=resid` records. Every row of a request using the variant gets the same fp32 rank-K
  removal as the o_proj edit: `y <- y - s[a,k] r[a,k] (r[a,k].y)`.
  - The edit applies to every input stream of that mix: 1 stream for local layers, 3 (q/k/v) for global layers.
  - It works in the stream's own dtype: bf16 normed mixes at pass-0 visits, fp32 pre-norm mixes inside the loop.
  - a = 47 is the final hidden state fed to the LM head.
  - a = 17 and a = 29: the pass restart throws away the captured mix and rebuilds the next pass's start
    input from the prelude state. The edit therefore goes to a copy of that start input (`prelude_dests`
    is shared by every pass).
  - The capture records the stream before the edit.
- Same kernel as the o_proj path, with one change: an `nrows` argument. Row r uses the variant of token
  r % nrows, so all streams share one token map. On the o_proj path nrows equals the row count, and its
  output is bitwise identical to 21add0ba8 (tested on all 48 apps).
- Rows with variant 0 (stock) are skipped, so stock rows are bitwise untouched. A step with only stock
  rows, or whose variants have no edit at that app, launches nothing.
- Hot add: the same directory rescan as the o_proj variants, so no restart is needed. The residual table
  is allocated on the first residual variant.
- Eager only, like the o_proj variants. The batch-invariant contract still relies on AEON_BATCH_INVARIANT
  (on by default with AEON_VARIANT_DIR).

## Unit tests (small shapes, GPU1, < 100 MB; GPUs were at 74% VRAM from the abliteration run)
`aeon/r5_unit.py`, `aeon/r5_unit_oproj_regress.py`, `aeon/r5_unit_api.py`: all pass.
- Kernel vs fp32 torch reference, all 48 apps:
  - bf16 [24,2304]: max diff 2.4e-4 (one bf16 ulp from reduction order);
  - fp32 [3,24,2304]: 4.8e-7;
  - o_proj path: 3.7e-9.
- Stock rows and padding rows are bitwise untouched at every app. o_proj-only rows are untouched by
  residual edits, and residual-only rows by o_proj edits.
- With s=1 the removal is exact: |r.x| goes from 1.30 to 4e-7 (fp32), and the result is within 2.4e-7
  of the closed form.
- A mixed-variant launch is bitwise equal to per-variant launches (apps 3, 5, 17, 29, 40, 47; bf16 2-D
  and fp32 3-D).
- Row results do not depend on row position or neighbours (permuted batch: bitwise equal).
- `resid_edit(copy=True)` leaves the shared pass-start tensor untouched. `end_step` turns the edit off.
- Hot add: a residual variant registered while the bank is live is applied on the next step that names it.
- HTTP register works for residual-only, both, o_proj-only, and by path. Bad requests get 400. List shows
  rank and resid_rank.
- loop.py imports cleanly from the worktree.
- Placement check: app 5 -> next visit input, 17 -> pass-1 start, 29 -> pass-2 start, 41 -> next visit
  input, 47 -> final hidden.

## Bakeability (numbers: `runs/r5/bake_analysis.json`, `aeon/r5_bake_analysis.py`, CPU, stock read-only)
Directions are the stock DoM `none_last_v{app}` vectors from ~/inkling/directions, and s = 1. A random
direction is the control.

### 1. Folding into the residual writers (embedding, o_proj, MoE down projections; no shared experts): not equivalent, and lossy
- **The semantics differ.** The hook edits one transient tensor, the next application's attention-input
  mix. The block stack that every later application reads is untouched. Folding into writers edits the
  stack, which makes it a global edit seen by all later applications (and mixes, including the MLP-input
  mixes). It is a different intervention, not an approximation of the hook.
- **There is no single residual sum.** In AttnRes, each consumer reads its own softmax mix of
  value-normalised blocks times its own RMSNorm gain. Removing v from the stream is not the same as
  removing it from a normed input. cos(g*v, v):
  - embed_norm 0.861;
  - final norm 0.942;
  - L11 q/k/v input norms 0.955 / 0.937 / 0.980;
  - L12 attn_norm 0.970.

  Consumers also disagree with each other: cos(g_L11q*v, g_L12*v) = 0.966.
- **Short convs.** Every writer output goes through a static 4-tap per-channel causal conv
  (attn_sconv / mlp_sconv) before the residual add.
  - Folding P into the writer weights before the conv leaves part of the v-component (iid inputs):
    - L10 attn 0.37, L10 mlp 0.46, L21 attn 0.65, L21 mlp 0.85;
    - error vs hook 1-7%.
  - Exact removal needs a rank-4 projection (span of c_j*v). That leaves ~5e-7 of v but changes the
    output by 8-28%.
- **Loop.** Layers 6-17 are shared by 3 passes, so per-pass scales or directions cannot bake. The DoM
  directions differ by pass: layer 10 apps 10/22/34 have cos 0.90 / 0.94, and apps 5 vs 17 have cos 0.14.
- **Re-quantisation of W' = (I - v v^T) W.** The quantisers reproduce the stock tensors bit-exactly on an
  unedited round trip.

  | Writer | v-energy left after requant | requant error / edit size |
  |---|---|---|
  | NVFP4 down proj, L10 | 76% | 1.00 |
  | NVFP4 down proj, L19 | 77% | 1.15 |
  | NVFP4 down proj, random v | 84% | |
  | MXFP8 down proj, L21 | 27% | 0.62 |
  | MXFP8 down proj, L23 | 25% | 0.58 |
  | MXFP8 down proj, random v | 28% | |
  | bf16 o_proj (wo_ud) and embedding rows | 0.6% | bf16 rel err 0.16% |

  - The edit is ~2% of ||W||, below the FP4 grid step, so nearest rounding mostly snaps back to the stock
    codes. Error-feedback (GPTQ-style) rounding could cut this, but it is not in the stock format pipeline.
  - MXFP8 scales are per row x 32 columns (not 32x32 tiles).

### 2. Folding into the consumer instead: exact where the consumer is unique and linear (no new module)
- The hooked tensor has exactly one consumer: application a+1's attention input projections.
  - In the engine that is the fused qkvr (wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv), plus the
    k/v dynamic-tap weight_proj on global layers.
  - For a = 47 the consumer is the unembed (`model.llm.unembed.weight`, not tied; the softcap comes after
    the linear).
- At pass-0 visits the mix is already normed, so W' = W (I - s v v^T) is exact in real arithmetic. bf16
  storage leaves 0.55% of v-energy (L11 wq_du) and a 0.16% weight error. These are stock bf16 tensors, so
  a shippable overlay is possible.
- Bakeable apps:
  - **Clean:** 0-3 and 41-45 (local consumer layers 1-4 and 18-22), and 47 (unembed).
  - **Same algebra, not yet validated:** 4 and 46 (global layers 5 and 23, with three streams and their
    weight_proj).
- **Not bakeable: loop-consumer apps 5-40.**
  - Consumer weights are shared across passes, so the edit must be equal at a, a+12 and a+24 (and at 5,
    17 and 29 for layer 6).
  - At looped visits the hooked tensor is the fp32 pre-norm mix. Loop messages are then blended in and it
    is RMS-normed, so the projection does not commute and pass-tied folding is only approximate there.
- The residual DoM peak (app 47, L2 ~54) falls in the exactly bakeable set.

### Verdict
- **Writer-side folding:** rejected. It is a different (global) edit; it fails through the short convs and
  norms; NVFP4 snaps back 76-84%; MXFP8 snaps back 25-28%.
- **The runtime hook in general:** a research tool.
- **Shippable subset, no new module:** residual edits at apps 0-4 and 41-47, folded into the next layer's
  qkvr / unembed columns. Error is bf16 rounding only (~0.5% of the removed component left).
- **Loop apps 5-40:** research only. They become a candidate bake only with pass-tied edits, and even then
  the bake is approximate.

## Full-model validation (run on 1x R9700, Nov 24 2026)
`aeon/r5_run.sh <gpu> prep|run` runs `aeon/r5_validate.py` from the worktree via PYTHONPATH.
- `run` refuses to start unless that GPU is at <= 15% VRAM. It runs in-process (1287 s), same row position:
  - **A** stock rows in mixed batches vs the plain engine, bitwise;
  - **B** every variant mixed vs alone, bitwise;
  - **C** zero-scale residual variant == stock;
  - **D** post-edit |r.x| ~ 0 on edited rows at apps 10/17/30/47 (and stock rows bitwise);
  - **E** hot add while serving == pre-registered copy;
  - **F** consumer fold (in-engine qkvr fold for apps 0-3 and 41-45) vs hook: greedy exact, KL, logprob diff;
  - **G** greedy generations, 100 harmful + 100 benign x {stock, r_res, r_res_h, r_both, op_only}, to
    runs/r5/gens.jsonl for the refusal judge;
  - **H** 8-row decode tok/s: stock / mixed / residual / plain.

### Results (`tools/r5_results/validate.json`), all PASS

| Check | Result |
|---|---|
| A stock rows in mixed batches vs plain engine | 16/16 greedy exact, KL 0, logprob diff 0 |
| B mixed vs alone, 8 variants (stock, r_res, r_both, r_res_h, r_cons, r_47, op_only, r_zero) | 16/16 bitwise each, KL 0 |
| C zero-scale residual variant vs stock | 16/16 bitwise |
| D post-edit abs(r . x), s = 1 | app 10 (bf16) 2.61 -> 1.3e-3; app 17 (fp32, pass-1 start) 3.24 -> 6e-7; app 30 (fp32) 11.7 -> 1.4e-6; app 47 (bf16, final hidden) 18.7 -> 7.6e-3; stock rows bitwise |
| E hot add while serving | r_late == r_res bitwise at 2 row positions |
| F consumer fold vs runtime hook (apps 0-3, 41-45, folded into qkvr in-engine) | 7/16 greedy exact, KL 1.67e-3 (o_proj baked-overlay vs runtime floor: 1.52e-3); hook vs stock KL 5.65e-3 |
| G greedy generations (5 variants x 100 harmful + 100 benign, 256 tokens) | written for the refusal judge (not included here) |
| H 8-row decode tok/s (eager, required for variants) | plain 221.9, stock 222.0, residual-only 221.0 (-0.4%), mixed 219.5 (-1.1%) |

The fold-vs-hook KL sits at the bf16 weight-rounding floor measured earlier for o_proj overlays, so the
consumer fold reproduces the hook up to rounding. Folds at apps 4 and 46 (global consumer layers 5 and 23:
three input streams plus k/v weight_proj) and at app 47 (unembed) are covered by the CPU analytic test only
and are NOT yet validated on GPU.

## Offline bake tool: `tools/r5_bake.py` (CPU only)

Consumer layout in the stock checkpoint (all bf16, `weight_transform: cast`, none quantized):
- local layers (1-4, 18-22): `layers.L.attn.{wq_du [3072,2304], wr_du [384,2304], gate_attn [3072,2304],
  attn_sink_proj [24,2304], wk_dv [1024,2304], wv_dv [1024,2304]}.weight`, separate tensors (the engine
  fuses them into `attn.qkvr` at load);
- global layers (5, 23): the same six plus `attn.k_sconv.weight_proj` and `attn.v_sconv.weight_proj`
  [32,2304] (dynamic conv taps that read the k/v input streams);
- unembed: `model.llm.unembed.weight` bf16 [201024,2304], not tied to `model.llm.embed.weight`;
- o_proj: `layers.L.attn.wo_ud.weight` bf16 [2304,3072].

```
python r5_bake.py --stock /path/Inkling-2-Mini --variant v.safetensors [--variant w.safetensors] \
    (--out-overlay edits.safetensors | --out-checkpoint /new/dir) [--report report.json]
```
- o_proj edits -> wo_ud (W' = P_{K-1}..P_0 W). Residual edits at apps 0-4 / 41-46 -> the next layer's
  attention input tensors (W' = W P_{K-1}..P_0); app 47 -> unembed. fp64 math, one rounding to the stored
  dtype; original names, dtypes and shapes.
- Refused with an error: residual edits at apps 5-40; o_proj edits whose dirs/scales differ across the
  passes of a shared loop layer (apps a, a+12, a+24); quantized or packed consumers.
- `--out-overlay`: only the edited tensors; loads with `AEON_WEIGHT_OVERLAY` (matched on checkpoint names
  before the engine's renames and qkvr fusion, so unembed and attention-input tensors work).
- `--out-checkpoint`: copies the stock directory and patches the edited tensors' bytes in place (headers
  identical), so it loads in the stock architecture anywhere. The stock directory is only read; writing
  inside it is refused.
- `tools/test_r5_bake.py` (CPU): baked vs analytic fp64 W (I - s v v^T) and (I - s v v^T) W for rank 1 and 2:
  max 0.58 bf16 ulp, 99.998% bitwise; refusals; full-checkpoint byte patching; stock files unchanged;
  AEON_WEIGHT_OVERLAY replaces all 55 baked tensors (unembed, qkvr shards, k/v weight_proj, wo_ud).
  On the real stock checkpoint (read-only): r_cons 54 tensors, r_47 1, op_only 20; storage rounding leaves
  at most 0.79% of a removed component (bf16 rel err <= 0.17%); r_res is refused (apps 5-40).
