# Per-request multi-variant ablation serving (patch 0007)

Original AEON work (Apache-2.0). Clean room: written from the AEON M1 engine,
the PR #1 spec docs, M1_RESULTS.md and upstream vLLM only; nothing from
vllm-redshift was read.

One vLLM server holds a bank of registered o_proj ablation *variants* and
applies the right one to every token of every request in a mixed batch. An
abliteration screen (many edit strengths / layer sets / directions) runs as
one stream of requests against one server, with no restart or weight reload
per variant.

## What a variant is

Inkling-2-Mini has 48 layer applications ("apps"): layers 0-5 (apps 0-5),
looped layers 6-17 three times (app = 6 + 12 * pass + (layer - 6), apps
6-41), layers 18-23 (apps 42-47). A variant gives, per app a, K unit
directions r[a, k] and scales s[a, k]. Right after attention o_proj (wo_ud)
of app a, every token row y of a request using the variant becomes

    for k in 0..K-1:   y <- y - s[a, k] * r[a, k] * (r[a, k] . y)

which is the runtime form of W' = (I - s_{K-1} r r^T) ... (I - s_0 r r^T) W.
K = 1 is exactly the AEON_ABLATE / baked-overlay edit W' = (I - s r r^T) W,
but with a separate scale per loop pass (e.g. 0.6x in the loop) and a
separate direction per app if wanted. Rank K > 1 applies the projections in
order (equal to removing a subspace when the directions are orthonormal and
s = 1).

Variant file: `$AEON_VARIANT_DIR/<name>.safetensors` with

| key | shape | meaning |
|-----|-------|---------|
| `dirs` | `[2304]` | one direction for every app |
| | `[24, 2304]` | one direction per layer (all passes of a looped layer share it) |
| | `[48, 2304]` | one direction per app |
| | `[48, K, 2304]` | rank K per app |
| `scales` | `[48]` or `[48, K]` | float; 0 = app untouched |

Directions are unit-normalised on load. Names match
`[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`; `stock` (variant 0) is the unedited model.

## Running

```
export AEON_VARIANT_DIR=/path/to/variants      # turns the feature on
vllm serve ~/inkling/models/Inkling-2-Mini --enforce-eager \
    --max-model-len 4096 --max-num-seqs 16 --max-logprobs 1000
```

Eager mode only for now: the worker refuses to start a step with any
cudagraph mode while AEON_VARIANT_DIR is set (a captured graph would silently
bake in "no variant"). Without AEON_VARIANT_DIR nothing changes.

### HTTP API (mounted only when AEON_VARIANT_DIR is set)

```
GET    /aeon/variants                 -> {"variants": [{name, rank, hidden, nonzero_apps, salt}, ...]}
POST   /aeon/variants                 register or replace
       {"name": "s06_l4_23", "dirs": [[...2304]...], "scales": [...48]}
       {"name": "s06_l4_23", "path": "/server/side/file.safetensors"}
       optional "replace": false -> 400 if the name exists
DELETE /aeon/variants/{name}          -> 404 if missing
```

Writing a file into AEON_VARIANT_DIR directly works too (atomic rename
recommended). The engine rescans the directory at most once per
`AEON_VARIANT_RESCAN_S` seconds (default 1.0) and at once when a request
names a variant it has not loaded.

### Selecting a variant per request

OpenAI chat or completions body: top-level `"aeon_variant": "<name>"`
(`extra_body` in the OpenAI Python client), or `"vllm_xargs":
{"aeon_variant": "<name>"}`. Omitted or `"stock"` = unedited. Unknown or
malformed names get HTTP 400. The API folds the variant name and a file
stamp into the prefix-cache salt, so KV computed under one variant (or an
older version of the same name) is never reused by another.

Offline `LLM`: `SamplingParams(extra_args={"aeon_variant": "<name>"})`.
With prefix caching on, also pass a per-variant `cache_salt` in the prompt
dict (or run with `enable_prefix_caching=False`).

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
r = c.chat.completions.create(
    model="inkling", messages=[{"role": "user", "content": prompt}],
    temperature=0, max_tokens=512, logprobs=True, top_logprobs=20,
    extra_body={"aeon_variant": "s06_l4_23",
                "chat_template_kwargs": {"reasoning_effort": "none"}})
```

## How it works

- `vllm/models/inkling2/amd/aeon_variants.py`: variant files, worker bank
  (fp32 `[slots, 48, K, 2304]` directions and `[slots, 48, K]` scales on
  the GPU, slot 0 = stock), per-step token map, Triton kernel, API helpers.
- `model_runner.py` (2 small hooks): on request add, remember the request's
  variant; after `prepare_attn`, build the per-token slot map from the batch
  order and `query_start_loc` (one small pinned H2D copy) and the set of
  apps any variant in the step edits.
- After o_proj of app a, if any row in the step uses a variant that edits a,
  one Triton launch (one program per token row) loads the row's slot and
  applies its K removals in fp32, writing the row back in bf16 in place.
  Stock rows return immediately and are never written. Steps with no variant
  rows launch nothing, so stock-only traffic runs the M1 code path.
- `aeon_variants_api.py` + `routers.py`: the HTTP endpoints; both OpenAI
  `protocol.py` files: the `aeon_variant` field, registry check, cache salt.

## Validation (one R9700, gfx1201, eager)

Scripts and raw results: `patches/inkling2/tools/abl_*`.
Directions: AEON stock refusal directions (effort none, last token, per app).
Test variants: `v_eq` (per-layer dirs, s = 1 on apps 4-47, overlay
expressible), `v_pass` (per-app dirs, loop passes at 1.0 / 0.6 / 0.3),
`v_s0..v_s4` (s = 0.25 ... 1.5), `v_r2`, `v_r3` (rank 2 and 3).
Prompts: 8 harmful + 8 benign eval prompts, greedy 96 tokens with top-50
logprobs, plus 16 teacher-forced sequences from the stock KL set.

Note on engine noise: this engine is not bit-reproducible across processes
(Triton autotune; plain M1 vs plain M1 in two processes: teacher-forced KL
1.5e-3, 3/16 greedy runs identical) and not row-position invariant
(identical requests in different batch rows differ: 30/180 identical over
64 tokens). Exactness checks therefore run in one process with the hooks
switched on the live model, and mixed-batch checks compare the same row
position.

1. **Stock through the variant server = plain M1.** Same process, same
   batch: 16/16 greedy sequences identical, teacher-forced logprobs
   identical (KL 0). Swapping in the baked overlay and restoring it also
   returns exactly to stock.
2. **Per-request variant = server-wide AEON_ABLATE = baked overlay.**
   - First edited application (app 4, identical input on every path):
     request path is bitwise equal to an fp32 torch reference of the
     formula; 93.6% of values bitwise equal to AEON_ABLATE (which does the
     math in bf16) and 54.5% to the baked overlay (weight rounding);
     remaining |r . y| 0.56 stock -> 0.0010 request, 0.0017 AEON_ABLATE,
     0.0007 overlay.
   - End to end, teacher-forced KL: request vs AEON_ABLATE 1.43e-3,
     request vs overlay 1.52e-3, while the two existing references differ
     from each other by 1.64e-3; per-pass `v_pass` vs AEON_ABLATE 1.65e-3.
     The edit itself moves the model by 1.5e-2 to 1.8e-2 (10x larger).
     So the three forms agree to the engine's rounding-amplification floor;
     they are not bit-identical because each rounds differently.
3. **Mixed batches.** 10 variants (stock + 9, including rank 2 and 3 and
   per-pass scales) in one batch, rotated so every variant sits at every row
   position, against a batch of 10 copies of the same variant at the same
   row: 200/200 greedy sequences identical with identical logprobs (KL 0).
   Kernel level: a mixed launch is bitwise equal to per-variant launches.
   Over HTTP (prefix caching on) a 12-request burst over 3 variants matched
   the single-request texts only up to the same row-position noise that a
   stock-only burst shows.
4. **Throughput** (8 concurrent, 8 distinct prompts, 256 new tokens,
   median of 3, same process):

   | traffic | tok/s |
   |---------|-------|
   | all stock | 135.5 |
   | 8 different variants | 133.5 (-1.5%) |
   | 1 variant x 8 | 133.2 (-1.7%) |
   | half stock, 4 variants | 133.7 (-1.3%) |
   | bs1 stock / variant | 19.8 / 19.5 |

HTTP API (server, prefix caching on): register by path and by JSON, list,
top-level and vllm_xargs selection agree, `stock` = omitted, unknown name /
bad name / bad shape -> 400, delete -> 200, request for a deleted variant ->
400, delete missing -> 404.

## Screening sweep recipe (20 variants x 100 harmful + 100 benign)

1. Register 20 variants (POST, or write 20 files) - seconds, no restart.
2. Send 20 x 200 = 4000 chat requests, each with its `aeon_variant`,
   capped `max_tokens`, `temperature=0`; keep 8-16 in flight (one GPU) and
   let the scheduler mix variants freely. Interleave variants rather than
   sending one variant at a time, so a slow variant does not idle the batch.
3. For KL on the benign set, send prompt + stock completion as one
   teacher-forced prompt with `prompt_logprobs` (completions API,
   `max_tokens=1`) under each variant: 20 x 100 more requests, prefill
   only, compared against the same request under `stock`.
4. Judge answers (refusal / coherence / loops) offline as before.
   At ~135 tok/s for 8 concurrent (M1 speed) the cost is the tokens
   generated, not 20 model reloads; M2 kernel speedups apply unchanged.

## Limits

- Eager only; cudagraph/graph mode refused at the first step (stretch goal:
  persistent bank and always-launch kernel inside graphs).
- Not wired into speculative decoding (DSpark) token maps; TP = 1 tested.
- Edits only the o_proj output (wo_ud); experts, residual and other weights
  are untouched by design.
- Replacing a variant while its requests are in flight changes their edit
  from the next step; new requests get a new cache salt.
- Directions are stored in fp32 on the GPU: 48 x K x 2304 x 4 B = 0.44 MB
  per slot per rank; the bank grows in powers of two (K = max rank seen).
