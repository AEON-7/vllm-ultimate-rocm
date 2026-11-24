# Ablation variants on the M2 kernels

This PR's patch (0007, on M1) is carried on top of the M2 speed series in
PR #4 as patch 0013, followed by patch 0014.

Why 0014 exists: the M2 decode kernels add multi-row expert paths (a
tensor-core dot path and an MXFP8 hybrid) whose reduction order depends on
how many rows of a step picked the same expert. With those paths a mixed
batch of stock + 9 variants matched same-variant batches at the same row
position in only 6-14 of 20 greedy generations per variant (stock rows
included). On M1 it was 20/20.

0014 adds `AEON_BATCH_INVARIANT`, on by default whenever `AEON_VARIANT_DIR`
is set: every routed row uses the per-row expert GEMV, and each token's
experts are summed in a fixed order. With it the mixed-batch test is back to
200/200 bitwise exact. Default serving (no `AEON_VARIANT_DIR`) is unchanged.

Variant server cost (eager, R9700, 8 concurrent): stock 213.2 tok/s,
8 different variants 211.8 (-0.7%); bs1 40.0 / 38.6. On M1 the same test
was 135.5 / 133.5. Details and raw results: PR #4,
`docs/inkling2-rocm/M2_RESULTS.md` and `patches/inkling2/tools/abl_results_m2/`.

To use the variant server on M2: apply 0001-0014 (PR #4), not this PR's 0007.
