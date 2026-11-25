# R6-1: all-readers residual fold (aeon-r6)

Status log (center time CST; ET = CST + 1)
- Branch aeon-r6 off aeon-r5 (3a363b4e9) in a NEW worktree wt-r6; wt-r5 checkout untouched.
- tools/inkling2/r5_bake.py: --readers all [--gain-mode oblique|plain] [--fold-queries];
  default --readers consumer is byte-identical to the R5 tool (tested).
- tools/inkling2/test_r6_bake.py: CPU unit test ALL PASS (see report).
- aeon/r6_validate.py + aeon/r6_run.sh: prep (CPU) done -> runs/r6/{variants,overlay_*.safetensors,bake_reports.json};
  run on GPU1 started.

Key facts
- R5 hook is LOCAL: edits only the attention-input mix of visit a+1 (final hidden at 47). The MLP-input mix,
  block stack and all later mixes are untouched. --readers all is a different (global, reader-side) edit.
- Round-6 named variants conv_reason_blend_l6-23_s1.25 / conv_lsv1_l4-23_s1.25 are o_proj-only (dirs/scales,
  rank 5, same dirs every pass). Their wo_ud fold is exact math (hook acts on wo_ud output before attn_sconv);
  any bake-vs-hook mismatch for them is bf16 storage rounding, not missing readers.
- Controllers exist only on loop layers 6-17: none read the stream after app 41. down_weight [64, 2*D] reads
  RMS-normed [dest, message] (no gain); forget/gain [D,64] are output-side gates, not stream readers.

## Results (GPU1 run 11:16-11:24 CST, 476 s; GPU1 back to 0% VRAM after)
Commit 0714f860c on aeon-r6 (wt-r6): tools/inkling2/{r5_bake.py,test_r6_bake.py,test_r5_bake.py,r6_validate.py,r6_run.sh}.
16 prompts, greedy, 64 tokens, row 0 of a 4-row batch, same process. S0: stock round trip through the in-engine swap
is bitwise (91 params), router weight_f32 refreshed; stock before vs after all swaps 16/16, hook repeat 16/16.
| comparison | exact | KL mean |
| n_reason r5 bake vs hook | 7/16 | 1.91e-3 |
| n_reason r5 bake vs fp32-math bake (rounding floor) | 7/16 | 1.59e-3 |
| n_lsv1 r5 bake vs hook | 6/16 | 2.45e-3 |
| n_lsv1 r5 bake vs fp32-math bake (floor) | 3/16 | 2.72e-3 |
| n_reason_r r5 bake vs hook | 7/16 | 1.76e-3 |
| n_reason_r r6 all oblique vs hook | 1/16 | 6.97e-3 |
| n_reason_r r6 all plain vs hook | 2/16 | 7.93e-3 |
| n_reason_r r6 all + queries vs hook | 1/16 | 8.05e-3 |
| n_reason_rx ext fold vs ext hook (bake correctness) | 7/16 | 1.20e-3 |
| ext hook vs local hook (semantic gap) | 2/16 | 7.78e-3 |
| hook vs stock (effect), n_reason / n_lsv1 / n_reason_r | 0/16 | 0.125 / 0.105 / 0.084 |
Expert inputs (not folded): per-layer mean ||y - E y|| / ||y|| on the MLP-input mix 0.072 (L18) .. 0.198 (L22), 0.171 (L23);
within layers 18-23, 30.5% of the all-readers edit energy is on the MLP mix (router folded, experts not).
