# Final-logit magnitude: dual vs output-only (addsub-L18-23-neuronaligned pair), target stream

Question: both decompositions reproduce the model's output distribution; do they differ in the
pre-softmax logits or the pre-RMSNorm residual in ways the norm/softmax erase?

Protocol (same as `emulate_decomp.py`): frozen Llama-3.1-8B in fp32 on CPU, "a+b=" grid 1..100,
layer 18 replaced at "=" only by the CI-masked, delta-off component forward (checkpoint 20000 +
step-20000 ab-grid CI; unsaved components CI 0). "all comps on" = every component at CI 1.
Job 11122 (2026-09-07). Raw data (`logitmag/*.npz`) and scripts (`logit_mag.py`, `logit_mag_plots.py`,
`logit_mag.sbatch`) live in `~/pd_scratch/dual_obj_jax/attn_alive/`; figures are copied to
`notes/plots/logit_magnitude/` (fig 1-7, `table.json`).

## Answer: no. The logits differ by tiny, structureless amounts in both runs.

Median over 10 000 prompts (logit std over the vocab is 2.34):

| condition | KL | acc | logit offset | slope | rms logit diff | of which shape | ‖final resid‖ / target | cos(final resid) |
|---|---|---|---|---|---|---|---|---|
| dual, output CI | 0.0065 | 0.947 | -0.000 | 0.998 | 0.13 | 0.12 | 0.992 | 0.998 |
| dual, hidden CI | 0.0039 | 0.951 | -0.004 | 0.999 | 0.08 | 0.07 | 0.997 | 0.999 |
| output-only, output CI | 0.0087 | 0.938 | -0.012 | 0.997 | 0.17 | 0.16 | 1.011 | 0.997 |
| dual, all comps on | 0.0018 | 0.949 | -0.003 | 0.999 | 0.07 | 0.06 | 0.998 | 1.000 |
| output-only, all comps on | 0.0037 | 0.946 | +0.004 | 0.999 | 0.13 | 0.12 | 1.015 | 0.998 |

- No hidden constant offset (softmax-invisible) and no temperature/scale change: per-prompt mean
  shift is within ±0.2 logits and the regression slope within 0.98-1.02 in every condition
  (fig 3, fig 5). The rms logit difference is 90% "shape" (per-token, not affine), i.e. the
  same error the KL already sees. Logit std, max logit, logsumexp, answer logit, margin and
  entropy all match the target to ~0.1 (fig 3); scatter plots sit on the identity (fig 4).
- The only magnitude effect lives in the RESIDUAL, and it is small: the output-only forward
  runs the residual norm 5.5% LOW right after layers 19-20 (dual: 3%; dual under hidden CI:
  <1%), then recovers and overshoots to +1.1% at the final RMSNorm input, while the dual run
  ends 0.8% low (fig 1, 2). The final RMSNorm removes all of this (post-norm norm 131.15-131.21
  in every condition vs 131.19 target). Direction is preserved (cos 0.997-0.999).
- The layer-18 deficit is magnitude-structured (fig 7): the output-only norm shortfall after
  layer 19 is worst for small a+b (0.90 at a+b < 25) and vanishes at a+b > 150; the dual's is
  flatter. The final-residual excess of the output-only run is largest for mid-range sums
  (a+b 50-175, +2%). None of this shows in the logits.
- Where the shrinkage comes from: at "=", the layer-18 attention output is reproduced at 65%
  of its norm by the dual (output CI; 84% hidden CI) and 48% by the output-only run (fig 2,
  right — one channel, o#73, see report_attn_alive.md); the MLP-18 output at 84% (dual, output
  CI), 94% (dual, hidden CI) and 93% (output-only). The output-only MLP output has the right
  size but the wrong direction (76% relative MSE), the dual's the right direction but is
  under-sized under the output CI.
- Even with ALL components on (no CI masking, delta off) the output-only run keeps a +1.5%
  final residual excess and a 3% layer-19 deficit; the dual is within 0.2% at the end and
  <0.5% everywhere. This is the sum-of-components-minus-delta faithfulness at "=", not a
  CI/sparsity effect.

So: the differences between the two decompositions are not in logit magnitude. They are in
the residual stream between layers 18 and ~24 (a few percent of norm, magnitude-structured),
which the final RMSNorm erases, and in the per-token logit shape, which the KL already
measures (output-only 0.0087 vs dual 0.0065 / 0.0039).

## Figures

Fig 1: Residual-stream norm at '=' per layer, relative to the target model

![Residual-stream norm at '=' per layer, relative to the target model](plots/logit_magnitude/fig1_resid_norm_layers.png)

Fig 2: Final residual norm ratio, direction, and layer-18 write norms

![Final residual norm ratio, direction, and layer-18 write norms](plots/logit_magnitude/fig2_final_resid.png)

Fig 3: Per-prompt logit statistics, decomposed minus target

![Per-prompt logit statistics, decomposed minus target](plots/logit_magnitude/fig3_logit_stats.png)

Fig 4: Decomposed vs target logits, full vocab

![Decomposed vs target logits, full vocab](plots/logit_magnitude/fig4_logit_scatter.png)

Fig 5: Logit difference split into offset / scale / shape

![Logit difference split into offset / scale / shape](plots/logit_magnitude/fig5_diff_decomposition.png)

Fig 6: (a, b) grids of final residual norm ratio and mean logit shift

![(a, b) grids of final residual norm ratio and mean logit shift](plots/logit_magnitude/fig6_grids.png)

Fig 7: Norm ratios and KL by a+b

![Norm ratios and KL by a+b](plots/logit_magnitude/fig7_by_sum.png)
