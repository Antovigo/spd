# Parameter decomposition and Llama's pre-block RMSNorms (addsub-all-layers -05)

Date 2026-09-29. Run p-ba5a0c05 (addsub-all-layers-4xh100-05), step 40000; the "decomposed model" is the
model every arith_repr analysis uses: ceiling-filter masks at CI > 0.01, 11,604 alive components, no
delta, last-position KL to Llama 0.052 on 20,000 `a+b=` / `a-b=` prompts.
Scope, as requested: **the 62 norms in front of the attention and MLP of blocks 0-30** (norm point
t = 2l attn, 2l+1 MLP). Block 31 and the final norm are left out except where stated (they hold the
entropy/temperature components, see the earlier study in `~/out/pod-backup/p-ba5a0c05/analysis/rmsnorm/README.md`).
KL is `KL(reference || condition)` at `=`, reference = the original model unless stated.

Code lives on the `experiment/arith_representations` worktree (`param_decomp/arith_repr/rmsnorm/{model,run,robust}.py`,
uncommitted there because that worktree is shared); arrays are in `<run>/analysis/rmsnorm/`. The figure and
summary script here is `make_figures.py`. Everything is a forward-pass counterfactual: a norm's rms is replaced
by a chosen value ("forced"), or the components' write is dropped from the stream seen by later norms
("norm-only") or from the stream itself keeping its energy in the norms ("direct-only").

## Answer in five lines

1. Removing the inactive components does change the rms (dec/orig rms ratio 0.86-1.08, per-prompt sd 1-5 %,
   and 15-20 % of the stream's energy is error the readers never read), but the decomposed model **is not
   sensitive to that**: a +-10 % rms error at any single inner norm of blocks 0-23 costs < 5e-4 KL (original
   Llama: 3e-4 to 3e-2), and 10 % iid noise on all 62 norms costs 0.05 KL (original: 0.58).
2. There are **no "RMS-keeper" components** in blocks 0-30: 89 % of the live components change the output by
   < 1e-4 KL when removed; only 24 have any norm-only effect > 1e-4 and none is norm-dominated at > 0.01
   (the biggest, L1.down.c0, is the BOS massive-activation writer whose direct and norm effects cancel).
3. Where the norm does matter is a **co-adaptation of late writers to the decomposed model's own smaller
   rms** (blocks 24-30 MLP norms): forcing the original's rms there costs +0.16 KL, forcing it everywhere
   (blocks 0-30) costs 1.46. So hypothesis 3 (a frozen / original norm would reconstruct better) is false; the
   opposite holds.
4. Freezing norms cannot be used as a counterfactual for the decomposed model: removing a whole block's
   active components costs 0.002-0.08 KL with free norms and 0.2-7 with frozen ones. The norm is a
   negative-feedback loop that repairs the ablation.
5. The robustness is partly a property of the masked forward, partly of Llama. Caveat: masks here are
   **static** (from the clean run); the real CI function would react to an rms change too, and that was not tested.

## 1. How different is the rms after removing the inactive components?

`fig1_scale.png`. Notation at `=`: decomposed stream x_d = alpha x_o + perp, r = rms_d / rms_o, so
r^2 = alpha^2 + perp^2 (perp relative to |x_o|).

![fig1](figures/fig1_scale.png)
*Fig. 1. Left: rms ratio, alpha and their quotient at `=`. Right: unread error norm; it equals sqrt(r^2 - alpha^2).*

| blocks | rms ratio r | alpha | perp | cos(x_d, x_o) |
|---|---|---|---|---|
| 0-3 | 0.95-1.01 | 0.94-1.00 | 0.14-0.22 | 0.975-0.99 |
| 10-15 | 0.98-1.08 | 0.93-0.99 | 0.28-0.44 | 0.95 |
| 20-25 | 0.94-1.06 | 0.87-0.99 | 0.36-0.44 | 0.92-0.94 |
| 28-30 | 0.86-0.91 | 0.78-0.84 | 0.33-0.36 | 0.91-0.93 |

* The unread error (`perp`, 15-45 % of the stream's norm) is what inflates the rms; the shrinkage of the
  active part (alpha 0.78-1.0) deflates it. The two partly cancel, leaving r within 6 % of 1 until block 25 and
  0.86-0.91 at blocks 28-30.
* Per-prompt, log alpha tracks log r (slope 0.85-1.27, corr 0.65-0.98). Mostly arithmetic (perp is roughly
  constant, so r and alpha co-move); it is not evidence of an active compensation by itself.
* The error energy is diffuse: at blocks 3-15 the top-8 residual dimensions carry 4-9 % of the original's energy
  and 5-6 % of the error's; from block ~20 a few massive dimensions dominate the original stream (dim 2352 alone: 11 %
  at block 20, 33-35 % at blocks 25-30) and the error avoids them (5-23 % of the error in the top 8).
  So the rms is a collective statistic of thousands of dimensions early, and of a couple of outlier dimensions late.
* What the active readers see after the norm: relative squared error of their inner activations
  |H_d - H_o|^2 / |H_o|^2 = 0.1 % (blocks 0-7), 0.4-0.8 % (8-23), 1.0-1.1 % (24-30) with their own norms, and about
  the same (0.1-1.5 %) if the ORIGINAL rms is used instead. Energy ratio |H_d|^2/|H_o|^2: 1.02-1.03 early,
  0.97-1.04 mid, 1.09-1.18 late with own rms, 0.88 (MLP) / 1.03 (attn) late with the original's.

## 2. Test of the three hypotheses (+ two more)

**H1: some components exist only to hold the rms in range.** Rejected for blocks 0-30
(Fig. 4; 10,599 live components on 500 prompts, `split/`, `ablate/`).

| | sum of ablation KL | direct-only | norm-only |
|---|---|---|---|
| blocks 0-30 | 2.03 | 2.28 | 0.26 |

![fig4](figures/fig4_direct_vs_norm.png)
*Fig. 4. Each dot is a live component: direct-only vs norm-only KL. Almost all lie below the diagonal and below 1e-3; none reaches the norm-dominated corner at a size that matters (floor at 1e-7).*

* 89 % of the live components: ablation KL < 1e-4. Components with either split effect > 1e-3: 252; among them
  norm-dominated (norm-only > direct-only) 9; at > 1e-2: 1 of 24.
* The norm-dominated ones are small (norm-only 0.001-0.010): L1.down.c0 (BOS massive-activation writer, ablation
  itself 2e-4, its direct and norm effects cancel), L2.q.c26, L3.q.c1, L3.k.c0, L24.down.c0, L22.up.c12,
  L24.gate.c586, L25.o.c394, L26.o.c19. * The norm route is ~11 % of the direct route overall (norm-only / direct-only per block group: L0 0.002/0.010, L1 0.014/0.010,
  L2-7 0.017/0.070, L8-15 0.010/0.252, L16-23 0.055/0.388, L24-30 0.161/1.549). In block 31 it dominates (norm-only 2.12 of
  2.38 total), the reason it is excluded here.
* So the model does not need "energy-only" components: dropping rank-one energy from the stream is
  never what a single component is needed for. A collective role of many tiny components is not excluded by a
  per-component test, but it is excluded by section 3 (whole-block removal is compensated by the free norms).

**H2: ablating inactive components moves the rms and magnitudes, and it does not matter downstream (a property of the model).**
Largely supported, and stronger for the decomposed forward than for Llama itself.

* One norm's rms scaled by 0.9 or 1.1, KL against the same model's clean run (Fig. 2, 1000 prompts):
  original Llama median 2-4e-3, max 0.04 (block 1-2 norms, embedding norm, block 30 MLP); decomposed median 2e-5, and
  > 0.01 only at block 30 MLP (0.049), block 29 MLP (0.007). Blocks 0-23: < 5e-4 everywhere.
![fig2](figures/fig2_single_norm.png)
*Fig. 2. Sensitivity of each norm to a +-10 % rms error, both models.*

* Random rms noise on all 62 norms (lognormal sigma), 2000 prompts (Fig. 3):

| sigma | original, every token | original, `=` only | decomposed, every token | decomposed, `=` only |
|---|---|---|---|---|
| 0.03 | 0.041 | 0.017 | 0.0043 | 0.0042 |
| 0.10 | 0.581 | 0.215 | 0.051 | 0.049 |
| 0.20 | 2.22 | 1.18 | 0.196 | 0.186 |

![fig3](figures/fig3_noise.png)
*Fig. 3. KL vs own clean run under log-normal rms noise on the 62 inner norms.*

  The rms mismatch actually observed (sd 1-5 %, mean offset up to -14 %) is inside the flat region of the decomposed
  model. In Llama itself, 10 % noise is not free, which fits "the norm is a stabiliser" (section 3) rather than
  "activations are magnitude-invariant".
* Uniform scaling of all norms (0.9/0.95/1.05/1.1) costs the decomposed model 0.116/0.027/0.022/0.078 and the original
  0.487/0.072/0.030/0.078: symmetric-ish and small around 1.
* Why the decomposed forward is more robust than the original (hypothesis, only partly tested): its readers are
  the ~10^4 alive directions with binary masks fixed in advance, so a rms error scales a handful of read
  coefficients but cannot switch on the many small directions, SiLU thresholds and attention pattern shifts that a
  dense reader feels. Not isolated: it needs the CI function in the loop (below).

**H3: the reconstruction would be better with a frozen / original RMSNorm.** Rejected.

* Decomposed model forced to the original's per-prompt rms (`robust_forced`, 10,000 prompts, reference KL 0.0513):

| forced norms | all positions | only `=` |
|---|---|---|
| blocks 0-7 | 0.054 | 0.052 |
| blocks 8-15 | 0.055 | 0.052 |
| blocks 16-23 | 0.053 | 0.052 |
| blocks 24-30 | **0.214** | 0.206 |
| blocks 0-30 | **1.51** (acc 0.02) | 0.59 (acc 0.27) |

  Any window of blocks 0-23 is harmless, blocks 24-30 hurt, all together break the model, mostly through other positions.
  The original's rms is not the target the decomposed model needs; its own is.
* All 65 norms frozen at the pool mean: NaN (own means) / KL 5.0 (original's means). Freezing all norms of Llama itself gives 15 % NaN prompts.
![fig7](figures/fig7_hybrid_forced.png)
*Fig. 7. Original Llama with one block decomposed: forcing the original rms back never lowers the median KL.*

* Original model with ONE block decomposed (`hybrid.npz`): KL 0.009-0.042 free (medians 0.008-0.026), and forcing the original
  rms back RAISES the median KL in every block 5-30 (+0.0016..+0.020), never helping (blocks 0-4 also blow up on a few prompts). The single-block substitution moves later rms by ~1 % (|dlog rms| 0.006-0.023).

**H4 (added): late writers are calibrated to the decomposed model's own smaller rms.** Supported for blocks 24-30 MLP norms.
The stream projection alpha falls to 0.78-0.85 and rms_d/rms_o to 0.86-0.91 in the same blocks; with the original's rms the
reads of MLP blocks 24-30 are 12 % too small in energy (0.88) while with its own they are 9 % too large (1.09), and forcing the
original's rms at just L30 MLP costs 0.106 vs 0.052. The training loss ran with the norm in the loop, so this is a
decomposition-side adaptation, not a property of the mapping being copied from the original.

**H5 (added): the norm repairs the removal (self-normalisation).** Whole-block ablation of all alive components
(`groups.npz`, 2000 prompts), free vs frozen norms:

![fig6](figures/fig6_block_ablation_free_frozen.png)
*Fig. 6. All alive components of one block ablated, norms free vs frozen at the unablated rms (2000 prompts).*

| block | free | frozen |
|---|---|---|
| 0 | 0.009 | 6.99 |
| 3 | 0.002 | 5.96 |
| 9 | 0.003 | 1.44 |
| 15 | 0.077 | 2.61 |
| 18 | 0.041 | 1.11 |
| 24 | 0.016 | 0.24 |
| 30 | 0.239 | 0.48 |

The later rms moves by 5-17 % (mean |dlog rms| at `=`) and the free norms rescale everything downstream: the removal
of energy is absorbed, and freezing the norms removes that repair.

**H6 (added): the rms mismatch sits mostly at other positions.** `_eq` vs all positions above (0.59 vs 1.51 for blocks 0-30):
BOS/early positions have massive activations (rms 7.7 vs 0.11 at `=`, block 10) and their K/V feed the last position.
Original-model numbers agree (noise only at `=`: KL a third to a half of the all-position noise).

## 3. Limits

* Masks are static (thresholded CI of the clean run). In the real decomposed model the CI network reads the stream and
  would respond to an rms change; the robustness above is that of the masked forward, an upper bound.
* No separate "output-only replacement" was run: the alive set is the dataset's single ceiling-filter set. The hidden-only
  vs output-causal split of the -05 components (the dual objective) is not exploited; the question "does dropping the
  hidden-only components change the rms differently" needs the hidden CI of the same run and is still open.
* One decomposition (-05), one task, one position (`=`, all positions for the forcing tests); KL means over 1000-10,000 prompts
  with a fixed subset per test, the split/ablate per-component numbers over 500 prompts (per-component KL below ~1e-4 is noise).
* alpha-vs-r correlations are partly arithmetic (see section 1); H4 rests on the reads energy ratios and the forced-window tests, not on it.

## 4. Suggested follow-ups

1. Put the CI function in the loop (torch port at `attn_alive/ci_torch.py`, or JAX eval): perturb the rms, recompute masks,
   measure KL. Decides how much of section 2 is sparsity-by-masking.
2. Hidden-only components: replace the model with the output-role alive set only and recompute rms ratios / forced-window tests;
   compare with the -05 outputs-only twin (addsub-all-layers-4xh100-05-outputs-only).
3. Test H4 causally: rescale the active writers of blocks 24-30 by 1/r so that the original's rms becomes the right target,
   and see whether forced-original stops hurting.
