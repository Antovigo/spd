# Layer 30 in a 32-block decomposition vs on its own: why the deep run keeps 2-3x more always-on components

2026-09-15. Runs compared: `addsub-all-layers-4xh100-04` (`p-ba4a0c04`, all 32 blocks
decomposed, checkpoint + ab-grid at step 16 000) and `addsub-all-layers-04-block30`
(`p-46ace901`, the same recipe restricted to block 30, checkpoint + ab-grid at step 8 000; still
training). Same neuron-aligned init, same per-point hidden/output coefficients, same imp-min and
frequency schedule; the only differences are the decomposed span (32 blocks vs 1), the mesh, and
the batch-per-rank.

Protocol for everything below: frozen HF Llama-3.1-8B in fp32 on CPU, "a+b=" grid, layer 30's
seven sites replaced by one run's components under a mask, layers 0-29 and 31 frozen (except §5).
CI is the torch port of the chunkwise CI fn on the clean taps, as in training. Scripts and raw
outputs live in `~/pd_scratch/dual_obj_jax/last_layers/` (`emu.py`, `static_compare.py`,
`directions.py`, `inner_grids.py`, `site_transform.py`, `early_recruit_plots.py`,
`followup.py`; results in `emu_clean/`, `emu_up/`, `followup.json`). Figures are in
`notes/plots/last_layers/`. Unless stated otherwise the emulation uses the 50x50 sub-grid
(a, b odd, 2 500 prompts); the grid-snapshot statistics use all 10 000.

## Summary

1. **The extra always-on components are extra pieces of the same write.** All layer-30
   always-on components are bias-like: their inner activation over the grid has mean 2-15 and sd
   below 1. Summed, the all-layers set (10 gate / 7 up / 7 down) and the block30 set (3 / 3 / 3)
   produce the same constant vector at "=" (cosine 0.79 gate, 0.77 up, 0.97 down, 0.95 at the
   MLP output), with the same 25-45 % residual against the clean output. The block30 run packed
   the vector into 3 denser pieces; the deep run kept it split over ~10 neuron-aligned ones.
2. **In the clean-upstream setting the block30 solution strictly dominates.** Layer 30
   CI-masked, everything else frozen: KL at "=" 0.0108 (block30) vs 0.0278 (all-layers), with
   half the imp-min and frequency penalty. Even with every component on and no delta the deep
   run's layer-30 factorisation is worse (0.0045 vs 0.0263): the deficit is in V·U, not in the CI.
3. **It does not collapse because imp-min never touches V/U and cannot lower a load-bearing
   CI.** Each extra piece costs 4e-3 to 0.5 nats when ablated, against a 4e-4 (KL-equivalent)
   penalty saving; every always-on piece sits at CI preactivation 1.3-2.1, where the mask squash
   passes no gradient and the imp-min leak passes 0.01x. Merging would need the surviving pieces
   to absorb the others' function *before* their CI drops, which nothing rewards.
4. **The pieces were recruited in the first 2 000 steps, when the last layers were the cheapest
   fix for a chaotic output, and then parked.** Layer-30/31 L0 was 460 at step 500 against a
   median of 76 for layers 2-29; every extra piece is already at CI 1 in the step-4000 snapshot.
   Components that are always-on at 4k either stay at 1 or die (bimodal), in every layer group.

## 1. What the always-on components are

Per-site counts at "=" from the grid snapshots (mean output-role CI over 10 000 prompts):

| site | C | all-layers @16k alive (≥0.05) / ≥0.9 | block30 @8k alive / ≥0.9 | block30 @12k alive / ≥0.9 |
|---|---|---|---|---|
| gate | 912 | 45 / 10 | 33 / 3 | 36 / 3 |
| up | 912 | 37 / 7 | 19 / 3 | 27 / 3 |
| down | 1024 | 37 / 7 | 33 / 3 | 40 / 3 |
| o | 512 | 2 / 2 | 1 / 0 | 2 / 0 |
| q, k, v | | 1 / 1, 1 / 0, 0 | 1 / 1, 1 / 0, 0 | 1 / 1, 1 / 0, 1 / 0 |

`fig1_mean_ci_sorted.png`: the mean CI of each site's components sorted; the deep run has a
plateau of ~10 at CI 1, the block30 run a plateau of 3 followed by the same tail of partially-on
components. `fig2_alwayson_union.png`: same indices in both runs (same init). The block30 set is a
strict subset of the all-layers set: gate {3, 43, 103} ⊂ {1, 3, 6, 15, 17, 21, 27, 43, 103,
299}; up {3, 47, 52} ⊂ {1, 3, 15, 21, 43, 47, 52}; down {1, 3, 52} ⊂ {1, 3, 17, 21, 27, 43, 52}.
The extras have mean CI 0.00-0.08 in the block30 run (gate c1 is the exception at 0.36-0.43).
Each component still carries its init neuron as its top neuron (11272, 3382, 6764, 6209, 12185,
10073, 11976, 11430, 11920, 2762 for gate), but with only 10-40 % of its energy there; block30's
three spread half their energy over 450-1 200 neurons, the all-layers pieces over 430-900.

**They are bias-like.** `fig10_inner_grids.png` shows x·V_c/|V_c| over the 100x100 grid for all
always-on components: means 2 to 15 in magnitude, sd 0.2 to 1.0, with a faint diagonal (a+b)
modulation. The CI-fn preactivation at "=" is 1.3-2.1 for every one of them (block30: 1.1-1.4),
i.e. all are in the leaky region above 1.

## 2. Do the two sets do the same thing when summed?

Yes. Site output of the always-on set only, y_on = (x V_A) U_A at "=", 2 500 prompts
(`site_transform_2500.txt`):

| site | cos(all-layers sum, block30 sum) | cos to clean y (all / b30) | rel. sq. err vs clean (all / b30) | ‖sum‖ all / b30 / clean |
|---|---|---|---|---|
| gate | 0.79 | 0.74 / 0.74 | 0.46 / 0.45 | 78 / 84 / 117 |
| up | 0.77 | 0.68 / 0.70 | 0.53 / 0.54 | 65 / 76 / 89 |
| down | 0.97 | 0.88 / 0.88 | 0.23 / 0.23 | 21 / 21 / 25 |
| MLP output | 0.95 | 0.87 / 0.87 | 0.24 / 0.28 | 21 / 17 / 25 |

The sum is essentially one constant vector per site (rms deviation over prompts 4-6 against a
mean norm of 65-84 at gate/up). Per-piece constant contributions (norms) at gate: all-layers
c299 32, c15 27, c6 19, c27 14, c17 13, c103 13, c43 10, c3 7, c1 3, c21 3; block30 c103 58,
c43 37, c3 8. So the deep run's extras are genuine parts of the sum, not cancelling pairs, and
block30's pieces are 2-5x larger. `fig6_alwayson_function.png` gives the per-prompt cosine and
error distributions.

The read/write directions themselves have drifted apart: same-index components have cosine
0.2-0.7 between runs, and block30's always-on V and U lie only 25-60 % inside the span of the
all-layers always-on set (`directions.txt`). The block30 run re-oriented its three pieces; it did
not keep three of the ten.

## 3. Which set gives the better loss?

Layer 30 replaced by each run's components, clean upstream, 2 500 prompts
(`emu_clean/results.json`, `fig3_kl_by_mask.png`):

| mask | KL at "=" all-layers / block30 | KL mean over 5 positions | MLP-30 rel. sq. err at "=" |
|---|---|---|---|
| CI (output head) | 0.0278 / 0.0108 | 0.0163 / 0.0063 | 0.133 / 0.102 |
| CI (hidden head) | 0.0284 / 0.0103 | 0.0165 / 0.0061 | 0.129 / 0.098 |
| rounded CI ≥ 0.5 | 0.0282 / 0.0113 | 0.0167 / 0.0066 | 0.133 / 0.103 |
| always-on set only | 0.0951 / 0.1235 | 0.294 / 0.307 | 0.243 / 0.281 |
| alive set (≥0.05) only | 0.0459 / 0.0525 | 0.258 / 0.293 | 0.157 / 0.145 |
| all components on, no delta | 0.0263 / 0.0045 | 0.0148 / 0.0023 | 0.119 / 0.082 |
| stochastic mask + delta (1 draw) | 0.0167 / 0.0061 | 0.0096 / 0.0034 | 0.085 / 0.062 |
| everything off | 0.527 | 0.482 | 1.0 |

Loss bookkeeping with the post-10k coefficients (1.5·KL over positions + 5e-5·imp-min +
2.5e-5·frequency, output head, layer 30 only): all-layers 0.0244 + 0.0014 + 0.0054; block30
0.0095 + 0.0008 + 0.0026. The block30 solution is better on every term. Note the all-components-on
row: the deep run's V·U for layer 30 is itself 6x worse than block30's, so part of the deficit is
not about CI at all. The always-on-set-only rows go the other way (the ten pieces alone are
slightly better than the three alone) because the block30 run leans more on its partially-on
tail. Restricting the all-layers V/U to block30's three indices gives KL 0.36: those three
components do not carry the write in the deep run.

`fig7_kl_heatmaps.png`: per-prompt KL over (a, b) for both runs' CI-masked forward.

## 4. Why gradient descent does not merge them

`fig4_ablations.png`: ablating a single always-on component from the CI-masked forward.

| all-layers extra pieces | ΔKL at "=" | block30 pieces | ΔKL at "=" |
|---|---|---|---|
| gate c1 / c6 / c15 / c17 / c21 / c27 / c299 | 0.018 / 0.006 / 0.022 / 0.010 / 0.004 / 0.044 / 0.206 | gate c3 / c43 / c103 | 0.203 / 0.015 / 0.130 |
| up c1 / c15 / c27 / c43 | 0.039 / 0.027 / 0.006 / 0.001 | up c3 / c47 / c52 | 0.090 / 0.050 / 0.172 |
| down c17 / c21 / c27 / c43 | 0.021 / 0.007 / 0.104 / 0.013 | down c1 / c3 / c52 | 0.147 / 0.402 / 0.040 |
| all 15 MLP extras at once | 0.327 | | |

Dropping one always-on component saves about 6e-4 per token in loss units (imp-min 5e-5 on
both heads, frequency f·log2(1+640f) ≈ 9.3 x 2.5e-5 on both), i.e. 4e-4 in KL units after the
1.5 recon weight. Every piece except up c43 is worth 10-500x that. Three mechanisms then hold
the configuration:

- **Imp-min acts only on CI, never on V/U.** Collapse would have to run as: CI of piece c goes
  down, recon degrades, the other pieces' V/U move to absorb c's function, c becomes redundant,
  its CI reaches 0. Step one is blocked while c is load-bearing, and nothing rewards the other
  pieces for absorbing a function that is still being provided.
- **Saturation removes the remaining force.** With preactivations at 1.3-2.1 the lower squash
  passes zero gradient (recon cannot push further) and the upper squash leaks 0.01, so the
  imp-min pull on a parked component is ~5e-7 per token.
- **The pieces are not interchangeable across positions.** Mean CI by position (BOS, a, +, b,
  =) for the all-layers gate set: c1 is on only at "=", c3 and c6 on at positions 1-4, c15 on
  at positions 2 and 4 only. A merged piece would be on at the union, paying imp-min there and
  on the fineweb stream at 4x the target coefficient (§6 for the fineweb measurement).

Why block30 nevertheless ended at 3: its recon was satisfied from step 100 (train KL 0.03,
layer-30 L0 8.5 at step 500) while imp-min sat at its 4x initial coefficient, so marginal
pieces were pushed below CI 1 before they became load-bearing. The deep run spent its first
2 000 steps at train KL 1.4 (`fig5_trajectories.png`).

## 5. Why the last layers, and why early

`figA_l0_per_layer.png` (eval L0 per layer over training, output head, all positions):

| step | layer 31 | layer 30 | median of layers 2-29 |
|---|---|---|---|
| 500 | 459 | 465 | 76 |
| 1 000 | 472 | 344 | 197 |
| 2 000 | 213 | 181 | 94 |
| 16 500 | 89 | 64 | 38 |

At step 500 layers 30 and 31 hold 6x the components of a typical layer; layers 22-23 and 28 are
the next largest and layers 0-1 are the other outliers. `figB_alwayson_per_layer.png` (grid
snapshots, "=" only): always-on counts at 4k/16k are 47/27 for layer 30 and 69/40 for layer 31
against 8-33 / 4-22 elsewhere. The last layers are the only place the output can still be
corrected before the unembed, their component gradients are the largest in the network (norm
1.74 at layer 31, 0.42 at layer 30, 0.18-0.29 for layers 9-29 at step 16 900), and during the
noisy start every direction that helps the logits gets recruited.

`figC_stickiness.png` and `figD_ci_traj_alwayson.png`: of the 116 components always-on at 4k in
layers 30-31, 66 are still ≥0.9 at 16k and 33 are dead, with almost nothing in between; layers
2-29 show the same bimodality (326 of 499 stay, 113 die). The parking mechanism is not specific
to the last layers; what is specific is how many pieces they recruit before it engages.

## 6. Pending measurements

- **Noisy upstream** (`emu_up/`): layer 30 by each run's components with layers 0-29 replaced by
  the all-layers run's CI-masked or stochastically masked components. Tests whether the ten
  pieces are worth more than the three when the input carries upstream reconstruction error.
- **Merged replacements** (`followup.json`): each run's always-on set replaced by its best
  rank-k least-squares merge, k = 1..3 (5), fitted on all positions. Measures directly how many
  rank-1 pieces the write needs.
- **Fineweb CI**: CI of the same components on 384 x 64 fineweb tokens; whether the pieces are
  used separately on the non-target stream.
