# Why the transformer CI network saturates — and how to fix it

**Question.** With the transformer CI network (`global_shared_transformer`), predicted
CIs are almost all exactly 0 or 1; the MLP CI networks leave intermediate values. The
practical cost: gamma annealing prunes intermediate-CI strays, so the transformer run
keeps many more stray subcomponents alive than the MLP run.

**Answer.** It is not that the transformer "can't calibrate" — it's that nothing
anchors its logit scale. Its output head reads an unnormalized residual stream (no
final norm), CI-fn weight decay is 0, and confident logits reduce stochastic-masking
loss variance, so logits inflate to ±100. The hard sigmoid's [0, 1] linear window then
covers ~1% of the network's dynamic range: almost no input can land in it, and a stray
parked at logit +30 sees only the 0.01-leaky importance-minimality gradient over a
30-unit round trip — it is frozen at CI = 1, out of the gamma anneal's reach. The MLP
CI nets are too weak to inflate logits (theirs live in [-1.5, +2.6]), which is exactly
why their strays linger at intermediate CI where the anneal kills them.

Testbed: `copy_training_v8d32_partial`; baseline runs from the completeness work,
all with `sigmoid_type: leaky_hard`, SmoothL0 (γ 1.0 → 0.01 annealed over the last
25% of training).

## Evidence

Pre-sigmoid logits at step 10000 — layerwise MLP (`joint_norm`, left) stays in
[-1.5, +2.6], hugging the sigmoid window; transformer (`joint_txci`, right) spans
[-80, +150]:

![pre-sigmoid, layerwise MLP](report_figures/presigmoid_norm.png)

![pre-sigmoid, transformer](report_figures/presigmoid_txci.png)

CI values at step 5000, *before* the gamma anneal starts (75%): the MLP run has
substantial intermediate mass (0.05–0.6) that the anneal later sweeps to zero; the
transformer is already near-binary, so the anneal has nothing to grab:

![CI hist 5000, layerwise MLP](report_figures/ci_hist_5000_norm.png)

![CI hist 5000, transformer](report_figures/ci_hist_5000_txci.png)

Final decompositions — MLP is textbook (1 q, 1 k, clean v/o diagonals); transformer
keeps stray/duplicate components in every block:

![active subcomponents, layerwise MLP](report_figures/active_norm.png)

![active subcomponents, transformer](report_figures/active_txci.png)

| run (step 10000) | total CI-L0 | stoch recon KL |
|---|---|---|
| layerwise MLP (`joint_norm`) | **3.50** | 1.1e-5 |
| global MLP (`joint_globalci`) | 4.49 | 3.5e-5 |
| transformer (`joint_txci`) | 5.25 | 1.7e-5 |
| transformer, beta 1 (`joint_txci_beta1`) | 12.92 | 6.5e-5 |

## Fix

Three standard-transformer changes to `GlobalSharedTransformerCiFn`
(`param_decomp/ci_fns.py`):

1. **Final RMS norm** before the output head — config `final_rms_norm: true`.
2. **Readout zero-init with bias 0.5** (unconditional) — all logits start at 0.5,
   mid-window.
3. **Tanh logit soft-cap** centered on the window — config `logit_softcap: 2.0`.

### How the soft-cap works

The output head produces a raw logit `x` per component; the CI is `hard_sigmoid(x)`
(= `clamp(x, 0, 1)`). The soft-cap replaces `x` with

```
x' = 0.5 + cap · tanh((x − 0.5) / cap)
```

before the sigmoid (Gemma-2 uses the uncentered form on attention/final logits).
Properties:

- **Identity near the window.** `tanh(u) ≈ u` for small `u`, so for logits near 0.5
  (the center of the sigmoid's [0, 1] linear region) `x' ≈ x` — calibrated
  intermediate values pass through unchanged.
- **Hard bound on saturation depth.** `tanh` is bounded in (−1, 1), so
  `x' ∈ (0.5 − cap, 0.5 + cap)` — with `cap = 2`, (−1.5, 2.5) — no matter how large
  the raw logit gets. A maximally confident component sits at most 1.5 units from the
  linear window instead of 100+.
- **Bounded hysteresis, not bigger gradients.** The cap does *not* increase the
  gradient on a saturated component — above the window the chain is still the
  sigmoid's 0.01 leak (times `tanh' < 1`). What it bounds is the *integrated
  distance* that small gradient must cover before the CI value changes at all:
  ≤ 1.5 units instead of 30–150. Uncapped, a stray at logit +30 is frozen (the
  gamma anneal ends long before the leak drags it back 29 units); capped, every
  stray is a few steps from re-entering the window. The same bound on the negative
  side helps dead components resurrect.
- **No incentive to drift.** Uncapped, the trunk can keep burying logits deeper
  (5 → 50 → 150) — confidence is loss-free and self-reinforcing. Capped, logits 5
  and 50 map to nearly the same `x'`, so growing confident weights buys nothing.

The choice `cap = 2` is loose: CI = 1 needs `x' ≥ 1`, i.e. `tanh ≥ 0.25` (raw logit
≈ 1.0), so confident components are cheap to represent, while the bound stays the
same order as the window itself. The cap alone (without the final RMS norm) still
bounds saturation depth, but lets the pre-cap logits — and the trunk driving them —
grow unboundedly confident; the RMS norm addresses that upstream. The two compose:
norm anchors the input scale of the head, cap bounds its output.

## Results

Three variants on `v8d32_partial`, same config as `joint_txci` otherwise
(`txci_newinit` = new head init only; `txci_fnorm` = + final RMS norm;
`txci_fnorm_cap2` = + soft-cap 2.0):

| run | CI-L0 | recon KL | v/o dupes | shared subs | skipped mechs |
|---|---|---|---|---|---|
| layerwise MLP (reference) | 3.50 | 1.1e-5 | 1 | 0 | 10/19 |
| transformer baseline | 5.25 | 1.7e-5 | 7 | 9 | 4/19 |
| + new init only | 5.56 | 5.8e-5 | 11 | 10 | 5/19 |
| + final RMS norm | **3.43** | 2.1e-5 | 3 | 1 | 10/19 |
| + RMS norm + soft-cap 2 | **3.24** | 1.9e-5 | 2 | 0 | 10/19 |

**The final RMS norm is the load-bearing change**; the soft-cap helps a little more;
the init alone does nothing. Both fixed variants now beat the layerwise-MLP reference
on sparsity at comparable recon, and skip exactly the same 10/19 redundant mechanisms
it does — the baseline transformer's apparent extra "coverage" (4/19 skipped) was
strays incidentally covering redundant mechanisms.

The mechanism is confirmed end-to-end. Fixed-transformer pre-sigmoid logits at
step 10000 sit in [-5, +3.5], anchored on the sigmoid window:

![pre-sigmoid, fixed transformer](report_figures/presigmoid_fnorm.png)

Mid-training (step 5000) CI values regain the intermediate mass the anneal needs —
compare with the near-binary baseline above:

![CI hist 5000, fixed transformer](report_figures/ci_hist_5000_fnorm.png)

Final decomposition, `fnorm_cap2` — block-0 v/o perfectly clean (1.00/input,
0 dupes, 0 shared), single q routing column; residual dirt is 3 alive k components
(vs 1) and one dupe in each of blocks.1.{v,o}:

![active subcomponents, fnorm_cap2](report_figures/active_fnorm_cap2.png)

## Rigorous comparison: 3 seeds × 3 sizes × 4 variants

Final CI-L0, mean [min–max] over seeds {0, 1, 2} (array 5464); recon stays within
the reference band for the fnorm variants at every size:

| variant | v8d32 | v12d64 | v16d32 |
|---|---|---|---|
| layerwise MLP (seed-0 ref) | 3.50 | 4.93 | 6.53 |
| transformer baseline | 4.90 [4.62–5.25] | 9.52 [7.10–14.34] | 17.44 [13.48–19.66] |
| + final RMS norm | **3.29 [3.19–3.43]** | **4.22 [4.01–4.39]** | **5.52 [5.37–5.66]** |
| + soft-cap only | 7.08, 1 crash | 9.33, 2 crashes | 13.76 [13.15–14.79] |
| + norm + soft-cap | 3.66 [3.24–3.87] | 4.41 [4.27–4.66] | 5.37/6.04 (s2 pending) |

Robust conclusions:

- **The final RMS norm is the consistent, measurable improvement.** Every seed at
  every size beats both the baseline transformer and the layerwise-MLP reference,
  with the tightest seed variance of any variant. At v16d32 it repairs the dense
  collapse outright (17.4 → 5.5 at 3–5× better recon).
- **The soft-cap adds nothing reliable on top of the norm.** Its seed-0 wins did
  not replicate; means are slightly worse than norm-alone at v8/v12.
- **The soft-cap alone is actively harmful**: it bounds saturation depth but leaves
  the trunk unanchored — with the tanh saturated, CI-fn gradients vanish
  (~1e-8) while Adam's normalized updates keep drifting the trunk until
  activations go non-finite. 3 of 9 seeds crashed this way; survivors are far
  from sparse.
- **Weight-decay control**: baseline + CI-fn Adam WD (0.01 / 0.1) gives L0
  5.32 / 6.83 — no rescue. Generic weight shrinkage is not a substitute for
  normalizing the head's input scale.

## Recommendation

Set `final_rms_norm: true` in `simple_transformer_ci_cfg` for all
`global_shared_transformer` runs; leave `logit_softcap` off (it is config-gated
and available, but adds nothing on top of the norm and must never be used without
it). Both fields default to the old behavior, so saved configs of existing txci
runs reload and evaluate unchanged. The readout zero-init (bias 0.5) is
unconditional — it only affects newly initialized CI networks.
