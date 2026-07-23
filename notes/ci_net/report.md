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
3. **Tanh logit soft-cap** centered on the window — config `logit_softcap: 2.0`,
   `0.5 + cap·tanh((x−0.5)/cap)` — logits can never leave gradient reach.

## Results

*(pending: `txci_newinit` / `txci_fnorm` / `txci_fnorm_cap2` runs on v8d32_partial,
jobs 5431–5433)*
