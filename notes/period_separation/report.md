# Period separation — findings (roadmap objectives)

2026-07-21. Findings for [roadmap.md](roadmap.md). Run naming:
`addsub-L18-08-<what><value>`. All runs are 4000-step decompositions with the
`addsub-L18-04-hidden` hyperparameters except where the run name says otherwise, with
these Objective-1 series baselines: **SmoothL0** importance minimality (γ = 1 constant,
coeff constant = the value in the run name), **no hidden-acts recon loss**, eval +
slow-eval every 250 steps. Separation is measured by the `PeriodSeparation` metric /
`score_period_separation` script: inner activations `x·V/‖V‖` on the a+b grid at the
answer position, CI gate 0.1, presence = class-bin power ≥ 20× the random-read-direction
null median, mixing counted over the canonical periods {2, 5, 10, 20, 50} (T=4/25/100
reported as diagnostics only).

## Objective 1 — optimal impmin coeff for initial training

Runs `addsub-L18-08-impmin{1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 1e-3, 3e-3}` (jobs 5141–5146,
5155; the only varied hyperparameter is the SmoothL0 `coeff`). Figure:
[figures/obj1_impmin_sweep.png](figures/obj1_impmin_sweep.png). All metrics at step 4000;
`coverage` = fraction of (matrix × canonical period) cells with ≥ 1 periodic component
(criterion: 1.0), `comp/period` = mean components per covered cell (criterion: → 1),
`mixed` = fraction of periodic components detecting ≥ 2 canonical periods (criterion: → 0).

| run (coeff) | PGD | rounded | SmoothL0 proxy (coeff-free) | n_active | coverage | comp/period | mixed |
|---|---|---|---|---|---|---|---|
| impmin1e-5 | 0.00524 | 0.00444 | 59.7 | 307 | **1.00** | 37.6 | 0.70 |
| impmin3e-5 | 0.00727 | 0.00616 | 25.4 | 164 | **1.00** | 22.7 | 0.67 |
| impmin5e-5 | *(running — interpolation probe)* | | | | | | |
| impmin1e-4 | 0.01246 | 0.01075 | 10.3 | 89 | 0.93 | 13.6 | 0.73 |
| impmin3e-4 | 0.01570 ✗gate | 0.01335 | 4.8 | 49 | 0.93 | 7.1 | 0.56 |
| impmin1e-3 | 0.02724 ✗ | 0.02352 | 2.4 | 25 | 0.93 | 3.8 | 0.60 |
| impmin3e-3 | 0.06497 ✗ | 0.06190 | 1.1 | 9 | 0.80 | 1.9 | 0.67 |

Readings:

- **Monotone dose–response on all axes**: coeff ↑ → recon ↓, active components ↓,
  redundancy ↓. The PGD < 0.015 gate passes only coeff ≤ 1e-4.
- **Coverage breaks between 3e-5 and 1e-4**: at 1e-4 the up_proj loses its T=2 cell
  (its census: no T2 among 28 active); 1e-5 and 3e-5 keep every canonical period in
  every matrix. At 3e-3 coverage collapses further (up_proj holds only T=5/T=10).
- **Mixing is flat (~0.6–0.73) across the whole sweep** — at 4000 steps with constant
  γ=1, the impmin dose does not separate periods; it only prunes. Separation will have
  to come from the later knobs (hidden-acts, γ-anneal, β).
- Redundancy at the coverage-preserving doses is still far from the ideal
  (22.7 components per covered period at 3e-5 vs the optimal 1): the sweet spot sits in
  the 3e-5–1e-4 gap, hence the `impmin5e-5` interpolation probe (result pending; will
  finalize the winner here).
- Diagnostics: T=4 and T=25 detections are common (components genuinely read
  non-canonical harmonic planes), T=100 detections track magnitude/trend reads.

**Provisional winner: `impmin3e-5`** (full coverage, PGD 0.0073 ≪ gate, half the
redundancy of 1e-5) — to be confirmed against `impmin5e-5` when it lands: if 5e-5 keeps
coverage 1.0 it takes the win (lower redundancy at compliant recon).

## Objective 2 — initialization (kaiming vs coupled) — pending

Blocked on the Objective-1 winner. Plan: cherry-pick the coupled-init feature
(`61ca40b9d` lineage) onto this branch, run `addsub-L18-08-init{kaiming,coupled}` at the
winning coeff, compare period mixing.

## Objective 3 — fused hidden-acts recon loss — pending

Re-implement `StochasticHiddenActsReconLoss` as an attribute of the StochasticRecon
loss (no extra forward passes), following `experiment/8B_targeted_jax`; then sweep
hidden-acts coeff ∈ {0, 0.001, 0.01, 0.1} at the winning recipe.

## Objective 4 — impmin β sweep — pending

β ∈ {0, 0.5, 0.75} at the best hyperparameters from Objectives 1–3.
