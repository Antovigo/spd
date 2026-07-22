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
| impmin5e-5 | 0.00958 | 0.00731 | 17.7 | 131 | 0.93 | 18.4 | 0.67 |
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
- The `impmin5e-5` interpolation probe located the coverage break precisely: **up_proj's
  T=2 cell dies between coeff 3e-5 and 5e-5** (the same cell missing at 1e-4) — up_proj
  carries only 2–3 period-2 components even at gentle doses, making it the fragile cell
  the coeff must protect.
- Diagnostics: T=4 and T=25 detections are common (components genuinely read
  non-canonical harmonic planes), T=100 detections track magnitude/trend reads.

### Revised mixing metric: `n_pure_periods`

Visual inspection of the panels showed `mixed_frac` mis-ranking the sweep: the top-CI
components at gentle doses are heavily mixed plaids while high doses leave some very
clean components. The revised headline metric (user-specified): **n_pure_periods** =
number of (matrix × canonical period) cells with ≥ 1 *single-period* component (pure =
detected at exactly one non-trend period; T=100 ignored, T=4/25 extras disqualify).
Counted per matrix, so the maximum is 3 × 5 = 15 for the MLP triple. Recomputed over
the sweep (pure census pooled across matrices):

| run | PGD | n_active | n_pure_periods (/15) | n pure comps | pure census |
|---|---|---|---|---|---|
| impmin1e-5 | 0.0052 | 307 | **14** | 62 | T2=6 T5=15 T10=9 T20=7 T50=25 |
| impmin3e-5 | 0.0073 | 164 | 11 | 43 | T2=2 T5=9 T10=10 T20=2 T50=20 |
| impmin5e-5 | 0.0096 | 131 | 11 | 27 | T2=3 T5=4 T10=7 T50=13 — no pure T20 |
| impmin1e-4 | 0.0125 | 89 | 10 | 17 | T2=2 T5=5 T10=2 T20=1 T50=7 |
| impmin3e-4 | 0.0157 ✗gate | 49 | 9 | 16 | T2=1 T5=6 T10=2 T20=1 T50=6 |
| impmin1e-3 | 0.0272 ✗ | 25 | 3 | 6 | T5=4 T50=2 |
| impmin3e-3 | 0.0650 ✗ | 9 | 1 | 1 | T50=1 |

Missing cells at the gentle end: `impmin1e-5` misses only up_proj-T2 (14/15);
`impmin3e-5` additionally loses gate_proj-{T5, T20} and up_proj-{T2, T20}. Under the
per-matrix definition the metric is monotone in the coeff — gentler is purer — where
the pooled version had flattened the top four runs to 5/5.

Reconciliation of panel vs metric: the panels show the top-20 by mean CI — at gentle
doses the *dominant* components are mixed while dozens of pure components sit at mean CI
0.15–0.35 below the panel cut; at 1e-3 the few survivors include spectacularly clean
ones (own-SNR 60–300, others < 20) but periods 2/10/20 lose their pure representative
entirely. Margin check: the gentle-dose pure components are mostly genuine (own-SNR
30–260 with other classes ≤ 10), except pure-T20 which is threshold-marginal
(max-other-SNR 16–19).

**Winner: `impmin3e-5` (coeff = 3e-5)** — chosen under the pooled metric as the largest
dose satisfying every criterion simultaneously (PGD 0.0073 ≪ 0.015 gate, full per-matrix
coverage, pooled n_pure 5/5); it is the base coeff for the in-flight Objectives 2–3
runs. Note the per-matrix revision reranks the top: `impmin1e-5` now leads (14/15 vs
11/15) with better recon (0.0052) at the cost of 2× the active components (307 vs 164)
and much higher redundancy (37.6 vs 22.7 comps/period).

## Objective 2 — initialization: kaiming wins on mixing

The coupled init was ported to this branch (`e5cdcbc80`; a raw cherry-pick would have
removed SmoothL0, so the net feature was applied surgically). Comparison at the winning
coeff 3e-5 — `addsub-L18-08-impmin3e-5` doubles as the kaiming arm (identical config,
default init); `addsub-L18-08-initcoupled` is the same + `weight_init: coupled`:

| run (init) | PGD | rounded | n_active | n_pure_periods (/15) | pure census |
|---|---|---|---|---|---|
| impmin3e-5 (kaiming) | 0.00727 | 0.00616 | 164 | **11** | T2=2 T5=9 T10=10 T20=2 T50=20 |
| initcoupled (coupled) | **0.00564** | **0.00467** | 113 | 5 | T2=2 T10=2 T20=1 T50=5 — no pure T5 |
| initwithinspan (within_span) | 0.00588 | 0.00494 | 120 | 7 | T2=2 T10=3 T20=1 T50=10 — no pure T5 |

Coupled init reconstructs better (−22% PGD) with fewer active components (113 vs 164),
but **loses period-5 purity entirely**: its best T5 components all carry exactly one
strong extra period (SNR 28–147) — genuine two-period mixes, not threshold near-misses.
The per-matrix metric widens the gap further (11 vs 5 cells): coupled's 10 pure
components sit only in down_proj (T2/T10/T50) and up_proj (T20/T50) — **gate_proj has
zero pure components**. The third arm, `within_span` (in-span like coupled but with the
two sides statistically independent), sits strictly between: recon and n_active near
coupled's (0.00588, 120), purity closer to kaiming's but still well short (7/15, worst
mixed_frac of the three at 0.83, gate_proj nearly empty with one pure T50, and — like
coupled — no pure T5 anywhere). So the purity loss of the in-span inits is driven by
the span restriction itself, not the U–V coupling: breaking the coupling recovers only
2 of the 6 cells coupled loses. On the roadmap's criterion for this objective (effect
on period mixing): **kaiming picked** for the rest of the series. Coupled remains
attractive if later objectives recover its lost pure cells at its better recon.

## Objective 3 — fused hidden-acts recon loss: free fidelity, mild purity gain at 0.1

Implementation: `HiddenActsReconAux` on `StochasticReconSubsetLossConfig` — the aux MSE
is computed from the host loss's already-masked forward (no extra passes), against
frozen `x@W+b` targets recomputed from the cached pre-weight acts
(`param_decomp/metrics/stochastic_recon_subset.py`; parity-tested against the
standalone metric). Sweep `addsub-L18-08-hid{0.001, 0.01, 0.1}` on the impmin3e-5
recipe (which doubles as the 0 arm). Figure:
[figures/obj3_hid_sweep.png](figures/obj3_hid_sweep.png).

| run (hid coeff) | PGD | rounded | n_active | hid MSE (CI-masked) | n_pure_periods (/15) | fraction_pure |
|---|---|---|---|---|---|---|
| impmin3e-5 (0) | 0.00727 | 0.00616 | 164 | 0.272 | 11 | 0.26 |
| hid0.001 | 0.00752 | 0.00631 | 161 | 0.224 | 10 | 0.17 |
| hid0.01 | 0.00722 | 0.00627 | 166 | 0.142 | 12 | 0.19 |
| hid0.1 | 0.00753 | 0.00609 | 169 | **0.054** | **12** | **0.26** |

Readings:

- **The aux is free**: output recon (PGD 0.0072–0.0075, all ≪ gate) and n_active
  (161–169) are flat across three orders of magnitude of coeff — the hidden-acts
  constraint does not trade against the main objectives at these doses.
- **It does its job**: CI-masked hidden-acts MSE falls monotonically, 5× at coeff 0.1
  (0.272 → 0.054; PGD-masked 0.123 → 0.030).
- **Separation improves modestly at the top of the sweep**: hid0.1 reaches 12/15 pure
  cells with full coverage and the best mixed_frac of the series (0.65), matching the
  0-arm's fraction_pure (0.26). Notably it fixes gate_proj — all 5 periods pure there
  (0-arm: 3) — leaving only up_proj-{T2, T20} and down_proj-T20 uncovered. The 0.001
  arm is slightly *worse* than 0 (10/15, fraction 0.17) — the weak dose perturbs
  without constraining.

**Winner: hid coeff 0.1** for the rest of the series — equal recon, much tighter
per-site fidelity, and the best purity of the sweep.



## Objective 4 — impmin β sweep — pending

β ∈ {0, 0.5, 0.75} at the best hyperparameters from Objectives 1–3.
