# Period separation — report

2026-07-19. Question: which hyperparameters / training recipes make the targeted L18
addsub decomposition assign **one operand period per subcomponent** (periods 2, 5, 10,
20, 50, 100 — the model's Fourier features), instead of mixing several periods in one?

Method: `score_period_separation` (2D-FFT orbit purity of the per-position CI grids; see
spec.md) over the existing 20k runs plus twelve 5000-step probe runs
(`addsub-L18-06-psep-*`, schedules compressed into 5k). Metrics that discriminate above
the seed-noise floor: the **answer-position active-component count `n`** (how
concentrated usage is) and **`n_orbits_50`** (how mixed each component is); median
band-purity differences at 5k are mostly seed noise (base vs base-s1: ±0.015).

## Findings

**1. Two independent levers, doing different jobs.**

- **Early impmin strength** (the coeff multiplier starts at its peak and decays to 1x —
  note it is *early* pressure, not a mid-training spike) concentrates usage: peak 2x →
  5x → 10x gives n = 71 → 43 → 28 active MLP components at the answer position, with
  reconstruction degrading roughly linearly (PGD 0.0095 → 0.0123 → 0.0151). A
  **separation–reconstruction Pareto with no cliff up to 10x**.
- **Early hidden-acts reconstruction pressure** (`StochasticHiddenActsReconLoss`
  coeff 0.1, exponentially decayed to ~0) reduces per-component mixing roughly for
  free: n50 33 → 24 at 2x impmin (recon unchanged), 33 → 17 at 5x. It pins components
  to period-specific neuron groups while the topology forms.

**2. Schedule shape matters more than magnitude — and in non-obvious directions.**

- *Press early, release late* (hidden-acts): constant 0.1 keeps n low but re-mixes
  components (n50 30 vs 17 decayed). Dose is non-monotonic: 0.3 loses the gain entirely.
- *The late concave-p phase is load-bearing*: keeping `p ≥ 1` in the winning combo
  blows mixing up to n50 48 (worst of all probes) at zero recon gain. Late concave
  impmin **purifies** the concentrated components rather than merging them — the naive
  "Lp<1 merges co-occurring features" story is wrong here. Softening impmin generally
  (p1 / nopeak alone) just inflates the circuit (315 / 260 alive vs 177) with equally
  mixed components.

**3. Recommended recipes** (all = coupled recipe + the listed changes):

| goal | recipe | evidence |
|---|---|---|
| best separation, recon secondary | impmin peak **10x**→1x + hidden-acts **0.1**→0 (`addsub-L18-07-10x-hid0.1`, 20k validation running) | 5k: n=28, n50=15.2, PGD 0.0151 |
| strong separation, moderate recon cost | peak **5x** + hidden-acts **0.1**→0 — the existing **`addsub-L18-05-hid_sched-5x`** | 20k: med_band 0.392 (vs 0.323 coupled), 34 active vs 59, 10/34 clean; PGD 0.0087 vs 0.0055 |
| free improvement, recon parity | peak 2x + hidden-acts **0.1**→0 — the existing **`addsub-L18-05-hid_sched`** | 5k: n50 24 vs 33 at identical PGD; 20k: recon ≈ coupled |

Keep the default p-anneal (2.0 → 0.5 over training) in all cases.

**4. Corroborating dynamics.** Purity improves late everywhere (coupled 15k→20k,
hid_sched 5k→20k, 5x 10k→20k): the early phase decides *which* components exist
(concentration), the late phase cleans *what each one holds* (purity). This matches the
user's prior that topology is set by ~10k, and locates the two levers on either side of
that point.

## Caveats / open threads

- 5k probes compress every schedule; absolute purity numbers at 5k undershoot 20k ones.
  All probe conclusions are probe-vs-probe at matched steps; the 5x direction is
  additionally validated at 20k (baseline table), the 10x one pending (run
  `addsub-L18-07-10x-hid0.1`).
- One seed replicate only; n and n50 cross the noise floor comfortably, med_band does
  not.
- The metric reads CI usage grids. Weight-side confirmation (`collect_inner_activations`
  → `compute_subcomp_periods`) on the final recommended run is the natural next check,
  along with whether the 10x recipe's anchored circuit stays functionally sufficient
  (its anchor rises with its rounded KL).
- `5x-hidconst` had the highest median band purity (0.314) despite bad n50 — if a
  future pass optimises band purity specifically, revisit constant-vs-decayed with more
  seeds.

## Full 5k probe table

`+` rows, answer position, MLP matrices; n50 = mean orbits to 50% FFT power:

| probe | n | clean | med_band | mw_band | n50 | PGD | rounded |
|---|---|---|---|---|---|---|---|
| base | 71 | 8 | 0.271 | 0.312 | 33.0 | 0.00951 | 0.00898 |
| base-s1 (seed) | 84 | 8 | 0.257 | 0.299 | 32.1 | 0.00934 | 0.00904 |
| p1 | 81 | 7 | 0.273 | 0.297 | 29.1 | 0.00854 | 0.00759 |
| nopeak | 92 | 8 | 0.253 | 0.261 | 27.3 | 0.00706 | 0.00607 |
| 5x | 49 | 3 | 0.270 | 0.296 | 32.7 | 0.01222 | 0.01152 |
| 5x-hid0.1 | 43 | 4 | 0.290 | 0.324 | 17.4 | 0.01231 | 0.01157 |
| 5xanneal0.5 | 54 | 5 | 0.283 | 0.289 | 21.2 | 0.01118 | 0.01055 |
| hid0.1 | 76 | 6 | 0.291 | 0.309 | 23.9 | 0.00936 | 0.00893 |
| hid0.3 | 80 | 9 | 0.232 | 0.296 | 31.7 | 0.00926 | 0.00884 |
| 5x-hidconst | 47 | 5 | 0.314 | 0.336 | 30.4 | 0.01197 | 0.01136 |
| 10x-hid0.1 | 28 | 4 | 0.274 | 0.326 | 15.2 | 0.01507 | 0.01451 |
| 5x-hid0.1-p1 | 51 | 7 | 0.238 | 0.296 | 48.3 | 0.01226 | 0.01163 |
