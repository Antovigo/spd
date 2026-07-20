# Period separation — report

2026-07-20 (final). Question: which hyperparameters / training recipes make the targeted L18
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

**3. Recommended recipe — the Pareto has a knee at 5x** (20k-validated):

| run (20k) | n | clean | med_band | n50 | PGD |
|---|---|---|---|---|---|
| coupled (baseline) | 59 | 9 | 0.323 | 17.4 | 0.0055 |
| hid_sched (2x + hid0.1→0) | 59 | 9 | 0.308 | 12.8 | 0.0056 |
| **hid_sched-5x (5x + hid0.1→0)** | **34** | **10** | **0.392** | **7.6** | 0.0087 |
| 10x-hid0.1 (`addsub-L18-07-10x-hid0.1`) | 26 | 6 | 0.401 | 15.2 | 0.0111 |

**`addsub-L18-05-hid_sched-5x` is the recommended recipe** (coupled + impmin peak 5x→1x
+ hidden-acts 0.1 exponentially decayed to ~0; default p-anneal kept): best mixing
(n50 7.6, 2.3× better than baseline), most clean components (10), 34 vs 59 active, at a
~60% recon premium. The 20k validation shows 10x is **past the knee**: usage concentrates
further (26 active, med_band 0.401) but per-component mixing and the clean count get
*worse* (n50 15.2, 6 clean) at higher recon cost — squeezing too few components forces
periods back together. If recon parity is required, `hid_sched` (2x + hid0.1→0) still
improves mixing for free (n50 12.8 vs 17.4).

**4. Corroborating dynamics.** Purity improves late everywhere (coupled 15k→20k,
hid_sched 5k→20k, 5x 10k→20k): the early phase decides *which* components exist
(concentration), the late phase cleans *what each one holds* (purity). This matches the
user's prior that topology is set by ~10k, and locates the two levers on either side of
that point.

## Caveats / open threads

- 5k probes compress every schedule; absolute purity numbers at 5k undershoot 20k ones.
  All probe conclusions are probe-vs-probe at matched steps; the 5x and 10x directions
  are both validated at 20k (table above). Note the 5k probes ranked 10x above 5x on
  n50 — the knee only appears at full training length; final calls need the 20k run.
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
