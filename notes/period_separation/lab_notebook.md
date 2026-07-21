# Period separation — lab notebook

Newest entries at the bottom. Runs referenced by their run id in `~/out/runs/`.

## 2026-07-17 — setup

- Built `score_period_separation` (2D-FFT orbit purity of the per-position CI grids; see
  spec.md). Sanity-checked on synthetic grids: pure sinusoids in a / b / a+b / a-b score
  purity 1.0 with correct labels; a 60/40 two-period mixture scores n_orbits ≈ 2; a binary
  period-10 square wave scores band_purity 1.0 (harmonic pooling works).
- First real-data run (addsub-L18-05-coupled final, `+` pos 4): median band_purity ~0.27–0.40
  across the three MLP matrices, 9/59 MLP components clean at band_purity > 0.5, and the
  top1 period labels land almost entirely on the canonical set (2, 5, 10, 20, 50, 100) —
  the metric picks up real structure without being told the periods.
- Two metric pitfalls found and fixed while iterating: always-on components (mass ≈ 1)
  have near-constant grids whose FFT is pure noise → excluded as `flat`; and near-binary
  stripes spread power over harmonics → `band_purity` pools them (raw top-orbit `purity`
  under-credits clean square patterns by ~2x).
- Scored all existing addsub runs via SLURM (job 4803) — baseline table to follow.
- User note to keep front of mind: **the dynamics may be quite subtle, especially the
  scheduling of the different hyperparameters** — treat schedule shape/timing as the
  primary experimental axis, not just coefficient magnitudes.

### Existing-run context (from the hid_sched grid, 2026-07-16/17)

| run | hidden-acts | impmin peak | PGDRecon | anchor | alive |
|---|---|---|---|---|---|
| addsub-L18-05-coupled | 0.001 const | 2x | 0.00549 | 0.00854 | 177 |
| addsub-L18-05-hid_sched | 0.1 → 0 | 2x | 0.00561 | 0.00814 | 177 |
| addsub-L18-05-hid_sched0.01 | 0.01 → 0 | 2x | 0.00689 | 0.00855 | 146 |
| addsub-L18-05-hid_sched-5x | 0.1 → 0 | 5x | 0.00868 | 0.01179 | 177 |
| addsub-L18-05-hid_sched0.01-5x-b | 0.01 → 0 | 5x | 0.00913 | 0.01224 | 146 |

Reading: early hidden-acts magnitude controls circuit size (topology knob); impmin peak
5x only hurts. Period-purity columns for these runs pending job 4803.

## 2026-07-17 — baseline period-purity table (job 4803)

`+` rows, answer position (pos 4), MLP matrices pooled; `n` = scored (non-flat)
subcomponents, `clean` = band_purity > 0.5, `med_band` = n-weighted median band_purity,
`n50` = mean orbits to 50% power:

| run | n | flat | clean | med_band | mw_band | n50 |
|---|---|---|---|---|---|---|
| L15-05-coupled | 7 | 9 | 4 | 0.594 | 0.678 | 14.0 |
| L16-04-init-proj | 12 | 5 | 3 | 0.352 | 0.407 | 5.8 |
| L17-04-init-proj | 14 | 3 | 7 | 0.516 | 0.416 | 22.1 |
| L18-05-coupled | 59 | 3 | 9 | 0.323 | 0.383 | 17.4 |
| L18-05-coupled @15k | 59 | 3 | 7 | 0.311 | 0.374 | 17.4 |
| L18-05-dense | 55 | 7 | 8 | 0.295 | 0.320 | 12.0 |
| L18-05-hid_sched | 59 | 2 | 9 | 0.308 | 0.344 | 12.8 |
| L18-05-hid_sched @5k | 54 | 8 | 5 | 0.255 | 0.285 | 13.3 |
| **L18-05-hid_sched-5x** | **34** | 5 | **10** | **0.392** | **0.428** | 7.6 |
| L18-05-hid_sched-5x @10k | 34 | 5 | 5 | 0.358 | 0.351 | 30.2 |
| L18-05-hid_sched0.01 | 49 | 4 | 8 | 0.339 | 0.324 | 16.4 |
| L18-05-hid_sched0.01-5x-b | 37 | 3 | 9 | 0.364 | 0.423 | 11.8 |
| L19-05 | 15 | 4 | 1 | 0.287 | 0.268 | 18.7 |
| L20-05-coupled | 21 | 2 | 2 | 0.275 | 0.287 | 32.2 |

Observations:

- **The 5x-impmin-peak runs are the cleanest L18 decompositions** (med_band 0.39/0.36 vs
  0.32 coupled) and have far fewer per-position-active MLP components (34–37 vs 59), with
  the same absolute number of clean ones. Against H1/H2's naive direction: *more* impmin
  pressure improved period separation — apparently by pruning/suppressing the noisy mixed
  components — while costing ~60% reconstruction. The interesting axis is now the
  **separation-vs-recon trade-off**, and whether p-anneal/peak *shape* can buy the
  separation without the recon cost.
- Purity improves over training everywhere (coupled 15k→20k, hid_sched 5k→20k, 5x
  10k→20k) — late training refines rather than merges, mildly against H1. 5k probes will
  read lower in absolute terms; only probe-vs-probe comparisons at 5k are valid.
- Cross-layer numbers (L15 strikingly clean with only 7 scored + 9 flat; L19/L20 messy)
  confound recipe with layer — recipe comparisons must stay within-layer.

## 2026-07-17 — probe wave 1 (design)

All probes: coupled recipe, `steps: 5000` (schedules compress — deliberate, see spec.md
probe protocol), 2 GPUs, ~3.5h. Names `addsub-L18-06-psep-*`:

- `psep-base` — unmodified coupled recipe at 5k. The control every probe compares against.
- `psep-p1` — `p_anneal_final_p: 1.0` (impmin never goes concave). Tests H1 (late
  concave impmin merges co-occurring periods).
- `psep-nopeak` — `coeff_peak_multiplier: 1.0` (no mid-training impmin surge). Tests H2
  (the 2x peak lands while topology forms and pushes merging).

Wave-2 candidates (pick after wave 1): p-anneal *timing* (start_frac 0.5 → concave only
after topology sets), hidden-acts 0.1-constant × p1 interaction, impmin coeff halved,
higher C on the MLP matrices, seed replicate of psep-base for the noise floor. The
baseline table adds a live candidate: probe the *5x direction* at matched recon (e.g.
peak 3–5x with a gentler base coeff) — separation may be buyable with schedule shape.

Launched: trains 4813 (base) / 4815 (p1) / 4817 (nopeak), analyses 4814/4816/4818
chained `afterok`. (First launch attempt 4804–4806 died on the known stale
`ci_fn_output_bias_init` field in the stage4-derived yaml — stripped and relaunched.)

## 2026-07-17 — wave 1 results: softening impmin does NOT separate

`+` pos-4 MLP aggregates at 5k steps; alive = full-sweep circuit size; anchor from the
run's own rounded circuit:

| probe | n | clean | med_band | mw_band | alive | PGD | rounded | anchor |
|---|---|---|---|---|---|---|---|---|
| base | 71 | 8 | 0.271 | 0.312 | 177 | 0.00951 | 0.00898 | 0.01275 |
| p1 (p stays ≥ 1) | 81 | 7 | 0.273 | 0.297 | 315 | 0.00854 | 0.00759 | 0.01252 |
| nopeak (mult 1x) | 92 | 8 | 0.253 | 0.261 | 260 | 0.00706 | 0.00607 | 0.01083 |

- H1 and H2 **refuted** in their naive direction: relaxing the p-anneal or the coeff
  multiplier buys reconstruction but *inflates* the circuit (315 / 260 vs 177 alive; 81 /
  92 vs 71 active at the answer position) with the same absolute number of clean
  components — i.e. the extra components are mixed/noisy, and per-component purity does
  not improve. Consistent with the baseline 5x finding from the other side.
- **Schedule reading corrected** (from `_get_coeff_multiplier`): with
  `warmup_frac 0, anneal 0→1`, the multiplier starts AT the peak and decays linearly to
  1x — the recipe applies its strongest impmin pressure *early*, while `p` is still
  convex (≈2). The 5x runs were therefore "strong early convex pressure", not a
  mid-training spike. New working model: **early convex impmin while the topology forms
  is the separation force** — it shrinks marginal CI usage proportionally (no
  winner-take-all merging at p≈2) so components crystallise around single features;
  concave-p late refines sparsity.
- psep-base at 5k reads med_band 0.271 vs the 20k coupled parent's 0.323 — confirms
  purity keeps improving with training; probes compare only within-wave.

## 2026-07-17 — wave 2 (design + launch)

Testing the "early strong convex impmin" model and its recon trade-off, all at 5k:

- `psep-5x` — coeff starts at 5x, decays to 1x over the run (probe-scale replicate of
  the direction that won in the 20k baselines).
- `psep-5x-hid0.1` — 5x + hidden-acts 0.1 → 0 (exact hid_sched-5x recipe; the baseline
  table's best cell. Does the hidden-acts schedule add anything over 5x alone?).
- `psep-5xanneal0.5` — 5x decaying to 1x by *mid-training* (`coeff_anneal_end_frac 0.5`),
  then flat: keeps the early pressure, releases late — aims for 5x-level separation
  without the late recon cost.
- `psep-base-s1` — base recipe, seed 1: the seed noise floor every comparison needs.

Trains 4823 / 4825 / 4827 / 4829 (s1 queued behind 4823 for the GPU cap), analyses
4824 / 4826 / 4828 / 4830 chained `afterok`.

## 2026-07-18 — wave 2 results: seed noise floor + which metrics discriminate

Full 5k probe table (`+` pos-4 MLP; alive = anchored sweep count, NOT comparable across
recipes with different recon — the anchor moves):

| probe | n | clean | med_band | mw_band | n50 | alive | PGD | rounded |
|---|---|---|---|---|---|---|---|---|
| base | 71 | 8 | 0.271 | 0.312 | 33.0 | 177 | 0.00951 | 0.00898 |
| base-s1 (seed 1) | 84 | 8 | 0.257 | 0.299 | 32.1 | 215 | 0.00934 | 0.00904 |
| p1 | 81 | 7 | 0.273 | 0.297 | 29.1 | 315 | 0.00854 | 0.00759 |
| nopeak | 92 | 8 | 0.253 | 0.261 | 27.3 | 260 | 0.00706 | 0.00607 |
| 5x | 49 | 3 | 0.270 | 0.296 | 32.7 | 315 | 0.01222 | 0.01152 |
| 5x-hid0.1 | 43 | 4 | 0.290 | 0.324 | 17.4 | 260 | 0.01231 | 0.01157 |
| 5xanneal0.5 | 54 | 5 | 0.283 | 0.289 | 21.2 | 260 | 0.01118 | 0.01055 |

- **Seed noise floor** (base vs base-s1): Δmed_band ≈ 0.015, Δn ≈ 13, Δalive ≈ 40.
  ⇒ med_band differences among probes (all 0.25–0.29) are *within noise* at 5k; the
  sweep-based `alive` is both noisy and anchor-confounded. The discriminative
  metrics at 5k are the **pos-4 active count `n`** and **`mean n_orbits_50`**.
- **Early-5x concentrates usage**: n = 49/43/54 for the 5x family vs 71–92 for 2x —
  well beyond noise, replicating the 20k finding at probe scale. Purity per component
  doesn't move yet at 5k (it emerged 10k→20k in the baselines).
- **Hidden-acts 0.1 is what reduces mixing**: n50 17.4 (5x-hid0.1) and 21.2
  (5xanneal0.5) vs ~32 for everything without it. hid_sched@5k (uncompressed schedule)
  also sat at n50 13.3. The n50 signal tracks the hidden-acts term, not the impmin
  strength.
- Releasing impmin at mid-run (5xanneal0.5) recovered less recon than hoped
  (0.0112 vs 0.0122 full-5x, base 0.0095) — early pressure does most of the recon
  damage too, not just the separation work.

Working model, updated: **early hidden-acts reconstruction pressure pins components to
period-specific neuron groups (low mixing / n50); early impmin strength prunes the
active set (low n) at a recon cost.** The two compose (hid_sched-5x was the 20k
winner).

## 2026-07-18 — wave 3 (design + launch): isolate the hidden-acts axis

All 5k, vs base/base-s1: `psep-hid0.1` (2x impmin + hidden-acts 0.1 → ~0 — compressed
hid_sched; does hid0.1 alone reduce n50 without the 5x recon cost?), `psep-hid0.3`
(3x that pressure — dose response), `psep-5x-hidconst` (5x + hidden-acts 0.1 constant,
no decay — is the *decay* needed, or is constant support better late?).

Trains 4935 / 4937 / 4939, analyses 4936 / 4938 / 4940 chained.

## 2026-07-19 — wave 3 results: hid-acts dose & shape

| probe | n | clean | med_band | mw_band | n50 | PGD |
|---|---|---|---|---|---|---|
| hid0.1 (2x) | 76 | 6 | 0.291 | 0.309 | 23.9 | 0.00936 |
| hid0.3 (2x) | 80 | 9 | 0.232 | 0.296 | 31.7 | 0.00926 |
| 5x-hidconst | 47 | 5 | 0.314 | 0.336 | 30.4 | 0.01197 |

- **hid0.1 alone reduces mixing for free**: n50 33 → 24 at unchanged recon (PGD 0.0094 ≈
  base) — but doesn't concentrate usage (n 76 ≈ base). The two levers really are
  independent.
- **Dose is non-monotonic**: 0.3 loses the n50 gain (31.7) and has the worst med_band.
  0.1 is near the sweet spot; don't push the coefficient, shape it.
- **The decay matters**: 5x + *constant* 0.1 keeps n low (47) but n50 back at 30 (vs
  17.4 decayed). Holding the hidden-acts constraint to the end freezes mixed patterns
  in; pressing early and releasing lets impmin clean up late. (Its med/mw_band are the
  highest, though — worth a second look when validating.)

## 2026-07-19 — wave 4 (design + launch): push and protect the winner

- `psep-10x-hid0.1` — impmin 10x → 1x, hid 0.1 → 0: does usage concentrate further
  (n < 43), and does recon degrade gracefully or collapse?
- `psep-5x-hid0.1-p1` — the winning combo but `p_anneal_final_p 1.0`: is late concave-p
  needed at all once the early pressure has done the work, and does dropping it buy
  recon?

Trains 5018 / 5020, analyses 5019 / 5021 chained.

## 2026-07-19 — wave 4 results: the frontier scales; late concave-p is load-bearing

| probe | n | clean | med_band | mw_band | n50 | PGD |
|---|---|---|---|---|---|---|
| 10x-hid0.1 | 28 | 4 | 0.274 | 0.326 | 15.2 | 0.01507 |
| 5x-hid0.1-p1 | 51 | 7 | 0.238 | 0.296 | 48.3 | 0.01226 |

- **10x keeps sliding the Pareto**: n 43 → 28, n50 17.4 → 15.2, recon 0.0123 → 0.0151.
  Degradation is roughly linear in the multiplier — no cliff up to 10x.
- **Dropping the late concave-p phase destroys mixing**: n50 48.3 (worst of all 12
  probes), recon unchanged. Combined with wave 1 (p1 alone ≈ base) the interaction is
  now clear: with strong early pressure, the late concave phase is what *purifies* the
  concentrated components. It refines, it does not merge — H1 fully inverted.
- 20k validation of `10x-hid0.1` launched as `addsub-L18-07-10x-hid0.1`
  (train 5036, analyze 5037).

Final 12-probe table lives in report.md; findings consolidated there.

## 2026-07-21 — probe series invalidated; reset

- The `steps: 5000` probes compressed every schedule (fracs of training) into 5k steps,
  so they never sampled the *actual* schedule dynamics — user call: this mostly
  invalidates the probe experiment. All 12 `addsub-L18-06-psep-*` runs deleted from
  disk and wandb (ids tombstoned — never reuse the names). Their conclusions are
  downgraded to *suggestive*; what stands as evidence is the full-length 20k grid
  (05-series + `addsub-L18-07-10x-hid0.1`).
- New feature landed on this branch: `SmoothL0ImportanceMinimalityLoss`
  (φ(c)=c²/(c²+γ²), γ-anneal replaces p-anneal; bounded gradient, no c→0 cliff).
- Pilot: `addsub-L18-08-smoothl0` (coeff 5e-5, job 5114) was cancelled at step 172 and
  relaunched as `addsub-L18-08-smoothl0-b` (job 5115) with **coeff 1e-4 flat — the
  04-hidden peak value (5e-5 × 2x) held throughout** — since the 2x→1x multiplier has no
  SmoothL0 equivalent. Otherwise identical: 04-hidden recipe, beta 0.75, γ 1 → 0.01
  over 24k steps. (5e-5 partial deleted from disk + wandb; id tombstoned.)
- report.md rewritten: objective hierarchy (recon ≻ separation ≻ parsimony), the
  three-phase dynamics model (P0 imprint / P1 crystallize / P2 purify), per-knob
  predictions, and the staged full-length experimental plan (S0–S4).
- SmoothL0 pilots (`-08-smoothl0`, `-08-smoothl0-b`) cancelled and purged per user —
  superseded by the plan below.

## 2026-07-21 — new mixing metric (inner activations) + θ-sensitivity

User redesign of the separation metric — CI-based n50 was wrong on three counts:
CI's saturating nonlinearity manufactures harmonic spread; a same-period read on both
operands (blob grid) is one period, not two; aperiodic components are fine. The
failure mode to measure is **one component reading ≥2 distinct periods**.

New `PeriodSeparation` eval metric (commit `4ada9c519`): answer-position inner
activations `x·V/‖V‖` over the full a+b grid (linear in the residual features → one
Fourier feature ≈ one spectral bin, no harmonic pooling needed); canonical period
classes T ∈ {2,4,5,10,20,25,50,100} via the bins a linear read can produce ((f,0),
(0,f), (f,±f)); component gated at mean CI > 0.1; `n_periods` = classes with share ≥ θ.
`mixed_frac` and `excess_periods` are over *periodic* components only. Also logs a
per-period census and an AB-heatmap-style inner-activation figure (top 20/matrix by
mean CI). Caveat: T=100 is indistinguishable from a smooth aperiodic trend on a
100-window.

θ-sensitivity on six full-length runs (jobs 5118/5119), mixed_frac:

| run | θ=0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|
| 04-hidden | 0.450 | 0.300 | 0.225 | 0.125 | 0.100 |
| 05-coupled | 0.490 | 0.367 | **0.184** | 0.083 | 0.042 |
| hid_sched | 0.458 | 0.271 | 0.188 | 0.125 | 0.064 |
| hid_sched-5x | 0.471 | 0.412 | 0.294 | 0.176 | 0.091 |
| hid_sched0.01 | 0.638 | 0.426 | 0.255 | 0.170 | 0.111 |
| 10x-hid0.1 | 0.500 | 0.429 | 0.214 | 0.185 | 0.074 |

- **Ranking is θ-unstable** (coupled: 4th at θ=0.10 → best at θ≥0.20; hid_sched-5x:
  3rd → worst at 0.20): many components hold a secondary period at 10–25% power.
  Mitigation: metric logs mixed_frac at θ ∈ {0.1, 0.2 (primary), 0.3} + a θ-free
  `secondary_share` (mean share of the 2nd-strongest class).
- Notably the *inner-activation* ranking inverts the old CI-based one — the 5x/10x
  runs that looked cleanest on CI grids are among the most mixed in weight-read space,
  while plain coupled is cleanest at θ≥0.2. CI cleanliness ≠ read-vector purity.
- periodic_frac ≈ 1.0 everywhere: essentially every used component is periodic on
  inner activations; the aperiodic escape hatch rarely triggers.
- Census@0.2 shows all canonical periods represented (T=50 dominates everywhere,
  ~40-50% of components).

## 2026-07-21 — S1 sweep launched: SmoothL0 coeff scan

Per user spec: 4000-step runs, γ = 1 constant (no anneal), coeff constant, **no
hidden-acts loss**, eval every 250 with slow_every 250 (fine-grained PeriodSeparation
curves + PGD/rounded recon). Constant hyperparameters → no schedule-compression
concern. Names `addsub-L18-09-sl0scan-<coeff>`, coeff ∈ {1e-5, 3e-5, 1e-4, 3e-4,
1e-3, 3e-3}; jobs 5120–5125 (3 concurrent, 3 chained). Selection metrics:
PGDReconLoss + rounded recon vs mixed_frac/secondary_share.

## 2026-07-21 — Objective 2 extended: within_span init arm

User-requested third init condition, `within_span` = coupled with the coupling
broken: `V_c ∝ Wᵀ g_c`, `U_c ∝ W h_c` with independent Gaussians — per-side
marginals identical to coupled (S²-weighted in-span directions, unit-norm narrow
side, W-image-scale wide side) but the sides are statistically independent, so
the component sum starts at zero mean and the delta carries all of W. Rationale:
span_proj (dropped pre-port in `61ca40b9d`) was in-span but threshold-dependent
and scale-shrunk by √(r/d); within_span keeps the in-span property with
W-natural scale and no rank threshold. Implemented + committed `0593b2a97`
(configs literal, `init_within_span_`, unit tests, roadmap Obj-2 revision).

Launched `addsub-L18-08-initwithinspan` (job 5173, yaml =
`psep2/obj1/addsub-L18-08-impmin3e-5.yaml` + `weight_init: within_span`),
dependency `afterany:5160:5161` to hold the 6-GPU cap while the Obj-3 hid sweep
runs; scoring chained `afterok` as job 5174
(`psep2/obj1/score_initwithinspan.sbatch`). Comparison vs kaiming
(`-08-impmin3e-5`) and coupled (`-08-initcoupled`) on n_pure_periods + PGD at
step 4000 to follow.

## 2026-07-22 — Objective 5 launched (1e-3 regime) + hid1 arm

Roadmap gained Objectives 5 (re-run init/hid/beta comparisons at impmin 1e-3) and 6
(gamma-anneal fine-tunes). Launched the 1e-3 trainable arms — `impmin1e-3` doubles as
the kaiming / hid-0 arm; beta arms at 1e-3 deferred until the 1e-3 init+hid winners are
known. Also added `hid1` (coeff 1.0) to the 3e-5 hid sweep per the roadmap edit.
All work queued as three strict serial 2-GPU chains (train -> score alternating), so
usage never exceeds the 6-GPU cap:

- slot 1 (behind initwithinspan 5201): score-WS -> hid1 (5224) -> score -> hid1-1e-3 (5230) -> score
- slot 2 (behind beta0 5220): score -> initcoupled-1e-3 (5233) -> score -> hid0.001-1e-3 (5235) -> score -> hid0.01-1e-3 (5237) -> score
- slot 3 (behind beta0.5 5221): score -> initwithinspan-1e-3 (5240) -> score -> hid0.1-1e-3 (5242) -> score

Yamls in `~/pd_scratch/psep2/obj5/` (from the impmin1e-3 yaml; init arms add
`weight_init`, hid arms add `hidden_acts_recon` to StochasticReconSubsetLoss).
Objective-6 readiness: SmoothL0 already supports gamma annealing
(`gamma_final`/`gamma_anneal_{start,end}_frac`); `training_4000.pth` snapshots exist in
all run dirs. Open detail: resume with edited steps/anneal config.
