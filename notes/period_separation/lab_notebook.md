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
