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
higher C on the MLP matrices, seed replicate of psep-base for the noise floor.
