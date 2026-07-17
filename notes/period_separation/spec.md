# Period separation — goal, metric, and scripts

Started 2026-07-17. Goal: find hyperparameters / training recipes for the targeted L18
addsub decomposition such that subcomponents for **different operand periods** (2, 5, 10,
20, 50, 100 — the model's Fourier features store integers modulo these) end up in
**different subcomponents**, instead of one subcomponent mixing several periods.

Companion files: [lab_notebook.md](lab_notebook.md) (dated observations),
[commands.md](commands.md) (runnable invocations), [report.md](report.md) (key findings).

## Why mixing happens — the force balance

Two families of loss terms pull in opposite directions:

- **Separation forces** — the masked-reconstruction losses (`StochasticReconSubsetLoss`,
  `PersistentPGDReconLoss`, `StochasticHiddenActsReconLoss`). A component whose weights
  carry two periods injects the *wrong* period's feature whenever a mask activates it for
  the other one, which costs KL / hidden-MSE. The hidden-acts loss is the sharpest of
  these: it demands the masked circuit reproduce the MLP's internal activations, which are
  themselves period-specific neurons.
- **Merging force** — `ImportanceMinimalityLoss` is a per-datapoint Lp norm on the CI
  vector. Once `p < 1` (concave), it prefers *fewer active components per datapoint*.
  All the mod-p features of an operand co-occur on essentially every prompt (the answer
  needs every residue), so two single-period components are strictly more expensive under
  concave impmin than one merged two-period component. The default schedule anneals
  `p: 2.0 → 0.5` across training and doubles the coeff mid-training
  (`coeff_peak_multiplier: 2.0`) — i.e. the merging pressure is applied exactly as the
  topology crystallises.
- `ci_scaled_component_weight_decay` (0.3) prunes rarely-used components; it shapes
  *how many* survive more than *what each one holds*.

Working hypotheses (tested by the probe runs):

1. **H1 — late concave impmin merges periods.** Keeping `p ≥ 1` (or annealing less deep /
   later) should yield cleaner per-period components, possibly at the cost of more alive
   components.
2. **H2 — the mid-training coeff peak is mistimed.** It lands while the topology is still
   forming (~step 3–8k of 20k); removing it (peak multiplier 1.0) should reduce merging
   with little sparsity cost (the 5x-peak experiment showed the reverse direction:
   5x peak → much worse recon, no sparsity gain).
3. **H3 — early hidden-acts pressure sets the topology.** hid_sched (0.1 early → 0 late)
   vs hid_sched0.01 (0.01 early) changed the final circuit size (177 vs 146) — the early
   hidden-acts coefficient is a topology knob. Higher early values should pin components
   to individual (period-specific) neurons harder.
4. **H0 — scheduling interactions are subtle** (user note): probe *schedules*, not just
   magnitudes, and expect interactions (e.g. impmin peak timing × p-anneal depth).

## Metric — `score_period_separation`

`param_decomp_lab/scripts/validation/score_period_separation.py` (full spec in
`param_decomp_lab/scripts/validation/spec.md`). Reads the per-position CI JSON that
`find_alive_subcomponents` already produces for every analysed run — no GPU pass.

Per (op, position, subcomponent): 2D FFT of the `[b, a]` CI grid → conjugate frequency
orbits labelled `a` / `b` / `a+b` / `a-b` / `mixed2d` + integer period.

- `band_purity` — power share of the top orbit *plus its harmonics* (near-binary CI
  stripes spread power over harmonics of one fundamental; those are one clean pattern,
  not a mixture). **The headline per-component number; > 0.5 = clean.**
- `purity` — top orbit alone (strict sinusoid purity).
- `n_orbits_50` — orbits needed for 50% of power (1 = clean; robust to the speckle-noise
  floor, unlike the 90% variant).
- `flat` — always-on grids (std < 0.05) carry no period; excluded from aggregates.

Run-level comparison numbers (in `period_separation_summary.tsv`, per op × pos × matrix):
`n_clean`, `n_flat`, median / mass-weighted `band_purity`, `mean_n_orbits_50`, and the
per-period component counts (`period_counts`, e.g. `a:10=5 b:50=6 ...`).

Caveats:

- Subtraction prompts cover a triangle only → 1D-marginal fallback; diagonal (`a±b`)
  structure is invisible there. **Compare runs on the `+` rows.**
- CI grids are read at the JSON's collection threshold (CI > 0.1 stored, else 0) — the
  metric sees the *usage* pattern, not the weight geometry. The weight-side twin
  (`collect_inner_activations` → `compute_subcomp_periods`) needs a GPU pass; use it to
  confirm a winner, not to screen.
- Comparisons across runs are only fair at matched checkpoints/steps and matched
  `--min-mass`.

## Probe protocol

The topology (which component holds which period) is set by ~10k steps, usually earlier —
so probes are **5000-step runs** with the full schedule compressed into those 5000 steps
(all schedule knobs are fractions of `steps`, so shortening `steps` compresses shapes,
it does not truncate them). A `psep-base` probe (the unmodified coupled recipe at 5000
steps) anchors every comparison; a probe result only counts if it beats `psep-base` at
the *same* step count. Winners get re-validated at 20k.

Probe naming: `addsub-L18-06-psep-<variant>`. Configs in
`~/pd_scratch/subspace_restriction/cifn_pipeline/psep/` (ad-hoc artifacts stay out of the
repo). Trainings launch from the `subspace_restriction` worktree (their configs validate
natively there); scoring/plots run from `8B_targeted`.

Per-probe analysis chain (one job each, see commands.md): `find_alive_subcomponents` on
the final checkpoint → `score_period_separation` + `plot_ab_heatmaps`.
