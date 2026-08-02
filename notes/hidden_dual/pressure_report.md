# More components, and more pressure on the hidden objective — the `-11` series

Status as of 2026-08-02: **three runs launched, no results yet.** The dual-CI scheme is
described in `report.md`; chronology in `lab_notebook.md`.

Two questions, both asked against `addsub-L18-10-dual-ppgd` as the control:

1. **Does simply making the decomposition bigger cost anything?** The reference sits at its
   alive-count ceiling — the hidden net had `q_proj` and `k_proj` at exactly 128/128 — so
   every density readout it produces is clipped. Raising C removes the clip, but a larger
   component pool could plausibly slow convergence or dilute the components. `bigc` measures
   that directly.
2. **Is the hidden objective underweighted?** Its two reconstruction coefficients are scaled
   2x and 5x, with everything on the output side held fixed.

## The runs

| run | C | hidden recon coeffs (stochastic / PPGD) |
|---|---|---|
| `addsub-L18-10-dual-ppgd` (control, already complete) | 1536 | 1.0 / 0.5 |
| `addsub-L18-11-bigc` | 6144 | 1.0 / 0.5 |
| `addsub-L18-11-press2` | 6144 | 2.0 / 1.0 |
| `addsub-L18-11-press5` | 6144 | 5.0 / 2.5 |

Each config is generated from the reference run's own YAML rather than from a descendant, so
the replication is exact by construction. Verified by diff:

- `bigc` vs the reference differs in **the label and the seven C values, nothing else**.
- `press2` / `press5` vs `bigc` differ in **the label and two coefficients, nothing else**.

Untouched in all three: `StochasticReconSubsetLoss` (1.0), `UnmaskedReconLoss` (0.5),
`PersistentPGDReconLoss` (0.5), both `SmoothL0ImportanceMinimalityLoss` instances (5e-5,
output and hidden), every optimizer and schedule, seed, batch sizes, the nontarget block,
and the eval metric list. So the pressure runs move the hidden reconstruction weight and
nothing else — in particular the sparsity penalty the hidden net competes against is
unchanged, which is what makes "more pressure" mean what it says.

20000 steps with `gamma_anneal_start_frac: 0.5` — annealed over the second half. That is
already the reference's schedule, so no change was needed.

Note this series carries **no `hidden_readout_sites`** and no residual-stream eval probes.
Both would have been harmless additions, but `site_patterns: null` means "every measurement
site", so declaring readout sites would silently widen the hidden objective from 7 matrices
to 9 and break the exact replication.

## What the earlier series established

The predecessor under this name (now superseded, see `site_targets_report.md`) compared
*where* the hidden objective is measured. Two of its results bear on this one:

- **C at 4x is not a compromise.** At matched step 10000 the 4x-C run beat the 1x-C
  reference on every output and hidden metric, and saturation fell from 88-100% to 28-47%.
  `bigc` is the clean 20000-step test of that, against the reference directly.
- **Per-site gradient weight is `1/n_sites`.** The hidden loss is a mean over sites, so
  changing the site set silently changes the pressure. This series changes the coefficient
  instead and holds the site set at all 7 matrices, which is the cleaner way to ask the
  pressure question.

## Read-outs

Same panel as before: absolute alive counts and `CI_L0` per locus for both CI nets with
MLP/attention subtotals, saturation against C, the anomaly census from the `ab_grids`
payloads, and output quality. The combined score is `PGDReconLoss * alive-either` — both
are costs, so they multiply; a ratio would credit a run for keeping more components.
`alive-either` is the union over both CI nets, since they score the same subcomponent pool.
