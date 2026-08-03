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
| `addsub-L18-11-bigc-mlp` | 6144 | 1.0 / 0.5, measured on `*.mlp.*` only |
| `addsub-L18-11-bigc-zeroinit` | 6144 | 1.0 / 0.5, `weight_init: coupled_zero_u` |

`bigc-mlp` is `bigc` with the hidden objective restricted to the three MLP matrices; it
differs from `bigc` in the label and the two `site_patterns` fields, nothing else.

It is not purely a locus change, and the interaction is the interesting part. The hidden
loss is a **mean over sites**, so restricting 7 sites to 3 raises the per-site gradient
weight by 7/3 = 2.33x. `bigc-mlp` therefore applies roughly `press2`-level pressure, but
concentrated on the MLP instead of spread across all seven matrices. The pair separates
*concentrating* pressure from *increasing* it — a confound the superseded series could not
untangle, because there the site set and the coefficient were never varied independently.

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

## `bigc` is worse under the fresh PGD probe — and why that is mostly a probe artifact

`bigc` finished first. Against the reference at step 20000:

| metric | reference | `bigc` | |
|---|---|---|---|
| `UnmaskedReconLoss` (clean) | 0.001755 | 0.001484 | -15%, `bigc` better |
| `PersistentPGDReconLoss/output_recon` (trained-against adversary) | 0.003593 | 0.003600 | identical |
| `PGDReconLoss` (fresh eval adversary) | 0.004429 | 0.004977 | +12%, `bigc` worse |
| `PGDHiddenActsReconLoss` | 0.03386 | 0.03223 | `bigc` better |
| `CI_L0` output / hidden | 23.87 / 57.79 | 24.62 / 59.25 | +3.2% / +2.5% |
| `NAlive` output / hidden | 1107 / 1387 | 1253 / 1899 | +13% / **+37%** |

Every objective the run is actually optimized against is flat or better. The whole
regression sits in the headroom between the persistent adversary and a fresh one.

`PGDReconLoss` is **not a C-invariant yardstick**, for two compounding reasons:

1. The eval mask is `ci + (1 - ci) * s` with `s` in `[0, 1]`, so the adversary can only push
   components *up* from their CI value. Its playground is exactly the near-zero-CI
   components: 1536 - 1107 = **429** of them in the reference, 6144 - 1253 = **4891** in
   `bigc`. An 11.4x larger attack surface buys a 12% worse number.
2. It is sign-PGD (`pgd_utils.py::_run_pgd_loop`) with a per-coordinate `L_inf` step and no
   budget on total injected mask mass. 20 steps at 0.1 saturates any coordinate, so the
   reachable set is the whole hypercube `[ci, 1]^C` — its dimension grows linearly with C.

Two further signs this is surface rather than quality: the gap **shrinks** over training
(1.28x at step 2000, 1.12x at 20000), and the fresh-vs-persistent ratio widens with C (1.23x
reference, 1.38x `bigc`) — the signature of a training adversary that under-covers a larger
mask space, not of a worse decomposition.

The one genuine cost, which should be tracked separately from PGD: **fragmentation**.
Per-position density is flat (+2.5%) while the hidden-net alive set grows 37%. Alive per
unit of `CI_L0` goes 24.0 -> 32.1: the same computation, spread over more and rarer
components.

### `bigc-zeroinit` — the direct test

`weight_init: coupled_zero_u` is `coupled` with `U` zeroed, added in `optimize.py`. The
component sum is exactly zero at init and the delta carries all of W; subcomponents acquire
norm only as the reconstruction losses demand it.

Zeroing both sides is a dead fixed point — `U`'s gradient is proportional to the component
acts `x @ V` and `V`'s is proportional to `U`, so both vanish. Zeroing `U` alone is the only
workable form, and it is also the right one: `V` is untouched, so `get_component_acts` still
feeds the CI nets a live signal, and `U` has a nonzero gradient from step 0.

This attacks the attack surface at its root. Under `coupled`, a subcomponent that is never
needed keeps W-natural norm forever — nothing decays it — so the adversary switching it on
injects real garbage. Under `coupled_zero_u` an unused subcomponent sits at exactly zero and
switching it on injects nothing. If the C-dependence of `PGDReconLoss` is the surface effect
diagnosed above, `bigc-zeroinit` should recover the reference's PGD number at 4x C.

The degenerate "components stay at zero forever" fixed point is not stable: the delta's mask
is `torch.rand` per position (`masks.py::calc_stochastic_component_mask_info`), so the delta
is partially ablated every step and the output is wrong unless the components carry the
weight.

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
