# Chained hidden recon + enforcing `CI_hidden >= CI_output`

Design note for the arm that follows the trunk / local / impmin sweep. Written before
implementation; the run results go in a sibling report.

## Where the last series left us

Nine runs, `1x / 2x / 5x` impmin peak crossed with `trunk` (chained) / `local` /
`trunk+local`. Two conclusions carried forward from that sweep:

1. The shared trunk does not harm the decomposition, and may steer it away from
   mechanically-unfaithful components that buy minimality cheaply.
2. The **local** hidden reconstruction gets better final metrics but cheats: the
   `ab_grids` applet shows many magenta subcomponents — output-important but
   hidden-unimportant. That is exactly the anomaly the merge view was built to catch.

So the local formulation is abandoned as a *recipe*. The code stays: deleting
`site_inputs` would make the six finished `local` runs' `experiment_config.yaml`
unloadable (`SavedLMRun.from_path` re-validates with `extra_forbidden`), and the
chained-minus-local difference remains the only direct read of the inherited error.

`site_inputs: masked_forward` is already the default, so the new arms simply do not
mention the field.

## Part A — let the ablations compound

The chained formulation runs a real forward pass and lets each site see the damage its
upstream neighbours did. But *how much* damage reaches a site is set by the **routing**,
and today the stochastic hidden loss routes with `uniform_k_subset`:

> for each position, draw `k ~ U[1, 7]`, route that position through a random `k`-subset
> of the seven decomposed matrices; the rest run the frozen weights.

A site is scored only at the positions where it is itself routed, and at those positions
each upstream site is routed with probability `E[k]/7 = 4/7`. So roughly **43% of the
upstream chain is frozen** at any scored position — the compounding is real but heavily
diluted, and a downstream matrix mostly sees clean inputs.

The fix is to route everywhere. `AllLayersRouter` already exists in `masks.py` and is
already what `PersistentPGDHiddenActsReconLoss` uses (`_router_for_cfg`'s `case _`), so
the adversarial half of the hidden objective has been running fully-replaced forwards all
along; only the stochastic half was subsetting. Adding an `all` member to the routing
union closes that gap:

```yaml
- type: StochasticHiddenReconSubsetLoss
  ci_role: hidden
  coeff: 2.0
  routing: {type: all}      # was {type: uniform_k_subset}
```

Consequences to expect, none of them blocking:

- Errors get larger, especially at `o_proj` and `down_proj` which sit downstream of the
  most replaced matrices. The relative-error denominators are unchanged, so the logged
  numbers are **not** comparable to the previous runs' — the new baseline arm exists to
  re-anchor them.
- Every position is scored rather than the routed subset, so the loss is an average over
  ~1.75x more entries. Cheaper per unit of signal, not more expensive: the forward pass
  was being run in full either way.
- The output objective keeps `uniform_k_subset`. That is press2's recipe and is not what
  this change is about.

Not turned on here, but adjacent and available: `pd.hidden_readout_sites` would let the
error be read further downstream (post-attention and post-MLP residual stream), which is
the only way to make ablations compound past the decomposed block itself. Worth a later
arm; it changes the denominators again, so not in the same step as the routing change.

## Part B — enforcing `CI_hidden >= CI_output`

### Why the ordering should hold

A subcomponent affects the model's output through exactly one channel: the output of the
matrix it lives in. So if masking it changes the logits, it changed that matrix's output
— which is precisely what the hidden objective measures. Output-important therefore
implies hidden-important, and the converse fails exactly where we want it to: a component
can move its own matrix's output and be cancelled downstream (interference, dead ends),
which is hidden-important but output-unimportant.

Two caveats worth stating, because they bound how much to trust the constraint:

- The implication needs **every decomposed site measured**. If the hidden loss's
  `site_patterns` excluded a site, a component living there could be output-important with
  no hidden error to show for it. Our configs measure all seven.
- It is qualitative, not quantitative. The errors are *relative* per site, so a component
  that perturbs its site by a hair which then gets amplified downstream is output-critical
  and hidden-negligible. The ordering is a prior with a good reason behind it, not a
  theorem about the learned mask values.

### Mechanism 1 (hard): floor the hidden logit at the output logit

Both CI nets emit a pre-sigmoid logit per (position, subcomponent), squashed by
`lower_leaky` (the mask) and `upper_leaky` (the minimality penalty). **Both squashes are
monotone non-decreasing**, so an ordering imposed on the logits is inherited by both
branches at once. That makes logit space the right place to put the constraint — one
expression, no special-casing per branch.

```
z_hidden = z_out.detach() + softplus(z_hidden_raw - z_out.detach(), beta=sharpness)
```

which is a smooth `max(z_hidden_raw, z_out)`. Why this form and not the obvious ones:

| form | fails |
|---|---|
| `z_out + softplus(h)` | at init both heads read `b=0.5`, so the offset starts at `softplus(0.5)≈0.97` and the hidden CI starts pinned at 1 with **zero** mask gradient (`lower_leaky` is flat above 1). Diverges from the baseline before step 1. |
| `max(z_out, h)` (hard) | gradient to the hidden head is identically zero wherever the floor binds, so a hidden logit that drifts under the floor can never come back. Dead unit. |
| smooth max (chosen) | at init `h = z_out = 0.5`, offset `= ln2/beta = 0.069` — the run starts where the baseline starts. Gradient is `sigmoid(beta·(h - z_out)) ∈ (0,1)`, never exactly zero, so the floor is escapable in both directions. |

`beta = 10` puts the knee about `0.1` wide in logit units, i.e. ~10% of the sigmoid's
active window. `F.softplus(x, beta=)` computes this with its own linear-regime threshold,
so no overflow guard is needed.

`z_out` is **detached**. Without that, the hidden objective's own sparsity pressure would
push the output net's logits down to satisfy itself — the hidden impmin would silently act
on the output CI, confounding exactly the comparison the dual setup exists to make. With
the detach the hidden objective trains only the hidden head (and the shared trunk through
it), which is what the baseline does too.

Placement: inside `ComponentModel.calc_causal_importances`, so *every* caller asking for
`role="hidden"` gets the floored value — the trainer, the eval metrics, `find_alive`, the
app, harvest. Putting it in the trainer instead would leave every analysis tool reading a
CI the model never actually used. The redundant output-net forward this costs on the
hidden path runs under `no_grad` and is ~0.25% of one target-model forward, so it is not
worth optimising away.

The guarantee is exact on `pre_sigmoid` and on `upper_leaky`. On `lower_leaky` under
`sampling: binomial` the two nets draw independent noise (`1.05z - 0.05u`), which can
invert the order when the gap is below `0.0476`. That is a property of the mask jitter, not
of the parameterization; the per-step assert therefore checks `pre_sigmoid`.

### Mechanism 2 (soft): penalise the shortfall

`HiddenCIShortfallLoss` — mean over sites of `mean(relu(ci_out.detach() - ci_hidden))`,
on the `lower_leaky` branch.

- **Linear hinge, not squared.** The linear form's value *is* the mean shortfall, so the
  training loss and the diagnostic are the same number and comparable across runs; and its
  gradient does not vanish for small violations, which a penalty meant to *enforce* an
  ordering needs.
- **`lower_leaky`, not `upper_leaky` or the raw logit.** The raw logit is wrong: `z_out=5`
  against `z_hidden=1.2` is a huge logit gap and no violation at all, both CIs being 1.
  Between the two squashes, `lower_leaky`'s custom backward leaks below zero *when the
  gradient is negative*, which is exactly the direction this penalty pushes — so it can
  revive a hidden CI that has saturated at 0. `upper_leaky` is flat there and cannot.
- **`ci_out` detached**, for the same reason as above: the penalty should raise the hidden
  CI, never lower the output CI to meet it.

Registered as both a loss and an eval metric (the `PGDReconLoss` precedent), because the
shortfall is worth logging on the arms that do not penalise it — including the hard arm,
where it should read ~0 and is a self-check on the parameterization.

### Which to run

Both, plus a baseline, all on top of Part A:

| arm | Part A | hard floor | shortfall penalty |
|---|---|---|---|
| `chained` | yes | — | eval-only |
| `soft` | yes | — | loss |
| `hard` | yes | yes | eval-only |

Three runs at ~20 h and 2 GPUs each fits the GPU cap concurrently. The baseline is not
optional: Part A changes the hidden loss's numbers, so neither of the constrained arms is
comparable to any existing run.

The penalty coefficient is set from a measurement, not a guess: job 8338 reports the
observed shortfall on the finished `press2-trunk`, `press2` and `trunk-imp2x`
checkpoints. If the shortfall is already negligible the whole exercise is moot and the
soft arm should be dropped rather than run.

## Implementation checklist

- `masks.py`: `AllRoutingConfig` (`type: "all"`) into the routing union; `get_subset_router`
  returns `AllLayersRouter`.
- `configs.py`: `PDConfig.hidden_ci_floor: HiddenCIFloorConfig | None`, validated to require
  `dual_hidden_ci`.
- `component_model.py`: apply the floor in `calc_causal_importances(role="hidden")`; store
  the config; new constructor argument threaded from both build sites (`optimize.py`,
  `component_model_io.py`).
- `optimize.py`: per-step assert of the ordering on `pre_sigmoid` when the floor is on.
- `metrics/hidden_ci_shortfall.py`: the loss; register in `dispatch.py`,
  `AnyLossMetricConfig`, `EVAL_METRIC_CLASSES`, `AnyEvalMetricConfig`.
- Tests: ordering holds under the floor; the floor is identity when the hidden logit is
  already above; the shortfall metric detects a planted violation; `all` routing routes
  every position.
- CLAUDE.md: `param_decomp/metrics/` (routing + the new loss), core config docs.
