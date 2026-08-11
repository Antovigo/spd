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

### What the measurements said before launching

Two jobs ran against the finished `addsub-L18-11` checkpoints before any coefficient was
chosen. Both point the same way: **the ordering already holds, nearly everywhere.**

Per *entry* — one (prompt, position, subcomponent) triple, `sampling: continuous`, 2048
prompts:

| run | mean CI out | mean CI hidden | mean shortfall | shortfall \| alive | frac alive with shortfall > 0.1 |
|---|---:|---:|---:|---:|---:|
| `press2` (independent nets) | 0.0040 | 0.0139 | 3e-5 | 0.0064 | 1.7% |
| `press2-trunk` | 0.0043 | 0.0138 | 2e-5 | 0.0041 | 1.1% |
| `trunk-imp2x` | 0.0031 | 0.0104 | 1e-5 | 0.0027 | 0.8% |

The hidden CI runs 3–4x *above* the output CI on average, and the mean excess (~0.008) is
some 300x the mean shortfall. The trunk runs violate less than the independent-net run,
which is weak support for the shared trunk steering away from cheating.

Per *subcomponent* — pool-mean CI, alive at 0.1, "magenta" = output-alive while the hidden
net rates it under half as important:

| run | alive (output) | alive (hidden) | both | output-only | hidden-only | magenta |
|---|---:|---:|---:|---:|---:|---:|
| `press2-trunk` | 45 | 107 | 45 | 0 | 62 | **0** |
| `trunk-imp2x` | 34 | 83 | 34 | 0 | 49 | **0** |
| `local` | 43 | 100 | 42 | 1 | 58 | **1** |

At the component level the constraint is already satisfied outright: every output-alive
subcomponent is also hidden-alive, in all three runs, with a single exception in the `local`
run. The hidden net marks ~2.4x as many components alive as the output net — the
interference population the dual objective was built to expose.

**So the intervention is narrower than the framing suggested, and the two units disagree
on purpose.** A component whose pool-mean is well-ordered can still violate on particular
prompts, which is what the ~1% of alive entries are — and what the `ab_grids` merge view
renders per (op, a, b) cell rather than per component. The constraint acts there: on
specific prompts, for components that look fine on average. Expect a small effect; a large
one would be surprising and would need explaining.

One correction to an earlier draft of this note, which said the floor makes the hidden head
"learn only the excess" over the output CI. That is wrong for the smooth-max form chosen
here: wherever the hidden logit is already above the floor — 99% of the time — the max is
the identity and the head learns the value directly. The floor is a *clipping* intervention
on the violating minority, not a reparameterization of everything. The smooth max is still
the right form, because it permits equality where `z_out + softplus(h)` would forbid it.

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

Both losses normalise the same way (sum over subcomponents, mean over positions), so their
coefficients compare directly. The hidden impmin's per-entry gradient is
`coeff * dφ/dc = 5e-5 * 0.65/γ`: 3.3e-5 at `γ=1` early, 3.3e-3 at `γ=0.01` late. The two
soft arms bracket it — `1e-3` is firm early and overrulable late, `1e-2` stays above the
late value throughout and should land close to the hard floor.

## The binomial jitter, and why the two roles must be computed together

Found during review and confirmed on hardware. The floor orders the *logits*, and both
squashes are monotone — but `sampling: binomial` mixes `-0.05 * rand_like` into the
lower-leaky branch before squashing, and the two roles were drawing that noise
independently, having come from two separate `calc_causal_importances` calls. The jittered
gap is `1.05·Δz − 0.05·(u_h − u_o)`, so the order inverts whenever `Δz < 0.0476` — and `Δz`
*is* small exactly where the floor binds (0.0135 at a logit gap of −0.2, 0.00067 at −0.5).
The guarantee failed precisely on the entries it exists for, and only on the branch that
actually masks components.

A 3-step probe on the real target measured it: `HiddenCIShortfallLoss` read **51.2 at step 0
of a floored run**, where the true shortfall is identically zero. That number is
`6144 components x 0.0083` — the expected jitter bias exactly. The diagnostic was reading
pure noise, two orders above the real signal, and the soft arm would have spent its early
training pushing hidden logits ~0.048 above output logits just to escape the noise floor.

`ComponentModel.calc_causal_importances_both_roles` computes both roles in one pass and
hands them one shared draw. Being a monotone transform of both, it preserves the ordering
exactly. It also removes a redundant output-net forward — the floor needs the output logits,
which are now the same tensor the output role returns — and with it a latent failure mode
where a single ULP of kernel nondeterminism between two forwards could trip the per-step
assert and abort a 20-hour run.

## Implementation checklist

- `masks.py`: `AllRoutingConfig` (`type: "all"`) into the routing union, renamed
  `SubsetRoutingType` -> `RoutingType` / `get_subset_router` -> `get_router`.
- `configs.py`: `PDConfig.hidden_ci_floor: HiddenCIFloorConfig | None`, validated to require
  `dual_hidden_ci`.
- `component_model.py`: apply the floor in `calc_causal_importances(role="hidden")`; store
  the config; new constructor argument threaded from both build sites (`optimize.py`,
  `component_model_io.py`).
- `optimize.py`: per-step assert of the ordering on `pre_sigmoid` when the floor is on.
- `metrics/hidden_ci_shortfall.py`: the loss; register in `dispatch.py`,
  `AnyLossMetricConfig`, `EVAL_METRIC_CLASSES`, `AnyEvalMetricConfig`.
- Tests: ordering holds under the floor, including under `binomial`; the floor is identity
  when the hidden logit is already above; the shared-trunk case leaves the output *head*
  ungradiented (the trunk itself is legitimately trained by both); the shortfall metric
  detects a planted violation and its `compute()` divides by the right two denominators;
  `all` routing routes every position.
- CLAUDE.md: `param_decomp/metrics/` (routing + the new loss), core config docs.

## Reviewed and deliberately not done

- **Splitting the CI fn into `trunk` / `readout`** so both roles read one trunk forward.
  Worth ~2.8x of CI-net cost under a shared trunk, but the CI net is ~0.5% of a step, and
  the 2x redundancy between the two roles predates this work. The joint entry point removed
  the *third* forward, which is the part this change added.
- **Narrowing `all` to the hidden loss's routing field.** It does make
  `StochasticReconSubsetLoss(routing={type: all})` an alias for `StochasticReconLoss`, and
  likewise for three other pairs — but `StaticProbabilityRoutingConfig(p=1.0)` already
  aliased all four, so the redundancy predates the new member, and a bespoke union for one
  field costs more than it saves.
- **Collapsing `HiddenCIFloorConfig` to a bare `float | None` on `PDConfig`.** The config
  block reads better in YAML and matches how `nontarget:` expresses an optional feature.
- **Consolidating the three near-identical `ComponentModel` test fixtures** across
  `test_hidden_ci_floor.py`, `test_dual_hidden_ci.py` and `tests/metrics/fixtures.py`. Real
  duplication, but it reaches well outside this change.
