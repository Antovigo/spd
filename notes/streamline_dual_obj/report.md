# Streamlining the dual CI objective

Design study, 2026-08-06. Two proposals for the dual-CI scheme before scale-up: **share a
trunk** between the two CI networks, and **enforce `CI_hidden >= CI_output` by
construction**. Written against `feature/dual_hidden_acts` at `af9212b00`; the empirical
inputs are `notes/hidden_dual/report.md` and the `addsub-L18-{09,10,11}` runs.

## Recommendation up front

Do both, as one change, in this shape:

```
                        ┌── base head ───────────────► logit_output
inputs ──► shared trunk ┤
                        └── surplus head ─► softplus ─┐
                                                      ├──► logit_hidden
                        clamp(logit_output.detach()) ─┘
```

One trunk (input projector + transformer blocks), two readout heads, hidden derived from
the *detached* output logit plus a non-negative surplus. That is `-46%` CI-fn parameters at
the current L18 config, exact monotonicity in all three CI branches, and it makes the
**surplus** — the quantity the whole experiment is about — a directly parameterized,
directly readable object rather than a difference of two independent estimates.

Three things gate it, all measurable before writing training code (§5), and one thing must
be conceded honestly: both proposals trade *measurement independence* for cost (§6).

---

## 1. Where the cost actually is

`addsub-L18-11-bigc`: 7 sites on L18, `total_C = 6144`, global transformer CI fn with
`d_model 512`, `n_blocks 4`, `mlp_hidden 2048`. Concatenated input dim is
`6 x 4096 + 14336 = 38912` (`down_proj` alone contributes 37% of it).

| part of one CI net | params | share |
|---|---:|---:|
| input projector (`38912 -> 512`) | 19.92 M | 56% |
| 4 transformer blocks | 12.59 M | 35% |
| output head (`512 -> 6144`) | 3.15 M | 9% |
| **one net** | **35.66 M** | |
| **two independent nets (today)** | **71.3 M** | |

Sharing the trunk and keeping two heads gives **38.8 M — a 46% cut**, i.e. 32.5 M
parameters, or roughly **0.5 GB** of the training footprint once gradients (4 B/param) and
Adam moments (8 B/param) are counted. That matches the `~0.55 GB per extra CI net` recorded
in `notes/hidden_dual/report.md`.

Two facts make this matter more at scale, not less:

- **The input projector — the dominant term — scales linearly in the number of decomposed
  sites.** At 3 blocks (21 sites) it is 59.8 M per net, so two nets cost ~151 M and a shared
  trunk ~79 M: **~1.2 GB saved**. The `addsub-L18to20-01-dual` run peaked at 42.2 GiB of a
  46 GiB card, and `addsub-L18-09-dual` was *rejected* at 427 MiB headroom. This saving is
  the same order as the decisions that were already forced (batch 128 → 96; 3 GPUs → 4).
- **C is a weak memory lever** (established: 304 → 228 bought under 1 GB, because the
  weight-delta tensors are full-weight-shaped). The CI nets are one of the few levers left
  that does not cost components or batch.

**Compute is not the argument.** One CI net costs ~0.44 TFLOP fwd+bwd per step here
(35.7 M params, 128 x 16 = 2048 tokens) against ~33 TFLOP for a *single* 8 B target forward,
of which the step runs several — so the second net is around 1% of the step and trunk sharing
recovers about half of that. The measured dual-vs-reference step time already showed the
scheme costing nothing. Do not sell this on speed.

## 2. The sharing spectrum

Ordered by how much is shared. Each row is a real design; the prediction column is what §3
argues.

| # | design | params (L18) | monotone for free? | prediction |
|---|---|---:|---|---|
| A | one net, one logit, per-site learned offset `tau >= 0` per role | 35.7 M | yes | **fails** — needs the two nets to agree on component *ranking* |
| B | shared trunk, two independent heads | 38.8 M | no | **works**, recommended |
| C | shared trunk, base head + non-negative surplus head | 38.8 M | yes | **works**, recommended |
| D | shared input projector, split blocks + heads | 51.4 M | no | fallback if B/C underfit |
| E | fully separate (today) | 71.3 M | no | the measurement instrument |

Note the degenerate endpoints. Sharing *everything* including the readout collapses to a
single CI net with two losses — which is the pre-dual setup (`addsub-L18-04-hidden`,
`StochasticReconSubsetLossConfig.hidden_acts_recon`) that the dual scheme was built to
replace because it conflates the two objectives. A is one scalar away from that collapse.

Monotonicity is a **separate axis** from sharing: C = B + constraint. They compose, and
option C is the joint implementation. But they can be landed and evaluated independently,
and if you want to attribute a result to one of them you should.

## 3. What the existing runs already say about which sharing works

The decisive evidence is in `notes/hidden_dual/report.md`:

- **Nesting is high but not from a common ranking.** Output-active implies hidden-active for
  97.7% of (component, position) entries; the converse fails for 46%. A pure threshold model
  (design A) predicts exactly that nesting — hidden = a looser cut of the same ranking. So
  nesting alone does *not* discriminate A from B.
- **k/v kills A.** At the answer position on `addsub-L18-11-bigc`, `k_proj` fires 0.01
  output components per prompt and `v_proj` 0.00 — medians and both quartiles are 0 over
  20 000 prompts — while the hidden net fires 6.41 and 11.75 there. Where the output net has
  no gradient signal, its *ranking is arbitrary*, and a threshold on an arbitrary ranking
  cannot reproduce a specific set of 6–12 components. Design A must fail on exactly the
  sites where the dual scheme's most interesting finding lives.
- **But the two nets need the same input features.** Both read the identical cached
  `pre_weight_acts`; both must answer "which components does *this* token/context engage".
  The featurization is objective-agnostic; only the scoring is not. That is precisely the
  trunk/head split.

So the evidence points at B/C: share the featurizer, let the heads re-rank freely. It also
predicts *where* B would show strain — attention q/k/v, where the two nets' answers are most
dissimilar — which is the thing to watch in the first shared-trunk run.

Additional argument for sharing that is easy to miss: the two nets currently learn their own
internal bases, and **the analysis compares them**. A shared trunk makes the comparison
happen in one representation. It also plausibly improves the hidden net's sample efficiency,
since it no longer relearns the featurization from scratch.

## 4. The monotonicity constraint

### 4.1 What it should mean

The falsifiable claim is *output-important implies hidden-important*. The clean encoding of
that is at the level of **supports**: the hidden mask must be a superset of the output mask.
The pointwise inequality `CI_hidden >= CI_output` is strictly stronger — it also orders the
intermediate values, and CI values are not calibrated across nets (one is a mask coefficient
for a KL in nats, the other for a dimensionless relative activation error). In principle
that is an overreach.

In practice it is nearly vacuous: measured on `addsub-L18-09-dual/model_20000.pth`, CI is
97.8–98.8% exactly 0 and 1.0–1.9% exactly 1 under `leaky_hard`, with 0.086–0.111% of entries
in `(0.01, 0.5)`. There is no usable middle band for the extra ordering to distort. Take the
pointwise version — it is far easier to enforce exactly and buys support nesting as a
corollary.

### 4.2 Direction: which net is free

**Make the output net free and derive the hidden net from it.** Two reasons:

- The 607 output-only entries are 2.3% of the output mask but carry 15% of output
  reconstruction quality (KL 0.004000 without vs 0.003401 with) — about 7x an average entry.
  These are the violations of the claim, and they are systematically high-value, not
  leakage. If the *hidden* net were free and the output net derived, the constraint would
  force those entries off in the output net (or drag the hidden net up to cover them by a
  path with no gradient reason to do so). Forcing them *on* in the hidden net instead costs
  the hidden objective 607 entries against 47 458 already-on — **1.3%**, negligible.
- The output objective is the one with an external anchor (KL against the target model).
  Constrain the derived quantity, not the anchored one.

### 4.3 Parameterization

Work in **logit space**, before the sigmoids:

```python
base = clamp(logit_output.detach(), min=0.0)          # detach: see 4.4
logit_hidden = base + softplus(surplus_head_out)
```

Then squash exactly as today. Four properties fall out:

- **Exact `>=` in every branch.** Both `lower_leaky_hard` (forward `clamp(x, 0, 1)`) and
  `upper_leaky_hard` are monotone non-decreasing, so ordering in the logit implies ordering
  in `lower_leaky`, `upper_leaky`, and `pre_sigmoid` alike. One parameterization covers all
  three fields of `CIOutputs`; no per-branch special-casing.
- **The clamp is free and is not a violation.** `clamp(o, min=0) >= o` always, so
  `logit_hidden >= logit_output` still holds; and below 0 the lower branch is already
  saturated at CI 0, so nothing is lost. Without it, an output logit that has drifted to −20
  would demand a surplus of 20 before the hidden CI could switch on — a scale mismatch that
  makes the surplus head's job artificially hard. (The impmin gradient dies at `x <= 0`
  under `upper_leaky_hard`, so logits are not driven arbitrarily negative — but the clamp
  costs nothing and removes the failure mode.)
- **bf16-safe.** `a + b` with `b >= 0` never rounds below `a` under round-to-nearest, so the
  inequality survives autocast. Assert it anyway, next to the existing
  `assert (lower_leaky_output <= 1.0).all()` — same style, one reduction.
- **The surplus is the object of study.** `softplus(surplus)` *is* "how much more important
  this component is for the activations than for the logits", available per (component,
  position) without differencing two independent estimates.

**Share the binomial noise draw.** `sampling: binomial` is live in the -11 configs, and
`_apply_sigmoid_to_ci_outputs` mixes `1.05 * x - 0.05 * rand_like(x)` into the lower-leaky
branch. Independent draws per role break the inequality by up to 0.05 even with identical
logits. `f(x) = 1.05x - 0.05u` is monotone in `x` for *fixed* `u`, so drawing `u` once and
reusing it for both roles restores exactness. This is the one silent-corruption trap in the
whole design.

**Init.** `zero_init_readout` currently starts both nets at logit exactly 0.5, so they emit
identical CI until gradients separate them — which is what makes the step-0 diagnostic table
readable. Under this parameterization exact equality at init needs `softplus(b) = 0`, i.e.
`b = -inf`. Pick a small positive initial surplus instead: `b = -3` gives `softplus = 0.049`
(hidden logit 0.549). Slightly denser at init, in the direction the run goes anyway. Record
it; the step-0 identity check changes meaning.

### 4.4 Why the detach is load-bearing

Without `.detach()` on the base, the hidden net's importance-minimality term penalizes
`upper_leaky(base + surplus)` — and it can reduce that by pushing **the output logit down**.
That is a first-order, direct incentive for the hidden objective to make the output net
sparser than the output objective wants. It would silently corrupt the very comparison the
run exists to make. Detaching kills that channel through the head.

Note that trunk sharing leaves a weaker version of the same coupling through the shared
features, which is unavoidable and is one of the costs of §2 option B. The head-level channel
is the one that is both first-order and free to remove.

Consequence for logging: `Phi_hidden` now structurally includes the base's contribution, so
its absolute value gains an uninformative offset (no gradient flows through it — only the
number moves). The meaningful quantity is `Phi_hidden - Phi_output`, which is already how
`notes/hidden_dual/report.md` reads it. Log the surplus explicitly.

### 4.5 Alternative: interpolation in CI space

`ci_hidden = ci_output + (1 - ci_output) * s`, `s in [0, 1]` — algebraically the same shape
as the PGD source→mask interpolation already in `pgd_utils.py`, bounded automatically, and
`s` reads as "fraction of the remaining headroom". Where `ci_output = 0` (98% of entries) the
hidden net is completely free, which is exactly where you want freedom.

Rejected because `lower_leaky` and `upper_leaky` are *different* functions of the logit, so
the interpolation has to be done twice with two different meanings, and `pre_sigmoid` for the
hidden role becomes ill-defined — breaking every downstream consumer that reads it. Logit
space is uniform across all three.

### 4.6 Soft variant, if the hard one disturbs training

Add `coeff * mean(relu(ci_output - ci_hidden))` to the loss and keep the nets structurally
independent. Pros: violations stay *measurable* (the residual hinge value is the violation
rate), no reparameterization, tunable pressure. Cons: another coefficient, no exact
guarantee, and it does not streamline anything — it adds a term. Keep it in the back pocket;
it is the right move only if the hard constraint visibly hurts the hidden objective.

### 4.7 Knock-on simplifications the constraint buys

- `ci_scaled_component_weight_decay` currently takes `torch.maximum` over both nets'
  per-component batch CI max (`optimize.py:829-836`). Under the constraint the hidden max
  dominates pointwise, so the `maximum` is a provable no-op. Keep the code, turn it into an
  assertion — it is a live check that the constraint is actually holding end-to-end.
- The `ab_grids` two-colour heatmaps **cannot** show magenta (output-important,
  hidden-unimportant). That was the primary step-5000 sanity check; it becomes a tautology.
  Replace it with the *binding rate*: the fraction of entries where the surplus is pinned at
  its floor while the hidden objective is still pushing down. That is the constraint's cost,
  and it is the honest replacement for the check you gave up.
- Alive sets nest by construction, so `find_alive_subcomponents`' dual-role lists and the
  `alive_plane_scatter` panels get a guaranteed containment structure instead of an empirical
  one.

## 5. Three measurements to make first — all cheap, none need a training run

**(1) Is a shared trunk expressive enough?** Decisive, and it needs no training. On
`addsub-L18-10-dual-ppgd/model_20000.pth` (or `-11-bigc`), hook the input of each net's
`_output_head` to cache the `d_model = 512` trunk representation over a few eval batches,
then least-squares fit a linear map from **net A's trunk** to **net B's logits**, per site.
If the shared trunk with separate heads is viable, a linear head on A's features recovers B's
answer. Report per-site R² and, more usefully, mask agreement at CI > 0.1. Pass ≈ agreement
> 95%; failure on q/k/v specifically is the predicted failure mode and would argue for
option D (split the last block) rather than E.

Caveat: this tests whether a *converged* trunk can express the other role, not whether joint
training finds such a trunk. A pass is good evidence; a failure is decisive.

**(2) Is the hidden mask just a lower threshold on the output ranking?** Per-site rank
correlation between `pre_sigmoid` logits of the two nets, plus AUC of the output logit as a
classifier for the hidden mask. Expect high on MLP, near-chance on `k_proj`/`v_proj`. If it
came out high *everywhere*, option A becomes viable and the second net collapses to one
scalar per site — a much stronger streamlining result, and worth 20 minutes to rule out.

**(3) What is the gradient-scale ratio between the two objectives?** This is already logged:
`component_grad_norms` runs every `train_log_every` (100) steps and namespaces the hidden
net under `grad_norms/ci_fns/hidden.*`. Pull `ci_fns/_input_projector.W` against
`ci_fns/hidden._input_projector.W` from any completed dual run. It matters because today the
two nets are disjoint and **Adam makes each net's updates invariant to its own loss scale**;
a shared trunk receives the *sum* of the two gradients through a single `total_loss.backward()`
(`optimize.py:863`), so the ratio suddenly sets which objective steers the shared features.
Within ~3x: proceed as-is. At ~30x: rescale, or normalize per objective before the trunk.

There is reassurance here already — the *components* pool has always received both losses'
gradients summed and it works — but the components are anchored by faithfulness while the
trunk is not.

## 6. Risks, stated plainly

- **Both proposals cost measurement independence.** Sharing a trunk correlates the two nets'
  outputs architecturally, so the 97.7% nesting statistic is no longer an independent
  measurement of it. The constraint converts that measurement into an assumption outright.
  This is acceptable *because the measurement has already been made* on L18 with independent
  nets — but any nesting claim at the new scale must either be caveated or checked against a
  separate-net control arm.
- **The "output net identical to a single-CI run" property is lost** under any trunk sharing,
  since hidden-loss gradients reach the trunk. The detach preserves it only at the head. The
  `addsub-L18to20-01-dual` vs `-ctrl` pair is the existing clean comparison; a shared-trunk
  run is not directly comparable to it.
- **Checkpoints break.** State-dict keys change. Per repo policy no migration shim, but land
  the change when nothing needs to resume: `pd.dual_hidden_ci` going from `bool` to a config
  object also breaks snapshot `pd_config` deserialization on resume, which has bitten before.
- **DDP unused-parameter hazard.** Harmless today because both losses fire every step. If the
  hidden loss ever gets a `start_frac` or a schedule, the *heads* become the unused
  parameters instead of a whole net — a smaller but still real trap.
- **The constraint does not fix the C ceiling.** Hidden `n_alive` was censored at C on four
  of seven sites in `addsub-L18-10-dual-ppgd`; that is a C-allocation problem (already
  addressed in `-11-bigc` at q/k 512, v/o 1024) and is orthogonal to everything here.
- **Expect to re-touch `ci_fn` LR.** Currently 1.6e-4 against 3.2e-4 for components. A shared
  trunk moves once under a summed gradient rather than twice independently; watch the CI L0
  curves in the first run rather than pre-emptively halving it.

## 7. Implementation shape

Config — replace the bool, keeping the two axes separable:

```yaml
pd:
  dual_hidden_ci:
    share_trunk: true
    monotone: true
```

`dual_hidden_ci: DualHiddenCIConfig | None = None` in `PDConfig`; omitted means single net.

Code:

- `make_ci_fn_wrapper(..., n_roles: int)`. Every CI fn already has the shape
  `input -> [..., C]` per layer, so "two roles" is a second readout on the same body. For
  `GlobalSharedTransformerCiFn` that is a second `Linear(d_model, total_c)` (prefer a second
  head module over widening the existing one to `2 * total_c`: cleaner state dict, and it
  lets `separate` / `shared_trunk` / `monotone` all live behind the one config). `MLPCiFn`'s
  final `ParallelLinear` goes to `n_roles` outputs instead of 1 — but the LM runs are all
  `global_shared_transformer`, so implement that path and assert on the others.
- The trunk must be run **once** for both roles. `_build_metric_context`
  (`optimize.py:174-191`) currently calls `calc_causal_importances` twice, which would
  recompute the shared trunk. Add a both-roles entrypoint returning
  `dict[CIRole, CIOutputs]` and use it there; keep the existing single-role
  `calc_causal_importances(role=...)` for the ~15 lab call sites, almost all of which want
  `"output"` only. `ComponentModel.ci_fn_for(role)` stays the public seam so nothing
  downstream — app, harvest, `find_alive_subcomponents`, `ab_grid_dataset` — changes.
  (`detach_inputs` must match across roles for a single trunk pass; it does today.)
- Monotonicity lands inside that both-roles path, before `_apply_sigmoid_to_ci_outputs`, and
  the shared binomial noise draw lands inside `_apply_sigmoid_to_ci_outputs`.
- Assertions: `logit_hidden >= logit_output` elementwise; `maximum` in the CI-scaled weight
  decay is a no-op under `monotone`.

Docs to update: `param_decomp/metrics/CLAUDE.md` (the *Dual CI networks (`CIRole`)* section,
including the weight-decay rule which the constraint changes), the root `CLAUDE.md` dual-CI
paragraph, and `notes/hidden_dual/report.md` with a pointer here.

## 8. Validation plan

Run at the smallest configuration you still trust the science on — `addsub-L18-11-bigc`'s
shape — before spending the scale-up budget:

| arm | `share_trunk` | `monotone` | purpose |
|---|---|---|---|
| control | false | false | reproduces `-11-bigc`; the reference |
| shared | true | false | isolates trunk sharing |
| both | true | true | the production instrument |

Compare on: output KL, `CIHiddenActsRecon_{outputCI,hiddenCI}`, `PGDHiddenActsReconLoss`,
per-site `n_alive` and `CI_L0` for both roles, and peak memory. What would send you back:
output KL degrading materially in `shared` (trunk contention — go to option D), or the hidden
net's relative error rising in `both` (the constraint is binding harder than the 1.3%-of-entries
estimate predicts — check the binding rate, consider the soft variant).

Then keep **one** separate-net arm alive at the scale-up config so nesting remains an
independently measured fact rather than an architectural assumption.

## 9. Things not to do

- **Do not collapse to one head with a per-site threshold** (option A) without measurement
  (2) passing — the k/v evidence predicts it fails exactly where the interesting result is.
- **Do not narrow `site_patterns` to the residual-stream writes** as part of this. Already
  settled: attention carries 47% of the hidden error at kappa 0.035–0.074 and is the most
  distinctively-hidden signal; narrowing would align the hidden objective with output
  relevance and discard it.
- **Do not try to unify the two importance-minimality coefficients** via the exchange rate.
  Already settled: `kappa` spans 200x by direction, and unit conversion prices the hidden-only
  surplus at ~0, which drives `lambda_hidden` to infinity and reproduces the output net. The
  standing recommendation is `lambda_hidden 5e-5 -> 1e-4` (surgical: shaves the near-threshold
  fringe at ~2x its keep-threshold, cannot touch the bulk at ~200x), `lambda_out` unchanged.
  Both are still 5e-5 in the -11 configs — worth applying independently of this work.
