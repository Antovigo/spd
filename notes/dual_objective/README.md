# Dual-objective decomposition in JAX

The JAX port of the torch `feature/dual_ci_shared_trunk` scheme: the CI network predicts two
causal-importance values per input and per component — one for reconstructing the model
**output**, one for reconstructing **hidden activations** — off a single shared trunk, with
only the readout head private per role.

Reference run to replicate: **`addsub-L18-11-trunk-imp2x`** (torch).
Base to derive from: **`addsub-L18-jax-mirror-01`** (JAX, single-objective), which already
mirrors the torch run field-for-field.

## What is here

| piece | where |
|---|---|
| dual CI fn (shared trunk, two heads) | `param_decomp/core/ci_fn.py` — `DualCI`, `CIRole`, `ci_for_role` |
| the hidden pass | `param_decomp/core/objective.py` — `HiddenPass`, `TargetedObjective.hidden` |
| four-pass step + pass scheduling | `param_decomp/core/train.py` — `_PassPlan`, `make_targeted_train_step` |
| hidden-only comparison | `param_decomp/core/recon.py` / `losses.py` — `HiddenActsOnlyReconstruction` |
| config surface | `pd.hidden`, `nontarget.hidden`, `decomposition.ci.dual`, `runtime.sequential_passes` |
| the run (NOT in-repo, per CONFIGS.md) | `~/pd_scratch/dual_obj_jax/` — config, sbatch, and the derivation script |
| SPEC | S36 (dual CI), T12 (the hidden pass); T1 and T5 amended |

## Three decisions worth knowing

**The trunk is shared by construction, not by identity.** The torch port shares it by module
identity (`adopt_trunk`), which needs a hand-rolled parameter dedupe so an optimizer cannot
count the trunk twice, and a load-time value comparison because a shared-trunk state dict is
key-identical to an independent pair's. Neither trap exists here: the trunk is one set of
arrays in the pytree, so it appears once to the optimizer, and the pytree *shape* differs
between the topologies, so a checkpoint cannot be misread. One `eqx.filter_vjp` pullback
returns a trunk gradient that is exactly the sum of the two objectives' — pinned by
`core/tests/test_dual_ci_fn.py`.

**The hidden objective is a PASS, not S35's per-term rider.** The rider adds
`coeff · hidden` on top of a term's end-to-end loss, so its CI stays shaped mostly by the
output objective. The hidden pass instead carries no output term at all
(`HiddenActsOnlyReconstruction`), so `recon_loss_fn` is never called on a hidden forward and
the full-vocabulary KL never enters that pass's graph.

**Sequential and fused pass scheduling are the same objective.** `runtime.sequential_passes`
scores one pass at a time and adds the gradients, with an `optimization_barrier` threading the
trainables through the accumulator so XLA cannot hoist the next pass's forwards ahead of the
previous pass's backward. Peak activation memory then holds one pass's masked forwards rather
than all four. Measured on the TMS fixture: every loss scalar is **bit-identical** between the
two paths and the decomposition agrees to ~1e-7 relative; only gradient-*norm* diagnostics
differ (~4e-4), and forcing fp32 compute makes the decomposition bit-identical and drops that
gap to ~1e-7 — so the residual is bf16 rounding in the backward, not a different objective.
Re-summing the same per-pass gradients in a different order accounts for only ~3e-8, which is
how we know it is not reassociation. Pinned by `core/tests/test_dual_objective.py`.

## What the hidden pass reconstructs

Every MLP output from the decomposed layer to the end of the network:
`layers.18.mlp.down_proj.out` … `layers.31.mlp.down_proj.out` (14 points).

No new tap vocabulary was needed. `<site>.out` already resolves to a per-block physical tap
(`_GLUTap.DOWN_OUTPUT`) captured during the scan regardless of whether that block is
decomposed, so undecomposed layers 19–31 were already addressable. `resolve()` rejects two
keys naming one array, and `assert_hidden_acts_reconstruction_points` rejects points masking
cannot reach (verified: L17 refused, L18–L31 accepted).

This deliberately **differs from the torch reference run**, which measures at the seven L18
site outputs via a forward that early-exits after layer 18. Measuring to the end of the
network makes early exit impossible by construction — which also means JAX never needs one.

MLP outputs rather than residual boundaries because the relative error divides by the point's
own clean scale: at `resid.N` the denominator is dominated by the carried residual stream,
which the L18 perturbation barely moves, so the metric is deflated and the gradient weak. An
MLP output's denominator is that MLP's own contribution.

## Divergences from the torch reference run

Recorded so the comparison is read correctly, not silently.

- **Hidden measurement points** (above): 14 MLP outputs L18→L31, vs the torch run's seven L18
  site outputs.
- **ΣC = 1768** (72/72/128/128/456/456/456), inherited from `addsub-L18-jax-mirror-01`, vs the
  torch run's 6144. Unchanged from the mirror on purpose: the dual run is a controlled
  comparison against *it*, not against the torch run directly.
- **A hidden pass on the non-target stream.** SPEC T5 previously refused internal-activation
  matching off-target, on the argument that with the delta pinned on it constrains exactly the
  behavior tPD declines to decompose. That argument is preserved in the amended T5 rather than
  deleted; the off-target hidden pass is an experimental choice, optional and absent by
  default. **This is the one place where the implementation goes against a written SPEC
  rationale, and it was an explicit instruction rather than a derivation.**

## Not done

- **`ABGridDataset` still emits the single-CI payload.** The torch applet's dual half (the
  green/magenta output-alive vs hidden-alive overlay) is not ported, so the applet falls back
  to single-colour rendering as it does today.
- **No GPU run has been executed.** The config and sbatch are staged; nothing has been
  launched.

## Launching

```bash
sbatch ~/pd_scratch/dual_obj_jax/addsub-L18-jax-dual-01.sbatch
```

2× L40, 20000 steps, 23 h wall clock, SIGTERM-save at 10 min remaining. The sbatch points at
this worktree and at the staged config. `derive_dual_config.py` regenerates that config from
`addsub-L18-jax-mirror-01.yaml`, so the diff against the single-objective run stays exactly the
dual-objective change — the same discipline the torch `press2-trunk` launch used.

Memory to watch on the first steps: the non-target stream is the expensive one for the hidden
pass, because it runs at seq_len 64 against the target stream's unpadded prompt length. Fourteen
captured points at `[128, 64, 4096]` bf16 is ~0.9 GB per retained forward there, against ~0.12 GB
on the target stream. `sequential_passes: true` is set for exactly this reason; if it still does
not fit, the next lever is dropping the non-target hidden pass (`nontarget.hidden`), which is
optional and the largest single contributor.
