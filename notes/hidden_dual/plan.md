# Dual CI networks: final-output CI + hidden-activation CI

Branch `feature/dual_hidden_acts`, forked from `experiment/8B_targeted`
(worktree `~/Code/param-decomp/dual_hidden_acts`). Started 2026-07-29.

## Motivation

Today PD reconstructs only the target model's final outputs. That cuts both ways:

- Reconstructing hidden activations forces us to reproduce internal *interference* even
  when it is cancelled downstream and never reaches the output. That costs sparsity
  twice over: extra components that only carry interference, and a weaker ablation
  signal (ablating causally-unimportant components is precisely how the important ones
  get gradient).
- But hidden-activation reconstruction gives every layer signal from immediately
  downstream instead of backpropagating from the logits, and should bias the search
  toward a basin whose mechanisms are actually faithful to the target model's. That
  should matter most for *chained* blocks and for targeted decomposition.

So: run both at once. **One shared pool of subcomponents, two CI networks.** One scores
"is this subcomponent needed to reconstruct the final output", the other "is it needed to
reconstruct the hidden activations". They are two different reconstruction losses over
the same components, nothing more.

**Falsifiable sanity check.** Output-importance should imply hidden-importance, but not
conversely. On the two-colour (a,b) CI heatmaps that means: white (both) and green
(hidden-only) regions are expected; **magenta (output-important, hidden-unimportant) is
an anomaly**.

## Specification

### Two CI networks

- `ComponentModel` gains `ci_fn_hidden`, built by the same `make_ci_fn_wrapper` from the
  same `pd.ci_config` (identical architecture, independent initialisation). Gated by a
  new `pd.dual_hidden_ci: bool = False`. No separate architecture override — not needed
  for these experiments.
- `calc_causal_importances(..., role: CIRole)` where `CIRole = Literal["output", "hidden"]`.
  `MetricContext` carries `ci` (output) and `ci_hidden: CIOutputs | None`, plus
  `ci_for(role)` which asserts availability.
- Both nets' parameters go into the **existing** `ci_fn_optimizer`. Adam is
  per-parameter and the two nets have disjoint parameters, so a shared optimizer and two
  optimizers with identical hyperparameters are mathematically identical; splitting would
  only buy per-net LR/betas, which we do not want.
- Metrics that read CI take a `ci_role` config field (default `"output"`, so every
  existing YAML stays valid).

### Losses

| loss | CI net | target | coeff |
|---|---|---|---|
| `StochasticReconSubsetLoss` | output | final logits (KL) | 1.0 |
| `StochasticHiddenReconSubsetLoss` **(new)** | hidden | per-site hidden acts (relative MSE) | 1.0 |
| `UnmaskedReconLoss` | — | final logits | 0.5 |
| `PersistentPGDReconLoss` | output | final logits | 0.5 |
| `SmoothL0ImportanceMinimalityLoss` | output | — | 5e-5 |
| `SmoothL0ImportanceMinimalityLoss` | hidden | — | 5e-5 |

Importance-minimality hyperparameters are identical for both nets (beta 0.5, gamma
1.0 → 0.01 annealed over the second half, `normalize_at_one`). `UnmaskedReconLoss` and
`PersistentPGDReconLoss` stay output-only, as specified.

#### The new hidden loss

Stochastic subset ablation exactly like `StochasticReconSubsetLoss` — same
`UniformKSubsetRouter`, same `ci + (1-ci)*source` masks, same `n_mask_samples` — but the
loss is measured at the decomposed sites instead of the logits.

- **Targets** are the *clean* per-site outputs `F.linear(x_clean, W, b)`, recomputed from
  the clean input activations already cached each step. This costs **no extra forward
  pass**, and it is the "global" semantics: it measures accumulated drift from the target
  model, not merely each site's local approximation error given an already-perturbed
  input. That is the quantity we actually care about.
- **Relative MSE per site**: `Σ(out - tgt)² / Σ tgt²` over the positions routed to
  components, then averaged over sites. Raw MSE would weight sites by their activation
  variance (`down_proj` dwarfing `q_proj`) and would not transfer across blocks.
- **Measurement scope**: `site_patterns: list[str] | None`. `None` (the default, and what
  we run first) measures every decomposed site. Setting
  `["*.mlp.down_proj", "*.self_attn.o_proj"]` restricts measurement to the residual-stream
  writes — expressed as fnmatch patterns so no Llama-specific literal leaks into core.
  Masking always covers all decomposed sites; only the measurement is filtered.
- **Early exit.** The loss reads nothing past the last decomposed module, so everything
  downstream is wasted compute *and* wasted memory (downstream activations would be
  retained for backward). New `ComponentModel.site_outputs(batch, mask_infos)` aborts the
  forward as soon as the cache holds every hooked module — detected by
  `len(cache) == len(mask_infos)`, so no execution-order bookkeeping is needed — via a
  sentinel exception raised from the caching hook and caught in the method. The aborted
  module's cached output keeps its autograd graph; only its return value is discarded.

### Forward / backward architecture

Per training step, target pass:

1. Clean forward, `cache_type="input"`. Builds no autograd graph (target frozen, no
   components applied) and arms the DDP reducer.
2. `ci_fn(x)` and `ci_fn_hidden(x)` — two CI-net forwards. ~34 M params each; negligible
   against an 8 B target.
3. `StochasticReconSubsetLoss(ci_output)` — 1 full masked forward.
4. `UnmaskedReconLoss` — 1 full masked forward.
5. `PersistentPGDReconLoss(ci_output)` — 3 forwards (2 warmup + 1) plus inner source
   backwards.
6. `StochasticHiddenReconSubsetLoss(ci_hidden)` — 1 **truncated** masked forward.
7. `SmoothL0(ci_output)` + `SmoothL0(ci_hidden)`.
8. One `total_loss.backward()`.

Nontarget pass, under `delta_override(1.0)`: clean forward → both CI nets →
`StochasticReconSubsetLoss(ci_output)` + `StochasticHiddenReconSubsetLoss(ci_hidden)` +
both `SmoothL0`s scaled by the shared `impmin_coeff_ratio: 2.0` → one backward.

Then one `components_optimizer.step()` and one `ci_fn_optimizer.step()`.

**Gradient accumulation is unchanged**: exactly two backwards per step, one per pass,
same as today.

**One backward per pass is forced, not chosen.** DDP re-arms its reducer on every
`wrapped_model(...)` forward and then requires every grad-requiring parameter to be
marked ready during the following backward. Splitting the output-recon and hidden-recon
backwards to free the first graph earlier would leave `ci_fn` untouched in the second and
hang the reducer. Both graphs are therefore live simultaneously; early exit is what keeps
the second one small (one block's internals rather than the whole tail of the model plus
`lm_head`).

**Marginal cost per step** over the current recipe: two extra CI-net forwards, one
truncated masked forward per pass, and one extra CI net's worth of optimizer state
(~34 M params × 16 B ≈ 0.55 GB).

### CI-scaled weight decay

`ci_scaled_component_weight_decay` currently shrinks subcomponents by
`1 - lr·coeff·(1 - max_batch_CI)`. With two nets it takes the **elementwise max over
both**. A subcomponent that is alive only in the hidden net is doing real work
(representing interference); decaying it would fight the hidden recon loss and push the
run back toward the output-only solution, defeating the experiment.

### Evals

Four recon probes, two per target:

| probe | target | CI net |
|---|---|---|
| `CEandKLLosses` → `kl_ci_masked` (exists) | logits | output |
| `PGDReconLoss` n_steps 20 (exists) | logits | output |
| `CIHiddenActsReconLoss` + `ci_role` | hidden acts | hidden *and* output |
| `PGDHiddenActsReconLoss` **(new)**, n_steps 20 | hidden acts | hidden |

`CIHiddenActsReconLoss` runs under **both** roles in the dual runs — that pairing is the
direct measurement of the interference hypothesis (how much hidden-act error does the
output CI net leave on the table?) and is what makes the ctrl run comparable. On the ctrl
run, both hidden-acts probes use the single (output) CI net.

Two instances of one dict-returning metric class need distinguishable identities, so:
`name` moves from `LossMetricConfig` up into a new `NamedMetricConfig` base that
`LossMetricConfig` extends, `Metric.instance_key` reads it from there, and the per-module
hidden-acts result keys are namespaced by `instance_key` rather than by class name.

`slow_every: 5000`, and `ABGridDataset` is in the slow-eval set for all three runs.

### ab_grids modification

- Both nets' CI is accumulated over the prompt pool.
- **Filter**: a subcomponent's full grid is saved when
  `max(mean_ci_output, mean_ci_hidden)` reaches `mean_ci_floor` at some position — i.e.
  the max over *either* net, so a hidden-only component is not silently dropped.
- **Heatmaps**: green/magenta merge, `rgb(ci_output, ci_hidden, ci_output)`.
  White = important to both. Green = hidden-only (expected and common). **Magenta =
  output-important but hidden-unimportant — the anomaly the sanity check looks for.**
  Black = neither.
- The payload keeps `ci_hidden` optional and the applet falls back to the existing
  single-colour rendering when it is absent (ctrl run, older snapshots).

## Runs

All on `compute` (unlimited wall time), 8 B Llama-3.1, `addition_subtraction_1-100`
prompts as target and fineweb as nontarget, 20 000 steps, `slow_every: 5000`.

| run | blocks | GPUs | scheme |
|---|---|---|---|
| `addsub-L18-09-dual` | 18 | 2 | dual |
| `addsub-L18to20-01-dual` | 18,19,20 | 3 | dual |
| `addsub-L18to20-01-ctrl` | 18,19,20 | 3 | single (output only) |

`addsub-L18-09-dual` mirrors `addsub-L18-09-one-im` — same C (456 MLP / 72,72,128,128
attn), same optimizers, same `coupled` init, same CI arch (d_model 512, 4 blocks) —
minus its `StochasticHiddenActsReconLoss`, plus the dual scheme. Batch 128/128; the
nontarget batch shrinks only if the two CI nets do not fit (prior art: nontarget KL OOMs
at 128 on a 45 GB card, 96 fits).

The two `L18to20` runs hold batch size fixed and shrink C to fit 3 GPUs: as close to 456
as possible, floor of 228 (the 6L sizing that fit 6 blocks on 4 GPUs at batch 48), attn C
scaled in proportion. **Both use identical batch and C** so the only difference is the
scheme.

Memory is probed empirically with `steps: 3` and no wandb, per the established recipe
(memory is ~deterministic in C).

## Commit plan

Cleanly separable so the scheme can be replayed onto other branches:

1. `feat(core): CIRole + second CI fn on ComponentModel` — `ci_fns` plumbing,
   `MetricContext.ci_hidden` / `ci_for`, `pd.dual_hidden_ci`, trainer wiring, CI-scaled
   WD max over both nets.
2. `feat(core): NamedMetricConfig — instance_key for non-loss metric configs`.
3. `feat(core): ComponentModel.site_outputs — early-exit forward to decomposed sites`.
4. `feat(metrics): StochasticHiddenReconSubsetLoss` + shared relative-error helpers.
5. `feat(eval): PGDHiddenActsReconLoss + ci_role on CIHiddenActsReconLoss`.
6. `feat(eval): ab_grids dual-CI green/magenta merge`.
7. `docs(notes)` + configs.

## Open risks

- The sentinel exception cuts against the repo's "no try/except for control flow" rule.
  It is the only way to abort a hooked forward; it is contained to one method and the
  exception type is private. Accepted deliberately.
- Relative MSE has a zero-denominator failure mode if a site's clean output is identically
  zero over the routed positions. Assert rather than guard.
- The hidden CI net will be much denser than the output net by construction. If
  `n_alive` on the hidden net saturates at C, the shared SmoothL0 coeff is too low for it
  — that is a result, not a bug, but it is the first thing to check at step 5000.
