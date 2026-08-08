# Refactor plan: local (clean-input) hidden-activation reconstruction

Plan for the change described in `notes/streamline_dual_obj/report.md`. Branch
`worktree-streamline-dual-obj`, based on `feature/dual_hidden_acts` at `6313d73e3`.

## Goal

Give every decomposed matrix its **own clean input** when measuring hidden-activation
reconstruction, instead of the input produced by a masked forward pass that upstream
decomposed matrices have already perturbed.

Today: prediction is `masked_matrix_i(perturbed input)`, target is `W_i · x_i` (clean).
After: prediction is `masked_matrix_i(x_i)`, same target. Same input on both sides.

Consequences: each matrix's error is its own doing, the matrices become independent, and
the truncated forward through the model disappears from this loss entirely — every input it
needs is already cached by the clean pass that runs anyway.

## Decisions taken

| question | decision |
|---|---|
| persistent adversary on the hidden objective | **port to local** — same `PersistentPGDState`, same sources and optimizer, only the attacked error function changes |
| shared CI trunk | **not in this change**; land it after this run is understood |
| monotone `CI_hidden >= CI_output` | **no** — keep it measurable as a diagnostic |
| coefficients | **press2's verbatim** (stochastic 2.0, persistent 1.0); the local loss is easier so its value will be smaller, and that is an accepted, recorded confound |
| eval probes | **both flavours** — 20-step PGD chained *and* local, plus CI-masked in both flavours, so `chained − local` is available per matrix at matched adversary strength |
| training losses | local only; no chained hidden loss in `pd.loss_metrics` |

## Design: one dispatch point, not four new classes

Four metrics measure hidden-activation error, and every one of them obtains its prediction
from the same single line, `self.model.site_outputs(ctx.batch, mask_infos)`:

| metric | where | role |
|---|---|---|
| `StochasticHiddenReconSubsetLoss` | core | training loss |
| `PersistentPGDHiddenActsReconLoss` | core | training loss (persistent adversary) |
| `PGDHiddenActsReconLoss` | core | eval probe (fresh adversary) |
| `CIHiddenActsReconLoss` | lab | eval probe (CI as the mask) |

So this is a config field on those four, not four new classes. Two reasons beyond the
smaller diff:

- Two of them must run in **both** flavours in the same run. `Metric.instance_key` /
  `name` already exists for exactly that (one class, several instances, distinct log keys),
  and `reduced_relative_errors` already keys every entry under `instance_key`.
- A duplicated class per flavour would duplicate the site selection, the relative-error
  accumulation and the DDP reduction, which is most of the code.

New field on all four configs:

```python
SiteInputs = Literal["clean", "masked_forward"]
```

named `site_inputs`, defaulting to `"masked_forward"`. The default preserves the meaning of
every existing config — this is not a compatibility shim, it is the pre-existing behaviour
keeping its name — and matters because `addsub-L18-11-press5` may still need resuming.

## Code changes

### 1. `param_decomp/metrics/hidden_acts.py`

Add the type and two functions; everything else in the module is untouched and shared.

```python
SiteInputs = Literal["clean", "masked_forward"]

def resolve_measured_sites(model, patterns, site_inputs) -> list[str]
    # select_sites(...) plus: assert no readout site is measured under "clean"

def masked_site_outputs(model, batch, pre_weight_acts, mask_infos, sites, site_inputs)
    # match site_inputs:
    #   "masked_forward" -> model.site_outputs(batch, mask_infos), filtered to `sites`
    #   "clean"          -> each site's components run on its own cached clean input
```

The `"clean"` branch is:

```python
components(pre_weight_acts[site],
           mask=mask_info.component_mask,
           weight_delta_and_mask=mask_info.weight_delta_and_mask)
```

Correctness notes that must hold, and are checked in review:

- **Bias cancels.** `LinearComponents.forward` adds `self.bias`; `clean_site_outputs` builds
  the target as `F.linear(x, W, components.bias)`. Both sides carry it.
- **Routing.** The chained path returns `where(routing_mask, components_out, frozen_out)`
  and `site_squared_errors` then keeps only routed positions, so the frozen branch is never
  scored. The local path therefore does not need the `where` at all — but it must keep
  passing `mask_infos` to `site_squared_errors` so the same positions are selected.
- **Delta component.** `weight_delta_and_mask` is forwarded unchanged, so the leftover-weight
  term behaves identically, including under the nontarget pass's forced-on delta.
- **Only measured sites are computed.** With no chain there is no reason to evaluate a
  matrix nobody measures. The chained branch keeps computing all of them (it has to) and is
  filtered afterwards.
- **Readout sites are rejected under `"clean"`, loudly.** A `hidden_readout_sites` target is
  a point in the residual stream, not a matrix output; feed every matrix its clean input and
  the stream is unchanged, so its error would be identically zero. Silently scoring zero is
  the dangerous outcome, so `resolve_measured_sites` asserts. (press2 defines no readout
  sites, so this run is unaffected.)

### 2. The four metrics

Each gains `site_inputs` on its config, swaps `select_sites` for `resolve_measured_sites` in
`bind`, and routes its one prediction line through `masked_site_outputs`. For
`PersistentPGDHiddenActsReconLoss` that line is inside `_objective`, which is the documented
single seam — its `_accumulate_eval` override already avoids the base class's extra
`cache_type="output"` forward, so nothing else moves.

### 3. Docs

`param_decomp/metrics/CLAUDE.md`, *Hidden-activation reconstruction* section: the three-row
table gains the input-source axis, plus the readout-site restriction and the reason both
flavours are worth logging.

## The run

`addsub-L18-11-local`, from `~/pd_scratch/hidden_site_targets/addsub-L18-11-press2.yaml`
(the plain press2, not the `-ntppgd` variant — that one carries fields this branch does not
have). Config lands in `~/pd_scratch/hidden_dual_local/`, not the repo.

Changes against press2:

- `StochasticHiddenReconSubsetLoss` → `site_inputs: clean` (coeff 2.0 unchanged)
- `PersistentPGDHiddenActsReconLoss` → `site_inputs: clean` (coeff 1.0 unchanged)
- eval `PGDHiddenActsReconLoss` ×2: `PGDHiddenActsRecon_chained` and `..._local`, both
  `n_steps: 20`
- eval `CIHiddenActsReconLoss` ×4: `{output,hidden}CI × {chained,local}`
- everything else byte-identical: 2 GPUs, 20000 steps, batch 128, nontarget 96, C
  (1024,1024,1024,512,512,1024,1024), both impmin at 5e-5, `sampling: binomial`,
  `sigmoid_type: leaky_hard`

### What to look at

1. **`q_proj`, `k_proj`, `v_proj` chained error must equal local error.** Their input is the
   block input, which nothing upstream touches, so both formulations are handed the same
   tensor. A mismatch means the implementation is wrong. This is the primary correctness
   check and it reads straight off the per-site eval keys.
2. `chained − local` at `o_proj` and `down_proj` — the compounding term, the number this
   whole change exists to expose.
3. `NAlive_hidden` against press2. It may fall, either because part of the hidden network's
   density was compensating for inherited error, or because the easier objective is being
   out-pushed by the unchanged 5e-5 sparsity coefficient. Those two are not distinguishable
   from `n_alive` alone; the per-site local error tells them apart.
4. Step time and peak memory against press2's, to confirm the removed forward shows up.

### Sequence

1. `uv sync` in the worktree.
2. Implement, `make check`, `make test`.
3. `simplify` skill over the diff, weighted to correctness.
4. 3-step probe, no wandb, to catch config and shape errors and read peak memory.
5. Launch on 2 GPUs via `run_ddp.sbatch` (repo path pointed at this worktree), `-J
   addsub-L18-11-local`.

## Review outcome

Five reviewers over the diff: the four `simplify` angles plus a correctness pass (the
`simplify` skill excludes correctness by design, and correctness was the priority here).

**One real defect, reproduced and fixed.** With `site_inputs="clean"` *and* a `site_patterns`
that excludes some decomposed site, every PGD/PPGD hidden-acts metric dies mid-step:
`torch.autograd.grad(loss, sources)` runs with `allow_unused=False`, sources are allocated per
decomposed site, and locally an excluded site's components never enter the loss at all.
Chained, such a source still reaches the loss by perturbing the sites downstream of it. Now a
bind-time assert (`assert_sources_reach_every_site`) with a test. The plan's own advice —
"restrict readouts to a separate `masked_forward` instance via `site_patterns`" — was the
fastest route into it.

**The cost claim was wrong.** "~100x cheaper per mask sample" ignored the leftover-weight term.
With `use_delta_component` on, each site computes `x @ delta.T`, a dense `d_in x d_out` matmul
costing exactly what running the frozen matrix costs: 218.1 of the local path's 295.1
MACs/token. Real figure is **~15x**. Corrected in `metrics/CLAUDE.md` and
`notes/streamline_dual_obj/report.md`, along with the two optimisations that would reach ~37x /
~107x (deriving the delta term from the target; hoisting the mask-independent work).

**Verified clean** by the correctness pass, by reading and by CPU experiment: bias cancels on
both sides in both paths; the routing-mask asymmetry is invisible to the loss because
`site_squared_errors` scores only routed positions in both paths (raw tensors differ by up to
1.78, scored numerators by 0.00); `weight_delta_and_mask` is forwarded verbatim and behaves
identically under `delta_override`; nothing consumed the dropped keys of the narrowed return
dict; the PPGD source state machine, `_accumulate_eval` and the source gradient path are
undisturbed when all sites are measured; `pre_weight_acts` is the right tensor, bit-identical
under bf16 autocast.

**Cleanups applied:** a `HiddenActsSitesConfig` mixin so the two fields are declared once
rather than four times; `masked_site_outputs` takes `ctx` instead of three of its fields and
absorbs the local branch; the duplicated `LinearComponents` assert moved to bind time; stale
class docstrings on the PGD and CI probes; test scaffolding tidied.

**Skipped:** a full `HiddenActsErrorMetric` base owning `bind`/`reset`/`compute` for the four
metrics — a genuine observation, but it restructures pre-existing code well outside this diff.
Adding `.detach()` to the local path's input — verified to be a no-op, since the frozen model's
cache never requires grad.

## Probe result: the q/k/v identity holds exactly

3-step probe (`probe_addsub-L18-11-local.yaml`, job 7408), `CIHiddenActsRecon_hiddenCI`
chained against local, per site, at the last eval:

| site | chained | local | chained − local |
|---|---:|---:|---:|
| `self_attn.q_proj` | 0.835663 | 0.835663 | **0.000e+00** |
| `self_attn.k_proj` | 0.736249 | 0.736249 | **0.000e+00** |
| `self_attn.v_proj` | 0.784129 | 0.784129 | **0.000e+00** |
| `mlp.gate_proj` | 0.762764 | 0.764863 | −2.1e-03 |
| `mlp.up_proj` | 0.898317 | 0.897145 | +1.2e-03 |
| `mlp.down_proj` | 0.999104 | 0.913550 | +8.6e-02 |
| `self_attn.o_proj` | 0.993997 | 0.903993 | +9.0e-02 |

Exactly the predicted structure. `q/k/v` read the block input, which nothing upstream
touches, and agree bit-for-bit — the correctness check the implementation was built around.
The gap is concentrated on the two most-downstream sites, `o_proj` (behind q/k/v) and
`down_proj` (behind everything), at ~0.09 against ~0.001 for `gate`/`up`.

Note `gate_proj`'s gap is slightly *negative*: inherited drift is not obliged to increase a
site's error, it can cancel against the local error. At an essentially untrained step-3 model
these small signed gaps are noise; the two large positive ones are not.

Cluster note: two earlier probe attempts (7404, 7406) died on GPUs SLURM believed were free.
Another user's interactive job reserves 2 GPUs but runs 3 processes, so SLURM hands out GPUs
that are already occupied. The sbatch guard now fails fast (exit 75) instead of proceeding
into an opaque OOM inside `set_device`; resubmitting is the remedy.

## Arms launched

All from `feature/trunk_plus_local` (`da5361a95`), the merge of `feature/dual_ci_shared_trunk`
into `feature/dual_hidden_acts`. The merge was clean — the two branches diverged from
`af9212b00` and touched disjoint files — and it also picked up the SmoothL0 coefficient
schedule another session had pushed to `dual_hidden_acts` in the meantime, which is what makes
the impmin sweep below expressible. That schedule is on *neither* single-feature branch, so the
merged tree is the only one that can run all three arms.

| run | trunk | hidden inputs | impmin | job |
|---|---|---|---|---|
| `addsub-L18-11-press2` | no | chained | 5e-5 | (done, the baseline) |
| `addsub-L18-11-press2-trunk` | yes | chained | 5e-5 | 7405 |
| `addsub-L18-11-local` | no | **clean** | 5e-5 | 7409 |
| `addsub-L18-11-trunk-local` | yes | **clean** | 7.5e-5 | 7417 |
| `addsub-L18-11-{trunk,local,trunk-local}-imp{2,5}x` | — | — | ×2 / ×5 peak | 7418–7423 |

`trunk-local` carries a deliberate coefficient rebalance, agreed with the user: every output
recon coeff to 1.0 (from 1.0 / 0.5 / 0.5) and both hidden recon coeffs equal at 2.25 (from
2.0 / 1.0), which holds press2's hidden:output recon ratio of 1.5 with the output total now at
3.0. Both impmin instances scale by the same 1.5 to 7.5e-5 so the recon:sparsity balance is
preserved too — without that, the 1.5x larger recon mass would have made both nets denser for
a reason unrelated to either variable. It does mean this arm differs from press2 in three
things, not one; the sweep rows inherit each arm's own base, so every {1x, 2x, 5x} triple stays
internally clean.

The sweep's peak is a real peak: held from step 0, then annealed back to the base coefficient
over the last 30% (`coeff_anneal_start_frac 0.7`, `coeff_anneal_end_frac 1.0`), applied to both
the output and hidden impmin instances. Note it overlaps the existing gamma anneal (0.5 → 1.0),
so in the final stretch the coefficient falls while gamma sharpens.

The trunk+local probe reproduced the q/k/v identity exactly (q 0.834938, k 0.742744, v 0.777202
in both formulations; gap 0.000e+00), with the same concentration on `o_proj` (0.096) and
`down_proj` (0.088) — so the shared trunk does not disturb it. Peak 41106 MiB on one GPU at the
2-GPU per-rank batch, against press2's 41710.

## Explicitly not in this change

- The shared CI trunk (deferred by decision above).
- The monotonicity constraint (rejected for now, on purpose).
- Hoisting `V x` out of the mask-sample loop. `LinearComponents.forward` recomputes it, which
  wastes roughly half the local cost — irrelevant beside removing a ~17 TFLOP forward, and
  worth doing only once `n_mask_samples` rises above 1.
- Raising `n_mask_samples`. It is a global `pd` field shared with the output loss, so raising
  it is not free and is a separate experiment.
- The legacy `StochasticHiddenActsReconLoss` (raw MSE, own forwards) — untouched.
