# JAX single-pool PD+PGD trainer — design log

A running log of decisions, torch↔JAX mappings, perf observations, and open
questions while building the single-pool VPD trainer in JAX. The COMPARISON
(does XLA optimize this loop better than torch?) is the deliverable this file
serves.

## What this is

A clean, packaged JAX implementation of the **single-pool** Parameter
Decomposition training loop with all four VPD losses + the persistent-PGD
adversary, mirroring the torch FSDP single-pool path
(`param_decomp_lab/fsdp/`, plan `fuzzy-tinkering-meteor.md`). It is the
research counterpart testing the "single-pool SPMD collapse" hypothesis from
`jax_spike/SYNTHESIS.md` / `HANDOFF.md`.

Lives in `nano_param_decomp_jax/jax_single_pool/`, a sibling package to the
existing `nano_pd_jax` (which it reuses for the model / decomposition / CI /
sigmoid primitives).

## Why a new package, not extending nano_pd_jax in place

`nano_pd_jax` v1 is deliberately scoped: TMS / toy-MLP / tiny-transformer,
3 losses, single device, no PGD, no persistent state, no sharding. Its
`trainer.py` `TrainState` and `make_step_fn` bake in that 3-loss shape. The
single-pool target adds (a) a 4th loss with persistent adversarial state and a
minimax inner loop, (b) GSPMD sharding, (c) a weight-delta source channel.
Threading those through the v1 trainer would force `if ppgd is not None` /
`if sharded` branches through a loop shared with the converging v1 experiments
— exactly the optionality-by-branching the repo guidelines (and the torch
plan) warn against. So: reuse the *primitives* (Linear shim, DecomposedLinear,
CIFn, sigmoid, mask sampling), build a *new* trainer.

## Sources of truth I'm mirroring

- **torch PPGD semantics**: `param_decomp/metrics/persistent_pgd_state.py` +
  `persistent_pgd_recon.py`. Load-bearing:
  - `mask = ci + (1 - ci) * source`, source ∈ [0,1] (sigmoid-param OR clamp).
  - `n_warmup_steps` supplemental source-only ascent iters, **then** the final
    fused fwd+bwd does one more source ascent → `n_warmup + 1` total source
    updates per training step.
  - weight-delta channel: when `use_delta_component`, `source_c = C + 1`; the
    extra channel masks `W_delta` (`mask = source[..., -1]` interpolated with
    `ci=1` effectively — torch interpolates the delta source directly).
  - adversary MAXimizes recon; params MINimize worst-case recon (+ the other
    three losses).
  - scopes set source leading dims: single `[1,...]`, broadcast
    `[1, *batch_dims[1:]]` (production default), repeat `[n, ...]`,
    per-batch-per-position `[*batch_dims]`.
- **single-pool SPMD step**: `jax_spike/stage8_train_distributed.py` (flat
  einsum prototype: 4 losses + PGD + 2 hand-rolled Adams + GSPMD, validated
  1/8/16 GPU GPU-count-invariant). I'm re-expressing this over the Equinox
  model instead of flat NamedTuple einsums.
- **PGD mechanics**: `jax_spike/stage6_pgd.py` (lax.scan inner loop bit-exact
  to a python loop; fused multi-argnums grad; minimax stop-gradient).
- **two-optimizer split**: `nano_pd_jax/trainer.py` (`eqx.partition` on a
  bool-pytree filter — V/U vs CI fn, no string-path matching).

## torch → JAX mapping (running)

| torch (FSDP single-pool)                  | JAX single-pool                              | maps cleanly? |
|-------------------------------------------|----------------------------------------------|---------------|
| 4 `Metric` objects + `MetricContext`      | 4 pure loss fns called inside one `loss_fn`  | cleaner in JAX |
| `before_backward`/`after_backward` PPGD   | fused multi-argnums `value_and_grad`         | much cleaner — no manual graph orchestration |
| `PersistentPGDState` (mutable, in-place)  | `PGDState` Equinox pytree carried in `TrainState` | clean (functional) |
| Adam-PGD optimizer (mutable m/v)          | functional adam over sources in the step     | clean |
| FSDP2 `fully_shard`                        | `NamedSharding` + `jit` auto-collectives     | the headline win — zero manual collectives |
| `replica_sync_group` broadcast/AVG-reduce | GSPMD reduces over sharded batch axis for free | eliminated by SPMD |
| DCP sharded save                           | (deferred — note below)                       | n/a yet |
| residual-start `use_cached_residual`       | (deferred — note below)                       | partial |

## Open questions / TODO

- [ ] residual-start analog: skip the frozen prefix. nano_pd_jax models have no
      residual cache. Deferred — note why in the residual section once built.
- [ ] real LM (Flax/Equinox Llama) at scale: stage10/11 have `vendored_jax`;
      out of scope for the CPU-runnable core but the step fn is model-agnostic.
- [ ] checkpoint/resume of the PGD state (sources + adam m/v) — pytree, trivial
      with `eqx.tree_serialise_leaves`; wire when needed.
- [ ] perf A/B vs torch on GPU — needs an accelerator (see "needs GPU/TPU").

## Perf / compilation observations

(filled in as smokes run)

## Needs a GPU/TPU to validate

(filled in as I hit accelerator-only paths)
