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

- [x] checkpoint/resume of the PGD state (sources + adam m/v) — `checkpoint.py`,
      flat pytree `.npz`. Test proves resume continues the trajectory bit-exactly
      (the adversary persists). Trivial because the whole adversary state is in the
      `TrainState` pytree — no torch-style `state_dict`/`load_state_dict` plumbing.
- [x] real model: `experiments/transformer_qkv.py` decomposes the (square) q/k/v
      sites of `nano_pd_jax.TinyTransformer`, pulling real pre-weight acts. Proves
      the step is model-agnostic (faith 0.043→0.005, stoch 0.30→0.064 on real
      attention projections).
- [ ] real LM (Equinox Llama) at scale: `jax_spike/vendored_jax/llama.py` +
      stage10/11 already have a bit-parity Equinox Llama. The single-pool step here
      is shape-compatible (stacked homogeneous MLP sites) but the full LM run needs
      a GPU and param-sharding for memory (the HANDOFF.md open TODO #1). Out of
      scope for the CPU core; the step fn drops in unchanged.
- [ ] perf A/B vs torch on GPU — needs an accelerator (see "needs GPU/TPU").

## Residual-start analog

The torch path's residual-start (`use_cached_residual`) skips recomputing the
frozen transformer prefix before the decomposed layer each step — a *target
forward* optimization (cache the residual stream entering the decomposed block,
replay it). In this JAX design it maps even more cleanly and is **already
implicit**: the single-pool step's recon is *layerwise* (site-local), so it never
runs the target's prefix at all inside the jit'd step — it consumes pre-computed,
stop-gradient'd pre-weight acts `x` produced by ONE frozen target forward
(`stacked_acts` in `transformer_qkv.py`). That single frozen forward is exactly
the "cache the residual / acts once" idea. At LM scale you'd run the frozen
target forward once per batch (outside the differentiated step), harvest the
decomposed sites' pre-weight acts + the sites' target outputs, then feed those to
the step — the residual-start saving (skip the frozen prefix on the *masked*
re-forward) is free because there is no masked re-forward through the prefix:
recon is computed directly from acts. So residual-start isn't a separate feature
here; it's a consequence of the layerwise-recon factoring. (The torch path needs
it because its recon re-forwards the whole model under masks; the JAX layerwise
factoring sidesteps that.) Caveat: *output*-recon (KL of final logits, not
layerwise MSE) WOULD need a masked re-forward through the suffix; that's the one
place a residual cache would re-enter. Deferred with the full-LM output-recon
variant.

## Perf / compilation observations

- **CPU single-device smoke (toy_stacked_sites, S=6 d=32 C=8 B=64, n_warmup=4)**:
  the whole 4-loss + PGD step compiles to one XLA executable and trains; faith
  0.122→0.042, stoch 1.10→0.33, ppgd stays *above* stoch throughout (0.73→1.05
  vs stoch 0.33) — the correct minimax signature (the adversary finds masks worse
  than random; the worst-case recon floor is higher than the stochastic one).
- **GSPMD multi-device on CPU** (`distributed_stacked_sites`, simulated 1 vs 4
  devices via `--xla_force_host_platform_device_count`): the step runs sharded,
  faith is **bit-mesh-invariant** (1.213e-1 at both 1 and 4 dev) — confirms params
  replicate + the param math is sharding-clean. BUT see the sharding bug below.

## GPU-count invariance: PASS (bit-identical) — SPMD collapse confirmed

`distributed_stacked_sites` at 1 vs 4 simulated CPU devices, FIXED global batch +
seed, broadcast (replicated) source scope: **bit-identical loss trajectories**
(`[2.64249, 3.00304, 3.13791, 3.2729, 3.59267, 3.64591]`, final `3.91742` at
both). This is the headline correctness signal — the full single-pool VPD+PGD
step is GPU-count-invariant under GSPMD with **zero manual collectives**, the
persistent adversary's replicated source included. XLA's autodiff of the global
mean correctly all-reduces the replicated source's cotangent; the torch
`reduce_source_grads` AVG-reduce is NOT needed — it's absorbed by the compiler.
This strengthens the SYNTHESIS claim rather than weakening it.

### A test-harness pitfall worth recording (not an algorithm bug)

I first saw ~3–6% divergence and suspected the replicated-source grad wasn't
all-reduced. It was a **harness artifact**: the SLURM-style `shard_batch`
(lifted from `jax_spike/distributed_util.py`) assumes 1 device per process and
slices `x[:, idx*per:(idx+1)*per]` by `process_index()`. On single-process
multi-device CPU (`--xla_force_host_platform_device_count=N`), `process_index()`
is always 0, so it placed the FIRST 1/N slice replicated on all devices — the
4-device run literally trained on different (repeated-first-slice) data. With a
fixed `shard_batch` (`make_array_from_process_local_data`, which handles both the
single-process-many-devices and multi-process-1-device topologies) the
trajectories go bit-identical. Lesson: validating SPMD invariance on simulated
multi-device CPU requires sharding the FULL global array, not the SLURM
per-process-slice idiom. The bit-exact isolation repro that nailed it: identical
`recon0` + `first_grad_sum` to 12 digits once the full array is `device_put` with
`P(None,'dp',None)`.

## Needs a GPU/TPU to validate

(filled in as I hit accelerator-only paths)
