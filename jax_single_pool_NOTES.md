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

## Full-LM Llama-8B output-recon variant (2026-06-09, GPU)

The layerwise core (`step.py`/`losses.py`) does *site-local* MSE recon — it does NOT
match the torch reference. The torch `StochasticReconLayerwiseLoss` masks ONE module
but reconstructs the **final logits** via a full masked re-forward (and PPGD masks all
sites at once). So the core's site-local recon is a simplification; the real workload
needs a masked re-forward through the suffix. That full-LM variant now lives in:

- `llama8b.py` — residual-start L18->L31 frozen suffix + decomposed L18 MLP (V/U +
  weight-delta) + **real HF safetensors loader** (no torch dep; reads the cached
  `meta-llama/Llama-3.1-8B` shards directly) + `make_real_target_residual` (one frozen
  L0->L17 prefix forward, the residual-start amortization).
- `ci_fn.py` — `global_shared_transformer` CI fn (d4096 / 4 bidir-RoPE blocks / 64 heads
  / mlp16384), per-site clean inputs concatenated, leaky-hard-sigmoid head.
- `llama8b_step.py` — the full 4-loss + persistent-PGD step, output-recon on logits.
  n_warmup pre-ascents (lax.scan, params+ci detached) + one post-update ascent. Source
  clamped to [0,1] (torch config `use_sigmoid_parameterization: false`). `jax.checkpoint`
  on the masked suffix forwards (recompute in bwd — trades compute for memory so a real
  batch fits).
- `llama8b_sharding.py` — the FSDP-analog GSPMD plan (see below).
- `experiments/llama8b_real.py` — runner + tok/s/GPU + MFU + `--real_weights` + `--shard`.
- `experiments/llama8b_slurm.sbatch` — 1-task/GPU multi-GPU launcher.

This is the clean re-homing of `jax_spike/stage10_real_pd_bench.py` into the package,
PLUS the two open HANDOFF TODOs: real HF weights, and working param sharding.

### Why output-recon, not the core's layerwise site-local recon
Matching torch apples-to-apples requires the masked re-forward (the recon target is the
clean *suffix logits*, not each weight's local output). The core's `forward.py` recon is
kept as the simplified layerwise variant; the 8B target deliberately uses its own step.

### Single-B200 results (this node, 183GB B200, PEAK 1715 TFLOP/s bf16)
bl4, seq2048, C=24576, n_warmup=2, all jit'd one executable:
- random weights:  4,538 tok/s/GPU
- **real HF weights: 4,600 tok/s/GPU** (loss ppgd 4.99 >> stoch 0.13 — strong minimax
  signature with real structure, vs ppgd 1.42 / stoch 0.37 on random weights)
- `--shard` on 1-device mesh (no-op): 4,524 tok/s/GPU (sharding code path validated)
- bl8 OOMs replicated on 183GB (157GB activation alloc) -> the shard path is the fix
- All HOST-BOUND (dispatch ~1.4s vs blocked ~1.8s): the many small jit sub-forwards have
  high Python dispatch overhead. Real multi-step training amortizes this; for the bench
  the device-bound number is the lower of blocked/dispatch.

MFU caveat: the reported ~87% over-counts — the FLOP model charges fwd+bwd=3x for the PGD
ascents (params detached -> bwd is cheaper) and doesn't separately account for checkpoint
recompute. Treat tok/s/GPU as the hard number; MFU is an upper-ish estimate.

### Sharding plan (the memory story / FSDP analog)
1-D `dp` mesh. Frozen suffix REPLICATED (~7.3GB/dev bf16, small vs activations). V/U +
their fp32 Adam states SHARDED over the C axis (`P(None,'dp')` for V (d_in,C); `P('dp',
None)` for U (C,d_out)) — every einsum stays valid, XLA inserts the reduce on the
contracted sharded C. CI fn + Adam sharded on the largest axis. PGD broadcast source
(1,T,C) sharded on C. Residual input + all activations BATCH-sharded (`P('dp')`) -> the
masked suffix re-forwards run on per-device sub-batches, so activation memory scales
1/n_dev. This is what lets a global batch that OOMs replicated fit across devices.

### Multi-GPU SPMD results (8× B200, 1 node, 1-task/GPU)
The single-pool 8B PD step SPMD-collapses cleanly — near-linear weak scaling, no manual
collectives (the headline SYNTHESIS claim, now at real-8B scale):

| mode    | per-dev bl | gbatch | tok/s total | tok/s/GPU | MFU* | note |
|---------|-----------:|-------:|------------:|----------:|-----:|------|
| 1 GPU   | 4          | 4      |       4,538 |     4,538 | ~87% | baseline |
| --shard | 1          | 8      |      31,098 |     3,887 | 74%  | below compute knee |
| --shard | 2          | 16     |      34,244 |     4,280 | 82%  | |
| --shard | 4          | 32     |      34,565 |     4,321 | 82%  | **95% of 8× single-GPU** |
| --shard | 8          | 64     |        OOM  |       —   |  —   | 223GB global `clean` logits |

`--shard` = `jax.jit` + C-sharded V/U/CI/Adam + batch-sharded resid +
`with_sharding_constraint`. Scales near-linearly. The bl8/dev OOM is NOT a sharding
gather of the per-device work — it's the un-checkpointed global `clean` logits
(gbatch,2048,128256) that XLA materializes once; same wall as the 1-GPU bl8 OOM.

`--shmap` (`jax.shard_map` over `dp`, params+source REPLICATED, per-shard losses +
source/imp grad `pmean`'d) keeps `clean` strictly per-shard — but it OOMs at bl4 on
8×B200 (100GB/dev) because replicating V/U + CI + their fp32 Adam states on every
device (no C-sharding) costs more than the activation-locality it buys back.
Numerically identical to --shard at mesh=1 (same loss to 4 dp).

**Finding: for this workload `--shard` (C-sharded optimizer) is the better memory story
than `--shmap` (pure DP, replicated optimizer).** The optimizer states (fp32 Adam m+v
over the ~2.1B V/U + ~0.6B CI = ~22GB replicated) dominate; C-sharding them across 8
devices frees ~19GB/dev, which more than pays for the one global `clean` logits tensor.
The ideal is a HYBRID — C-shard the optimizer AND shard_map the batch so neither the
optimizer nor `clean` is global — left as the next step. `--shmap` stays as the
guaranteed-correct DP reference and the multi-node path where activation-locality wins.

*MFU is an upper estimate (the FLOP model charges fwd+bwd=3× for PGD ascents whose
backward is cheaper with params detached, and doesn't separately count ckpt recompute).
tok/s/GPU is the hard number. torch reference: ~3,050 tok/s/GPU (2-pool, 80-GPU lore
baseline) / ~1,658 (1-pool bl2). JAX single-pool ~4,300 tok/s/GPU @ 8 GPU is competitive
WITHOUT the pool split — same conclusion as the prior 1-GPU 4-way, now at multi-GPU.

## 1 -> N decomposed layers (2026-06-09, GPU) — matched-Llama-8B 12-layer prep

The full-LM variant was generalized from "decompose ONLY L18 MLP (3 sites)" to
"decompose MLP on a **contiguous range** of layers (3N sites)". Default range is
`20..31` (12 layers, 36 sites) to prep a matched Llama-8B 12-layer A/B vs torch.

### What changed (1 -> N)
- **`llama8b.py`**: `DECOMPOSED_LAYER=18` + `SITES=(gate,up,down)` →
  `LayerRange(first,last)` + `KINDS=(gate,up,down)`. `Target.l18_*` (one layer's frozen
  attn/lns/MLP weights) → `Target.decomp_layers: list[DecompLayerFrozen]` (one per
  decomposed layer) + `Target.tail: list[FrozenBlock]` (fully-frozen layers above
  `last`). `DecompVU` arrays gained a **leading layer axis `L`** (`Vg: (L,d,C)` etc).
  `suffix_logits` loops the decomposed layers (frozen attn + masked-decomposed MLP),
  then the tail blocks. `all_site_inputs` harvests all 3N clean CI inputs in
  `(layer,kind)` order, threading the *clean* MLP output through so layer i+1's site
  inputs see layer i's clean output (matches the torch target forward). `weight_deltas`
  vmaps over the layer axis. Residual-start prefix harvest loads L0..`first-1`.
- **`ci_fn.py`**: ONE shared `global_shared_transformer` over ALL 3N sites (torch's
  `GlobalSharedTransformerCiFn` is one transformer with all sites concatenated, NOT
  per-layer). Inputs (3N of them) RMS-normed + concatenated → `d_model` → blocks →
  out_head emits `3*L*C` logits, reshaped `(b,t,L,3,C)` and split per kind to
  `{kind: (b,t,L,C)}`. `CIFnDims` gained `n_layers`; `total_in = L*(2d+di)`.
- **`llama8b_step.py`**: masks per kind are `(b,t,L,C)`; `_layerfirst` moves L to axis 0
  so `suffix_logits` indexes `masks[k][i]`. PGD source `{kind: (1,T,L,C)}`. Stoch masks
  ONE kind across all its layers (matches "mask one module type" — torch masks one
  module, here one kind-across-layers is the natural N-site batched analog; ppgd masks
  all sites). `make_llama8b_step(..., n_layers, ...)` and `_shmap` take `n_layers`.
- **`llama8b_sharding.py`**: V `(L,d,C)` shards axis 2, U `(L,C,d)` shards axis 1
  (C on `dp`); source `(1,T,L,C)` shards C. CI out_head `(d_model, 3LC)` shards last.
- **`experiments/llama8b_real.py`**: `--first_layer/--last_layer/--C/--per_gpu_batch`
  are the only knobs; FLOP model counts `first..n_layer-1` suffix blocks + `3LC` CI
  head; reports tok/s/GPU + MFU + **peak GB/device** (`memory_stats`).

### Validation (single B200, 183GB)
- **12L random weights**, bl1 seq2048 C2048 replicated: compiles to one executable,
  6 steps, **3,003 tok/s/GPU**, 101 GB/dev peak, ppgd 0.666 > stoch 0.448 (minimax OK).
- **12L real HF weights** (L0..L19 prefix harvested, L20..L31 decomposed), bl1 C2048
  `--shard` on 1-dev mesh: see table below.

The 1→12 extension is mechanically clean — no NCCL/sharding surprises; the leading-L
axis threads through V/U, source, masks, Adam, and the C-shard plan unchanged. Peak mem
scales ~linearly in N (12× the sites ⇒ 12× the V/U + 12× the per-layer activations of
the masked re-forwards). At full C the matched run needs the multi-GPU C-shard + batch
shard (the established `--shard` story), exactly as the L18-only run did.

### Launch incantations (parametrized by C / batch / mesh)
`C`, `per_gpu_batch`, and world size are picked LATER from the torch 2-pool sweep; the
structure (layers 20..31, 4 losses, n_warmup=2, seq2048, real weights) is fixed.

```bash
WT=/mnt/home/oli/pd-nano-jax-jaxsp; cd $WT/nano_param_decomp_jax
# (each run: source .venv-cuda/bin/activate; export PYTHONPATH=$WT/jax_spike;
#  HF_HUB_CACHE=/mnt/data/artifacts/hf_cache/hub)

# MATCHED run (plug in the fixed C + per_gpu_batch the torch agent settles on; N GPUs):
sbatch --nodes=$NODES jax_single_pool/experiments/llama8b_slurm.sbatch \
  --real_weights --shard --first_layer 20 --last_layer 31 \
  --C $C --per_gpu_batch $BL --steps 12 --n_warmup 2
# gbatch = BL * (8*NODES). Match torch's global batch by choosing BL and NODES.

# MAX-BATCH sweep (find the per-GPU batch knee at the fixed C; bump --per_gpu_batch
# until OOM, single node is enough to find the per-dev ceiling):
sbatch jax_single_pool/experiments/llama8b_slurm.sbatch \
  --real_weights --shard --C $C --per_gpu_batch 1 --steps 8   # then 2, 4, ...

# MIN-GPU run (smallest world size that fits the fixed C+batch — start at 1 node,
# drop GPUs by editing --gres / --ntasks-per-node in the sbatch, or run 1-GPU directly):
python -m jax_single_pool.experiments.llama8b_real \
  --real_weights --shard --C $C --per_gpu_batch $BL --steps 8
```

The sbatch (`llama8b_slurm.sbatch`) is 1-task/GPU, `--qos=opportunistic`, passes its
args straight through to `llama8b_real`. For >12h or multi-node bump `--time`/`--nodes`.

### Llama-8B 12-layer perf (real HF weights, bl1 seq2048 C2048, --shard)

| mode / topo        | gbatch | tok/s tot | tok/s/GPU | peak GB/dev | stoch | ppgd | note |
|--------------------|-------:|----------:|----------:|------------:|------:|-----:|------|
| 1 GPU, 1-dev       |      1 |     4,558 |     4,558 |        84.4 |  0.95 | 4.97 | MFU ~78% (upper est) |
| 8 GPU (1 node)     |      8 |    27,749 |     3,469 |        64.3 |  0.48 | 5.18 | SPMD collapse OK; bl1 below the compute knee |

The 8-GPU run (job 49449, opportunistic) confirms the SPMD collapse holds at 12 layers —
no manual collectives; V/U + CI + Adam + source C-sharded, batch sharded; the GRPC/NCCL
lines at job end are routine `jax.distributed.shutdown` teardown noise, not a failure.
bl1/8GPU sits below the compute knee (3,469/GPU < the 1-GPU 4,558), same shape as the
prior L18-only scaling — per-dev throughput recovers at higher per-GPU batch, which the
max-batch sweep finds. Peak drops 84 -> 64 GB/dev because the C-shard splits V/U + Adam.

The single-routed-stoch-forward (torch `recon_plan: subset`) is BOTH more faithful and
faster than the prior 3-per-kind-forward stoch: at 12L it cut a step from ~678ms to
~449ms (3,020 -> 4,558 tok/s/GPU) and peak from 101 -> 84 GB/dev — one routed forward
replaces three. Strong minimax holds (ppgd 4.97 >> stoch 0.95 on real structure). At
C2048/bl1 the 12L footprint is ~84 GB/dev; full C (8192+) at a useful per-GPU batch
needs the multi-GPU C-shard + batch-shard (the established `--shard` story).

These C/batch are placeholders for the validation — the matched run plugs in the C +
per-GPU batch the torch 2-pool sweep fixes.

### (legacy) single-layer L18 perf, pre-generalization, for reference
