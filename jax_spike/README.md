# Spike: feasibility of rewriting the 2-pool PD training strategy in JAX

Isolated CPU spike (`.venv-jax`, `jax 0.10.1`, x64). Toy models — **correctness
results are bit-exact and scale-independent; performance/memory are NOT tested.**

The 2-pool architecture (torch, `param_decomp_lab/three_pool/`): two *heterogeneous*
process pools. Pool A holds the CI fn + PPGD adversary; Pool B (chunkwise) holds the
V/U components sharded by site. They exchange masks (A→B) and gradients (B→A) every
step over typed NCCL portals, with cross-pool autograd stitched by hand
(`torch.autograd.backward(graph, gradient=seed)`).

## What each stage tests

| Stage | File | Question | Result |
|---|---|---|---|
| 1 | `stage1_autograd_split.py` | Does the cross-pool autograd seam (masks fwd, cotangents back, vjp stitch) match a monolithic grad? | **PASS** 0.0 |
| 2 | `stage2_ppermute_transport.py` | Does `ppermute` compose with autograd so cotangents return *automatically*? | **PASS** 0.0 |
| 3 | `stage3_heterogeneous_shapes.py` | Can one mesh host pools with different shapes / different work? | **PASS w/ tax** |
| 4 | `stage4_worker.py` + `stage4_launch.py` | Multi-process: does transport survive a real process boundary with each process holding only its shard? | **PASS** 0.0 |

Run: `source .venv-jax/bin/activate`
- `python stage1_autograd_split.py`
- `XLA_FLAGS="--xla_force_host_platform_device_count=2" python stage2_ppermute_transport.py`
- `XLA_FLAGS="--xla_force_host_platform_device_count=2" python stage3_heterogeneous_shapes.py`
- `python stage4_launch.py`

## Findings

**Two of the torch design's hardest parts get *easier* in JAX:**

1. **Cross-pool cotangent stitch → `jax.vjp`.** No `retain_graph`, no manual
   `.grad` accumulation, no `torch.autograd.backward(…, gradient=seed)`. (Stage 1)

2. **Cross-pool cotangent *transport* is free.** `ppermute` is differentiable —
   ship masks A→B inside the differentiated region and autograd ships the cotangents
   B→A via its transpose. This **deletes** the manual `g_CI` return, the typed
   portals, and the load-bearing *"recv g_CI first, then send g_VU or deadlock"*
   ordering. XLA schedules the collectives. (Stages 2, 4)

3. **Per-process shard locality works.** In multi-process JAX, a globally-sharded
   array is only ever locally materialized as its addressable shards — each process
   holds only its own. This is exactly how within-pool component/CI-fn sharding +
   DP would scale, and it's GSPMD-native. (Stage 4)

**The one genuine friction: cross-pool *shape* asymmetry.** The 2-pool's whole point
is that Pool B never holds the CI fn and Pool A never replicates components. Under one
SPMD mesh:
- heterogeneous shapes can't be a single sharded array (Stage 3, probe 1);
- the working idiom (per-pool pytrees + `axis_index` branch) is correct but **traces
  both branches on both devices** (Stage 3 NOTE) — naively defeating the memory win.
- recovering it needs `lax.cond`-guarded compute + per-pool param placement, with
  collectives kept unconditional/matched. Achievable, bespoke, untested at scale here.

## Bottom line

More feasible than a first read suggests. JAX eliminates the 2-pool's nastiest
engineering (manual cross-pool cotangent plumbing + deadlock-prone P2P ordering).
The risk narrows to: (a) cross-pool shape asymmetry under one mesh, and (b) ecosystem
migration (HF targets → Flax/weight-conversion; the torch downstream
harvest/autointerp/app pipeline; re-validating grad-checks + checkpoint/resume).

**Strategic note:** the manual pool split partly exists because the torch
`ComponentModel` lacked ergonomic auto-sharding (FSDP). JAX/GSPMD shards across one
mesh natively — so the highest-leverage JAX design may not be a faithful 2-pool port
at all, but the **single-pool SPMD collapse** (shard CI fn + components across one
mesh, one batched forward, independent PPGD source per rank) that the design-space
notes already floated. The spike supports it: uniform sharding + differentiable
collectives just work.

## Untested (the real go/no-go before committing)
- Performance / memory at scale; XLA-vs-`torch.compile` parity (torch: 2.74×
  chunkwise validated at 160 GPU).
- True heterogeneous-memory asymmetry on GPU (does Pool B's device avoid allocating
  the CI fn?).
- PPGD inner loop as `lax.scan` + `jax.vjp` at scale; bf16 numerics.
- Flax port of a real HF target + the torch-checkpoint interop boundary.
