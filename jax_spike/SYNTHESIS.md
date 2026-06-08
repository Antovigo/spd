# Two JAX spikes, compared

Two independent JAX feasibility spikes now exist. They are **complementary** — one
went vertical (the algorithm), one went horizontal (the distribution) — and together
they cover most of the question.

## What each covered

| | `feature/nano-pd-jax` (prior) | `jax_spike/` (this) |
|---|---|---|
| Axis | **Algorithm** port, single-device | **Distribution** / 2-pool topology |
| Framework | Equinox + Optax | raw JAX (+ stolen sigmoid) |
| Scope | TMS, toy MLP, tiny transformer — all converge | cross-pool autograd, transport, multi-process |
| Method | 3-variant bake-off → `eqx.tree_at` substitution | 5 staged grad-checks |
| Key win | real PD trains in JAX; clean 2-optimizer split | differentiable `ppermute` deletes manual cotangent plumbing |
| PGD / adversary | deferred to v2 | not covered |
| GPU perf/mem | not measured | not measured |

## Where they agree (independent convergence = high confidence)

1. **The two-optimizer structure is trivial in JAX.** Prior spike: `eqx.partition`
   splits V/U from CI-fn, one optimizer each, no string-path matching. This spike:
   the two grad sets fall out of one `jax.grad` over `(ci_params, vu_params)`.
2. **The pure-mask-arg forward is the right contract.** Prior spike's winning variant
   threads `mask` as a forward kwarg (`__call__(x, mask=None)`); the torch 3-pool
   independently converged on the *same* thing (vendored `ComponentGPT2`, masks as a
   forward arg → pure forward). Both worlds agree the hook-based impure forward is the
   thing to drop.
3. **Functional autograd fits PD cleanly** — `custom_vjp` for the leaky sigmoid,
   `vjp` for the cross-pool seam. Stage 5 here confirms the prior spike's `custom_vjp`
   sigmoid composes bit-exact through this spike's distributed transport.

## What I stole from the prior spike

- **Framework choice: Equinox + Optax.** Better than raw JAX for a real port.
  `eqx.partition`/`eqx.combine` is the clean 2-optimizer (and N-optimizer) answer.
- **`lower_leaky_hard_sigmoid` via `custom_vjp`** — used verbatim in Stage 5.
- **`sample_masks` + the faith/imp/stoch loss forms** — used in Stage 5.
- **The W_delta correctness note**: `W_delta = W_target − V@U` recomputed each step,
  NOT frozen at init (a 3-orders-of-magnitude bug the bake-off caught).
- **The real friction they found**: mask-threading + `forward_with_acts` duplication
  grows with model depth (48 mask keys at 12 layers). This is the ergonomic cost of
  the substitution approach — and it's exactly what GSPMD + a pure-mask-arg forward
  should absorb.

## The combined picture

Vertical (theirs) × horizontal (mine) now meet in Stage 5: the **real PD losses run
split across two pools with bit-exact gradients**. The building blocks for a JAX PD
trainer all exist and check out:

- decomposition + substitution (`eqx.tree_at`) ✓ theirs
- two optimizers (`eqx.partition`) ✓ theirs
- CI sigmoid (`custom_vjp`) ✓ theirs, ✓ distributed here
- stochastic mask ✓ theirs, ✓ distributed here
- faith/imp/stoch losses ✓ theirs, ✓ distributed here
- cross-pool transport (differentiable `ppermute`) ✓ here
- per-process shard locality ✓ here

## PGD / the adversary — covered by Stage 6 (correct on CPU + GPU)

Persistent-PGD is the compute bottleneck, the reason the pool split exists, and the
only stateful piece. Stage 6 (`stage6_pgd.py`) prototypes it and all four mechanics
check out, bit-exact on CPU and re-verified on an H200:

- **persistent state** — adversarial `sources` carried in `TrainState`, warm-started.
- **PGD inner loop** — `lax.scan` over n_warmup ascent steps; bit-exact to a python
  loop, no compile blow-up.
- **fused multi-`argnums` grad** — one backward over (V/U, CI, sources); CI graph flows.
- **minimax stop-gradient** — inner loop ascends sources (params detached); outer
  descends params. Jitted step trains (worst-case recon ↓), sources persist.

GPU scaling (1× H200, S=12, d=512, C=64; `remote/gpu.sh`):

| inner-loop depth (n_warmup) | ms/step | | batch (n_warmup=10) | ms/step |
|---|---|---|---|---|
| 1  | 0.25 | | 64   | 0.65 |
| 10 | 0.98 | | 256  | 0.99 |
| 20 | 1.72 | | 1024 | 1.95 |
| 40 | 3.17 | | 4096 | 5.50 |
| 80 | 6.14 | | | |

Cost is **linear in scan depth** (~0.075 ms/PGD-step over a ~0.18 ms fixed outer-grad
floor) and ~linear in batch once compute-bound — predictable, exactly the regime a
pool split is meant to balance. `lax.scan` handles the inner loop with no unroll/compile
pathology. (These are toy-shape numbers — not a model-scale claim.)

## Multi-GPU / multi-node — Stages 7 & 8 (real H200, up to 16 GPUs / 2 nodes)

**Stage 7 (`stage7_distributed_hello.py`) — distributed bring-up.** `jax.distributed`
under SLURM + a cross-mesh `psum`. PASS at 8 GPUs (1 node) and **16 GPUs (2 nodes)**
(`psum(device_ids)=120`). The launch recipe that works on this cluster (the part that
took the most iteration):
- `srun` with all GPUs visible per task (`--gres=gpu:8`, no per-task GPU binding),
- each process claims `local_device_ids=[SLURM_LOCALID]` (per-task binding +
  ordinal-0-for-all tripped 'invalid device ordinal' inside NCCL),
- barrier (`sync_global_devices`) before `jax.distributed.shutdown()` to avoid a
  coordinator-teardown gRPC race.

**Stage 8 (`stage8_train_distributed.py`) — the whole stack, GSPMD-sharded.** The full
single-pool SPMD step: real PD math + the four losses (faith / imp / stoch / PGD-recon)
+ the PGD adversary + two hand-rolled Adam optimizers, with data sharded `P('dp')`,
params replicated, and `jax.jit` inserting the grad all-reduce automatically — **no
manual collectives anywhere.**

*Correctness:* pure data-parallelism is GPU-count-invariant, so a fixed global batch +
seed must give the same loss trajectory at any scale. It does — 1 / 8 / 16 GPUs agree to
~5 sig figs (final total 0.45582 / 0.45582 / 0.45583; bit-level reduction-order drift
only). The multi-node 16-GPU run is correct.

*Weak scaling* (256 samples/GPU, S=12 d=512 C=64 n_warmup=10):

| GPUs | global batch | ms/step | samples/s | efficiency |
|---|---|---|---|---|
| 1  | 256  | 155 | 1,648  | — |
| 8  | 2048 | 157 | 13,084 | 99% |
| 16 (2 nodes) | 4096 | 168 | 24,310 | 92% |

Step time stays ~flat as batch + GPUs grow together → near-linear weak scaling, holding
across the node boundary (the 8% drop at 16 is the cross-node grad all-reduce).

## Remote GPU tooling (built here, reusable)

Git-based: edit locally → commit → `remote/gpu.sh` pushes, the cluster worktree pulls,
a SLURM job runs, the log streams back.

- `remote/gpu.sh "python …"` — `NODES`/`GPN`/`PART` env knobs pick the scale
  (e.g. `NODES=2 GPN=8` → 16 GPUs). Pushes the branch, pulls on the cluster, submits
  `srun` (1 task/GPU), waits, prints the log.
- `remote/job.sbatch` — generic; scale overridden on the sbatch CLI. All GPUs visible
  per task; `_remote_cmd.sh` carries the command.
- `remote/setup_cluster.sh` — one-time: `feature/nano-pd-jax` worktree at `~/pd-nano-jax`
  + persistent `jax[cuda12]` venv.
- `distributed_util.py` — `init_distributed()` (the LOCALID recipe), `dp_mesh()`,
  `shard_dp()`, `replicate()`.
- SSH bypasses the forced-tmux login config with `-o RemoteCommand=none -o RequestTTY=no`.

## Net recommendation (now demonstrated end-to-end on GPU)

Don't port the heterogeneous 2-pool faithfully. The **single-pool SPMD collapse** is
real and it works: the entire PD+PGD stack runs as one `jax.jit`'d step, GSPMD-sharded,
training correctly and scaling ~linearly to 16 GPUs / 2 nodes — with zero manual
collectives, zero pool-coordination code, zero deadlock surface. The manual pool split
is largely a workaround for torch's missing ergonomic auto-sharding (FSDP); GSPMD
provides that natively.

**Remaining before a real port decision:** model-shaped scale (a genuine HF/Flax target,
not toy einsums) with param sharding (not just data-parallel) for the memory story;
XLA-vs-`torch.compile` throughput parity; and the interpretation/checkpoint pipeline
boundary (still torch).
