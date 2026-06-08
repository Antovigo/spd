# HANDOFF — JAX port + the {jax|torch}×{1-pool|2-pool} benchmark (2026-06-08, "jaxer")

You are a cluster-launched Claude continuing this work autonomously. Full record is also in
**lore: `2026-06-08--jax-port-handoff-4way-benchmark`** (read it if the lore MCP is available;
otherwise this file is self-contained).

## Mission
Complete a **fresh, controlled 4-way throughput comparison** — `{jax, torch} × {1-pool, 2-pool}`
— on the **real Llama-3.1-8B layer-18-MLP PD workload**. Metric: **tok/s/GPU** (+ total tok/s),
each cell at its natural topology. The user has explicitly OK'd using many H200s.

## Ground truth / layout
- Work from **`~/pd-nano-jax`** (git worktree, branch `feature/nano-pd-jax`, rebased on
  `feature/multipool` so it has both the torch ref impl AND the JAX spike). Commit progress there;
  update lore as you go.
- **Venvs**: torch lab env = `~/pd-nano-jax/.venv` (torch 2.11+cu128; run torch from here with
  `PYTHONPATH=. .venv/bin/python` so multipool code, not main's, is used). JAX env =
  `~/pd-nano-jax/jax_spike/.venv-cuda` (jax cuda + equinox + optax).
- **SSH from a laptop** (if needed): `ssh -o RemoteCommand=none -o RequestTTY=no a-login`.

## Controlled workload spec (from `param_decomp_lab/experiments/lm/_llama8b/llama8b_l18_b512_2pool_lr_mid.yaml`)
Llama-3.1-8B, decompose `layers.18.mlp.{gate,up,down}`, **C=24576** (overcomplete), seq2048,
batch512, bf16, residual-start. CI fn `global_shared_transformer` d4096 / **4 blocks / 64 heads** /
mlp16384. Losses: Faithfulness 1e5, ImportanceMinimality (pnorm2, p-anneal→0.4),
StochasticReconLayerwise 0.5, PersistentPGDRecon 0.5 (**scope broadcast_across_batch**, n_warmup 2).
`use_delta_component: true`. comp LR 1.5e-4 / ci LR 5e-5. 2-pool topology: pool_a bl8 (n=64) +
chunkwise bl32 (dp=16) = **80 GPU / 10 nodes**.

## The 4 cells + next actions
1. **torch 2-pool** — baseline already in lore (~4.86 s/step, **~3,050 tok/s/GPU @ 80 GPU**). Run
   fresh for consistency: `cd ~/pd-nano-jax && PYTHONPATH=. .venv/bin/python -m
   param_decomp_lab.experiments.lm.two_pool_run param_decomp_lab/experiments/lm/_llama8b/llama8b_l18_b512_2pool_lr_mid.yaml --dp 80`
   then read steady-state tok/s (after compile) and scancel.
2. **torch 1-pool** — config authored:
   `param_decomp_lab/experiments/lm/_llama8b/llama8b_l18_b512_seq2048_1pool.yaml` (kind:hf hook-based
   ComponentModel, matched losses). Launch `... -m param_decomp_lab.experiments.lm.run <that> --dp ~64`.
   This also tests "does 1-pool even fit?" (replicates everything per rank).
3. **jax 1-pool** — extend `jax_spike/stage9_pd_bench.py` + `vendored_jax/llama.py:random_init` to the
   real workload (L18-MLP only, C=24576, **layerwise** recon, 4-block d4096 CI transformer, broadcast
   PGD, weight-delta, bf16). Replicated WON'T fit → add GSPMD **param sharding**
   (`NamedSharding`/`PartitionSpec` on V/U + CI-fn across the mesh). Launch via `remote/gpu.sh`.
4. **jax 2-pool** — faithful pool-split from the spike (stages 1–5): pool A = CI transformer + PGD
   adversary, pool B = component recon, masks/grads via differentiable `jax.lax.ppermute`.

## Load-bearing gotchas (don't relearn these)
- **JAX timing**: warm up with `jax.block_until_ready` before measuring (else XLA compile time leaks
  in — burned us ~20×). Measure dispatch-only too (enqueue w/o block) to spot host-bound.
- **JAX multinode**: `srun --gres=gpu:8` (all GPUs visible), each proc
  `jax.distributed.initialize(local_device_ids=[int(os.environ["SLURM_LOCALID"])])`,
  `sync_global_devices(...)` barrier before `jax.distributed.shutdown()`. See `distributed_util.py`.
- **TF32**: JAX defaults to full fp32 on GPU, torch uses TF32. Match precision for fair perf;
  for *parity* force `jax_default_matmul_precision="highest"` + torch `allow_tf32=False`.
- **NFS `__pycache__` ENOTEMPTY** during uv installs: clear pycache + retry.
- **git worktrees**: main checkout can't be on `feature/nano-pd-jax` (worktree holds it) — expected.
- **compute nodes run git 2.34.1** (login node is newer). A global `merge.conflictstyle=zdiff3`
  (needs git ≥2.35) makes EVERY torch DDP snapshot checkout fail instantly with
  `fatal: unknown style 'zdiff3'` (job FAILED in ~5s, looks like a scheduler drop). Fix once:
  `git config --global merge.conflictstyle diff3`.

## Already proven (don't redo)
Parity: JAX Equinox Llama + GPT-2 are **bit-parity vs torch vendored** (rel-L2 ~1e-7 fwd / ~1e-6
grads), fwd+bwd, clean+masked (`jax_spike/parity/`). Multinode JAX validated to 16 GPU / 2 nodes.
Toy-config perf (synthetic L12/d2048/C32): JAX 1-pool ~1.74× torch — SUPERSEDED by this real 4-way.

## 4-way DONE (2026-06-08) — see lore `2026-06-08--4way-results-jax-torch-1pool-2pool`
Real-workload bench code: `stage10_real_pd_bench.py` (1-pool) + `stage11_real_pd_2pool.py` (2-pool),
both faithful residual-start 14-layer suffix, L18-MLP C=24576 weight-delta, 4-block d4096 CI
transformer, layerwise stoch + persistent broadcast PGD. Cluster was saturated → controlled
small-topology, per-rank batch matched, tok/s/GPU metric.
- **jax 1-pool**: bl8 OOMs, **bl4 fits = 2,864 tok/s/GPU** (1 H200) — ≈ torch 2-pool's ~3,050/GPU on
  the same suffix workload, WITHOUT the pool split. Headline: jax single-pool is competitive.
- **torch 1-pool** (full 32L hook): OOMs at bl8 (even fp32); bf16 bl2 = ~1,658/GPU. Doesn't fit at bl8.
- **torch 2-pool**: ~3,050/GPU @ 80 GPU (lore baseline, not re-run).
- The 1-pool is memory-bound in both frameworks → the 2-pool is a *memory* tool (shards V/U), not a
  throughput/GPU win (at matched batch 2-pool ≤ 1-pool).

### Open TODOs (the two unfinished bits)
1. **jax 1-pool multi-GPU**: data-parallel `jit` OOMs — replicated params+Adam = ~80GB/GPU floor AND
   GSPMD materializes a global-batch tensor (~87GB) instead of sharding it. Need FSDP-style
   param/optimizer sharding + keep batch sharded (`shard_map` or `with_sharding_constraint`); the
   `--shard_params` flag in stage10 is stubbed/unimplemented. Then weak-scale to multi-node.
2. **Same-hardware A/B**: when the cluster frees up, run torch 2-pool fresh + jax 1-pool at equal GPU
   count / equal global batch for a true apples-to-apples (current torch-2pool is the 80-GPU lore #).
