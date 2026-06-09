# jax_single_pool

A JAX implementation of the **single-pool** Parameter Decomposition (VPD)
training loop — the four-term loss (faithfulness + importance-minimality +
stochastic-recon + persistent-PGD adversarial-recon) with the persistent
adversary, run as one `jax.jit`'d step and GSPMD-sharded data-parallel.

This is the research counterpart to the torch FSDP single-pool path
(`param_decomp_lab/fsdp/`). It tests the "single-pool SPMD collapse" hypothesis
from `jax_spike/SYNTHESIS.md`: that XLA + `jax.jit` over the whole step, with
GSPMD sharding, optimizes this complex minimax loop as well as (or better than)
the abandoned hand-written-NCCL multi-pool torch design — with **zero manual
collectives**.

## What's here

| file | what |
|---|---|
| `forward.py` | site-local masked decomposed forward (layerwise recon) + weight-delta channel |
| `losses.py` | the four VPD losses over a stacked-site `Decomposition`; `mask = ci + (1-ci)*source` |
| `scopes.py` | PPGD source scopes (single / broadcast / repeat / per-batch-per-position) |
| `pgd.py` | functional persistent adversary — sources + Adam moments in state; `n_warmup` `lax.scan` ascent + one post-update ascent |
| `step.py` | the whole step as one `jax.jit` fn — fused grad over (V/U, CI), two functional Adams, minimax stop-gradient |
| `sharding.py` | GSPMD helpers (mesh / replicate / `shard_batch`) — the FSDP analog |
| `checkpoint.py` | flat-pytree save/resume of `TrainState` (adversary state included) |
| `llama8b.py` | full-LM Llama-3.1-8B target: residual-start L18->L31 suffix + decomposed L18 MLP + real HF safetensors loader |
| `ci_fn.py` | `global_shared_transformer` CI fn for the 8B target |
| `llama8b_step.py` | full-LM **output-recon** step (recon on suffix logits, not site-local); `--shard` (jit+constraint) and `--shmap` (shard_map DP) variants |
| `llama8b_sharding.py` | FSDP-analog GSPMD plan for the 8B step |
| `experiments/` | runnable CPU smokes + the GSPMD distributed runner + `llama8b_real.py` (tok/s/GPU + MFU) + `llama8b_slurm.sbatch` |
| `tests/` | pure-fn unit tests, sharding tests, checkpoint resume, llama8b step |

## Run

```bash
cd nano_param_decomp_jax
uv venv .venv && source .venv/bin/activate && uv pip install -e .

# single-device CPU smokes
python jax_single_pool/experiments/toy_stacked_sites.py     # synthetic stacked sites
python jax_single_pool/experiments/transformer_qkv.py       # real TinyTransformer q/k/v

# GSPMD GPU-count invariance (simulated devices on CPU)
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  python -m jax_single_pool.experiments.distributed_stacked_sites --steps 20 --global_batch 64
# trajectory must match the 1-device run bit-for-bit (fixed batch + seed)

# multi-GPU / multi-node (under SLURM, via jax_spike/remote/gpu.sh):
#   NODES=2 GPN=8 ... python -m jax_single_pool.experiments.distributed_stacked_sites

pytest jax_single_pool/tests/
```

## Design

- **Stacked-site representation.** The decomposition is `(V, U, W_target)` stacked
  along a leading site axis `S`; the CI fn is a per-site linear head. Sites must be
  homogeneous (equal `d_in`/`d_out`) — the production target ("decompose layer-18
  MLP") is a fixed same-shape weight set. The step is otherwise model-agnostic: it
  consumes pre-weight activations `x: [S, B, ..., d_in]` and reconstructs each
  site's output (layerwise recon).
- **One jit'd step.** `make_step` returns a `jax.jit` function: frozen acts → CI
  envelope → four losses → fused `value_and_grad` over `(decomp, ci)` → two
  functional Adam updates (frozen `W_target` grad zeroed) → one post-update source
  ascent. The PGD adversary's persistent sources + Adam moments are carried in
  `TrainState` and threaded through.
- **GSPMD, not pools.** Data sharded `P('dp')`, params + sources replicated,
  `jax.jit` inserts the grad all-reduce. Validated GPU-count-invariant
  (bit-identical trajectories at 1 vs N devices). The replicated-source grad
  reduction the torch path does explicitly (`reduce_source_grads`) is absorbed by
  XLA's autodiff of the global mean.

See `../../jax_single_pool_NOTES.md` for the torch↔JAX mapping, the perf /
invariance results, and open questions.
