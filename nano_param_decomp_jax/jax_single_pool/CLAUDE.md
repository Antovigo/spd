# jax_single_pool — agent notes

Single-pool VPD (4-loss + persistent-PGD) training step in JAX. Sibling to
`nano_pd_jax` (whose primitives it reuses) and the research counterpart to the
torch FSDP path in `param_decomp_lab/fsdp/`. See `README.md` for the file map and
`../../jax_single_pool_NOTES.md` for the design log + torch↔JAX mapping.

## Invariants to preserve

- **The step is one `jax.jit` fn over a pure `TrainState`.** No in-place mutation,
  no Python-level state. The persistent adversary (sources + Adam moments) lives in
  `TrainState.pgd` and is threaded through — that's the whole point (functional
  minimax vs torch's `before_backward`/`after_backward` orchestration).
- **`n_warmup + 1` source updates per step.** `pgd_warmup` (lax.scan, `n_warmup`
  ascents) then `pgd_final_ascend` (one more, post-param-update). Matches the torch
  PPGD `warmup` + the fused final step. Don't collapse to `n_warmup`.
- **Frozen `W_target`.** Its grad is zeroed before the main optimizer; the new
  decomp re-pins `state.decomp.W_target`. `faithfulness = mean((W_target - V@U)^2)`
  is recomputed each step (NOT frozen at init — the 3-orders-of-magnitude bug the
  nano bake-off caught).
- **mask convention:** `mask = ci + (1-ci)*source` for component channels; the
  weight-delta source channel (last, when `use_delta_component`) is passed through
  raw (no ci interpolation). Mirrors `param_decomp/masks.py`.
- **GPU-count invariance.** Any change to the step or sharding must keep the
  `distributed_stacked_sites` trajectory bit-identical at 1 vs N devices (fixed
  batch + seed, broadcast scope). That's the SPMD-correctness contract.

## Gotchas

- **`shard_batch` topology.** Uses `make_array_from_process_local_data` so it's
  correct for BOTH single-process-many-devices (CPU sim, 1-process-N-GPU) and
  multi-process-1-device (SLURM). Do NOT revert to the per-`process_index()`-slice
  idiom — it silently replicates one slice on single-process multi-device CPU (see
  NOTES "test-harness pitfall").
- **Homogeneous sites only.** The stacked `[S, ...]` einsums assume equal site
  shapes. Heterogeneous sites need padding or per-site (no S-stacking) — deferred.
- **Stochastic-mask RNG under sharding.** `jax.random.uniform` over a sharded batch
  partitions per shard; this is fine for training (independent noise per element)
  but means the *stochastic* loss isn't bit-invariant across shard layouts (the
  recon/faith/ppgd terms are). The invariance check fixes the seed and relies on
  the deterministic terms; the trajectory still matched bit-for-bit in practice
  because the global batch + seed were fixed.

## When changing things

- Keep the package out of the repo `[tool.pyright]` `include` (it's a JAX sibling;
  the torch venv type-checks the torch side). Still: `basedpyright jax_single_pool/`
  must be clean, and `pytest jax_single_pool/tests/` green at 1 AND 4 sim devices.
- Update `../../jax_single_pool_NOTES.md` with any new perf / invariance result —
  that comparison is the deliverable.
