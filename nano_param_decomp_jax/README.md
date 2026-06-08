# nano-pd-jax

A JAX port of the param-decomp core, living as a self-contained sibling of
`nano_param_decomp/` (the PyTorch reference). v1 scope: TMS + toy MLP +
tiny transformer, all converging. SimpleStories 2L and the full loss family
are deferred to v2.

## Layout

- **`nano_pd_jax/`** — the library (~400 LOC, 8 files)
  - `linear.py` — mask-tolerant `Linear` shim (so user models don't need
    isinstance dispatch between Linear and DecomposedLinear)
  - `decomposed.py` — `DecomposedLinear` (dual-mode `__call__`: target vs
    decomposed, gated on whether `mask` is None; `W_delta` recomputed each
    call from current V·U) + `substitute_decomposed` (eqx.tree_at swap) +
    `collect_site_paths` walker
  - `ci_sigmoids.py` — `lower_leaky_hard_sigmoid` via `jax.custom_vjp`
  - `ci_fn.py` — per-site `SiteCI` MLP + `CIFn` dict (flattens arbitrary
    leading dims so batch/seq/etc. all work)
  - `masks.py` — stochastic mask sampling, `m = ci + (1 − ci) · u`
  - `losses.py` — faithfulness, importance-minimality, stochastic recon
  - `trainer.py` — `TrainState`, `make_trainable_filter`, `init_state`,
    `make_step_fn`. Two optimizers via `eqx.partition` (no
    `optax.multi_transform`, no string-path matching).

- **`experiments/`** — runnable training scripts
  - `tms.py` — pretrain TMS 5→2→5, decompose. Faith → ~1e-7, stoch → 3e-5.
  - `toy_mlp.py` — random 2-layer teacher MLP (d=64, d_ff=128), 4 sites C=16.
    Faith → rank-16 floor, stoch → 1.5e-4.
  - `tiny_transformer.py` — 2-layer transformer (d=64, n_heads=4, vocab=256,
    seq=32), 13 sites C=8. Random init, random int tokens. Stoch → 6e-5.

- **`bake-off/`** — historical: the 3-variant architecture bake-off whose
  winner became the library. Excluded from root ruff. See
  `bake-off/README.md` and `bake-off/SHARED_SPEC.md`.

- **`SHARED_SPEC.md`** — the algorithm spec the bake-off variants and the
  library all implement.

- **`pyproject.toml`** — self-contained uv project (NOT a workspace member
  of the root). Has its own `.venv` to avoid conflicts with the root
  PyTorch env.

## Setup

```bash
cd nano_param_decomp_jax
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python experiments/tms.py
python experiments/toy_mlp.py
python experiments/tiny_transformer.py
```

CPU JAX is sufficient for v1 (all three experiments train in seconds to a
few minutes). CUDA JAX (`jax[cuda12]` / `jax[cuda13]`) is deferred to v2
when we start needing real wall-clock at scale.

## Architecture decision history

This library is the output of a 3-way agent bake-off (see `bake-off/`)
comparing three architectural factorings of the same algorithm:

1. **forward returns everything** — user writes two parallel forwards.
   Hazardous past TMS scale.
2. **typed pytree leaves** — `Decomposed` struct + polymorphic `linop`.
   Doesn't generalize past single-matmul sites.
3. **`eqx.tree_at` substitution** — module swap via Equinox. Won, with two
   fixes (mask-tolerant `Linear` shim + acts inlined into `__call__`).

The bake-off also surfaced a math bug in the original spec: `W_delta`
must be recomputed each step from `V·U`, not frozen at init. Variant 1's
agent caught this from reading nano's `ComponentLinear.weight_delta()`.
Three orders of magnitude difference on the faith loss.

## Related artifacts

- **Library walkthrough:** <https://nano-pd-jax.pages.goodfire.pub/>
- **Scaling tour (JAX on NVIDIA GPUs, with the 3-pool case study):**
  <https://jax-distributed-tour.pages.goodfire.pub/>
- **PyTorch core:** `../param_decomp/` (this is what's being ported from)
- **PyTorch reference impl:** `../nano_param_decomp/` (the compact 1219-LOC
  PyTorch impl that nano-pd-jax mirrors algorithmically)
