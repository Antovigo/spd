# variant-2 — typed pytree leaves + polymorphic `linop`

Decomposition is a *leaf type*. At each decomposed site in the params pytree, the raw
weight array is replaced by a `Decomposed(V, U, W_delta)` struct (an `eqx.Module`, so
pytree-registered). The user's forward function is written **once**, factored through a
single dispatch function:

```python
def linop(leaf, x, mask=None):
    if isinstance(leaf, Decomposed):
        assert mask is not None
        return ((x @ leaf.V) * mask) @ leaf.U + x @ leaf.W_delta
    return x @ leaf
```

Two pytrees with identical structure cover both modes:

- `target_params       = {"W1": Array, "b1": Array, "W2": Array, ...}`
- `decomposed_params   = {"W1": Decomposed(V, U, W_delta), "b1": Array, ...}`

The same `model_forward(params, x, masks=None)` runs both — the difference is *which*
pytree you pass in and whether you supply masks.

## Files

- `decomposed.py` — `Decomposed` leaf struct, `linop` dispatch, `init_decomposed`.
- `ci_fn.py` — per-site `SiteCI` MLP (the only other `eqx.Module`) + lower-leaky-hard
  sigmoid via `jax.custom_vjp` for the asymmetric backward.
- `losses.py` — faithfulness (against frozen `target_weights`), `mean(ci^p)` importance
  minimality, MSE stochastic recon, mask sampling.
- `trainer.py` — `build_decomposed_params`, the two-optimizer pattern (see below), the
  jitted train step, `train()` driver.
- `train_tms.py` / `train_toy_mlp.py` — the two SHARED_SPEC targets. Each file's job is
  basically: (a) write `model_forward` factored through `linop`, (b) init target params,
  (c) call `train()`.

## How masks are threaded

Masks live in a `dict[site_name, Array]` whose keys match decomposed-site keys in the
params pytree. The user's forward reads `masks.get("W1")` at each call to `linop`. In
target mode the caller passes `masks=None` and every `linop` sees `mask=None`. The
trainer samples one mask dict per step from the current CI dict.

## Two-optimizer pattern

- **Main optimizer** (V and U of every `Decomposed` leaf, plus implicit freeze of
  `W_delta` and any non-`Decomposed` leaves like biases): built with
  `optax.multi_transform`. A label tree marks paths ending in `GetAttrKey('V'|'U')` as
  `"train"` (adamw) and everything else as `"freeze"` (`optax.set_to_zero`).
- **CI optimizer** (per-site `SiteCI` MLPs): a plain `optax.adamw`. The CI fns are
  partitioned with `eqx.partition(_, eqx.is_array)` so optax only sees the trainable
  array leaves.

`jax.value_and_grad(..., argnums=(0, 1))` returns gradients into both pytrees in one
call; we then update each with its own optimizer.

## Adding a new target model

1. Write `def model_forward(params, x, masks=None) -> (out, dict_of_pre_acts)` that
   factors every weighted op through `linop(params[site], x, masks.get(site) if masks else None)`.
2. Write `init_target_params(key) -> dict` that returns the raw target params.
3. Pick a `c_per_site: dict[str, int]` covering the sites to decompose.
4. Call `train(key, target_params, model_forward, data_fn, c_per_site, cfg)`.

No subclassing, no module surgery, no special protocol. The user owns the forward; the
trainer is a pure function of `forward_fn` + the two pytrees.

## Running

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python train_tms.py
python train_toy_mlp.py
```
