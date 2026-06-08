# Variant 1 — Forward returns everything

## The pattern

The user owns the model entirely. They provide **two pure forward functions**:

```python
def target_forward(params, x) -> (out, pre_acts)

def decomposed_forward(params, components, masks, x) -> (out, pre_acts)
```

- `params` is the user's normal weight dict.
- `pre_acts` is `dict[site_id, array]` — the pre-weight activations at each
  decomposable site. It exists purely for the CI fn to read.
- `components[site_id] = {"V": ..., "U": ..., "W_delta": ...}` — the trainer
  threads these in.
- `masks[site_id]` — the trainer samples these from CI.

The trainer never inspects, traces, or modifies the user's forward. It just
calls them twice per step (once for the target output, once for the decomposed
output) and threads `components` + `masks` via positional args.

## How the trainer accesses the model

It doesn't, except by calling the two functions. There is no module hierarchy
to walk, no `tree_at` substitution, no `hasattr` checks. The whole interface
is two function objects and a `target_weights` dict (so the trainer can compute
faithfulness and re-materialise `W_delta = W_target - V@U` each step).

## How masks flow

1. Trainer calls `target_forward(params, x)` to get `pre_acts`.
2. Trainer applies the CI fn (a `dict[site_id, eqx.nn.MLP]`) to `pre_acts`,
   producing `cis[site_id] ∈ [0, 1]^{..., C}` via `lower_leaky_hard_sigmoid`.
3. Trainer samples `masks[site_id] = ci + (1 - ci) * U[0, 1]`.
4. Trainer calls `decomposed_forward(params, components, masks, x)` to get
   the decomposed output.

The user's forward simply does `((x @ V) * mask) @ U + x @ W_delta` at each
decomposed site. That's it.

## Two-optimizer pattern

`vu` (the `{site: {V, U}}` dict) and `ci_fn` (the dict of MLPs) are
co-differentiated via `eqx.filter_value_and_grad((vu, ci_fn), ...)`. Each gets
its own `optax.adam` instance and state. After the single backward pass, we
apply each set of updates independently. Adding separate LR schedules is one
extra line per optimizer.

`W_delta` is excluded from the optimizer simply by not being part of `vu` — it's
recomputed inside `loss_fn` from `target_weights - V @ U`.

## Adding a new target model

Write the two forwards. That's the whole API surface for a user. See
`train_tms.py` (~120 LOC including TMS pretraining) and `train_toy_mlp.py`
(~100 LOC) for end-to-end examples.

## Files

- `ci_fn.py` — per-site MLPs + `lower_leaky_hard_sigmoid` (custom VJP)
- `losses.py` — faithfulness, importance-minimality, stochastic-recon, mask sampling
- `trainer.py` — the train loop and jitted train step
- `train_tms.py`, `train_toy_mlp.py` — the two target-model demos

Total: ~600 LOC.

## How to run

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python train_tms.py
python train_toy_mlp.py
```
