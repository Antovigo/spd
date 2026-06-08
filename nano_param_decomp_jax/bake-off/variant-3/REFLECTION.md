# REFLECTION — variant-3

## Final losses (5000 steps, GPU)

- **TMS** (5 → 2 → 5, two sites, C=5 each): total=0.0052, faith=7e-5, imp=0.30, stoch=0.0048
- **Toy MLP** (d=64, d_ff=128, four sites, C=16): total=0.0032, faith=0.0022, imp=0.39, stoch=0.0006

Both targets train cleanly. Faithfulness drops by ~3 orders of magnitude on TMS and
~8x on the MLP; stochastic-recon halves; importance saturates at a plateau set by
how many components the data wants to keep alive.

## What was elegant

`eqx.tree_at` for substitution is genuinely satisfying. The user writes a normal
`eqx.Module` with `eqx.nn.Linear` sublayers and the trainer hot-swaps decomposed
modules at named paths. The substitution mechanism is one line of library code per
site, and the result type-checks because `DecomposedLinear` shares the call signature
shape (a leading positional `x`).

`eqx.partition` + `eqx.combine` for the two-optimizer split is the cleanest factoring
I've used in JAX. Once I learned that filter specs need scalar booleans rather than
bool *arrays*, it took ~10 lines to express "trainable iff V/U inside a
DecomposedLinear" and the rest fell out. No manual leaf-name surgery, no `optax.masked`,
no parallel optimizer state.

## What was friction

The mask-threading invasion is real. The user's model has to (a) accept `masks=None`
in `__call__`, (b) route per-site by hand with `_apply(layer, x, mask)` helpers, and
(c) implement a separate `forward_with_acts(x)` to expose pre-weight activations.
For TMS that's small (two layers). For a transformer block — attention has 4 sites,
each MLP has 2 — it doubles. By the time you're at a 12-layer transformer the user is
plumbing 48 mask keys through their forward. Doable, but it does mean the
"natural Equinox model" framing erodes as you scale.

The `forward_with_acts` duplication is the worst of it: it's essentially the same
forward written twice. A future version could machine-generate it via tracing a
`forward(x, capture=True)` run, but that adds magic.

I also briefly fought `eqx.filter_value_and_grad`'s return shape (grads come back
matching the input pytree, so packing `(trainable, ci_fn)` into a tuple was the
ergonomic fix) and the filter-spec-as-bool-array gotcha.

## Would I bet on this for the full library port?

**Yes, with reservations.** For research models where the user already writes an
`eqx.Module`, the substitution mechanism is the cleanest of the three options. The
`eqx.partition` story for two optimizers scales straightforwardly to N optimizers.
The friction is concentrated in two API obligations on the user model (masks kwarg +
`forward_with_acts`), which becomes more annoying as the model grows. If the
transformer port hits that wall, a hybrid where `eqx.nn.State` carries masks (so the
user's `__call__` is unchanged but `forward_with_acts` still exists) might be the
escape valve. But for the bake-off, I'd score variant-3 as the most idiomatic JAX of
the three.
