# REFLECTION — variant-2

## Final loss values

**TMS (5000 steps, ~12s on one A/H100):** `total=0.0079  faith=0.00048  imp=0.073  stoch=0.0067`
(faith fell 1300x, stoch fell ~1.2x because it was already tiny at init)

**Toy MLP (5000 steps, ~25s):** `total=0.022  faith=0.0195  imp=1.0  stoch=0.0016`
(stoch fell 190x; faith stalled — random teacher of dim 64x128 is not low-rank-16-recoverable, so most loss budget is the irreducible faith floor)

Total: **651 LOC** across 6 files.

## What was elegant

The polymorphic `linop` is genuinely the prettiest thing about variant 2. The user
writes their forward *once*, in a way that's nearly indistinguishable from a normal
forward (`linop(W, x, m)` instead of `x @ W`). The same closure runs both the target
forward (clean array leaves, `masks=None`) and the decomposed forward (Decomposed
leaves, mask dict). No dual code path. `jax.value_and_grad(..., argnums=(0,1))` then
just falls out — V, U, and CI fn params all get correct grads in one pass.

Pre-act collection through the explicit `pre_acts` dict return is also clean — no
hooks, no global mutable state, just data flowing through the return value.

## Where there was friction

1. **Freezing `W_delta`.** The "leaf is a struct of mixed trainable + frozen arrays"
   story doesn't map cleanly to optax. I ended up with `optax.multi_transform` + a label
   tree built via `tree_map_with_path` matching on `GetAttrKey('V'|'U')`. It works but
   it's *spooky*: a string match on attribute names inside a pytree path. If someone
   later renames `V` to `basis_in`, the freeze logic silently breaks. `eqx.partition`
   with a filter spec would be more honest but ugly with the `Decomposed`-as-leaf
   pattern since it splits the struct into halves.

2. **CI fns are equinox modules in a dict-of-modules.** `optax.adamw().init(ci_fns)`
   chokes on non-array leaves (the `SiteCI` modules carry shape ints). I had to
   `eqx.partition` once at trainer init, thread the static half through the jit, and
   `eqx.combine` it back inside the loss closure. Workable but feels like two opposing
   pytree philosophies (typed-leaf vs filter-and-partition) glued together.

3. **`isinstance` check inside `jit`-traced `linop`.** It works because the leaf type
   is a *trace-time* property (decided when jit specializes on the pytree structure),
   but it's the kind of thing that would explode if someone tried to do leaf dispatch
   based on a runtime flag. Fine here, surprising elsewhere.

## Would I bet on this for the full library port?

Lukewarm yes for a small/medium codebase, **no** for the full param-decomp library.

The polymorphic `linop` is beautiful at this scale but doesn't generalize. Real targets
have attention with QKV projections, conv layers, embeddings, RMSNorm — each needs its
own `op(leaf, x, *extra)` dispatch. You'd end up with `linop_attn`, `linop_conv`,
`linop_embed`... each duplicating the `isinstance(leaf, Decomposed)` branch. Variant 3
(module substitution) handles this without changes to the user's forward.

The frozen-`W_delta`-via-label-tree workaround would also rot under refactoring. In a
production system I'd want either: (a) a separate `FrozenDecomposed(V, U)` pytree-leaf
that doesn't carry `W_delta` (kept in a sidecar dict), or (b) `eqx.partition` with a
filter that's not a string-name match.

**Verdict:** the cleanest user-facing API of the three for a single op type, but the
weakest scaling story. Pick variant 3 for the library port.
