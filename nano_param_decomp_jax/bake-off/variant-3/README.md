# variant-3 — Equinox-native module swap

The target model is an `eqx.Module` written naturally with `eqx.nn.Linear` sublayers.
Decomposition is performed by substituting each named `eqx.nn.Linear` with a
`DecomposedLinear` module via `eqx.tree_at`. The user's model code stays close to a
plain Equinox model — the only invasion is that it must thread masks through
`__call__` and expose a `forward_with_acts` helper.

## Architectural pattern

**Substitution.** `decomposed_linear.substitute_decomposed(model, {path: C})` walks
each path string (e.g. `"layer1.up"`), pulls the matching `eqx.nn.Linear`, and uses
`eqx.tree_at(lambda m: m.<path>, model, DecomposedLinear(...))` to install a
replacement. The new model has the same overall pytree structure aside from those
swapped leaves.

**Mask threading.** Masks are an explicit kwarg on the user model's `__call__`:
`model(x, masks=None)` where `masks: dict[str, Array]` keys per site path. The user
routes per-site masks in by hand (`m1 = masks["layer1"] if masks else None`). When
`masks=None` (or when a particular layer isn't decomposed) the layer takes the target
path. We considered `eqx.nn.State` and closures, but explicit args win for JAX
purity: no hidden mutation, no extra pytree to thread separately. Cost: the user's
forward picks up some boilerplate (see the `_apply` helper in the train scripts).

**Pre-weight activations.** The user implements `forward_with_acts(x) -> (out, acts)`
on their model. `acts` is a dict keyed by the same site paths. This is the second
piece of API the user owns. Equivalent to a manual "hook" — explicit but verbose.

**Two-optimizer pattern.** `eqx.partition(model, filter_spec)` splits the model into
`(trainable, frozen)` where the filter spec is a same-shape pytree with scalar booleans:
`True` for `V` / `U` arrays inside every `DecomposedLinear`, `False` for `W_target`,
`W_delta`, `bias`, and every leaf of the unmodified user model. Optimizer A operates
on `trainable`; optimizer B operates on the CI fn directly. `eqx.combine(trainable,
frozen)` reconstitutes the model for forward passes inside the loss. Gradients flow
only through the trainable leaves because `eqx.filter_value_and_grad` only differentiates
arrays in the params arg.

**JIT.** `eqx.filter_jit` over the step function. `target_model` is captured as a
closure constant; static fields (`d_in`, `d_out`, `C`) are marked `eqx.field(static=True)`.

## How to add a new target model

1. Write the model as an `eqx.Module` with `eqx.nn.Linear` sublayers.
2. Make `__call__` accept `masks: dict[str, Array] | None = None` and route per-site.
3. Add `forward_with_acts(x) -> (out, acts)` returning a dict of pre-weight acts.
4. Call `substitute_decomposed(model, {path: C, ...})` then `init_state` and
   `make_step_fn`.

See `train_tms.py` and `train_toy_mlp.py` for end-to-end examples.
