"""Train state, jitted step, outer loop.

Architecture (v3-with-fixes from the bake-off):
- The user's model is an `eqx.Module` with `nano_pd_jax.Linear` sublayers.
  After `substitute_decomposed`, the SAME model serves both modes via the
  `mask` kwarg (None → target forward, dict → decomposed forward). One pytree.
- `eqx.partition` with a scalar-bool filter splits trainable V/U inside every
  DecomposedLinear from everything else. Optimizer A operates on `trainable`;
  optimizer B operates on `ci_fn` directly.
- Forward returns `(out, acts)` always; the trainer collects acts on the
  target pass and ignores them on the decomposed pass.
"""

import equinox as eqx
import jax
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from nano_pd_jax.ci_fn import CIFn
from nano_pd_jax.decomposed import DecomposedLinear
from nano_pd_jax.losses import (
    faithfulness_loss,
    importance_minimality_loss,
    stochastic_recon_loss,
)
from nano_pd_jax.masks import sample_masks


def make_trainable_filter(model: eqx.Module) -> eqx.Module:
    """Bool-pytree marking V, U leaves inside DecomposedLinear True; else False.

    Filter spec for `eqx.partition` must be scalar bools per leaf — we replace
    whole arrays with True/False sentinels of the same pytree shape.
    """

    def per_node(node):
        if isinstance(node, DecomposedLinear):
            out = eqx.tree_at(lambda m: m.V, node, True)
            out = eqx.tree_at(lambda m: m.U, out, True)
            out = eqx.tree_at(lambda m: m.W_target, out, False)
            if node.bias is not None:
                out = eqx.tree_at(lambda m: m.bias, out, False)
            return out
        return jax.tree.map(lambda _: False, node)

    return jax.tree.map(per_node, model, is_leaf=lambda n: isinstance(n, DecomposedLinear))


class TrainState(eqx.Module):
    trainable: eqx.Module
    frozen: eqx.Module
    ci: CIFn
    opt_state_main: optax.OptState
    opt_state_ci: optax.OptState


def init_state(
    decomposed_model: eqx.Module,
    ci_fn: CIFn,
    opt_main: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
) -> TrainState:
    filter_spec = make_trainable_filter(decomposed_model)
    trainable, frozen = eqx.partition(decomposed_model, filter_spec)
    return TrainState(
        trainable=trainable,
        frozen=frozen,
        ci=ci_fn,
        opt_state_main=opt_main.init(eqx.filter(trainable, eqx.is_array)),
        opt_state_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
    )


def current_model(state: TrainState) -> eqx.Module:
    return eqx.combine(state.trainable, state.frozen)


def make_step_fn(
    site_paths: list[str],
    coeff_faith: float,
    coeff_imp: float,
    coeff_stoch: float,
    p_value: float,
    opt_main: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
):
    @eqx.filter_jit
    def step(
        state: TrainState,
        x: Float[Array, "B d_in"],
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        def loss_fn(params, key):
            trainable, ci = params
            model = eqx.combine(trainable, state.frozen)

            target_out, acts = jax.vmap(model)(x)
            target_out = jax.lax.stop_gradient(target_out)
            acts = jax.tree.map(jax.lax.stop_gradient, acts)

            ci_dict = ci(acts)
            masks = sample_masks(key, ci_dict)
            decomp_out, _ = jax.vmap(model)(x, masks)

            l_faith = faithfulness_loss(model, site_paths)
            l_imp = importance_minimality_loss(ci_dict, p_value)
            l_stoch = stochastic_recon_loss(decomp_out, target_out)
            total = coeff_faith * l_faith + coeff_imp * l_imp + coeff_stoch * l_stoch
            return total, (l_faith, l_imp, l_stoch)

        (total, (l_faith, l_imp, l_stoch)), (g_train, g_ci) = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )((state.trainable, state.ci), key)

        upd_main, new_state_main = opt_main.update(g_train, state.opt_state_main, state.trainable)
        new_trainable = eqx.apply_updates(state.trainable, upd_main)

        upd_ci, new_state_ci = opt_ci.update(g_ci, state.opt_state_ci, state.ci)
        new_ci = eqx.apply_updates(state.ci, upd_ci)

        return (
            TrainState(
                trainable=new_trainable,
                frozen=state.frozen,
                ci=new_ci,
                opt_state_main=new_state_main,
                opt_state_ci=new_state_ci,
            ),
            {"total": total, "faith": l_faith, "imp": l_imp, "stoch": l_stoch},
        )

    return step
