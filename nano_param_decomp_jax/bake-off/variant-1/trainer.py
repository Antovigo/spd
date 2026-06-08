"""Trainer for variant 1: user supplies (target_forward, decomposed_forward).

Architecture: the user writes two pure forward functions:

    target_forward(params, x) -> (out, pre_acts)
    decomposed_forward(params, components, masks, x) -> (out, pre_acts)

The trainer never inspects or modifies either function — it just calls them.
Masks are threaded as an explicit positional dict argument. Two optimizers:
one for (V, U) across all sites, one for the CI fn. W_delta is frozen.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import optax
from ci_fn import CIFn, apply_ci_fn
from jaxtyping import Array, Float, PRNGKeyArray
from losses import (
    Components,
    Masks,
    TargetWeights,
    faithfulness_loss,
    importance_minimality_loss,
    init_components,
    sample_masks,
    stochastic_recon_loss,
)

TargetForward = Callable[[dict, Any], tuple[Float[Array, "..."], dict[str, Float[Array, "..."]]]]
DecomposedForward = Callable[
    [dict, Components, Masks, Any],
    tuple[Float[Array, "..."], dict[str, Float[Array, "..."]]],
]


def _materialize_components(
    vu: Components, target_weights: TargetWeights
) -> Components:
    """Add a fresh `W_delta = W_target - V @ U` per site for the decomposed forward.

    Recomputed every step from current V/U (matches nano's `ComponentLinear.weight_delta`).
    """
    return {
        name: {"V": c["V"], "U": c["U"], "W_delta": target_weights[name] - c["V"] @ c["U"]}
        for name, c in vu.items()
    }


def make_train_step(
    target_forward: TargetForward,
    decomposed_forward: DecomposedForward,
    target_weights: TargetWeights,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    coeff_faith: float,
    coeff_imp: float,
    coeff_stoch: float,
    imp_p: float,
):
    """Returns a jitted train_step.

    Closes over forwards, target_weights, optimizers, and loss coefficients. `components`
    is the full pytree {V, U, W_delta}; only V/U receive grads — W_delta is frozen.
    """

    def loss_fn(
        trainable: tuple[Components, CIFn],
        params: dict,
        x: Any,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        vu, ci_fn = trainable
        components = _materialize_components(vu, target_weights)
        _y_target_for_grad, pre_acts = target_forward(params, x)
        cis = apply_ci_fn(ci_fn, pre_acts)
        masks = sample_masks(cis, key)
        y_decomp, _ = decomposed_forward(params, components, masks, x)
        y_target, _ = target_forward(params, x)

        l_faith = faithfulness_loss(components, target_weights)
        l_imp = importance_minimality_loss(cis, p=imp_p)
        l_stoch = stochastic_recon_loss(y_decomp, jax.lax.stop_gradient(y_target))
        total = coeff_faith * l_faith + coeff_imp * l_imp + coeff_stoch * l_stoch
        return total, {"faith": l_faith, "imp": l_imp, "stoch": l_stoch, "total": total}

    @eqx.filter_jit
    def train_step(
        vu: Components,
        ci_fn: CIFn,
        opt_vu_state: optax.OptState,
        opt_ci_state: optax.OptState,
        params: dict,
        x: Any,
        key: PRNGKeyArray,
    ):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            (vu, ci_fn), params, x, key
        )
        grad_vu, grad_ci = grads
        updates_vu, new_opt_vu = opt_vu.update(grad_vu, opt_vu_state, vu)
        new_vu = optax.apply_updates(vu, updates_vu)

        ci_diff, ci_static = eqx.partition(ci_fn, eqx.is_inexact_array)
        grad_ci_diff, _ = eqx.partition(grad_ci, eqx.is_inexact_array)
        updates_ci, new_opt_ci = opt_ci.update(grad_ci_diff, opt_ci_state, ci_diff)
        new_ci_diff = optax.apply_updates(ci_diff, updates_ci)
        new_ci_fn = eqx.combine(new_ci_diff, ci_static)

        return new_vu, new_ci_fn, new_opt_vu, new_opt_ci, aux

    return train_step


def train(
    target_forward: TargetForward,
    decomposed_forward: DecomposedForward,
    params: dict,
    target_weights: TargetWeights,
    ci_fn: CIFn,
    sample_batch: Callable[[PRNGKeyArray], Any],
    n_steps: int,
    key: PRNGKeyArray,
    c_per_site: dict[str, int],
    lr_vu: float = 1e-3,
    lr_ci: float = 1e-3,
    coeff_faith: float = 1.0,
    coeff_imp: float = 1e-3,
    coeff_stoch: float = 1.0,
    imp_p: float = 0.9,
    log_every: int = 100,
) -> tuple[Components, CIFn, list[dict[str, float]]]:
    init_key, key = jax.random.split(key)
    vu = init_components(target_weights, c_per_site, init_key)

    opt_vu = optax.adam(lr_vu)
    opt_ci = optax.adam(lr_ci)
    opt_vu_state = opt_vu.init(vu)
    ci_diff, _ = eqx.partition(ci_fn, eqx.is_inexact_array)
    opt_ci_state = opt_ci.init(ci_diff)

    train_step = make_train_step(
        target_forward,
        decomposed_forward,
        target_weights,
        opt_vu=opt_vu,
        opt_ci=opt_ci,
        coeff_faith=coeff_faith,
        coeff_imp=coeff_imp,
        coeff_stoch=coeff_stoch,
        imp_p=imp_p,
    )

    history: list[dict[str, float]] = []
    for step in range(n_steps):
        batch_key, mask_key, key = jax.random.split(key, 3)
        x = sample_batch(batch_key)
        vu, ci_fn, opt_vu_state, opt_ci_state, aux = train_step(
            vu, ci_fn, opt_vu_state, opt_ci_state, params, x, mask_key
        )
        if step % log_every == 0 or step == n_steps - 1:
            losses = {k: float(v) for k, v in aux.items()}
            losses["step"] = step
            history.append(losses)
            print(
                f"step {step:5d}  total={losses['total']:.4e}  faith={losses['faith']:.4e}  "
                f"imp={losses['imp']:.4e}  stoch={losses['stoch']:.4e}"
            )

    final_components = _materialize_components(vu, target_weights)
    return final_components, ci_fn, history
