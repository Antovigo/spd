"""Trainer wiring two pytrees (target and decomposed) + per-site CI fns into a JIT step.

Key design:
- The user supplies ONE `model_forward(params, x, masks=None) -> (out, pre_acts)`. It
  factors every weighted op through `linop(leaf, x, mask)`, so the same code handles
  both target (`mask=None`, leaf is a plain array) and decomposed (leaf is `Decomposed`,
  mask is the sampled mask) modes.
- The trainer:
    1. forwards through `target_params` for `y_target` and `pre_acts_target`
    2. forwards through `decomposed_params` with sampled masks for `y_decomposed`
    3. computes faith + imp + stoch_recon
    4. `jax.grad` over `decomposed_params` and `ci_fns` separately (two optimizers)

Freezing W_delta: we use `optax.masked` (via `optax.multi_transform`) with a label tree
marking W_delta leaves as `"freeze"` -> `optax.set_to_zero()`. Cleaner than zeroing
grads or partitioning the pytree.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.tree_util as jtu
import optax
from ci_fn import SiteCI, compute_ci, init_ci_fns
from decomposed import decomposed_sites, init_decomposed
from jaxtyping import Array, Float, PRNGKeyArray
from losses import (
    faithfulness_loss_against_targets,
    importance_minimality_loss,
    sample_masks,
    stochastic_recon_loss,
)

# (params, x, masks) -> (output, pre_acts). See docstring atop file for contract.
ForwardFn = Callable[..., tuple[Array, dict[str, Array]]]


@dataclass
class TrainConfig:
    n_steps: int
    main_lr: float = 1e-3
    ci_lr: float = 1e-3
    coeff_faith: float = 1.0
    coeff_imp: float = 1e-2
    coeff_stoch: float = 1.0
    log_every: int = 100


def build_decomposed_params(
    key: PRNGKeyArray, target_params: dict, c_per_site: dict[str, int]
) -> dict:
    """Replace each named site in `target_params` with `Decomposed(V, U, W_delta)`.

    Non-decomposed leaves (biases) pass through. Output has the same dict-keys as input.
    """
    names = sorted(c_per_site.keys())
    keys = jax.random.split(key, len(names))
    key_map = dict(zip(names, keys))
    out: dict[str, Any] = {}
    for k, v in target_params.items():
        if k in c_per_site:
            out[k] = init_decomposed(key_map[k], v, c_per_site[k])
        else:
            out[k] = v
    return out


def extract_target_weights(
    target_params: dict, c_per_site: dict[str, int]
) -> dict[str, Float[Array, "d_in d_out"]]:
    return {name: target_params[name] for name in c_per_site}


def _trainable_label_tree(decomposed_params: dict) -> dict:
    """Per-leaf label: 'train' for V/U inside Decomposed; 'freeze' for everything else.

    Uses tree_map_with_path so the final path entry tells us which Decomposed field we
    are at (V, U, or W_delta) — for non-Decomposed leaves, the final entry is the dict
    key, which we route to 'freeze'.
    """

    def label(path, _leaf):
        if path and isinstance(path[-1], jtu.GetAttrKey) and path[-1].name in ("V", "U"):
            return "train"
        return "freeze"

    return jtu.tree_map_with_path(label, decomposed_params)


def make_main_optimizer(decomposed_params: dict, lr: float) -> optax.GradientTransformation:
    labels = _trainable_label_tree(decomposed_params)
    return optax.multi_transform(
        {"train": optax.adamw(lr, weight_decay=0.0), "freeze": optax.set_to_zero()},
        labels,
    )


def _loss_and_aux(
    decomposed_params: dict,
    ci_arrays,
    ci_static,
    x: Float[Array, "B ..."],
    target_params: dict,
    target_weights: dict[str, Float[Array, "d_in d_out"]],
    mask_key: PRNGKeyArray,
    forward_fn: ForwardFn,
    cfg: TrainConfig,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    ci_fns = eqx.combine(ci_arrays, ci_static)

    # Target forward (no masks). Stop grads — target_params are frozen, no leak.
    target_params_sg = jax.lax.stop_gradient(target_params)
    y_target, pre_acts_target = forward_fn(target_params_sg, x, None)
    y_target = jax.lax.stop_gradient(y_target)
    pre_acts_target = {k: jax.lax.stop_gradient(v) for k, v in pre_acts_target.items()}

    ci = compute_ci(ci_fns, pre_acts_target)

    masks = sample_masks(mask_key, ci)
    y_decomposed, _ = forward_fn(decomposed_params, x, masks)

    l_faith = faithfulness_loss_against_targets(decomposed_params, target_weights)
    l_imp = importance_minimality_loss(ci)
    l_stoch = stochastic_recon_loss(y_decomposed, y_target)
    total = cfg.coeff_faith * l_faith + cfg.coeff_imp * l_imp + cfg.coeff_stoch * l_stoch
    return total, {"faith": l_faith, "imp": l_imp, "stoch": l_stoch, "total": total}


def make_train_step(
    forward_fn: ForwardFn,
    cfg: TrainConfig,
    main_opt: optax.GradientTransformation,
    ci_opt: optax.GradientTransformation,
):
    @eqx.filter_jit
    def step(
        decomposed_params: dict,
        ci_arrays,
        ci_static,
        main_state,
        ci_state,
        x: Array,
        target_params: dict,
        target_weights: dict,
        mask_key: PRNGKeyArray,
    ):
        def loss_wrt_both(dp, ca):
            return _loss_and_aux(
                dp, ca, ci_static, x, target_params, target_weights, mask_key, forward_fn, cfg
            )

        (_, aux), (g_dp, g_ca) = jax.value_and_grad(loss_wrt_both, argnums=(0, 1), has_aux=True)(
            decomposed_params, ci_arrays
        )

        upd_dp, new_main_state = main_opt.update(g_dp, main_state, decomposed_params)
        new_dp = optax.apply_updates(decomposed_params, upd_dp)

        upd_ca, new_ci_state = ci_opt.update(g_ca, ci_state, ci_arrays)
        new_ca = optax.apply_updates(ci_arrays, upd_ca)

        return new_dp, new_ca, new_main_state, new_ci_state, aux

    return step


def train(
    key: PRNGKeyArray,
    target_params: dict,
    forward_fn: ForwardFn,
    data_fn: Callable[[PRNGKeyArray], Array],
    c_per_site: dict[str, int],
    cfg: TrainConfig,
) -> tuple[dict, dict[str, SiteCI], list[dict[str, float]]]:
    dec_key, ci_key, train_key = jax.random.split(key, 3)

    decomposed_params = build_decomposed_params(dec_key, target_params, c_per_site)
    target_weights = extract_target_weights(target_params, c_per_site)
    ci_fns = init_ci_fns(ci_key, decomposed_sites(decomposed_params))

    ci_arrays, ci_static = eqx.partition(ci_fns, eqx.is_array)

    main_opt = make_main_optimizer(decomposed_params, cfg.main_lr)
    ci_opt = optax.adamw(cfg.ci_lr, weight_decay=0.0)
    main_state = main_opt.init(decomposed_params)
    ci_state = ci_opt.init(ci_arrays)

    step_fn = make_train_step(forward_fn, cfg, main_opt, ci_opt)

    history: list[dict[str, float]] = []
    for step_i in range(cfg.n_steps):
        train_key, data_key, mask_key = jax.random.split(train_key, 3)
        x = data_fn(data_key)
        decomposed_params, ci_arrays, main_state, ci_state, aux = step_fn(
            decomposed_params, ci_arrays, ci_static, main_state, ci_state,
            x, target_params, target_weights, mask_key,
        )
        if step_i % cfg.log_every == 0 or step_i == cfg.n_steps - 1:
            row = {"step": step_i, **{k: float(v) for k, v in aux.items()}}
            history.append(row)
            print(
                f"step={step_i:6d}  total={row['total']:.6f}  faith={row['faith']:.6f}  "
                f"imp={row['imp']:.6f}  stoch={row['stoch']:.6f}",
                flush=True,
            )

    return decomposed_params, eqx.combine(ci_arrays, ci_static), history
