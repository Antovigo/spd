"""Trainer: substitutes DecomposedLinear at named sites, runs the three losses with
two optimizers, jits the step.

Two-optimizer pattern via `eqx.partition`:

  - The model pytree is split into (trainable, frozen) where `trainable` keeps V and
    U arrays inside every DecomposedLinear and `frozen` keeps W_delta, bias, and the
    rest of the (untouched) user model. Optimizer A operates on `trainable`.
  - The CI fn is fully trainable. Optimizer B operates on it directly.
  - `eqx.combine(trainable, frozen)` reconstitutes the model for forward passes.

Pre-weight activation collection:

  - We require the user's model to expose `forward_with_acts(x) -> (out, acts)`,
    where `acts: dict[str, Array]` keys each decomposed site by its path. The user
    writes this once per model. Equinox idiom: simple, explicit, and stays in plain
    Python — no eqx.nn.State, no implicit side channels.
"""

import equinox as eqx
import jax
import optax
from ci_fn import CIFn
from decomposed_linear import DecomposedLinear
from jaxtyping import Array, Float, PRNGKeyArray
from losses import faithfulness_loss, importance_minimality_loss, stochastic_recon_loss


def make_trainable_filter(model: eqx.Module) -> eqx.Module:
    """Bool-pytree matching `model`'s array-leaf layout: True at V and U leaves inside
    any DecomposedLinear, False everywhere else (W_target, W_delta, bias, plus every
    array elsewhere in the user model). Used as the `eqx.partition` filter spec — each
    leaf must be a scalar bool, so we replace whole arrays with True/False sentinels."""

    def per_node(node: object) -> object:
        if isinstance(node, DecomposedLinear):
            out = eqx.tree_at(lambda m: m.V, node, True)
            out = eqx.tree_at(lambda m: m.U, out, True)
            out = eqx.tree_at(lambda m: m.W_target, out, False)
            out = eqx.tree_at(lambda m: m.W_delta, out, False)
            if node.bias is not None:
                out = eqx.tree_at(lambda m: m.bias, out, False)
            return out
        return jax.tree.map(lambda _: False, node)

    return jax.tree.map(
        per_node, model, is_leaf=lambda n: isinstance(n, DecomposedLinear)
    )


def collect_site_paths(model: eqx.Module) -> list[str]:
    """Walk the model and return dotted attribute paths to every DecomposedLinear,
    in sorted order so the CI fn module-key layout is deterministic."""
    paths: list[str] = []

    def visit(prefix: str, node: object) -> None:
        if isinstance(node, DecomposedLinear):
            paths.append(prefix)
            return
        if isinstance(node, eqx.Module):
            for f in node.__class__.__dataclass_fields__:
                child = getattr(node, f)
                sub = f"{prefix}.{f}" if prefix else f
                visit(sub, child)

    visit("", model)
    return sorted(paths)


class TrainState(eqx.Module):
    trainable: eqx.Module  # V, U arrays only — rest are sentinels
    frozen: eqx.Module  # W_delta, bias, and the rest of the model
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


def make_step_fn(
    target_model: eqx.Module,
    site_paths: list[str],
    coeff_faith: float,
    coeff_imp: float,
    coeff_stoch: float,
    p_value: float,
    opt_main: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
):
    """Build a JIT-compiled step. `target_model` is captured as a closure constant."""

    @eqx.filter_jit
    def step(
        state: TrainState,
        x: Float[Array, "B d_in"],
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        # Frozen target outputs.
        target_out = jax.vmap(target_model)(x)

        def loss_fn(params, key: PRNGKeyArray):
            trainable, ci_fn = params
            model = eqx.combine(trainable, state.frozen)
            _, acts = jax.vmap(model.forward_with_acts)(x)
            ci = ci_fn(acts)
            k_stoch, _ = jax.random.split(key)
            l_faith = faithfulness_loss(model, site_paths)
            l_imp = importance_minimality_loss(ci, p_value)
            l_stoch = stochastic_recon_loss(model, ci, x, target_out, key=k_stoch)
            total = coeff_faith * l_faith + coeff_imp * l_imp + coeff_stoch * l_stoch
            return total, (l_faith, l_imp, l_stoch)

        (total, (l_faith, l_imp, l_stoch)), (grads_trainable, grads_ci) = (
            eqx.filter_value_and_grad(loss_fn, has_aux=True)((state.trainable, state.ci), key)
        )

        updates_main, new_opt_main = opt_main.update(
            grads_trainable, state.opt_state_main, state.trainable
        )
        new_trainable = eqx.apply_updates(state.trainable, updates_main)

        updates_ci, new_opt_ci = opt_ci.update(grads_ci, state.opt_state_ci, state.ci)
        new_ci = eqx.apply_updates(state.ci, updates_ci)

        return (
            TrainState(
                trainable=new_trainable,
                frozen=state.frozen,
                ci=new_ci,
                opt_state_main=new_opt_main,
                opt_state_ci=new_opt_ci,
            ),
            {"total": total, "faith": l_faith, "imp": l_imp, "stoch": l_stoch},
        )

    return step


def current_model(state: TrainState) -> eqx.Module:
    """Convenience: reconstitute the full decomposed model from the partitioned state."""
    return eqx.combine(state.trainable, state.frozen)
