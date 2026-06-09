"""The whole single-pool VPD training step, as one jit-compilable function.

This is the SPMD-collapse target: clean target outputs (frozen) → CI envelope →
the four losses (faith, imp-min, stochastic recon, persistent-PGD recon) → fused
grads for components (V/U) and the CI fn, with the PGD adversary's sources
carried as persistent state. Two optimizers (functional Adam over V/U and over
the CI fn). Everything is pure; the only state is `TrainState` in/out.

Minimax discipline (mirrors `jax_spike/stage6_pgd.py` + the torch PPGD metric):
  1. n_warmup supplemental source-only ascent iters refine the persistent source.
  2. ONE fused outer backward over (decomp, ci) computes the param grads; the
     PPGD term uses the refined source (the adversary is fixed for the param
     descent — torch detaches it likewise in `before_backward`).
  3. params DESCEND; then one more source ASCEND against the fresh params, persisted.

The data axis is the leading batch dim of `x` / `target`-shaped tensors. Under
GSPMD that axis is sharded `P('dp')`; the mean-losses reduce over it so `jax.jit`
inserts the grad all-reduce — no manual collectives (see sharding.py + stage8).
"""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from jax_single_pool.losses import (
    CIParams,
    Decomposition,
    ci_envelope,
    faithfulness_loss,
    importance_minimality_loss,
    interpolate_mask,
    layerwise_recon_loss,
    sample_stochastic_source,
)
from jax_single_pool.pgd import (
    PGDConfig,
    PGDState,
    adversarial_mask,
    pgd_final_ascend,
    pgd_warmup,
)


class LossCoeffs(NamedTuple):
    faith: float
    imp: float
    stoch: float
    ppgd: float
    p_imp: float


class TrainState(NamedTuple):
    decomp: Decomposition  # V/U trainable, W_target frozen
    ci: CIParams
    opt_main: optax.OptState  # over (V, U)
    opt_ci: optax.OptState  # over CI fn
    pgd: PGDState
    step: Float[Array, ""]


def _stochastic_mask(
    key: PRNGKeyArray,
    ci: Float[Array, "S ... C"],
    source_c: int,
    use_delta_component: bool,
) -> Float[Array, "S ... source_c"]:
    src = sample_stochastic_source(key, (*ci.shape[:-1], source_c))
    return interpolate_mask(ci, src, use_delta_component)


def make_step(
    coeffs: LossCoeffs,
    pgd_cfg: PGDConfig,
    opt_main: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    source_c: int,
    use_delta_component: bool,
):
    @jax.jit
    def step(
        state: TrainState,
        x: Float[Array, "S B ... d_in"],
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        x = jax.lax.stop_gradient(x)
        batch_dims = x.shape[1:-1]

        ci_for_pgd = ci_envelope(state.ci, x)
        refined_pgd = pgd_warmup(state.decomp, x, ci_for_pgd, state.pgd, batch_dims, pgd_cfg)

        def loss_fn(decomp: Decomposition, ci_params: CIParams):
            ci = ci_envelope(ci_params, x)

            l_faith = faithfulness_loss(decomp)
            l_imp = importance_minimality_loss(ci, coeffs.p_imp)

            stoch_mask = _stochastic_mask(key, ci, source_c, use_delta_component)
            l_stoch = layerwise_recon_loss(decomp, x, stoch_mask, use_delta_component)

            adv_mask = adversarial_mask(
                ci,
                jax.lax.stop_gradient(refined_pgd.sources),
                batch_dims,
                use_delta_component,
            )
            l_ppgd = layerwise_recon_loss(decomp, x, adv_mask, use_delta_component)

            total = (
                coeffs.faith * l_faith
                + coeffs.imp * l_imp
                + coeffs.stoch * l_stoch
                + coeffs.ppgd * l_ppgd
            )
            return total, (l_faith, l_imp, l_stoch, l_ppgd)

        (total, (l_faith, l_imp, l_stoch, l_ppgd)), (g_decomp, g_ci) = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(state.decomp, state.ci)

        # W_target is frozen: zero its grad so the main optimizer never moves it.
        g_main = Decomposition(
            V=g_decomp.V, U=g_decomp.U, W_target=jnp.zeros_like(g_decomp.W_target)
        )
        upd_main, new_opt_main = opt_main.update(g_main, state.opt_main, state.decomp)
        upd_main = cast(Decomposition, upd_main)
        new_decomp = Decomposition(
            V=state.decomp.V + upd_main.V,
            U=state.decomp.U + upd_main.U,
            W_target=state.decomp.W_target,
        )

        upd_ci, new_opt_ci = opt_ci.update(g_ci, state.opt_ci, state.ci)
        upd_ci = cast(CIParams, upd_ci)
        new_ci = CIParams(w=state.ci.w + upd_ci.w, b=state.ci.b + upd_ci.b)

        ci_after = ci_envelope(new_ci, x)
        new_pgd = pgd_final_ascend(new_decomp, x, ci_after, refined_pgd, batch_dims, pgd_cfg)

        new_state = TrainState(
            decomp=new_decomp,
            ci=new_ci,
            opt_main=new_opt_main,
            opt_ci=new_opt_ci,
            pgd=new_pgd,
            step=state.step + 1,
        )
        metrics = {
            "total": total,
            "faith": l_faith,
            "imp": l_imp,
            "stoch": l_stoch,
            "ppgd": l_ppgd,
        }
        return new_state, metrics

    return step


def init_train_state(
    decomp: Decomposition,
    ci: CIParams,
    pgd: PGDState,
    opt_main: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
) -> TrainState:
    return TrainState(
        decomp=decomp,
        ci=ci,
        opt_main=opt_main.init(decomp),
        opt_ci=opt_ci.init(ci),
        pgd=pgd,
        step=jnp.array(0.0),
    )
