"""The full Llama-8B single-pool VPD step: output-recon through the frozen suffix.

This is the full-LM variant the layerwise core (`step.py`) deferred. Unlike the
site-local core, recon here is MSE on the **final logits** of a masked suffix
re-forward — matching the torch `StochasticReconLayerwiseLoss` / `PersistentPGDReconLoss`
semantics (mask one/all sites, re-forward the whole suffix, compare to the clean
suffix output).

Loss structure mirrors `llama8b_l18_mlp_fsdp.yaml`:
  faith  = mean weight-delta^2 over the 3 sites           coeff 1e5
  imp    = mean(clip(ci,0,1)^p), p annealed -> 0.4         coeff 5e-6
  stoch  = mean over sites of MSE(masked-one-site logits, clean)   coeff 0.5
  ppgd   = MSE(all-sites-masked-by-persistent-source logits, clean) coeff 0.5

Persistent PGD: a broadcast (1, T, C) source per site, clamped to [0,1] (the torch
config sets `use_sigmoid_parameterization: false`). Source updates per step =
n_warmup ascents inside the step body + 1 fused-with-the-loss isn't done here; we do
n_warmup pre-ascents then one post-update ascent, matching the torch warmup + final.

Everything is one `jax.jit` over a pure `Llama8BState`. The frozen `Target` is a
runtime arg (replicated), not an HLO constant (a multi-GB constant made compilation
pathological — see stage10 note).
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random, shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from jax_single_pool.ci_fn import CIFn
from jax_single_pool.llama8b import (
    DT,
    SITES,
    DecompVU,
    Target,
    l18_resid_to_mlp_input,
    mlp_site_inputs,
    suffix_logits,
    weight_deltas,
)


class LossCoeffs(NamedTuple):
    faith: float
    imp: float
    stoch: float
    ppgd: float
    p_imp: float


class Llama8BState(NamedTuple):
    vu: DecompVU
    ci_fn: CIFn
    opt_vu: optax.OptState
    opt_ci: optax.OptState
    source: dict[str, Float[Array, "1 t C"]]  # broadcast persistent PGD source per site
    step: Array


def _clamp_source(src: dict[str, Array]) -> dict[str, Array]:
    return {s: jnp.clip(src[s], 0.0, 1.0) for s in SITES}


def _recon(a: Array, b: Array) -> Array:
    return jnp.mean((a.astype(jnp.float32) - b.astype(jnp.float32)) ** 2)


def make_llama8b_step(
    coeffs: LossCoeffs,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    pgd_lr: float,
    n_warmup: int,
    mesh: Mesh | None,
):
    """n_warmup pre-ascent iters refine the persistent source; one post-update ascent
    persists it warm-started against the fresh params (torch warmup + final = n_warmup+1).

    `mesh` (when given) pins every batch-leading activation to `P('dp', ...)` via
    `with_sharding_constraint` — without it XLA propagation gathers the suffix
    activations to the FULL global batch and OOMs (the open HANDOFF TODO). With it,
    every masked re-forward stays on the per-device sub-batch (activation mem 1/n_dev)."""

    def bshard(x: Array, ndim: int) -> Array:
        if mesh is None:
            return x
        spec = ["dp"] + [None] * (ndim - 1)
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    @jax.jit
    def step(state: Llama8BState, frozen: Target, resid: Float[Array, "b t d"], key: PRNGKeyArray):
        dm = {s: jnp.ones((1, 1, 1), DT) for s in SITES}  # weight-delta mask = 1
        nomask = {s: None for s in SITES}
        Wg, Wu, Wd = frozen.l18_Wg, frozen.l18_Wu, frozen.l18_Wd
        resid = bshard(resid, 3)

        def suffix(*a):
            return bshard(suffix_logits(*a), 3)

        ckpt_suffix = jax.checkpoint(suffix)  # recompute masked fwd in bwd (memory)

        clean = jax.lax.stop_gradient(suffix(frozen, state.vu, resid, nomask, dm))
        mlp_in = bshard(l18_resid_to_mlp_input(frozen, resid), 3)
        site_in = mlp_site_inputs(Wg, Wu, mlp_in)

        # n_warmup source-only ascents (params + ci detached) against the current params
        vu_det = jax.lax.stop_gradient(state.vu)
        ci_pre = jax.lax.stop_gradient(state.ci_fn(site_in))

        def adv_pre(src):
            masks = {s: ci_pre[s] * jnp.clip(src[s], 0.0, 1.0) for s in SITES}
            return _recon(suffix(frozen, vu_det, resid, masks, dm), clean)

        def warmup_body(src, _):
            g = jax.grad(adv_pre)(src)
            return jax.tree.map(lambda s, gg: s + pgd_lr * gg, src, g), None

        refined_src, _ = jax.lax.scan(warmup_body, state.source, None, length=n_warmup)
        refined_src = jax.lax.stop_gradient(refined_src)

        def loss_fn(trainable):
            vu, ci_fn = trainable
            ci = ci_fn(site_in)

            wd = weight_deltas(vu, Wg, Wu, Wd)
            l_faith = sum((d.astype(jnp.float32) ** 2).sum() for d in wd.values()) / sum(
                d.size for d in wd.values()
            )
            l_imp = jnp.mean(
                jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** coeffs.p_imp) for v in ci.values()])
            )

            l_stoch = jnp.array(0.0)
            for i, s in enumerate(SITES):
                u = random.uniform(random.fold_in(key, i), ci[s].shape, dtype=DT)
                m = ci[s] + (1 - ci[s]) * u
                masks = {**nomask, s: m}
                l_stoch = l_stoch + _recon(ckpt_suffix(frozen, vu, resid, masks, dm), clean)
            l_stoch = l_stoch / len(SITES)

            ppgd_masks = {s: ci[s] * refined_src[s] for s in SITES}
            l_ppgd = _recon(ckpt_suffix(frozen, vu, resid, ppgd_masks, dm), clean)

            tot = (
                coeffs.faith * l_faith
                + coeffs.imp * l_imp
                + coeffs.stoch * l_stoch
                + coeffs.ppgd * l_ppgd
            )
            return tot, (l_faith, l_imp, l_stoch, l_ppgd, ci)

        (tot, (l_faith, l_imp, l_stoch, l_ppgd, ci)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )((state.vu, state.ci_fn))

        upd_vu, os_vu = opt_vu.update(grads[0], state.opt_vu, state.vu)
        upd_ci, os_ci = opt_ci.update(grads[1], state.opt_ci, state.ci_fn)
        new_vu = eqx.apply_updates(state.vu, upd_vu)
        new_ci = eqx.apply_updates(state.ci_fn, upd_ci)

        # post-update ascent: warm-start the persisted source against the fresh params
        new_vu_det = jax.lax.stop_gradient(new_vu)
        ci_post = jax.lax.stop_gradient(ci)

        def adv_post(src):
            masks = {s: ci_post[s] * jnp.clip(src[s], 0.0, 1.0) for s in SITES}
            return _recon(suffix(frozen, new_vu_det, resid, masks, dm), clean)

        g = jax.grad(adv_post)(refined_src)
        new_src = jax.tree.map(lambda s, gg: s + pgd_lr * gg, refined_src, g)
        new_src = _clamp_source(jax.lax.stop_gradient(new_src))

        new_state = Llama8BState(
            vu=new_vu,
            ci_fn=new_ci,
            opt_vu=os_vu,
            opt_ci=os_ci,
            source=new_src,
            step=state.step + 1,
        )
        metrics = {
            "total": tot,
            "faith": l_faith,
            "imp": l_imp,
            "stoch": l_stoch,
            "ppgd": l_ppgd,
        }
        return new_state, metrics

    return step


def make_llama8b_step_shmap(
    coeffs: LossCoeffs,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    pgd_lr: float,
    n_warmup: int,
    mesh: Mesh,
):
    """`shard_map` data-parallel step — the guaranteed-no-gather variant.

    Each shard runs the full step on its `bl`-sized local sub-batch with params + PGD
    source REPLICATED. `with_sharding_constraint` is a hint XLA can ignore (it gathered
    `clean` to the global batch -> OOM at large gbatch); `shard_map` makes the batch
    parallelism explicit so no activation ever exceeds the per-device sub-batch.

    Cross-shard reductions are explicit: every per-shard MEAN loss is `pmean`'d over
    `dp` (so the reported loss and — critically — the grads of the replicated params are
    the GLOBAL means, identical on every shard => params stay replicated). The PGD
    source grad is likewise `pmean`'d (the torch `reduce_source_grads` analog). Faith /
    imp don't depend on the batch, so they're already shard-identical.
    """
    repl = P()
    bdp = P("dp")

    def _pmean(x):
        return jax.lax.pmean(x, axis_name="dp")

    def local_step(state: Llama8BState, frozen: Target, resid, key):
        dm = {s: jnp.ones((1, 1, 1), DT) for s in SITES}
        nomask = {s: None for s in SITES}
        Wg, Wu, Wd = frozen.l18_Wg, frozen.l18_Wu, frozen.l18_Wd
        ckpt_suffix = jax.checkpoint(suffix_logits)

        clean = jax.lax.stop_gradient(suffix_logits(frozen, state.vu, resid, nomask, dm))
        mlp_in = l18_resid_to_mlp_input(frozen, resid)
        site_in = mlp_site_inputs(Wg, Wu, mlp_in)

        vu_det = jax.lax.stop_gradient(state.vu)
        ci_pre = jax.lax.stop_gradient(state.ci_fn(site_in))

        def adv_pre(src):
            masks = {s: ci_pre[s] * jnp.clip(src[s], 0.0, 1.0) for s in SITES}
            return _pmean(_recon(suffix_logits(frozen, vu_det, resid, masks, dm), clean))

        def warmup_body(src, _):
            g = jax.grad(adv_pre)(src)
            return jax.tree.map(lambda s, gg: s + pgd_lr * gg, src, g), None

        refined_src, _ = jax.lax.scan(warmup_body, state.source, None, length=n_warmup)
        refined_src = jax.lax.stop_gradient(refined_src)

        def loss_fn(trainable):
            vu, ci_fn = trainable
            ci = ci_fn(site_in)
            wd = weight_deltas(vu, Wg, Wu, Wd)
            l_faith = sum((d.astype(jnp.float32) ** 2).sum() for d in wd.values()) / sum(
                d.size for d in wd.values()
            )
            l_imp = jnp.mean(
                jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** coeffs.p_imp) for v in ci.values()])
            )
            l_stoch = jnp.array(0.0)
            for i, s in enumerate(SITES):
                u = random.uniform(random.fold_in(key, i), ci[s].shape, dtype=DT)
                m = ci[s] + (1 - ci[s]) * u
                l_stoch = l_stoch + _pmean(
                    _recon(ckpt_suffix(frozen, vu, resid, {**nomask, s: m}, dm), clean)
                )
            l_stoch = l_stoch / len(SITES)
            ppgd_masks = {s: ci[s] * refined_src[s] for s in SITES}
            l_ppgd = _pmean(_recon(ckpt_suffix(frozen, vu, resid, ppgd_masks, dm), clean))
            tot = (
                coeffs.faith * l_faith
                + coeffs.imp * l_imp
                + coeffs.stoch * l_stoch
                + coeffs.ppgd * l_ppgd
            )
            return tot, (l_faith, l_imp, l_stoch, l_ppgd, ci)

        (tot, (l_faith, l_imp, l_stoch, l_ppgd, ci)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )((state.vu, state.ci_fn))

        upd_vu, os_vu = opt_vu.update(grads[0], state.opt_vu, state.vu)
        upd_ci, os_ci = opt_ci.update(grads[1], state.opt_ci, state.ci_fn)
        new_vu = eqx.apply_updates(state.vu, upd_vu)
        new_ci = eqx.apply_updates(state.ci_fn, upd_ci)

        new_vu_det = jax.lax.stop_gradient(new_vu)
        ci_post = jax.lax.stop_gradient(ci)

        def adv_post(src):
            masks = {s: ci_post[s] * jnp.clip(src[s], 0.0, 1.0) for s in SITES}
            return _pmean(_recon(suffix_logits(frozen, new_vu_det, resid, masks, dm), clean))

        g = jax.grad(adv_post)(refined_src)
        new_src = jax.tree.map(lambda s, gg: s + pgd_lr * gg, refined_src, g)
        new_src = _clamp_source(jax.lax.stop_gradient(new_src))

        new_state = Llama8BState(
            vu=new_vu, ci_fn=new_ci, opt_vu=os_vu, opt_ci=os_ci,
            source=new_src, step=state.step + 1,
        )  # fmt: skip
        return new_state, {
            "total": tot, "faith": l_faith, "imp": l_imp, "stoch": l_stoch, "ppgd": l_ppgd,
        }  # fmt: skip

    mapped = shard_map(
        local_step,
        mesh=mesh,
        in_specs=(
            repl,
            repl,
            bdp,
            repl,
        ),  # state replicated, target replicated, resid dp, key replicated
        out_specs=(repl, repl),
        check_vma=False,
    )
    return jax.jit(mapped)
