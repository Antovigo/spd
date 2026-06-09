"""The full Llama-8B single-pool VPD step: output-recon through the frozen suffix.

This is the full-LM variant the layerwise core (`step.py`) deferred. Unlike the
site-local core, recon here is MSE on the **final logits** of a masked suffix
re-forward — matching the torch `StochasticReconLayerwiseLoss` / `PersistentPGDReconLoss`
semantics (mask one/all sites, re-forward the whole suffix, compare to the clean
suffix output).

Generalized to N decomposed layers (3N sites). Masks per kind carry a leading layer
axis `L`: `masks[kind]` is `(b, t, L, C)` (or `None` for the clean forward); the
suffix indexes layer i with `masks[kind][:, :, i]`. PGD sources are likewise
`{kind: (1, T, L, C)}` (broadcast over batch, per-layer per-position).

Loss structure mirrors `llama8b_l18_b512_2pool_lr_mid.yaml` (extended to a layer range):
  faith  = mean weight-delta^2 over all 3N sites              coeff 1e5
  imp    = mean(clip(ci,0,1)^p), p annealed -> 0.4            coeff 5e-6
  stoch  = mean over kinds of MSE(masked-one-kind logits, clean)  coeff 0.5
  ppgd   = MSE(all-sites-masked-by-persistent-source logits, clean) coeff 0.5

Everything is one `jax.jit` over a pure `Llama8BState`. The frozen `Target` is a
runtime arg (replicated), not an HLO constant.
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
    KINDS,
    DecompVU,
    Target,
    all_site_inputs,
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
    source: dict[str, Float[Array, "1 t L C"]]  # broadcast persistent PGD source per kind
    step: Array


def _clamp_source(src: dict[str, Array]) -> dict[str, Array]:
    return {k: jnp.clip(src[k], 0.0, 1.0) for k in KINDS}


def _recon(a: Array, b: Array) -> Array:
    return jnp.mean((a.astype(jnp.float32) - b.astype(jnp.float32)) ** 2)


def _no_routes() -> dict:
    return {k: None for k in KINDS}


def _sample_uniform_k_subset_routes(
    key: Array, n_layers: int, lead: tuple[int, ...]
) -> dict[str, Array]:
    """torch `uniform_k_subset` routing over ALL 3*n_layers sites, per leading position.

    Each position draws k ~ U[1, n_sites], then a random k-subset of the sites routes to
    the decomposed module (True); the rest route to the clean target module (False).
    Returns {kind: (L, *lead, 1)} bool, ready to broadcast over the C / d axis."""
    n_sites = len(KINDS) * n_layers
    k_key, perm_key = random.split(key)
    k = random.randint(k_key, (*lead, 1), 1, n_sites + 1)  # (*lead, 1) threshold per position
    perms = random.uniform(perm_key, (n_sites, *lead, 1)).argsort(axis=0)  # random rank per site
    routed = perms < k  # (n_sites, *lead, 1) bool
    routed = routed.reshape(n_layers, len(KINDS), *lead, 1)
    return {kind: routed[:, j] for j, kind in enumerate(KINDS)}


def make_llama8b_step(
    coeffs: LossCoeffs,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    pgd_lr: float,
    n_warmup: int,
    n_layers: int,
    mesh: Mesh | None,
):
    """n_warmup pre-ascent iters refine the persistent source; one post-update ascent
    persists it warm-started against the fresh params (torch warmup + final = n_warmup+1).

    `mesh` (when given) pins every batch-leading activation to `P('dp', ...)` so XLA
    keeps the masked re-forwards on the per-device sub-batch (activation mem 1/n_dev)."""

    def bshard(x: Array, ndim: int) -> Array:
        if mesh is None:
            return x
        spec = ["dp"] + [None] * (ndim - 1)
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    dm = {k: jnp.ones((n_layers, 1, 1), DT) for k in KINDS}  # weight-delta mask = 1, per layer

    @jax.jit
    def step(state: Llama8BState, frozen: Target, resid: Float[Array, "b t d"], key: PRNGKeyArray):
        nomask = {k: None for k in KINDS}
        no_routes = _no_routes()
        resid = bshard(resid, 3)

        def suffix(*a):
            return bshard(suffix_logits(*a), 3)

        ckpt_suffix = jax.checkpoint(suffix)  # recompute masked fwd in bwd (memory)

        clean = jax.lax.stop_gradient(suffix(frozen, state.vu, resid, nomask, dm, no_routes))
        site_in = all_site_inputs(frozen, resid)
        b, t = resid.shape[0], resid.shape[1]

        vu_det = jax.lax.stop_gradient(state.vu)
        ci_pre = jax.lax.stop_gradient(state.ci_fn(site_in))

        def adv_pre(src):
            masks = {k: ci_pre[k] * jnp.clip(src[k], 0.0, 1.0) for k in KINDS}
            return _recon(suffix(frozen, vu_det, resid, _layerfirst(masks), dm, no_routes), clean)

        def warmup_body(src, _):
            g = jax.grad(adv_pre)(src)
            return jax.tree.map(lambda s, gg: s + pgd_lr * gg, src, g), None

        refined_src, _ = jax.lax.scan(warmup_body, state.source, None, length=n_warmup)
        refined_src = jax.lax.stop_gradient(refined_src)

        # one stochastic forward over ALL sites with uniform-k-subset routing (torch's
        # recon_plan: subset, n_samples=1): each (site, position) is routed clean-or-masked.
        stoch_routes = _sample_uniform_k_subset_routes(random.fold_in(key, 1), n_layers, (b, t))

        def loss_fn(trainable):
            vu, ci_fn = trainable
            ci = ci_fn(site_in)

            wd = weight_deltas(vu, frozen.decomp_layers)
            l_faith = sum((d.astype(jnp.float32) ** 2).sum() for d in wd.values()) / sum(
                d.size for d in wd.values()
            )
            l_imp = jnp.mean(
                jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** coeffs.p_imp) for v in ci.values()])
            )

            u = {k: random.uniform(random.fold_in(key, 10 + i), ci[k].shape, dtype=DT)
                 for i, k in enumerate(KINDS)}  # fmt: skip
            stoch_masks = _layerfirst({k: ci[k] + (1 - ci[k]) * u[k] for k in KINDS})
            l_stoch = _recon(ckpt_suffix(frozen, vu, resid, stoch_masks, dm, stoch_routes), clean)

            ppgd_masks = _layerfirst({k: ci[k] * refined_src[k] for k in KINDS})
            l_ppgd = _recon(ckpt_suffix(frozen, vu, resid, ppgd_masks, dm, no_routes), clean)

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
            masks = {k: ci_post[k] * jnp.clip(src[k], 0.0, 1.0) for k in KINDS}
            return _recon(
                suffix(frozen, new_vu_det, resid, _layerfirst(masks), dm, no_routes), clean
            )

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


def _layerfirst(masks: dict) -> dict:
    """Move the layer axis to the front so `suffix_logits` can index `masks[k][i]`.

    CI / source masks are `(b, t, L, C)` (or `None`); the suffix wants per-layer
    `(b, t, C)` via `masks[k][i]`. Transpose L to axis 0 -> `(L, b, t, C)`."""
    return {k: (None if masks[k] is None else jnp.moveaxis(masks[k], -2, 0)) for k in KINDS}


def make_llama8b_step_shmap(
    coeffs: LossCoeffs,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    pgd_lr: float,
    n_warmup: int,
    n_layers: int,
    mesh: Mesh,
):
    """`shard_map` data-parallel step — the guaranteed-no-gather variant.

    Each shard runs the full step on its `bl`-sized local sub-batch with params + PGD
    source REPLICATED. Cross-shard reductions are explicit: every per-shard MEAN loss is
    `pmean`'d over `dp` (so the reported loss and the grads of the replicated params are
    the GLOBAL means). The PGD source grad is likewise `pmean`'d (the torch
    `reduce_source_grads` analog)."""
    repl = P()
    bdp = P("dp")
    dm = {k: jnp.ones((n_layers, 1, 1), DT) for k in KINDS}

    def _pmean(x):
        return jax.lax.pmean(x, axis_name="dp")

    def local_step(state: Llama8BState, frozen: Target, resid, key):
        nomask = {k: None for k in KINDS}
        no_routes = _no_routes()
        ckpt_suffix = jax.checkpoint(suffix_logits)

        clean = jax.lax.stop_gradient(suffix_logits(frozen, state.vu, resid, nomask, dm, no_routes))
        site_in = all_site_inputs(frozen, resid)
        b, t = resid.shape[0], resid.shape[1]

        vu_det = jax.lax.stop_gradient(state.vu)
        ci_pre = jax.lax.stop_gradient(state.ci_fn(site_in))

        def adv_pre(src):
            masks = {k: ci_pre[k] * jnp.clip(src[k], 0.0, 1.0) for k in KINDS}
            return _pmean(
                _recon(
                    suffix_logits(frozen, vu_det, resid, _layerfirst(masks), dm, no_routes), clean
                )
            )

        def warmup_body(src, _):
            g = jax.grad(adv_pre)(src)
            return jax.tree.map(lambda s, gg: s + pgd_lr * gg, src, g), None

        refined_src, _ = jax.lax.scan(warmup_body, state.source, None, length=n_warmup)
        refined_src = jax.lax.stop_gradient(refined_src)

        stoch_routes = _sample_uniform_k_subset_routes(random.fold_in(key, 1), n_layers, (b, t))

        def loss_fn(trainable):
            vu, ci_fn = trainable
            ci = ci_fn(site_in)
            wd = weight_deltas(vu, frozen.decomp_layers)
            l_faith = sum((d.astype(jnp.float32) ** 2).sum() for d in wd.values()) / sum(
                d.size for d in wd.values()
            )
            l_imp = _pmean(
                jnp.mean(
                    jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** coeffs.p_imp) for v in ci.values()])
                )
            )
            u = {k: random.uniform(random.fold_in(key, 10 + i), ci[k].shape, dtype=DT)
                 for i, k in enumerate(KINDS)}  # fmt: skip
            stoch_masks = _layerfirst({k: ci[k] + (1 - ci[k]) * u[k] for k in KINDS})
            l_stoch = _pmean(
                _recon(ckpt_suffix(frozen, vu, resid, stoch_masks, dm, stoch_routes), clean)
            )
            ppgd_masks = _layerfirst({k: ci[k] * refined_src[k] for k in KINDS})
            l_ppgd = _pmean(
                _recon(ckpt_suffix(frozen, vu, resid, ppgd_masks, dm, no_routes), clean)
            )
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
            masks = {k: ci_post[k] * jnp.clip(src[k], 0.0, 1.0) for k in KINDS}
            return _pmean(
                _recon(
                    suffix_logits(frozen, new_vu_det, resid, _layerfirst(masks), dm, no_routes),
                    clean,
                )
            )

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
        in_specs=(repl, repl, bdp, repl),
        out_specs=(repl, repl),
        check_vma=False,
    )
    return jax.jit(mapped)
