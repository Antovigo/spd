"""The full Llama-8B single-pool VPD step: output-recon through the frozen suffix.

This is the full-LM variant the layerwise core (`step.py`) deferred. Recon here is on
the **final logits** of a masked suffix re-forward, matching the torch 2-pool reference
(`feature/fsdp-lm-trainer`, commit 484786f0) term-for-term:

  faith  = Σ‖W − V@U‖² / Σ numel  over all 3N sites            (FaithfulnessLoss)
  imp    = Σ_components [mean(ci_upper+eps)^p + beta·mean·log2(1+sum)]
           on the UPPER-leaky CI, p annealed                    (ImportanceMinimalityLoss)
  stoch  = mean over the 12 per-chunk forwards of KL/n_positions, each chunk = one
           decomposed layer's 3 sites under per-position uniform-k-subset routing
           (ChunkwiseSubsetReconLoss / SubsetReconPlan, sites_per_chunk=3, n_samples=1)
  ppgd   = KL/n_positions of the all-sites-masked persistent-source forward
           (PersistentPGDReconLoss; broadcast_across_batch source, weight-delta channel)

Recon is **KL**, not MSE (`recon_loss_kl`: P=softmax(clean), logQ=log_softmax(masked),
Σ P·(logP−logQ) / n_positions). The per-chunk stochastic recon REVERTS the earlier
"single fused forward" optimization — equivalence to the torch chunkwise pool beats the
~N× speed, which is the deliverable here.

Generalized to N decomposed layers (3N sites). CI / source masks per kind carry a
leading layer axis `L`: CI is `(b, t, L, C)`; PGD sources are `{kind: (1, T, L, C+1)}`
(broadcast over batch, per-layer per-position; the trailing channel is the weight-delta
source). Everything is one `jax.jit` over a pure `Llama8BState`.
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
    imp_beta: float
    imp_eps: float


class Llama8BState(NamedTuple):
    vu: DecompVU
    ci_fn: CIFn
    opt_vu: optax.OptState
    opt_ci: optax.OptState
    # broadcast persistent PGD source per kind, with a trailing weight-delta channel:
    # `(1, t, L, C+1)`. `[..., :-1]` are the component sources, `[..., -1]` the delta source.
    source: dict[str, Float[Array, "1 t L Cp1"]]
    step: Array


# ───────────────────────────── recon loss (KL) ─────────────────────────────


def _kl_per_position(pred: Array, clean: Array) -> Array:
    """`recon_loss_kl(pred, clean) / n_positions` in fp32.

    Torch: `F.kl_div(log_softmax(pred), softmax(clean), reduction="sum") / n_positions`
    = `Σ P·(logP − logQ) / n_positions`, with `P = softmax(clean)`, `Q = softmax(pred)`,
    `n_positions = b·t`. `clean` is the detached target; `pred` is the masked re-forward."""
    pred = pred.astype(jnp.float32)
    clean = clean.astype(jnp.float32)
    log_q = jax.nn.log_softmax(pred, axis=-1)
    log_p = jax.nn.log_softmax(clean, axis=-1)
    p = jnp.exp(log_p)
    n_positions = pred.shape[0] * pred.shape[1]
    return jnp.sum(p * (log_p - log_q)) / n_positions


# ─────────────────────────── source / mask helpers ───────────────────────────


def _clamp_source(src: dict[str, Array]) -> dict[str, Array]:
    return {k: jnp.clip(src[k], 0.0, 1.0) for k in KINDS}


def _no_routes() -> dict:
    return {k: None for k in KINDS}


def _layerfirst(masks: dict) -> dict:
    """Move the layer axis to the front so `suffix_logits` can index `masks[k][i]`.

    CI / source masks are `(b, t, L, C)` (or `None`); the suffix wants per-layer
    `(b, t, C)` via `masks[k][i]`. Transpose L to axis 0 -> `(L, b, t, C)`."""
    return {k: (None if masks[k] is None else jnp.moveaxis(masks[k], -2, 0)) for k in KINDS}


def _layerfirst_delta(delta: dict) -> dict:
    """Layer-axis move for the per-layer weight-delta mask `(b, t, L, 1)` -> `(L, b, t, 1)`.

    The trailing singleton broadcasts against `(b, t, d_out)` inside `_proj`."""
    return {k: jnp.moveaxis(delta[k], -2, 0) for k in KINDS}


def _ones_delta(n_layers: int) -> dict[str, Array]:
    """Per-layer weight-delta mask = 1 (broadcast), for the clean forward."""
    return {k: jnp.ones((n_layers, 1, 1), DT) for k in KINDS}


def _ppgd_masks_and_deltas(
    ci: dict[str, Array], source: dict[str, Array]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """torch `get_ppgd_mask_infos`: split the trailing delta channel off each source,
    interpolate components `mask = ci + (1 − ci)·source[..., :-1]`, use `source[..., -1]`
    directly as the per-position weight-delta mask. Sources are clamped to [0,1] (the
    `use_sigmoid_parameterization: false` path)."""
    comp_masks: dict[str, Array] = {}
    delta_masks: dict[str, Array] = {}
    for k in KINDS:
        s = jnp.clip(source[k], 0.0, 1.0)  # (1, t, L, C+1) broadcast over batch
        comp_src = s[..., :-1]
        comp_masks[k] = ci[k] + (1.0 - ci[k]) * comp_src
        delta_masks[k] = s[..., -1:]  # (1, t, L, 1) broadcast over batch (trailing 1 for _proj)
    return comp_masks, delta_masks


# ─────────────────────────── stochastic per-chunk recon ───────────────────────────


def _sample_chunk_routes(key: Array, lead: tuple[int, ...]) -> dict[str, Array]:
    """torch `uniform_k_subset` routing over ONE chunk's 3 sites (gate/up/down of a
    single decomposed layer), per leading position.

    Each position draws `k ~ U[1, 3]`, then a random k-subset of the 3 sites routes to
    the decomposed module (True); the rest route to the clean target module (False).
    Returns `{kind: (*lead, 1)}` bool, ready to broadcast over the C / d axis."""
    n_sites = len(KINDS)
    k_key, perm_key = random.split(key)
    k = random.randint(k_key, (*lead, 1), 1, n_sites + 1)  # (*lead, 1) threshold per position
    perms = random.uniform(perm_key, (n_sites, *lead, 1)).argsort(axis=0)  # random rank per site
    routed = perms < k  # (n_sites, *lead, 1) bool
    return {kind: routed[j] for j, kind in enumerate(KINDS)}


def _stoch_one_chunk(
    frozen: Target,
    vu: DecompVU,
    resid: Array,
    clean: Array,
    ci_lower: dict[str, Array],
    chunk_idx: int,
    n_layers: int,
    key: Array,
    suffix_fn,
) -> Array:
    """One per-chunk stochastic recon forward (KL / n_positions).

    The chunk is decomposed layer `chunk_idx`'s 3 sites. torch `recon_one_forward`:
    `u ~ U[0,1]` per site, `mask = ci + (1 − ci)·u`; a random per-position weight-delta
    mask `~ U[0,1]` per site; per-position uniform-k-subset routing over the chunk's 3
    sites; KL vs the clean suffix logits. Only layer `chunk_idx` is decomposed; every
    other decomposed layer runs its frozen target MLP (`decompose_layer`). `suffix_fn`
    lets the caller wrap `suffix_logits` (e.g. `jax.checkpoint`) without changing math."""
    b, t = resid.shape[0], resid.shape[1]
    u_key, dm_key, route_key = random.split(key, 3)
    chunk_routes = _sample_chunk_routes(route_key, (b, t))  # {kind: (b, t, 1)}

    masks: dict[str, Array] = {}
    delta_masks: dict[str, Array] = {}
    routes: dict[str, Array] = {}
    for j, k in enumerate(KINDS):
        ci_k = ci_lower[k]  # (b, t, L, C)
        u = random.uniform(random.fold_in(u_key, j), ci_k.shape, dtype=DT)
        masks[k] = ci_k + (1.0 - ci_k) * u
        delta_masks[k] = random.uniform(random.fold_in(dm_key, j), (b, t, n_layers, 1), dtype=DT)
        # route is (b, t, L, 1): the chunk's layer carries the sampled per-position route,
        # other layers are unused (decompose_layer=False there).
        routes[k] = jnp.zeros((b, t, n_layers, 1), bool).at[:, :, chunk_idx, :].set(chunk_routes[k])

    decompose = tuple(i == chunk_idx for i in range(n_layers))
    pred = suffix_fn(
        frozen, vu, resid,
        _layerfirst(masks), _layerfirst_delta(delta_masks), _layerfirst(routes),
        decompose,
    )  # fmt: skip
    return _kl_per_position(pred, clean)


def _stochastic_recon(
    frozen: Target,
    vu: DecompVU,
    resid: Array,
    clean: Array,
    ci_lower: dict[str, Array],
    n_layers: int,
    key: Array,
    suffix_fn,
) -> Array:
    """Mean over the `n_layers` per-chunk forwards of KL/n_positions.

    Matches `chunkwise_subset_recon`: `Σ_chunks (KL_c / n_positions) / n_forwards` with
    `n_forwards = n_layers` (12 for layers 20..31, sites_per_chunk=3, n_samples=1). Each
    chunk is one decomposed layer. `suffix_fn` lets the caller wrap `suffix_logits` (e.g.
    `jax.checkpoint`) without changing the math."""
    total = jnp.zeros((), jnp.float32)
    for chunk_idx in range(n_layers):
        total = total + _stoch_one_chunk(
            frozen, vu, resid, clean, ci_lower, chunk_idx, n_layers,
            random.fold_in(key, chunk_idx), suffix_fn,
        )  # fmt: skip
    return total / n_layers


# ─────────────────────────── importance-minimality ───────────────────────────


def _imp_min(ci_upper: dict[str, Array], p: float, beta: float, eps: float) -> Array:
    """torch `finalize_imp_min` on the UPPER-leaky CI (single-process; no DP reduce).

    The torch CI fn keys `upper_leaky` PER SITE — one `(b, t, C)` tensor per (layer, kind)
    — and `per_component_lp_sums` sums each over its position dims (b·t) only, NOT across
    sites. So here each (kind, layer) is its OWN site: `sum = Σ_{b,t} (ci+eps)^p`,
    `mean = sum / (b·t)`, term `mean + beta·mean·log2(1+sum)`, summed over that site's C
    components; then summed over every (kind, layer). Treating all L layers of a kind as
    one group (n=b·t·L, one log2) would NOT match torch's per-site grouping (the convex
    log2 needs the per-site sum). No clip — `upper_leaky` may exceed 1, as in torch."""
    total = jnp.zeros((), jnp.float32)
    for k in KINDS:
        ci = ci_upper[k].astype(jnp.float32)  # (b, t, L, C)
        n_positions = ci.shape[0] * ci.shape[1]  # b·t per site (NOT including L)
        powed = (ci + eps) ** p  # (b, t, L, C)
        site_sums = jnp.sum(powed, axis=(0, 1))  # (L, C) — Σ over b,t per (layer, component)
        per_component_mean = site_sums / n_positions  # (L, C)
        total = total + jnp.sum(
            per_component_mean + beta * per_component_mean * jnp.log2(1.0 + site_sums)
        )
    return total


# ─────────────────────────── PPGD adversary objective ───────────────────────────


def _ppgd_recon(
    frozen: Target,
    vu: DecompVU,
    resid: Array,
    clean: Array,
    ci_lower: dict[str, Array],
    source: dict[str, Array],
    suffix_fn,
) -> Array:
    """KL/n_positions of the all-sites-masked persistent-source forward.

    torch PPGD masks EVERY decomposed site at once (router = AllLayersRouter → route
    everywhere), `mask = ci + (1−ci)·source[:,:-1]`, delta mask = `source[:,-1]`. KL vs
    the clean suffix logits. Every decomposed layer participates (no `decompose_layer`)."""
    comp_masks, delta_masks = _ppgd_masks_and_deltas(ci_lower, source)
    pred = suffix_fn(
        frozen, vu, resid,
        _layerfirst(comp_masks), _layerfirst_delta(delta_masks), _no_routes(), None,
    )  # fmt: skip
    return _kl_per_position(pred, clean)


# ───────────────────────────────── the step ─────────────────────────────────


def _faith_loss(vu: DecompVU, decomp_layers) -> Array:
    wd = weight_deltas(vu, decomp_layers)
    num = sum((d.astype(jnp.float32) ** 2).sum() for d in wd.values())
    den = sum(d.size for d in wd.values())
    return num / den


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

    dm_ones = _ones_delta(n_layers)
    nomask = {k: None for k in KINDS}
    no_routes = _no_routes()

    def suffix(frozen, vu, resid, masks, delta_masks, routes, decompose=None):
        return bshard(suffix_logits(frozen, vu, resid, masks, delta_masks, routes, decompose), 3)

    ckpt_suffix = jax.checkpoint(suffix, static_argnums=(6,))  # recompute masked fwd in bwd

    @jax.jit
    def step(state: Llama8BState, frozen: Target, resid: Float[Array, "b t d"], key: PRNGKeyArray):
        resid = bshard(resid, 3)
        clean = jax.lax.stop_gradient(suffix(frozen, state.vu, resid, nomask, dm_ones, no_routes))
        site_in = all_site_inputs(frozen, resid)

        vu_det = jax.lax.stop_gradient(state.vu)
        ci_pre = jax.lax.stop_gradient(state.ci_fn(site_in).lower)

        def adv_pre(src):
            return _ppgd_recon(frozen, vu_det, resid, clean, ci_pre, src, suffix)

        def warmup_body(src, _):
            g = jax.grad(adv_pre)(src)
            return jax.tree.map(lambda s, gg: s + pgd_lr * gg, src, g), None

        refined_src, _ = jax.lax.scan(warmup_body, state.source, None, length=n_warmup)
        refined_src = jax.lax.stop_gradient(_clamp_source(refined_src))

        def loss_fn(trainable):
            vu, ci_fn = trainable
            ci = ci_fn(site_in)
            l_faith = _faith_loss(vu, frozen.decomp_layers)
            l_imp = _imp_min(ci.upper, coeffs.p_imp, coeffs.imp_beta, coeffs.imp_eps)
            l_stoch = _stochastic_recon(
                frozen, vu, resid, clean, ci.lower, n_layers, random.fold_in(key, 1), ckpt_suffix
            )
            l_ppgd = _ppgd_recon(frozen, vu, resid, clean, ci.lower, refined_src, ckpt_suffix)
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
        ci_post = jax.lax.stop_gradient(ci.lower)

        def adv_post(src):
            return _ppgd_recon(frozen, new_vu_det, resid, clean, ci_post, src, suffix)

        g = jax.grad(adv_post)(refined_src)
        new_src = jax.tree.map(lambda s, gg: s + pgd_lr * gg, refined_src, g)
        new_src = _clamp_source(jax.lax.stop_gradient(new_src))

        new_state = Llama8BState(
            vu=new_vu, ci_fn=new_ci, opt_vu=os_vu, opt_ci=os_ci,
            source=new_src, step=state.step + 1,
        )  # fmt: skip
        metrics = {
            "total": tot, "faith": l_faith, "imp": l_imp, "stoch": l_stoch, "ppgd": l_ppgd,
        }  # fmt: skip
        return new_state, metrics

    return step


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
    source REPLICATED. Every per-shard MEAN loss is `pmean`'d over `dp` so the reported
    loss and the replicated params' grads are the GLOBAL means; the PGD source grad is
    likewise `pmean`'d (the torch `reduce_source_grads` AVG analog for a shared source)."""
    repl = P()
    bdp = P("dp")
    dm_ones = _ones_delta(n_layers)
    nomask = {k: None for k in KINDS}
    no_routes = _no_routes()

    def _pmean(x):
        return jax.lax.pmean(x, axis_name="dp")

    ckpt_suffix = jax.checkpoint(suffix_logits, static_argnums=(6,))

    def local_step(state: Llama8BState, frozen: Target, resid, key):
        clean = jax.lax.stop_gradient(
            suffix_logits(frozen, state.vu, resid, nomask, dm_ones, no_routes)
        )
        site_in = all_site_inputs(frozen, resid)

        vu_det = jax.lax.stop_gradient(state.vu)
        ci_pre = jax.lax.stop_gradient(state.ci_fn(site_in).lower)

        def adv_pre(src):
            return _pmean(_ppgd_recon(frozen, vu_det, resid, clean, ci_pre, src, suffix_logits))

        def warmup_body(src, _):
            g = jax.grad(adv_pre)(src)
            return jax.tree.map(lambda s, gg: s + pgd_lr * gg, src, g), None

        refined_src, _ = jax.lax.scan(warmup_body, state.source, None, length=n_warmup)
        refined_src = jax.lax.stop_gradient(_clamp_source(refined_src))

        def loss_fn(trainable):
            vu, ci_fn = trainable
            ci = ci_fn(site_in)
            l_faith = _faith_loss(vu, frozen.decomp_layers)
            l_imp = _pmean(_imp_min(ci.upper, coeffs.p_imp, coeffs.imp_beta, coeffs.imp_eps))
            l_stoch = _pmean(
                _stochastic_recon(
                    frozen,
                    vu,
                    resid,
                    clean,
                    ci.lower,
                    n_layers,
                    random.fold_in(key, 1),
                    ckpt_suffix,
                )  # fmt: skip
            )
            l_ppgd = _pmean(
                _ppgd_recon(frozen, vu, resid, clean, ci.lower, refined_src, ckpt_suffix)
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
        ci_post = jax.lax.stop_gradient(ci.lower)

        def adv_post(src):
            return _pmean(
                _ppgd_recon(frozen, new_vu_det, resid, clean, ci_post, src, suffix_logits)
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
        local_step, mesh=mesh,
        in_specs=(repl, repl, bdp, repl), out_specs=(repl, repl), check_vma=False,
    )  # fmt: skip
    return jax.jit(mapped)
