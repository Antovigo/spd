"""The generic single-pool VPD training step over a `DecomposedLM` (SPEC §4).

One `jax.jit` step: clean target → CI envelope → n_warmup adversary ascents → the four
losses → one fused backward over (vu, ci_fn, src) → optimizer updates → the final
(n_warmup+1)-th source ascent from the same graph (SPEC S13/S14). All trainable state
is fp32 masters (SPEC N1); forwards run in bf16 via explicit casts. The persistent
adversary (sources + its Adam moments) lives in `TrainState` and is projected to [0,1]
after every update (SPEC S15).

Schedules (imp-min p anneal, source-LR warmup) are computed inside the step from
`state.step`, so the jit signature is stable across the whole run (SPEC S9, S13).
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from jax_single_pool.ci_fn import CIFn
from jax_single_pool.lm import DecomposedLM, chunk_sites

COMPUTE_DT = jnp.bfloat16


def _cast_floating(tree: Any, dtype: Any) -> Any:
    return jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


# ───────────────────────────── configs & state ─────────────────────────────


class LossCoeffs(NamedTuple):
    faith: float
    imp: float
    stoch: float
    ppgd: float


class ImpMinConfig(NamedTuple):
    """`p` anneals linearly `p_start → p_final` over `[anneal_start_frac, anneal_end_frac]`
    of training (SPEC S9)."""

    beta: float
    eps: float
    p_start: float
    p_final: float
    anneal_start_frac: float
    anneal_end_frac: float


class SourceAdamConfig(NamedTuple):
    """The persistent adversary's Adam (SPEC §3): `lr` with linear warmup over
    `lr_warmup_frac` of training then constant; `n_warmup` supplemental ascents/step."""

    lr: float
    lr_warmup_frac: float
    beta1: float
    beta2: float
    eps: float
    n_warmup: int


class SrcAdamState(NamedTuple):
    m: dict[str, Array]
    v: dict[str, Array]
    step_count: Float[Array, ""]


class TrainState(NamedTuple):
    vu: Any  # LM-specific trainable pytree, fp32 masters
    ci_fn: CIFn  # fp32 masters
    opt_vu: optax.OptState
    opt_ci: optax.OptState
    src: dict[str, Float[Array, "1 T Cp1"]]  # per-site raw sources, always in [0,1]
    src_adam: SrcAdamState
    step: Array


def init_sources(
    site_names: tuple[str, ...], site_cs: tuple[int, ...], seq_len: int, key: PRNGKeyArray
) -> dict[str, Array]:
    """broadcast_across_batch scope (SPEC §1.6): `(1, T, C+1)` per site, init U[0,1]
    (SPEC S15; clamp parameterization). Trailing channel = the weight-delta source."""
    keys = random.split(key, len(site_names))
    return {
        name: random.uniform(k, (1, seq_len, c + 1), jnp.float32)
        for name, c, k in zip(site_names, site_cs, keys, strict=True)
    }


def init_src_adam(src: dict[str, Array]) -> SrcAdamState:
    zeros = {s: jnp.zeros_like(v) for s, v in src.items()}
    return SrcAdamState(
        m=zeros, v={s: jnp.zeros_like(v) for s, v in src.items()}, step_count=jnp.zeros(())
    )


def src_adam_ascend_project(
    src: dict[str, Array],
    grad: dict[str, Array],
    st: SrcAdamState,
    lr: Array,
    cfg: SourceAdamConfig,
) -> tuple[dict[str, Array], SrcAdamState]:
    """One Adam ASCENT on the sources, then project to [0,1] (SPEC S13/S15).

    The variation point `SRC_STEP` (SPEC §6): a `sign` variant would replace the Adam
    update with `lr * sign(g)` (stateless) — same projection contract."""
    count = st.step_count + 1.0
    m = {s: cfg.beta1 * st.m[s] + (1 - cfg.beta1) * grad[s] for s in src}
    v = {s: cfg.beta2 * st.v[s] + (1 - cfg.beta2) * grad[s] * grad[s] for s in src}
    bc1 = 1 - cfg.beta1**count
    bc2 = 1 - cfg.beta2**count
    new = {
        s: jnp.clip(src[s] + lr * (m[s] / bc1) / (jnp.sqrt(v[s] / bc2) + cfg.eps), 0.0, 1.0)
        for s in src
    }
    return new, SrcAdamState(m=m, v=v, step_count=count)


# ───────────────────────────── schedules ─────────────────────────────


def annealed_p(t: Array, total_steps: int, cfg: ImpMinConfig) -> Array:
    span = max(cfg.anneal_end_frac - cfg.anneal_start_frac, 1e-9)
    progress = jnp.clip((t / total_steps - cfg.anneal_start_frac) / span, 0.0, 1.0)
    return jnp.asarray(cfg.p_start + (cfg.p_final - cfg.p_start) * progress)


def warmup_then_constant_lr(t: Array, total_steps: int, lr: float, warmup_frac: float) -> Array:
    warmup_steps = jnp.maximum(jnp.floor(total_steps * warmup_frac), 1.0)
    return jnp.where(t < warmup_steps, lr * t / warmup_steps, lr)


# ───────────────────────────── losses (SPEC §2) ─────────────────────────────


def kl_per_position(pred: Array, clean: Array) -> Array:
    """`Σ_{b,t} KL(softmax(clean) ‖ softmax(pred)) / (B·T)` in fp32 (SPEC §2.3, N3)."""
    pred = pred.astype(jnp.float32)
    clean = clean.astype(jnp.float32)
    log_q = jax.nn.log_softmax(pred, axis=-1)
    log_p = jax.nn.log_softmax(clean, axis=-1)
    p = jnp.exp(log_p)
    n_positions = pred.shape[0] * pred.shape[1]
    return jnp.sum(p * (log_p - log_q)) / n_positions


def faithfulness_loss(weight_deltas: dict[str, Array]) -> Array:
    """`Σ_s ‖Δ_s‖² / Σ_s numel` over fp32 deltas (SPEC S17)."""
    num = sum(
        ((d.astype(jnp.float32) ** 2).sum() for d in weight_deltas.values()),
        start=jnp.zeros((), jnp.float32),
    )
    den = sum(d.size for d in weight_deltas.values())
    return num / den


def importance_minimality_loss(
    ci_upper: dict[str, Array], p: Array, beta: float, eps: float
) -> Array:
    """Per-site grouping with the global-batch sum inside the log2 (SPEC S7/S8).

    Under GSPMD the `(b, t)` axes are the global batch, so `jnp.sum` IS the exact
    global per-component sum — XLA reduces across shards inside the graph."""
    total = jnp.zeros((), jnp.float32)
    for ci in ci_upper.values():
        ci = ci.astype(jnp.float32)  # (B, T, C)
        n_positions = ci.shape[0] * ci.shape[1]
        site_sums = jnp.sum((ci + eps) ** p, axis=(0, 1))  # (C,)
        mean = site_sums / n_positions
        total = total + jnp.sum(mean + beta * mean * jnp.log2(1.0 + site_sums))
    return total


def uniform_k_subset_routes(
    key: PRNGKeyArray, chunk: tuple[str, ...], lead: tuple[int, ...]
) -> dict[str, Array]:
    """Per position: `k ~ U{1..|chunk|}`, then a uniform k-subset of the chunk routes
    True (SPEC S11). Distributionally identical to torch's double-argsort ranks."""
    n = len(chunk)
    k_key, perm_key = random.split(key)
    k = random.randint(k_key, lead, 1, n + 1)
    perms = random.uniform(perm_key, (n, *lead)).argsort(axis=0)
    routed = perms < k
    return {name: routed[j] for j, name in enumerate(chunk)}


def make_ppgd_masks(
    ci_lower: dict[str, Array], src: dict[str, Array], sites: tuple[str, ...]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """`mask = ci + (1−ci)·src[:, :C]`; delta mask = raw trailing channel (SPEC S1).
    Sources broadcast over the batch dim (broadcast_across_batch scope). The fp32
    source state is cast to the CI dtype here (torch-under-autocast behavior); the
    source gradient flows back through the cast."""
    masks = {}
    delta_masks = {}
    for s in sites:
        src_c = src[s].astype(ci_lower[s].dtype)
        masks[s] = ci_lower[s] + (1.0 - ci_lower[s]) * src_c[..., :-1]
        delta_masks[s] = src_c[..., -1]
    return masks, delta_masks


# ───────────────────────────── the step factory ─────────────────────────────


def make_train_step(
    lm: DecomposedLM,
    coeffs: LossCoeffs,
    imp_cfg: ImpMinConfig,
    src_cfg: SourceAdamConfig,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    total_steps: int,
    sites_per_chunk: int,
    n_samples: int,
    mesh: Mesh | None,
):
    """Build the jit'd `step(state, frozen, resid, key) -> (state, metrics)`.

    `mesh` (when given) pins every batch-leading activation to `P('dp', ...)` so the
    masked re-forwards stay on per-device sub-batches (activation memory 1/n_dev)."""
    site_names = lm.site_names
    chunks = chunk_sites(site_names, sites_per_chunk)

    def bshard(x: Array) -> Array:
        if mesh is None:
            return x
        spec = ["dp"] + [None] * (x.ndim - 1)
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    def masked(
        frozen: Any,
        vu_c: Any,
        resid: Array,
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
    ) -> Array:
        return bshard(lm.masked_logits(frozen, vu_c, resid, masks, delta_masks, routes, live))

    # Recompute each masked forward in backward — bounds activation memory to one
    # forward at a time (the torch 2-pool streaming profile).
    ckpt_masked = jax.checkpoint(masked, static_argnums=(6,))

    def ppgd_recon(
        frozen: Any,
        vu_c: Any,
        ci_lower: dict[str, Array],
        src: dict[str, Array],
        resid: Array,
        cln: Array,
        masked_fn: Any,
    ) -> Array:
        masks, delta_masks = make_ppgd_masks(ci_lower, src, site_names)
        pred = masked_fn(frozen, vu_c, resid, masks, delta_masks, None, site_names)
        return kl_per_position(pred, cln)

    def stoch_recon(
        frozen: Any,
        vu_c: Any,
        ci_lower: dict[str, Array],
        resid: Array,
        cln: Array,
        key: PRNGKeyArray,
    ) -> Array:
        b, t = resid.shape[0], resid.shape[1]
        total = jnp.zeros((), jnp.float32)
        forward_idx = 0
        for chunk in chunks:
            for _ in range(n_samples):
                fkey = random.fold_in(key, forward_idx)
                forward_idx += 1
                u_key, dm_key, route_key = random.split(fkey, 3)
                masks = {}
                delta_masks = {}
                for j, s in enumerate(chunk):
                    ci_s = ci_lower[s]
                    u = random.uniform(random.fold_in(u_key, j), ci_s.shape, COMPUTE_DT)
                    masks[s] = ci_s + (1.0 - ci_s) * u
                    delta_masks[s] = random.uniform(random.fold_in(dm_key, j), (b, t), COMPUTE_DT)
                routes = uniform_k_subset_routes(route_key, chunk, (b, t))
                pred = ckpt_masked(frozen, vu_c, resid, masks, delta_masks, routes, chunk)
                total = total + kl_per_position(pred, cln)
        return total / (len(chunks) * n_samples)

    @jax.jit
    def step(state: TrainState, frozen: Any, resid: Float[Array, "b t d"], key: PRNGKeyArray):
        t = state.step.astype(jnp.float32)
        p_t = annealed_p(t, total_steps, imp_cfg)
        src_lr = warmup_then_constant_lr(t, total_steps, src_cfg.lr, src_cfg.lr_warmup_frac)

        resid = bshard(resid)
        cln = jax.lax.stop_gradient(bshard(lm.clean_logits(frozen, resid)))
        site_in = lm.site_inputs(frozen, resid)

        # ── supplemental adversary ascents: params + CI detached (SPEC §4.5) ──
        vu_det = jax.lax.stop_gradient(_cast_floating(state.vu, COMPUTE_DT))
        ci_det = jax.lax.stop_gradient(_cast_floating(state.ci_fn, COMPUTE_DT))
        lo_det = ci_det(site_in).lower

        def adv_loss(src: dict[str, Array]) -> Array:
            return ppgd_recon(frozen, vu_det, lo_det, src, resid, cln, masked)

        def warmup_body(
            carry: tuple[dict[str, Array], SrcAdamState], _: None
        ) -> tuple[tuple[dict[str, Array], SrcAdamState], None]:
            src, adam = carry
            g = jax.grad(adv_loss)(src)
            src, adam = src_adam_ascend_project(src, g, adam, src_lr, src_cfg)
            return (src, adam), None

        (refined_src, src_adam), _ = jax.lax.scan(
            warmup_body, (state.src, state.src_adam), None, length=src_cfg.n_warmup
        )
        refined_src = jax.lax.stop_gradient(refined_src)

        # ── main losses: live vu/ci; ppgd's source participates in the graph so its
        # gradient comes from the SAME backward (SPEC S14); it is NOT detached here,
        # but vu/ci grads through it are what torch gets too (src is a leaf). ──
        def loss_fn(diff: tuple[Any, CIFn, dict[str, Array]]):
            vu, ci_fn, src = diff
            vu_c = _cast_floating(vu, COMPUTE_DT)
            ci_c = _cast_floating(ci_fn, COMPUTE_DT)
            ci = ci_c(site_in)
            l_faith = faithfulness_loss(lm.weight_deltas(frozen, vu))
            l_imp = importance_minimality_loss(ci.upper, p_t, imp_cfg.beta, imp_cfg.eps)
            l_stoch = stoch_recon(frozen, vu_c, ci.lower, resid, cln, random.fold_in(key, 1))
            l_ppgd = ppgd_recon(frozen, vu_c, ci.lower, src, resid, cln, ckpt_masked)
            tot = (
                coeffs.faith * l_faith
                + coeffs.imp * l_imp
                + coeffs.stoch * l_stoch
                + coeffs.ppgd * l_ppgd
            )
            return tot, (l_faith, l_imp, l_stoch, l_ppgd)

        (tot, (l_faith, l_imp, l_stoch, l_ppgd)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )((state.vu, state.ci_fn, refined_src))
        g_vu, g_ci, g_src_scaled = grads
        # The backward saw coeff·L_ppgd; the adversary ascends on L_ppgd itself.
        g_src = {s: g / coeffs.ppgd for s, g in g_src_scaled.items()}

        # ── the (n_warmup+1)-th source ascent, from the fused graph (SPEC S13/S14) ──
        new_src, src_adam = src_adam_ascend_project(refined_src, g_src, src_adam, src_lr, src_cfg)

        upd_vu, os_vu = opt_vu.update(g_vu, state.opt_vu, eqx.filter(state.vu, eqx.is_array))
        upd_ci, os_ci = opt_ci.update(g_ci, state.opt_ci, eqx.filter(state.ci_fn, eqx.is_array))
        new_vu = eqx.apply_updates(state.vu, upd_vu)
        new_ci = eqx.apply_updates(state.ci_fn, upd_ci)

        new_state = TrainState(
            vu=new_vu, ci_fn=new_ci, opt_vu=os_vu, opt_ci=os_ci,
            src=new_src, src_adam=src_adam, step=state.step + 1,
        )  # fmt: skip
        metrics = {
            "total": tot, "faith": l_faith, "imp": l_imp, "stoch": l_stoch, "ppgd": l_ppgd,
            "p_imp": p_t, "src_lr": src_lr,
        }  # fmt: skip
        return new_state, metrics

    return step


# ───────────────────────────── faithfulness warmup (SPEC S21) ─────────────────────────────


def make_faith_warmup_step(
    lm: DecomposedLM, opt: optax.GradientTransformation
) -> Callable[[Any, optax.OptState, Any], tuple[Any, optax.OptState, Array]]:
    @jax.jit
    def warmup_step(
        vu: Any, opt_state: optax.OptState, frozen: Any
    ) -> tuple[Any, optax.OptState, Array]:
        def loss_fn(vu_: Any) -> Array:
            return faithfulness_loss(lm.weight_deltas(frozen, vu_))

        loss, g = eqx.filter_value_and_grad(loss_fn)(vu)
        upd, opt_state = opt.update(g, opt_state, eqx.filter(vu, eqx.is_array))
        return eqx.apply_updates(vu, upd), opt_state, loss

    return warmup_step
