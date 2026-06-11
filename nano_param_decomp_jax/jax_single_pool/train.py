"""The generic single-pool VPD training step over a `DecomposedLM` (SPEC §4).

One `jax.jit` step: clean target → CI envelope → n_warmup adversary ascents → the four
losses → one fused backward over (components, ci_fn, sources) → optimizer updates →
the final (n_warmup+1)-th source ascent from the same graph (SPEC S13/S14). All
trainable state is fp32 masters (SPEC N1); forwards run in bf16 via explicit casts.
The persistent adversary (sources + its Adam moments) lives in `TrainState` and is
projected to [0,1] after every update (SPEC S15).

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

from jax_single_pool.ci_fn import CIFn, CIValues
from jax_single_pool.lm import DecomposedLM, chunk_sites

COMPUTE_DT = jnp.bfloat16


def cast_floating(tree: Any, dtype: Any) -> Any:
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


class FreshPGDConfig(NamedTuple):
    """Fresh per-batch sign-PGD adversary (torch `PGDReconLoss` as a TRAINING loss):
    sources are re-initialized every step, ascended `n_steps` times by
    `step_size * sign(grad)` with clamp to [0,1], and carry NO state across steps —
    `TrainState.sources` stays empty for this variant. `scope` picks the source shape:
    `unique_per_datapoint` -> `(B, T, C+1)` per site; `shared_across_batch` ->
    `(1, 1, C+1)` (the eval probe's shape)."""

    init: str
    step_size: float
    n_steps: int
    scope: str


AdversaryConfig = SourceAdamConfig | FreshPGDConfig


class SourcesAdamState(NamedTuple):
    m: dict[str, Array]
    v: dict[str, Array]
    step_count: Float[Array, ""]


class TrainState(NamedTuple):
    components: Any  # LM-specific trainable pytree (V/U), fp32 masters
    ci_fn: CIFn  # fp32 masters
    components_opt_state: optax.OptState
    ci_fn_opt_state: optax.OptState
    sources: dict[str, Float[Array, "1 T Cp1"]]  # per-site raw sources, always in [0,1]
    sources_adam_state: SourcesAdamState
    step: Array


def init_sources(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    seq_len: int,
    key: PRNGKeyArray,
) -> dict[str, Array]:
    """broadcast_across_batch scope (SPEC §1.6): `(1, T, C+1)` per site, init U[0,1]
    (SPEC S15; clamp parameterization). Trailing channel = the weight-delta source."""
    keys = random.split(key, len(site_names))
    return {
        name: random.uniform(k, (1, seq_len, c + 1), jnp.float32)
        for name, c, k in zip(site_names, site_component_counts, keys, strict=True)
    }


def init_fresh_pgd_sources(
    sites: tuple[Any, ...],
    cfg: FreshPGDConfig,
    batch: int,
    seq: int,
    key: PRNGKeyArray,
) -> dict[str, Array]:
    """Per-site fresh adversarial sources (torch `_init_adv_sources`): trailing channel
    is the weight-delta source; shape per `cfg.scope`; values per `cfg.init`."""
    match cfg.scope:
        case "unique_per_datapoint":
            leading = (batch, seq)
        case "shared_across_batch":
            leading = (1, 1)
        case _:
            raise AssertionError(f"unsupported fresh-PGD scope {cfg.scope!r}")
    keys = random.split(key, len(sites))
    sources = {}
    for site, site_key in zip(sites, keys, strict=True):
        shape = (*leading, site.C + 1)
        match cfg.init:
            case "random":
                sources[site.name] = random.uniform(site_key, shape, jnp.float32)
            case "ones":
                sources[site.name] = jnp.ones(shape, jnp.float32)
            case "zeroes":
                sources[site.name] = jnp.zeros(shape, jnp.float32)
            case _:
                raise AssertionError(f"unsupported fresh-PGD init {cfg.init!r}")
    return sources


def init_sources_adam_state(sources: dict[str, Array]) -> SourcesAdamState:
    return SourcesAdamState(
        m={site: jnp.zeros_like(v) for site, v in sources.items()},
        v={site: jnp.zeros_like(v) for site, v in sources.items()},
        step_count=jnp.zeros(()),
    )


def sources_adam_ascend_project(
    sources: dict[str, Array],
    sources_grad: dict[str, Array],
    adam_state: SourcesAdamState,
    lr: Array,
    cfg: SourceAdamConfig,
) -> tuple[dict[str, Array], SourcesAdamState]:
    """One Adam ASCENT on the sources, then project to [0,1] (SPEC S13/S15).

    The variation point `SRC_STEP` (SPEC §6): a `sign` variant would replace the Adam
    update with `lr * sign(grad)` (stateless) — same projection contract."""
    step_count = adam_state.step_count + 1.0
    m = {s: cfg.beta1 * adam_state.m[s] + (1 - cfg.beta1) * sources_grad[s] for s in sources}
    v = {
        s: cfg.beta2 * adam_state.v[s] + (1 - cfg.beta2) * sources_grad[s] * sources_grad[s]
        for s in sources
    }
    bias_correction1 = 1 - cfg.beta1**step_count
    bias_correction2 = 1 - cfg.beta2**step_count
    new_sources = {
        s: jnp.clip(
            sources[s]
            + lr * (m[s] / bias_correction1) / (jnp.sqrt(v[s] / bias_correction2) + cfg.eps),
            0.0,
            1.0,
        )
        for s in sources
    }
    return new_sources, SourcesAdamState(m=m, v=v, step_count=step_count)


# ───────────────────────────── schedules ─────────────────────────────


def annealed_pnorm(step_f32: Array, total_steps: int, cfg: ImpMinConfig) -> Array:
    span = max(cfg.anneal_end_frac - cfg.anneal_start_frac, 1e-9)
    progress = jnp.clip((step_f32 / total_steps - cfg.anneal_start_frac) / span, 0.0, 1.0)
    return jnp.asarray(cfg.p_start + (cfg.p_final - cfg.p_start) * progress)


def warmup_then_constant_lr(
    step_f32: Array, total_steps: int, lr: float, warmup_frac: float
) -> Array:
    warmup_steps = jnp.maximum(jnp.floor(total_steps * warmup_frac), 1.0)
    return jnp.where(step_f32 < warmup_steps, lr * step_f32 / warmup_steps, lr)


# ───────────────────────────── losses (SPEC §2) ─────────────────────────────


def kl_per_position(masked_logits: Array, clean_logits: Array) -> Array:
    """`Σ_{b,t} KL(softmax(clean) ‖ softmax(masked)) / (B·T)` in fp32 (SPEC §2.3, N3)."""
    masked_logits = masked_logits.astype(jnp.float32)
    clean_logits = clean_logits.astype(jnp.float32)
    log_q = jax.nn.log_softmax(masked_logits, axis=-1)
    log_p = jax.nn.log_softmax(clean_logits, axis=-1)
    p = jnp.exp(log_p)
    n_positions = masked_logits.shape[0] * masked_logits.shape[1]
    return jnp.sum(p * (log_p - log_q)) / n_positions


def faithfulness_loss(weight_deltas: dict[str, Array]) -> Array:
    """`Σ_s ‖Δ_s‖² / Σ_s numel` over fp32 deltas (SPEC S17)."""
    numerator = sum(
        ((delta.astype(jnp.float32) ** 2).sum() for delta in weight_deltas.values()),
        start=jnp.zeros((), jnp.float32),
    )
    denominator = sum(delta.size for delta in weight_deltas.values())
    return numerator / denominator


def importance_minimality_loss(
    ci_upper: dict[str, Array], pnorm: Array, beta: float, eps: float
) -> Array:
    """Per-site grouping with the global-batch sum inside the log2 (SPEC S7/S8).

    Under GSPMD the `(b, t)` axes are the global batch, so `jnp.sum` IS the exact
    global per-component sum — XLA reduces across shards inside the graph."""
    total = jnp.zeros((), jnp.float32)
    for ci in ci_upper.values():
        ci = ci.astype(jnp.float32)  # (B, T, C)
        n_positions = ci.shape[0] * ci.shape[1]
        per_component_sums = jnp.sum((ci + eps) ** pnorm, axis=(0, 1))  # (C,)
        per_component_means = per_component_sums / n_positions
        total = total + jnp.sum(
            per_component_means + beta * per_component_means * jnp.log2(1.0 + per_component_sums)
        )
    return total


def uniform_k_subset_routes(
    key: PRNGKeyArray, live_sites: tuple[str, ...], batch_seq_shape: tuple[int, ...]
) -> dict[str, Array]:
    """Per position: `k ~ U{1..|live|}`, then a uniform k-subset of the live sites
    routes True (SPEC S11). Distributionally identical to torch's double-argsort ranks."""
    n_sites = len(live_sites)
    k_key, perm_key = random.split(key)
    k = random.randint(k_key, batch_seq_shape, 1, n_sites + 1)
    perms = random.uniform(perm_key, (n_sites, *batch_seq_shape)).argsort(axis=0)
    routed = perms < k
    return {name: routed[j] for j, name in enumerate(live_sites)}


# ───────────────────────────── recon plans (SPEC S10/S11) ─────────────────────────────


Routes = dict[str, Array] | None
RoutingSampler = Callable[[PRNGKeyArray, tuple[int, int]], tuple[Routes, ...]]
"""`(key, (B, T)) -> (routes, ...)` — a STATICALLY-sized family of routing draws, each
`{site: bool[B, T]}` (or None = route everywhere) becoming ONE forward. The torch
`Router.get_masks` made pure: fresh draws per step require the key threaded in —
samplers run INSIDE the jitted step, so they must be traceable (SPEC R1). Returning
several draws from one invocation enables JOINTLY-sampled families (independent
repeats, antithetic/complementary subsets, per-step random covers) that duplicated
plan entries with independent keys cannot express. The plan's structure — live-sets,
sampler identities, family sizes — is static; only the key varies per step."""


class ReconForward(NamedTuple):
    """One plan entry: which sites run their decomposed path (`live_sites` — everything
    else takes the frozen `x @ W` path, the ~9x-cheaper non-decomposed matmul) and a
    sampler producing this entry's family of routing draws. Each draw is one forward,
    with its own fresh mask/delta sources."""

    live_sites: tuple[str, ...]
    sample_routing: RoutingSampler


ReconPlan = tuple[ReconForward, ...]
"""The stochastic-recon loss is the mean over ALL forwards (every draw of every entry)
of KL/(B·T). Live-sets may differ across entries (each traces its own forward); the
plan is fixed across steps (varying it would retrace)."""


def uniform_k_routing(live_sites: tuple[str, ...], n_draws: int) -> RoutingSampler:
    """`n_draws` independent per-position uniform-k-subset draws over `live_sites`."""

    def sample(key: PRNGKeyArray, batch_seq_shape: tuple[int, int]) -> tuple[Routes, ...]:
        return tuple(
            uniform_k_subset_routes(draw_key, live_sites, batch_seq_shape)
            for draw_key in random.split(key, n_draws)
        )

    return sample


def route_all(_key: PRNGKeyArray, _batch_seq_shape: tuple[int, int]) -> tuple[Routes, ...]:
    """One draw routing every position to every live site."""
    return (None,)


def subset_chunk_plan(
    site_names: tuple[str, ...], sites_per_chunk: int, n_samples: int
) -> ReconPlan:
    """The production plan: partition into sequential chunks, `n_samples` uniform-k
    forwards per chunk (torch `SubsetReconPlan` over `ThreePoolTopology` chunks)."""
    return tuple(
        ReconForward(live_sites=chunk, sample_routing=uniform_k_routing(chunk, n_samples))
        for chunk in chunk_sites(site_names, sites_per_chunk)
    )


def per_site_plan(site_names: tuple[str, ...]) -> ReconPlan:
    """One forward per site, routed everywhere — the historical "layerwise" loop
    (torch `PerSitePlan` / `StochasticReconLayerwiseLoss`)."""
    return tuple(ReconForward(live_sites=(site,), sample_routing=route_all) for site in site_names)


def make_ppgd_masks(
    ci_lower: dict[str, Array], sources: dict[str, Array], site_names: tuple[str, ...]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """`mask = ci + (1−ci)·source[:, :C]`; delta mask = raw trailing channel (SPEC S1).
    Sources broadcast over the batch dim (broadcast_across_batch scope). The fp32
    source state is cast to the CI dtype here (torch-under-autocast behavior); the
    source gradient flows back through the cast."""
    masks = {}
    delta_masks = {}
    for site in site_names:
        source_bf16 = sources[site].astype(ci_lower[site].dtype)
        masks[site] = ci_lower[site] + (1.0 - ci_lower[site]) * source_bf16[..., :-1]
        delta_masks[site] = source_bf16[..., -1]
    return masks, delta_masks


def _grad_norm_metrics(components_grad: Any, ci_fn_grad: Any) -> dict[str, Array]:
    """Pre-clip gradient L2 norms, matching the torch `component_grad_norms` families:
    per-leaf `grad_norms/components<path>` / `grad_norms/ci_fns<path>` (paths are this
    pytree's own — e.g. `.vu['layers.18.mlp.gate_proj'][0]` for the per-site Llama
    layout, vs torch's per-site names) and the overlay-critical
    `grad_norms/summary/{components,ci_fns,total}`."""
    out: dict[str, Array] = {}

    def family(grad_tree: Any, prefix: str) -> Array:
        sum_sq = jnp.zeros((), jnp.float32)
        for path, leaf in jax.tree_util.tree_flatten_with_path(grad_tree)[0]:
            leaf_sum_sq = jnp.sum(leaf.astype(jnp.float32) ** 2)
            out[f"grad_norms/{prefix}{jax.tree_util.keystr(path)}"] = jnp.sqrt(leaf_sum_sq)
            sum_sq = sum_sq + leaf_sum_sq
        out[f"grad_norms/summary/{prefix}"] = jnp.sqrt(sum_sq)
        return sum_sq

    total_sq = family(components_grad, "components") + family(ci_fn_grad, "ci_fns")
    out["grad_norms/summary/total"] = jnp.sqrt(total_sq)
    return out


# ───────────────────────────── the step factory ─────────────────────────────


def make_train_step(
    lm: DecomposedLM,
    coeffs: LossCoeffs,
    imp_cfg: ImpMinConfig,
    adversary: AdversaryConfig,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    recon_plan: ReconPlan,
    remat_recon_forwards: bool,
    mesh: Mesh | None,
):
    """Build the jit'd `step(state, frozen, residual, key) -> (state, metrics)`.

    `mesh` (when given) pins every batch-leading activation to `P('dp', ...)` so the
    masked re-forwards stay on per-device sub-batches (activation memory 1/n_dev)."""
    site_names = lm.site_names
    assert recon_plan, "empty recon plan"
    for recon_forward in recon_plan:
        assert recon_forward.live_sites and set(recon_forward.live_sites) <= set(site_names), (
            recon_forward
        )

    def batch_sharded(x: Array) -> Array:
        if mesh is None:
            return x
        spec = ["dp"] + [None] * (x.ndim - 1)
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    def batch_sharded_ci(ci_values: CIValues) -> CIValues:
        """Reshard the CI-fn output to batch-sharded ONCE, here. The CI head's `out_w`
        is ΣC-sharded, so its output is born C-sharded; without this pin GSPMD reshards
        it separately for EVERY consumer (each chunk forward + PPGD + imp-min, fwd and
        bwd) — at 36 sites those ~1.2 GB all-to-all buffers dominated the temp arena
        (the 109 GiB `jit_step` OOM, job 50542; XLA dump `memprobe_mc_50581`)."""
        return CIValues(
            lower={site: batch_sharded(v) for site, v in ci_values.lower.items()},
            upper={site: batch_sharded(v) for site, v in ci_values.upper.items()},
        )

    def masked_forward(
        frozen: Any,
        components_bf16: Any,
        residual: Array,
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live_sites: tuple[str, ...],
    ) -> Array:
        return batch_sharded(
            lm.masked_logits(
                frozen, components_bf16, residual, masks, delta_masks, routes, live_sites
            )
        )

    # Recomputing each masked forward in backward bounds activation memory to one
    # forward at a time (the torch 2-pool streaming profile) at the cost of the
    # recompute; with few recon forwards and memory headroom, remat off is faster.
    checkpointed_masked_forward = (
        jax.checkpoint(masked_forward, static_argnums=(6,))
        if remat_recon_forwards
        else masked_forward
    )

    def ppgd_recon_loss(
        frozen: Any,
        components_bf16: Any,
        ci_lower: dict[str, Array],
        sources: dict[str, Array],
        residual: Array,
        clean_logits: Array,
        masked_forward_fn: Any,
    ) -> Array:
        masks, delta_masks = make_ppgd_masks(ci_lower, sources, site_names)
        masked = masked_forward_fn(
            frozen, components_bf16, residual, masks, delta_masks, None, site_names
        )
        return kl_per_position(masked, clean_logits)

    def stochastic_recon_loss(
        frozen: Any,
        components_bf16: Any,
        ci_lower: dict[str, Array],
        residual: Array,
        clean_logits: Array,
        key: PRNGKeyArray,
    ) -> Array:
        batch, seq = residual.shape[0], residual.shape[1]
        total = jnp.zeros((), jnp.float32)
        n_forwards = 0
        for entry_idx, recon_forward in enumerate(recon_plan):
            entry_key, routing_key = random.split(random.fold_in(key, entry_idx))
            for draw_idx, routes in enumerate(
                recon_forward.sample_routing(routing_key, (batch, seq))
            ):
                mask_source_key, delta_mask_key = random.split(random.fold_in(entry_key, draw_idx))
                masks = {}
                delta_masks = {}
                for site_idx, site in enumerate(recon_forward.live_sites):
                    ci_site = ci_lower[site]
                    stochastic_source = random.uniform(
                        random.fold_in(mask_source_key, site_idx), ci_site.shape, COMPUTE_DT
                    )
                    masks[site] = ci_site + (1.0 - ci_site) * stochastic_source
                    delta_masks[site] = random.uniform(
                        random.fold_in(delta_mask_key, site_idx), (batch, seq), COMPUTE_DT
                    )
                masked = checkpointed_masked_forward(
                    frozen,
                    components_bf16,
                    residual,
                    masks,
                    delta_masks,
                    routes,
                    recon_forward.live_sites,
                )
                total = total + kl_per_position(masked, clean_logits)
                n_forwards += 1
        assert n_forwards > 0, "recon plan produced no forwards"
        return total / n_forwards

    @jax.jit
    def step(state: TrainState, frozen: Any, residual: Float[Array, "b t d"], key: PRNGKeyArray):
        step_f32 = state.step.astype(jnp.float32)
        pnorm = annealed_pnorm(step_f32, total_steps, imp_cfg)

        residual = batch_sharded(residual)
        clean_logits = jax.lax.stop_gradient(batch_sharded(lm.clean_logits(frozen, residual)))
        site_inputs = lm.site_inputs(frozen, residual)

        # ── supplemental adversary ascents: params + CI detached (SPEC §4.5) ──
        components_detached = jax.lax.stop_gradient(cast_floating(state.components, COMPUTE_DT))
        ci_fn_detached = jax.lax.stop_gradient(cast_floating(state.ci_fn, COMPUTE_DT))
        ci_lower_detached = batch_sharded_ci(ci_fn_detached(site_inputs)).lower

        def adversary_loss(sources: dict[str, Array]) -> Array:
            return ppgd_recon_loss(
                frozen,
                components_detached,
                ci_lower_detached,
                sources,
                residual,
                clean_logits,
                masked_forward,
            )

        match adversary:
            case SourceAdamConfig() as persistent_adam_config:
                sources_lr = warmup_then_constant_lr(
                    step_f32, total_steps, adversary.lr, adversary.lr_warmup_frac
                )

                def warmup_body(
                    carry: tuple[dict[str, Array], SourcesAdamState], _: None
                ) -> tuple[tuple[dict[str, Array], SourcesAdamState], None]:
                    sources, adam_state = carry
                    sources_grad = jax.grad(adversary_loss)(sources)
                    sources, adam_state = sources_adam_ascend_project(
                        sources, sources_grad, adam_state, sources_lr, persistent_adam_config
                    )
                    return (sources, adam_state), None

                (refined_sources, sources_adam_state), _ = jax.lax.scan(
                    warmup_body,
                    (state.sources, state.sources_adam_state),
                    None,
                    length=adversary.n_warmup,
                )
            case FreshPGDConfig() as fresh_pgd_config:
                sources_lr = None
                sources_adam_state = state.sources_adam_state
                batch, seq = residual.shape[0], residual.shape[1]
                fresh_sources = init_fresh_pgd_sources(
                    lm.sites, adversary, batch, seq, random.fold_in(key, 2)
                )

                def sign_ascend_body(
                    sources: dict[str, Array], _: None
                ) -> tuple[dict[str, Array], None]:
                    sources_grad = jax.grad(adversary_loss)(sources)
                    return {
                        site: jnp.clip(
                            sources[site]
                            + fresh_pgd_config.step_size * jnp.sign(sources_grad[site]),
                            0.0,
                            1.0,
                        )
                        for site in sources
                    }, None

                refined_sources, _ = jax.lax.scan(
                    sign_ascend_body, fresh_sources, None, length=adversary.n_steps
                )
        refined_sources = jax.lax.stop_gradient(refined_sources)

        # ── main losses: live components/ci; ppgd's source participates in the graph so
        # its gradient comes from the SAME backward (SPEC S14); it is NOT detached here,
        # but components/ci grads through it are what torch gets too (sources are leaves). ──
        def loss_fn(trainable: tuple[Any, CIFn, dict[str, Array]]):
            components, ci_fn, sources = trainable
            components_bf16 = cast_floating(components, COMPUTE_DT)
            ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
            ci = batch_sharded_ci(ci_fn_bf16(site_inputs))
            faith_loss = faithfulness_loss(lm.weight_deltas(frozen, components))
            imp_loss = importance_minimality_loss(ci.upper, pnorm, imp_cfg.beta, imp_cfg.eps)
            stoch_loss = stochastic_recon_loss(
                frozen, components_bf16, ci.lower, residual, clean_logits, random.fold_in(key, 1)
            )
            ppgd_loss = ppgd_recon_loss(
                frozen,
                components_bf16,
                ci.lower,
                sources,
                residual,
                clean_logits,
                checkpointed_masked_forward,
            )
            total_loss = (
                coeffs.faith * faith_loss
                + coeffs.imp * imp_loss
                + coeffs.stoch * stoch_loss
                + coeffs.ppgd * ppgd_loss
            )
            return total_loss, (faith_loss, imp_loss, stoch_loss, ppgd_loss)

        (total_loss, (faith_loss, imp_loss, stoch_loss, ppgd_loss)), grads = (
            eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                (state.components, state.ci_fn, refined_sources)
            )
        )
        components_grad, ci_fn_grad, sources_grad_scaled = grads
        grad_norm_metrics = _grad_norm_metrics(components_grad, ci_fn_grad)

        match adversary:
            case SourceAdamConfig():
                assert sources_lr is not None
                # The backward saw coeff·L_adv; the adversary ascends on L_adv itself.
                sources_grad = {s: g / coeffs.ppgd for s, g in sources_grad_scaled.items()}
                # ── the (n_warmup+1)-th source ascent, from the fused graph (SPEC S13/S14) ──
                new_sources, sources_adam_state = sources_adam_ascend_project(
                    refined_sources, sources_grad, sources_adam_state, sources_lr, adversary
                )
            case FreshPGDConfig():
                # fresh sources die with the step; the cotangent wrt them is unused
                new_sources = state.sources

        components_updates, new_components_opt_state = components_optimizer.update(
            components_grad,
            state.components_opt_state,
            eqx.filter(state.components, eqx.is_array),
        )
        ci_fn_updates, new_ci_fn_opt_state = ci_fn_optimizer.update(
            ci_fn_grad, state.ci_fn_opt_state, eqx.filter(state.ci_fn, eqx.is_array)
        )
        new_components = eqx.apply_updates(state.components, components_updates)
        new_ci_fn = eqx.apply_updates(state.ci_fn, ci_fn_updates)

        new_state = TrainState(
            components=new_components,
            ci_fn=new_ci_fn,
            components_opt_state=new_components_opt_state,
            ci_fn_opt_state=new_ci_fn_opt_state,
            sources=new_sources,
            sources_adam_state=sources_adam_state,
            step=state.step + 1,
        )
        adversary_metric_key = "ppgd" if isinstance(adversary, SourceAdamConfig) else "pgd"
        metrics = {
            "total": total_loss,
            "faith": faith_loss,
            "imp": imp_loss,
            "stoch": stoch_loss,
            adversary_metric_key: ppgd_loss,
            "p_imp": pnorm,
            **grad_norm_metrics,
        }
        if sources_lr is not None:
            metrics["src_lr"] = sources_lr
        return new_state, metrics

    return step


# ───────────────────────────── faithfulness warmup (SPEC S21) ─────────────────────────────


def make_faith_warmup_step(
    lm: DecomposedLM, opt: optax.GradientTransformation
) -> Callable[[Any, optax.OptState, Any], tuple[Any, optax.OptState, Array]]:
    @jax.jit
    def warmup_step(
        components: Any, opt_state: optax.OptState, frozen: Any
    ) -> tuple[Any, optax.OptState, Array]:
        def loss_fn(components_: Any) -> Array:
            return faithfulness_loss(lm.weight_deltas(frozen, components_))

        loss, grad = eqx.filter_value_and_grad(loss_fn)(components)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(components, eqx.is_array))
        return eqx.apply_updates(components, updates), opt_state, loss

    return warmup_step
