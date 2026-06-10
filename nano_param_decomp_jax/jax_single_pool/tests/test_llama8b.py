"""CPU tests for the Llama target + generic trainer at a tiny config.

Validates the `DecomposedLM` contract (clean == all-frozen masked forward, shapes) and
the full SPEC step (trains, VPD loss signature, adversary state advances), without real
weights or a GPU.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from vendored_jax.llama import LlamaConfig, llama3_inv_freq

from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.llama8b import (
    DecompLayerFrozen,
    DecompVU,
    FrozenAttn,
    FrozenBlock,
    FrozenMLP,
    LayerRange,
    Target,
    init_decomp_vu,
    llama_decomposed_lm,
)
from jax_single_pool.train import (
    ImpMinConfig,
    LossCoeffs,
    SourceAdamConfig,
    TrainState,
    init_sources,
    init_src_adam,
    make_faith_warmup_step,
    make_train_step,
    subset_chunk_plan,
)


def _tiny_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        n_layer=8,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=128,
    )


def _tiny_target(cfg: LlamaConfig, rng: LayerRange, key: jax.Array) -> Target:
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape: tuple[int, ...], s: float | None = None) -> jax.Array:
        return jax.random.normal(next(ks), shape) * (s or d**-0.5)

    def fattn():
        return FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep,
        )  # fmt: skip

    def dlayer():
        return DecompLayerFrozen(
            jnp.ones((d,)), jnp.ones((d,)), fattn(), n((di, d)), n((di, d)), n((d, di))
        )

    def fblock():
        return FrozenBlock(
            jnp.ones((d,)), jnp.ones((d,)), fattn(),
            FrozenMLP(n((di, d)), n((di, d)), n((d, di))), cfg.rms_norm_eps,
        )  # fmt: skip

    return Target(
        decomp_layers=[dlayer() for _ in range(rng.n_layers)],
        tail=[fblock() for _ in range(cfg.n_layer - rng.last - 1)],
        norm=jnp.ones((d,)), lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg), eps=cfg.rms_norm_eps,
    )  # fmt: skip


@pytest.mark.parametrize("rng", [LayerRange(4, 4), LayerRange(3, 6)])
def test_clean_path_and_masked_identity(rng: LayerRange):
    cfg = _tiny_cfg()
    tgt = _tiny_target(cfg, rng, jax.random.PRNGKey(0))
    C = 8
    lm = llama_decomposed_lm(cfg, rng, C)
    vu = init_decomp_vu(cfg, C, rng.n_layers, jax.random.PRNGKey(1))
    b, t = 2, 16
    resid = jax.random.normal(jax.random.PRNGKey(2), (b, t, cfg.n_embd)) * 0.5

    clean = lm.clean_logits(tgt, resid)
    assert clean.shape == (b, t, cfg.vocab_size)

    # SPEC S2: a masked forward with NO live sites is the frozen path — bit-identical
    # to the clean target.
    none_masked = lm.masked_logits(tgt, vu, resid, {}, {}, None, ())
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    # All-live, masks=1, delta=1, route-everywhere reconstructs the frozen path up to
    # decomposition rounding (the V@U + (W − V@U) identity; exact only in exact math).
    names = lm.site_names
    ones_masks = {s: jnp.ones((b, t, C)) for s in names}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = lm.masked_logits(tgt, vu, resid, ones_masks, ones_delta, None, names)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    site_in = lm.site_inputs(tgt, resid)
    assert set(site_in) == set(names)
    deltas = lm.weight_deltas(tgt, vu)
    d, di = cfg.n_embd, cfg.n_intermediate
    assert deltas[names[0]].shape == (di, d)  # gate: (d_out, d_in)
    assert deltas[names[2]].shape == (d, di)  # down
    assert all(v.dtype == jnp.float32 for v in deltas.values())


@pytest.mark.parametrize("rng", [LayerRange(4, 4), LayerRange(3, 6)])
def test_step_trains_and_has_vpd_signature(rng: LayerRange):
    cfg = _tiny_cfg()
    tgt = _tiny_target(cfg, rng, jax.random.PRNGKey(0))
    C = 8
    seq = 16
    n_warmup = 2
    lm = llama_decomposed_lm(cfg, rng, C)
    vu = init_decomp_vu(cfg, C, rng.n_layers, jax.random.PRNGKey(1))
    ci_fn = init_ci_fn(CIArch(d_model=16, n_blocks=2, n_heads=2, mlp_hidden=32),
                       lm.sites, jax.random.PRNGKey(2))  # fmt: skip
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    src = init_sources(lm.site_names, tuple(s.C for s in lm.sites), seq, jax.random.PRNGKey(3))
    state = TrainState(
        vu=vu, ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        src=src, src_adam=init_src_adam(src), step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    step = make_train_step(
        lm=lm,
        coeffs=LossCoeffs(faith=1e5, imp=5e-6, stoch=0.5, ppgd=0.5),
        imp_cfg=ImpMinConfig(
            beta=0.2,
            eps=1e-12,
            p_start=2.0,
            p_final=0.4,
            anneal_start_frac=0.0,
            anneal_end_frac=1.0,
        ),  # fmt: skip
        src_cfg=SourceAdamConfig(
            lr=0.01, lr_warmup_frac=0.025, beta1=0.5, beta2=0.99, eps=1e-8, n_warmup=n_warmup
        ),  # fmt: skip
        opt_vu=opt_vu,
        opt_ci=opt_ci,
        total_steps=100,
        recon_plan=subset_chunk_plan(lm.site_names, 3, 1),
        mesh=None,
    )

    resid = jax.random.normal(jax.random.PRNGKey(4), (2, seq, cfg.n_embd)) * 0.5
    n_steps = 4
    losses = []
    for i in range(n_steps):
        state, m = step(state, tgt, resid, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})

    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert int(state.step) == n_steps
    # SPEC S13: n_warmup + 1 source-Adam updates per training step, moments persist.
    assert float(state.src_adam.step_count) == n_steps * (n_warmup + 1)
    # SPEC S15: sources stay projected to [0,1].
    for v in state.src.values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    # SPEC S9: p annealed below its 2.0 start by step 4 of 100.
    assert losses[-1]["p_imp"] < 2.0
    # fp32 masters preserved through updates (SPEC N1).
    assert state.vu.Vg.dtype == jnp.float32
    assert state.ci_fn.in_proj_w.dtype == jnp.float32


def test_faith_warmup_decreases_faith():
    cfg = _tiny_cfg()
    rng = LayerRange(3, 4)
    tgt = _tiny_target(cfg, rng, jax.random.PRNGKey(0))
    lm = llama_decomposed_lm(cfg, rng, C=8)
    vu = init_decomp_vu(cfg, 8, rng.n_layers, jax.random.PRNGKey(1))
    opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(lm, opt)
    ostate = opt.init(eqx.filter(vu, eqx.is_array))
    first: float | None = None
    loss = None
    for _ in range(30):
        vu, ostate, loss = wstep(vu, ostate, tgt)
        first = float(loss) if first is None else first
    assert first is not None and loss is not None
    assert float(loss) < first * 0.9, (first, float(loss))


def test_decomp_vu_shapes_fp32():
    cfg = _tiny_cfg()
    C = 8
    rng = LayerRange(3, 6)
    vu = init_decomp_vu(cfg, C, rng.n_layers, jax.random.PRNGKey(1))
    d, di = cfg.n_embd, cfg.n_intermediate
    assert vu.Vg.shape == (rng.n_layers, d, C) and vu.Ug.shape == (rng.n_layers, C, di)
    assert vu.Vd.shape == (rng.n_layers, di, C) and vu.Ud.shape == (rng.n_layers, C, d)
    assert isinstance(vu, DecompVU)
    assert vu.Vg.dtype == jnp.float32
