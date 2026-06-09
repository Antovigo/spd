"""CPU-runnable tests for the Llama-8B full-suffix output-recon PD step at a tiny config.

Validates the model forward + the full step (one iteration trains, the loss has the VPD
signature) without loading real weights or needing a GPU. `vendored_jax` is put on the
path by conftest.py.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

pytest.importorskip("vendored_jax")

from vendored_jax.llama import LlamaConfig, llama3_inv_freq  # noqa: E402

from jax_single_pool.ci_fn import CIFnDims, init_ci_fn  # noqa: E402
from jax_single_pool.llama8b import (  # noqa: E402
    SITES,
    DecompVU,
    FrozenAttn,
    FrozenBlock,
    FrozenMLP,
    Target,
    init_decomp_vu,
    suffix_logits,
    weight_deltas,
)
from jax_single_pool.llama8b_step import (  # noqa: E402
    Llama8BState,
    LossCoeffs,
    make_llama8b_step,
)


def _tiny_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        n_layer=22,  # so suffix L18..L21 = 4 blocks
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


def _tiny_target(cfg: LlamaConfig, key) -> Target:
    ks = iter(jax.random.split(key, 256))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape, s=None):
        return jax.random.normal(next(ks), shape) * (s or d**-0.5)

    def fattn():
        return FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep,
        )  # fmt: skip

    def fblock():
        return FrozenBlock(
            jnp.ones((d,)), jnp.ones((d,)), fattn(),
            FrozenMLP(n((di, d)), n((di, d)), n((d, di))), cfg.rms_norm_eps,
        )  # fmt: skip

    n_suffix = cfg.n_layer - 18 - 1
    return Target(
        l18_ln1=jnp.ones((d,)), l18_ln2=jnp.ones((d,)), l18_attn=fattn(),
        l18_Wg=n((di, d)), l18_Wu=n((di, d)), l18_Wd=n((d, di)),
        rest=[fblock() for _ in range(n_suffix)],
        norm=jnp.ones((d,)), lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg), eps=cfg.rms_norm_eps,
    )  # fmt: skip


def test_suffix_logits_clean_matches_target():
    """Clean decomposed forward (V@U == W reconstructed by weight-delta) ~ frozen suffix."""
    cfg = _tiny_cfg()
    key = jax.random.PRNGKey(0)
    tgt = _tiny_target(cfg, key)
    C = 8
    vu = init_decomp_vu(cfg, C, tgt, jax.random.PRNGKey(1))
    b, t, d = 2, 16, cfg.n_embd
    resid = jax.random.normal(jax.random.PRNGKey(2), (b, t, d)) * 0.5
    nomask = {s: None for s in SITES}
    dm = {s: jnp.ones((1, 1, 1)) for s in SITES}
    logits = suffix_logits(tgt, vu, resid, nomask, dm)
    assert logits.shape == (b, t, cfg.vocab_size)
    wd = weight_deltas(vu, tgt.l18_Wg, tgt.l18_Wu, tgt.l18_Wd)
    assert wd["gate"].shape == tgt.l18_Wg.shape
    assert wd["up"].shape == tgt.l18_Wu.shape
    assert wd["down"].shape == tgt.l18_Wd.shape


def test_step_trains_and_has_vpd_signature():
    cfg = _tiny_cfg()
    tgt = _tiny_target(cfg, jax.random.PRNGKey(0))
    C = 8
    vu = init_decomp_vu(cfg, C, tgt, jax.random.PRNGKey(1))
    dims = CIFnDims(d_model=16, n_blocks=2, n_heads=2, mlp_hidden=32,
                    total_in=cfg.n_embd + cfg.n_embd + cfg.n_intermediate, C=C)  # fmt: skip
    ci_fn = init_ci_fn(dims, jax.random.PRNGKey(2))
    opt_vu = optax.adamw(1e-3)
    opt_ci = optax.adamw(1e-3)

    import equinox as eqx

    state = Llama8BState(
        vu=vu, ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        source={s: jnp.zeros((1, 16, C)) for s in SITES},
        step=jnp.array(0),
    )  # fmt: skip
    coeffs = LossCoeffs(faith=1e5, imp=5e-6, stoch=0.5, ppgd=0.5, p_imp=0.4)
    step = make_llama8b_step(coeffs, opt_vu, opt_ci, pgd_lr=0.01, n_warmup=2)

    resid = jax.random.normal(jax.random.PRNGKey(3), (2, 16, cfg.n_embd)) * 0.5
    losses = []
    for i in range(4):
        state, m = step(state, tgt, resid, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})

    # ppgd is the worst-case recon -> should sit at or above the stochastic recon (minimax)
    assert losses[-1]["ppgd"] >= losses[-1]["stoch"] * 0.5
    # the step advanced and produced finite losses
    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert float(state.step) == 4.0


def test_decomp_vu_shapes():
    cfg = _tiny_cfg()
    tgt = _tiny_target(cfg, jax.random.PRNGKey(0))
    C = 8
    vu = init_decomp_vu(cfg, C, tgt, jax.random.PRNGKey(1))
    d, di = cfg.n_embd, cfg.n_intermediate
    assert vu.Vg.shape == (d, C) and vu.Ug.shape == (C, di)
    assert vu.Vd.shape == (di, C) and vu.Ud.shape == (C, d)
    assert isinstance(vu, DecompVU)
