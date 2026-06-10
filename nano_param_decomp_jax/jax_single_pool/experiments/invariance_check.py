"""Device-count invariance of the generic trainer (SPEC D4), on the tiny Llama target.

Runs the SAME fixed global batch + seed through the full step twice on this host:
once single-layout (mesh=None — everything on device 0), once GSPMD batch-sharded over
ALL visible devices — and asserts the per-step metric trajectories match up to
floating-point reassociation (rel ≤ 1e-4; cross-shard reduction order differs, so
bit-exactness is not achievable for the batch-reduced terms). That is the
SPMD-correctness contract: sharding layout must be semantically invisible.

Simulated multi-device CPU run:

  XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    python -m jax_single_pool.experiments.invariance_check --steps 3

(JAX's counter-based RNG is value-deterministic for a fixed key regardless of
sharding, so the stochastic terms draw identical values — only summation order
differs across layouts.)
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random

from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.llama8b import LayerRange, init_decomp_vu, llama_decomposed_lm
from jax_single_pool.sharding import dp_mesh, shard_batch
from jax_single_pool.tests.test_llama8b import _tiny_cfg, _tiny_target
from jax_single_pool.train import (
    ImpMinConfig,
    LossCoeffs,
    SourceAdamConfig,
    TrainState,
    init_sources,
    init_src_adam,
    make_train_step,
    subset_chunk_plan,
)


def _run(steps: int, sharded: bool) -> list[dict[str, float]]:
    cfg = _tiny_cfg()
    rng = LayerRange(3, 6)
    tgt = _tiny_target(cfg, rng, random.PRNGKey(0))
    C, seq, gbatch = 8, 16, 8
    lm = llama_decomposed_lm(cfg, rng, C)
    vu = init_decomp_vu(cfg, C, rng.n_layers, random.PRNGKey(1))
    ci_fn = init_ci_fn(CIArch(16, 2, 2, 32), lm.sites, random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    src = init_sources(lm.site_names, tuple(s.C for s in lm.sites), seq, random.PRNGKey(3))
    resid = random.normal(random.PRNGKey(4), (gbatch, seq, cfg.n_embd)) * 0.5

    mesh = dp_mesh() if sharded else None
    if mesh is not None:
        resid = shard_batch(resid, mesh, batch_axis=0)

    state = TrainState(
        vu=vu, ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        src=src, src_adam=init_src_adam(src), step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    step = make_train_step(
        lm=lm,
        coeffs=LossCoeffs(faith=1e5, imp=5e-6, stoch=0.5, ppgd=0.5),
        imp_cfg=ImpMinConfig(0.2, 1e-12, 2.0, 0.4, 0.0, 1.0),
        src_cfg=SourceAdamConfig(0.01, 0.025, 0.5, 0.99, 1e-8, n_warmup=2),
        opt_vu=opt_vu, opt_ci=opt_ci,
        total_steps=100, recon_plan=subset_chunk_plan(lm.site_names, 3, 1), mesh=mesh,
    )  # fmt: skip

    out = []
    for i in range(steps):
        state, m = step(state, tgt, resid, random.PRNGKey(100 + i))
        out.append({k: float(v) for k, v in m.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    n_dev = len(jax.devices())
    print(f"devices: {n_dev}")
    single = _run(args.steps, sharded=False)
    sharded = _run(args.steps, sharded=True)

    REL = 1e-4
    ok = True
    worst = 0.0
    for i, (a, b) in enumerate(zip(single, sharded, strict=True)):
        for k in a:
            rel = abs(a[k] - b[k]) / (abs(a[k]) + 1e-30)
            worst = max(worst, rel)
            if rel > REL:
                ok = False
                print(f"step {i} {k}: single {a[k]!r} vs sharded({n_dev}) {b[k]!r} rel {rel:.2e}")
    assert ok, "trajectory diverged across shardings — SPMD correctness broken (SPEC D4)"
    print(
        f"OK: {args.steps}-step trajectory matches 1-layout vs {n_dev}-device GSPMD "
        f"(worst rel {worst:.2e} <= {REL:.0e}; reassociation-only)"
    )


if __name__ == "__main__":
    main()
