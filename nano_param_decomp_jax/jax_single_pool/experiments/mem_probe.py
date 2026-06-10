"""AOT memory probe for the train step — no data, no HF weights, no execution.

Compiles `jit_step` at the smoke topology (8 GPU, B=32, L18, C=24576) and prints the
per-device memory analysis; with `--xla_dump_to` set, XLA writes the
buffer-assignment table that names the largest allocations (debugging the smoke-v4
107 GiB OOM). Run under SLURM like the trainer:

  XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text" srun ... mem_probe.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jax_single_pool.ci_fn import CIArch, init_ci_fn

# the smoke's _random_target lives in the runner
from jax_single_pool.experiments.llama8b_real import (
    _random_target,  # pyright: ignore[reportPrivateUsage]
)
from jax_single_pool.llama8b import (
    LayerRange,
    init_decomp_vu,
    llama31_8b_config,
    llama_decomposed_lm,
)
from jax_single_pool.llama8b_sharding import (
    dp_mesh,
    replicate_target,
    shard_ci_fn,
    shard_decomp_vu,
    shard_source,
)
from jax_single_pool.sharding import init_distributed
from jax_single_pool.train import (
    ImpMinConfig,
    LossCoeffs,
    SourceAdamConfig,
    TrainState,
    init_sources,
    init_sources_adam_state,
    make_train_step,
    subset_chunk_plan,
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--per_gpu_batch", type=int, default=4)
    ap.add_argument("--no_remat", action="store_true",
                    help="disable the recon-forward rematerialization (memory A/B)")  # fmt: skip
    args = ap.parse_args()
    init_distributed()
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0

    cfg = llama31_8b_config()
    rng = LayerRange(18, 18)
    C = 24576
    seq = 2048
    gbatch = args.per_gpu_batch * ndev
    lm = llama_decomposed_lm(cfg, rng, C)

    target = replicate_target(_random_target(cfg, rng, random.PRNGKey(0)), mesh)
    vu = shard_decomp_vu(init_decomp_vu(cfg, C, rng.n_layers, random.PRNGKey(1)), mesh)
    ci_fn = shard_ci_fn(init_ci_fn(CIArch(4096, 4, 64, 16384), lm.sites, random.PRNGKey(2)), mesh)
    src = shard_source(
        init_sources(lm.site_names, tuple(s.C for s in lm.sites), seq, random.PRNGKey(3)), mesh
    )
    opt_vu = optax.chain(
        optax.clip_by_global_norm(0.01),
        optax.adamw(1.5e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0),
    )
    opt_ci = optax.adamw(5e-5, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources=src, sources_adam_state=init_sources_adam_state(src), step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    step_fn = make_train_step(
        lm=lm,
        coeffs=LossCoeffs(1e5, 5e-6, 0.5, 0.5),
        imp_cfg=ImpMinConfig(0.2, 1e-12, 2.0, 0.4, 0.0, 1.0),
        src_cfg=SourceAdamConfig(0.01, 0.025, 0.5, 0.99, 1e-8, n_warmup=2),
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100, recon_plan=subset_chunk_plan(lm.site_names, 3, 1),
        remat_recon_forwards=not args.no_remat, mesh=mesh,
    )  # fmt: skip

    resid = jax.device_put(
        jnp.zeros((gbatch, seq, cfg.n_embd), jnp.bfloat16), NamedSharding(mesh, P("dp"))
    )
    lowered = step_fn.lower(state, target, resid, random.PRNGKey(7))
    compiled = lowered.compile()
    if is0:
        ma = compiled.memory_analysis()
        assert ma is not None, "backend returned no memory analysis"
        gib = 1024**3
        print(
            f"[mem] ndev={ndev} bl={args.per_gpu_batch} remat={not args.no_remat} | "
            f"temp {ma.temp_size_in_bytes / gib:.1f} GiB | "
            f"args {ma.argument_size_in_bytes / gib:.1f} GiB | "
            f"out {ma.output_size_in_bytes / gib:.1f} GiB | "
            f"alias {ma.alias_size_in_bytes / gib:.1f} GiB",
            flush=True,
        )
        print("[mem] PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
