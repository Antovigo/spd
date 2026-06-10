"""The training entrypoint: YAML config -> full SPEC-compliant run on the Llama target.

    jsp-train configs/llama8b_l18_b512.yaml          # fresh, or resume-in-place

Composition root + the only I/O layer: data serving (`data.py`), HF weight loading,
metrics jsonl (+ optional wandb), orbax checkpoints, SIGTERM-save for SLURM requeue.
The step itself is the pure jit'd `make_train_step`. Resume restores the full
trajectory (SPEC S22) and fast-forwards the data schedule by step arithmetic.

Multi-process: launched one process per GPU under SLURM (`init_distributed`); every
process computes the same global schedule and contributes its local batch slice.
"""

import argparse
import json
import signal
import time
from pathlib import Path
from types import FrameType

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jax_single_pool.checkpoint import make_checkpoint_manager, restore_latest, save_state
from jax_single_pool.ci_fn import init_ci_fn
from jax_single_pool.config import ExperimentConfig, load_config
from jax_single_pool.data import BatchSchedule, ShardServer, scan_shards
from jax_single_pool.llama8b import (
    LayerRange,
    Prefix,
    Target,
    init_decomp_vu,
    llama31_8b_config,
    llama_decomposed_lm,
    load_prefix_from_hf,
    load_target_from_hf,
    prefix_residual,
)
from jax_single_pool.llama8b_sharding import (
    replicate_target,
    shard_ci_fn,
    shard_decomp_vu,
    shard_source,
)
from jax_single_pool.lm import DecomposedLM
from jax_single_pool.sharding import dp_mesh, init_distributed
from jax_single_pool.train import (
    TrainState,
    init_sources,
    init_src_adam,
    make_faith_warmup_step,
    make_train_step,
)

_sigterm_received = False


def _install_sigterm_flag() -> None:
    def handler(_signum: int, _frame: FrameType | None) -> None:
        global _sigterm_received
        _sigterm_received = True

    signal.signal(signal.SIGTERM, handler)


def _build_optimizers(cfg: ExperimentConfig):
    sched_vu = optax.cosine_decay_schedule(cfg.vu_optimizer.lr, cfg.steps, alpha=0.1)
    sched_ci = optax.cosine_decay_schedule(cfg.ci_optimizer.lr, cfg.steps, alpha=0.1)
    opt_vu = optax.chain(
        optax.clip_by_global_norm(cfg.vu_optimizer.grad_clip_norm),
        optax.adamw(sched_vu, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0),
    )
    opt_ci = optax.adamw(sched_ci, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
    return opt_vu, opt_ci


def _ensure_global(tree: object, mesh: Mesh) -> object:
    """Pin process-local leaves (eagerly created scalars: step counters, Adam counts)
    to a replicated mesh sharding. Multi-process orbax can only save GLOBAL arrays;
    leaves that haven't passed through the jitted step are otherwise per-process
    `SingleDeviceSharding` arrays and the save raises."""
    repl = NamedSharding(mesh, P())

    def fix(a: object) -> object:
        if eqx.is_array(a) and not isinstance(a.sharding, NamedSharding):  # pyright: ignore[reportAttributeAccessIssue]
            return jax.device_put(a, repl)
        return a

    return jax.tree.map(fix, tree)


def _global_token_batch(local: np.ndarray, mesh: Mesh, global_batch: int) -> jax.Array:
    sharding = NamedSharding(mesh, P("dp"))
    return jax.make_array_from_process_local_data(sharding, local, (global_batch, local.shape[1]))


# wandb keys match the torch trainer's (`train_step.py` emits `loss/<ClassName>`,
# `optimize.py` prefixes `train/`) so a torch-vs-jax run pair overlays on one panel.
# The stoch alias is exact for a single-chunk config (L18: chunkwise-subset over one
# chunk == StochasticReconSubsetLoss); multi-chunk configs diverge from that torch
# class but keep the name for comparability of the same-coeff term.
_METRIC_KEYS = {
    "total": "train/loss/total",
    "faith": "train/loss/FaithfulnessLoss",
    "imp": "train/loss/ImportanceMinimalityLoss",
    "stoch": "train/loss/StochasticReconSubsetLoss",
    "ppgd": "train/loss/PersistentPGDReconLoss",
    "p_imp": "train/schedules/p_imp",
    "src_lr": "train/schedules/lr/src",
    "step_time_s": "train/perf/step_time_s",
    "tok_per_s": "train/perf/tok_per_s",
    "tok_per_s_per_gpu": "train/perf/tok_per_s_per_gpu",
}


class MetricsSink:
    """Process-0 metrics fan-out: jsonl always, wandb when configured."""

    def __init__(self, cfg: ExperimentConfig, raw_cfg: dict[str, object], is_main: bool):
        self._jsonl = None
        self._wandb = None
        if not is_main:
            return
        self._jsonl = (cfg.run_dir / "metrics.jsonl").open("a")
        if cfg.wandb is not None:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.run_name,
                id=cfg.run_name,
                resume="allow",
                config=raw_cfg,
            )
            self._wandb = wandb

    def log(self, step: int, record: dict[str, float]) -> None:
        if self._jsonl is None:
            return
        record = {_METRIC_KEYS[k]: v for k, v in record.items()}
        self._jsonl.write(json.dumps({"step": step, **record}) + "\n")
        self._jsonl.flush()
        print(
            f"[step {step}] " + " ".join(f"{k}={v:.4g}" for k, v in record.items()),
            flush=True,
        )
        if self._wandb is not None:
            self._wandb.log(record, step=step)


def train(
    cfg: ExperimentConfig,
    raw_cfg: dict[str, object],
    lm: DecomposedLM,
    frozen: Target,
    prefix: Prefix,
    mesh: Mesh,
) -> None:
    is_main = jax.process_index() == 0
    n_proc = jax.process_count()
    ndev = mesh.devices.size
    assert cfg.data.global_batch % ndev == 0, (cfg.data.global_batch, ndev)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    opt_vu, opt_ci = _build_optimizers(cfg)

    key = random.PRNGKey(cfg.seed)
    init_key, src_key, run_key = random.split(key, 3)
    llama_cfg = llama31_8b_config()
    rng = LayerRange(cfg.target.first_layer, cfg.target.last_layer)
    vu = shard_decomp_vu(init_decomp_vu(llama_cfg, cfg.target.C, rng.n_layers, init_key), mesh)
    ci_fn = shard_ci_fn(init_ci_fn(cfg.ci_fn, lm.sites, random.fold_in(init_key, 1)), mesh)
    src = shard_source(
        init_sources(lm.site_names, tuple(s.C for s in lm.sites), cfg.data.seq_len, src_key),
        mesh,
    )
    state = TrainState(
        vu=vu,
        ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        src=src,
        src_adam=init_src_adam(src),
        step=jnp.zeros((), jnp.int32),
    )
    state = _ensure_global(state, mesh)
    assert isinstance(state, TrainState)

    mgr = make_checkpoint_manager(cfg.run_dir / "ckpts", cfg.cadence.keep_last)
    restored = restore_latest(mgr, state)
    if restored is not None:
        state, ckpt_step = restored
        start_step = ckpt_step
        if is_main:
            print(f"resumed from checkpoint step {ckpt_step}", flush=True)
    else:
        start_step = 0
        if cfg.faith_warmup.steps > 0:
            wopt = optax.adamw(cfg.faith_warmup.lr, weight_decay=0.0)
            wstate = wopt.init(eqx.filter(state.vu, eqx.is_array))
            wstep = make_faith_warmup_step(lm, wopt)
            vu_w = state.vu
            t0 = time.time()
            wloss = None
            for _ in range(cfg.faith_warmup.steps):
                vu_w, wstate, wloss = wstep(vu_w, wstate, frozen)
            assert wloss is not None
            jax.block_until_ready(wloss)
            new_opt_vu = _ensure_global(opt_vu.init(eqx.filter(vu_w, eqx.is_array)), mesh)
            state = state._replace(vu=vu_w, opt_vu=new_opt_vu)
            if is_main:
                print(
                    f"faith warmup: {cfg.faith_warmup.steps} steps in {time.time() - t0:.0f}s, "
                    f"final faith {float(wloss):.3e}",
                    flush=True,
                )
        save_state(mgr, 0, state)

    step_fn = make_train_step(
        lm=lm,
        coeffs=cfg.losses,
        imp_cfg=cfg.imp_min,
        src_cfg=cfg.ppgd,
        opt_vu=opt_vu,
        opt_ci=opt_ci,
        total_steps=cfg.steps,
        sites_per_chunk=cfg.recon.sites_per_chunk,
        n_samples=cfg.recon.n_samples,
        mesh=mesh,
    )

    def _harvest(pfx: Prefix, idx: jax.Array) -> jax.Array:
        resid = prefix_residual(pfx, idx)
        return jax.lax.with_sharding_constraint(resid, NamedSharding(mesh, P("dp")))

    harvest = jax.jit(_harvest)

    schedule = BatchSchedule(scan_shards(cfg.data.dir), cfg.data.global_batch, cfg.seed)
    server = ShardServer(schedule, cfg.data.seq_len, jax.process_index(), n_proc)
    assert server.per_process % jax.local_device_count() == 0, (
        server.per_process, jax.local_device_count(),
    )  # fmt: skip

    sink = MetricsSink(cfg, raw_cfg, is_main)
    tokens_per_step = cfg.data.global_batch * cfg.data.seq_len
    window_t0 = time.time()
    last_logged = start_step

    for step in range(start_step, cfg.steps):
        idx = _global_token_batch(server.local_batch(step), mesh, cfg.data.global_batch)
        resid = harvest(prefix, idx)
        state, metrics = step_fn(state, frozen, resid, random.fold_in(run_key, step))

        now_step = step + 1
        if now_step % cfg.cadence.log_every == 0 or now_step == cfg.steps:
            jax.block_until_ready(metrics["total"])
            dt = time.time() - window_t0
            per_step = dt / max(now_step - last_logged, 1)
            last_logged = now_step
            record = {k: float(v) for k, v in metrics.items()}
            record["step_time_s"] = per_step
            record["tok_per_s"] = tokens_per_step / per_step
            record["tok_per_s_per_gpu"] = tokens_per_step / per_step / ndev
            sink.log(now_step, record)
            window_t0 = time.time()

        if now_step % cfg.cadence.save_every == 0 or now_step == cfg.steps or _sigterm_received:
            save_state(mgr, now_step, state)
            if is_main:
                print(f"checkpoint saved @ step {now_step}", flush=True)
            window_t0 = time.time()
        if _sigterm_received:
            if is_main:
                print("SIGTERM: checkpoint saved, exiting for requeue", flush=True)
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    _install_sigterm_flag()
    init_distributed()
    mesh = dp_mesh()
    cfg = load_config(args.config)

    is_main = jax.process_index() == 0
    if is_main:
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_copy = cfg.run_dir / "config.yaml"
        if cfg_copy.exists():
            assert cfg_copy.read_text() == args.config.read_text(), (
                f"{cfg_copy} differs from {args.config} — refusing to resume with a changed config"
            )
        else:
            cfg_copy.write_text(args.config.read_text())
        print(
            f"run {cfg.run_name} | {mesh.devices.size} GPU / {jax.process_count()} proc | "
            f"B={cfg.data.global_batch} seq={cfg.data.seq_len} "
            f"layers={cfg.target.first_layer}..{cfg.target.last_layer} C={cfg.target.C} "
            f"steps={cfg.steps}",
            flush=True,
        )

    llama_cfg = llama31_8b_config()
    rng = LayerRange(cfg.target.first_layer, cfg.target.last_layer)
    lm = llama_decomposed_lm(llama_cfg, rng, cfg.target.C)
    frozen = replicate_target(load_target_from_hf(cfg.target.model_name, llama_cfg, rng), mesh)
    prefix = jax.device_put(
        load_prefix_from_hf(cfg.target.model_name, llama_cfg, rng),
        NamedSharding(mesh, P()),
    )

    train(cfg, yaml.safe_load(args.config.read_text()), lm, frozen, prefix, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
