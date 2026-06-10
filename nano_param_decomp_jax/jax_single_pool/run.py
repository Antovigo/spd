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
from jax_single_pool.config import ExperimentConfig, load_config
from jax_single_pool.data import BatchSchedule, ShardServer, scan_shards
from jax_single_pool.eval import make_eval_step
from jax_single_pool.llama8b import (
    LayerRange,
    Prefix,
    Target,
    llama31_8b_config,
    llama_decomposed_lm,
    load_prefix_from_hf,
    load_target_from_hf,
    prefix_residual,
)
from jax_single_pool.llama8b_sharding import replicate_target
from jax_single_pool.lm import DecomposedLM
from jax_single_pool.run_state import build_optimizers, init_train_state
from jax_single_pool.sharding import dp_mesh, init_distributed
from jax_single_pool.train import (
    TrainState,
    make_faith_warmup_step,
    make_train_step,
    subset_chunk_plan,
)

_sigterm_received = False


def _install_sigterm_flag() -> None:
    def handler(_signum: int, _frame: FrameType | None) -> None:
        global _sigterm_received
        _sigterm_received = True

    signal.signal(signal.SIGTERM, handler)


def _ensure_global(tree: object, mesh: Mesh) -> object:
    """Re-materialize the NON-mesh array leaves (eagerly created scalars: step
    counters, Adam counts) as well-formed GLOBAL replicated arrays via an identity
    jit. Multi-controller orbax can only save global arrays — and an eager
    `device_put(local, replicated-NamedSharding)` yields arrays whose
    `addressable_shards` raise (jax 0.10 multi-process), while jit outputs with the
    same sharding are well-formed.

    Leaves that already carry a NamedSharding pass through UNTOUCHED: routing them
    through the identity jit re-materializes the whole state in one executable,
    which OOM'd at the multi-chunk config's ~110 GB global state (job 50458,
    168 GiB alloc in jit__identity_fn)."""
    repl = NamedSharding(mesh, P())

    def is_mesh_placed(a: object) -> bool:
        return eqx.is_array(a) and isinstance(a.sharding, NamedSharding)  # pyright: ignore[reportAttributeAccessIssue]

    mesh_placed, stragglers = eqx.partition(tree, is_mesh_placed)
    straggler_shardings = jax.tree.map(lambda _a: repl, stragglers)
    fixed = jax.jit(lambda t: t, out_shardings=straggler_shardings)(stragglers)
    return eqx.combine(mesh_placed, fixed)


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
        record = {
            _METRIC_KEYS.get(k, f"train/{k}" if k.startswith("grad_norms/") else k): v
            for k, v in record.items()
        }  # keys already starting "train/" pass through verbatim
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
    opt_vu, opt_ci, (sched_vu, sched_ci) = build_optimizers(cfg)

    key = random.PRNGKey(cfg.seed)
    init_key, src_key, run_key = random.split(key, 3)
    state = _ensure_global(init_train_state(cfg, lm, opt_vu, opt_ci, init_key, src_key, mesh), mesh)
    assert isinstance(state, TrainState)
    assert isinstance(state, TrainState)

    checkpoint_manager = make_checkpoint_manager(cfg.run_dir / "ckpts", cfg.cadence.keep_last)
    restored = restore_latest(checkpoint_manager, state)
    if restored is not None:
        state, ckpt_step = restored
        start_step = ckpt_step
        if is_main:
            print(f"resumed from checkpoint step {ckpt_step}", flush=True)
    else:
        start_step = 0
        if cfg.faith_warmup.steps > 0:
            faith_warmup_optimizer = optax.adamw(cfg.faith_warmup.lr, weight_decay=0.0)
            faith_warmup_opt_state = faith_warmup_optimizer.init(
                eqx.filter(state.components, eqx.is_array)
            )
            faith_warmup_step = make_faith_warmup_step(lm, faith_warmup_optimizer)
            warmed_components = state.components
            t0 = time.time()
            faith_warmup_loss = None
            for _ in range(cfg.faith_warmup.steps):
                warmed_components, faith_warmup_opt_state, faith_warmup_loss = faith_warmup_step(
                    warmed_components, faith_warmup_opt_state, frozen
                )
            assert faith_warmup_loss is not None
            jax.block_until_ready(faith_warmup_loss)
            new_opt_vu = _ensure_global(
                opt_vu.init(eqx.filter(warmed_components, eqx.is_array)), mesh
            )
            state = state._replace(components=warmed_components, components_opt_state=new_opt_vu)
            if is_main:
                print(
                    f"faith warmup: {cfg.faith_warmup.steps} steps in {time.time() - t0:.0f}s, "
                    f"final faith {float(faith_warmup_loss):.3e}",
                    flush=True,
                )
        save_state(checkpoint_manager, 0, state)

    step_fn = make_train_step(
        lm=lm,
        coeffs=cfg.losses,
        imp_cfg=cfg.imp_min,
        src_cfg=cfg.ppgd,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=cfg.steps,
        recon_plan=subset_chunk_plan(lm.site_names, cfg.recon.sites_per_chunk, cfg.recon.n_samples),
        remat_recon_forwards=cfg.recon.remat_forwards,
        mesh=mesh,
    )

    def _harvest(prefix_weights: Prefix, token_ids: jax.Array) -> jax.Array:
        residual = prefix_residual(prefix_weights, token_ids)
        return jax.lax.with_sharding_constraint(residual, NamedSharding(mesh, P("dp")))

    harvest = jax.jit(_harvest)

    schedule = BatchSchedule(scan_shards(cfg.data.dir), cfg.data.global_batch, cfg.seed)
    server = ShardServer(schedule, cfg.data.seq_len, jax.process_index(), n_proc)
    assert server.per_process % jax.local_device_count() == 0, (
        server.per_process, jax.local_device_count(),
    )  # fmt: skip

    # Eval mirrors the torch reference's `eval_split: train`: an independent stream over
    # the SAME corpus (own seed), advanced one block of `n_steps` batches per eval pass.
    eval_step_fn = None
    eval_server = None
    if cfg.eval is not None:
        assert cfg.eval.every % cfg.cadence.log_every == 0, (
            "eval must land on a train-log step: the tok/s window resets after eval, so a "
            "mid-window eval would corrupt the next step-time estimate"
        )
        eval_schedule = BatchSchedule(scan_shards(cfg.data.dir), cfg.eval.batch_size, cfg.seed + 1)
        eval_server = ShardServer(eval_schedule, cfg.data.seq_len, jax.process_index(), n_proc)
        assert eval_server.per_process % jax.local_device_count() == 0, (
            eval_server.per_process, jax.local_device_count(),
        )  # fmt: skip
        eval_pgd = (cfg.eval.pgd.n_steps, cfg.eval.pgd.step_size) if cfg.eval.pgd else None
        eval_step_fn = make_eval_step(
            lm, cfg.eval.rounding_threshold, cfg.eval.ci_alive_threshold, eval_pgd, mesh
        )

    sink = MetricsSink(cfg, raw_cfg, is_main)
    tokens_per_step = cfg.data.global_batch * cfg.data.seq_len
    window_t0 = time.time()
    last_logged = start_step

    for step in range(start_step, cfg.steps):
        token_ids = _global_token_batch(server.local_batch(step), mesh, cfg.data.global_batch)
        residual = harvest(prefix, token_ids)
        state, metrics = step_fn(state, frozen, residual, random.fold_in(run_key, step))

        now_step = step + 1
        dense = cfg.cadence.dense_log_phase
        log_now = (
            now_step % cfg.cadence.log_every == 0
            or now_step == cfg.steps
            or (dense is not None and now_step <= dense.until_step and now_step % dense.every == 0)
        )
        if log_now:
            jax.block_until_ready(metrics["total"])
            dt = time.time() - window_t0
            per_step = dt / max(now_step - last_logged, 1)
            last_logged = now_step
            record = {k: float(v) for k, v in metrics.items()}
            record["step_time_s"] = per_step
            record["tok_per_s"] = tokens_per_step / per_step
            record["tok_per_s_per_gpu"] = tokens_per_step / per_step / ndev
            record["train/schedules/lr/components"] = float(jnp.asarray(sched_vu(now_step)))
            record["train/schedules/lr/ci_fn"] = float(jnp.asarray(sched_ci(now_step)))
            mem_stats = jax.local_devices()[0].memory_stats()
            if mem_stats is not None:
                record["train/mem/peak_gb_per_rank"] = mem_stats["peak_bytes_in_use"] / 1e9
            sink.log(now_step, record)
            window_t0 = time.time()

        if cfg.eval is not None and now_step % cfg.eval.every == 0:
            assert eval_step_fn is not None and eval_server is not None
            eval_pass_index = now_step // cfg.eval.every
            metric_sums: dict[str, jax.Array] = {}
            for j in range(cfg.eval.n_steps):
                eval_tokens = _global_token_batch(
                    eval_server.local_batch(eval_pass_index * cfg.eval.n_steps + j),
                    mesh,
                    cfg.eval.batch_size,
                )
                eval_residual = harvest(prefix, eval_tokens)
                # fold values >= cfg.steps never collide with the train step keys
                eval_key = random.fold_in(
                    run_key, cfg.steps + eval_pass_index * cfg.eval.n_steps + j
                )
                eval_metrics = eval_step_fn(
                    state.components, state.ci_fn, frozen, eval_tokens, eval_residual, eval_key
                )
                for k, v in eval_metrics.items():
                    metric_sums[k] = metric_sums.get(k, jnp.zeros(())) + v
            eval_record = {f"eval/{k}": float(v) / cfg.eval.n_steps for k, v in metric_sums.items()}
            sink.log(now_step, eval_record)
            if is_main:
                headline = {
                    k: eval_record[f"eval/{k}"]
                    for k in ("ce_kl/kl_ci_masked", "ce_kl/ce_unrecovered_ci_masked")
                }
                print(f"[eval @ {now_step}] {headline}", flush=True)
            window_t0 = time.time()

        if now_step % cfg.cadence.save_every == 0 or now_step == cfg.steps or _sigterm_received:
            save_state(checkpoint_manager, now_step, state)
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
