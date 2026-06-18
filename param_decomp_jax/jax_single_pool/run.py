"""The training entrypoint: wrapper YAML -> full SPEC-compliant run on a vendored target.

    jsp-train <wrapper.yaml>     # normally via pd-jax-lm, which stamps run_id into
                                 # the workspace copy; re-running resumes in place

Composition root + the only I/O layer: data serving (`data.py`), HF weight loading,
metrics jsonl (+ optional wandb), orbax checkpoints, SIGTERM-save for SLURM requeue.
The step itself is the pure jit'd `make_train_step`. Resume restores the full
trajectory (SPEC S22) and fast-forwards the data schedule by step arithmetic.

Multi-process: launched one process per GPU under SLURM (`init_distributed`); every
process computes the same global schedule and contributes its local batch slice.
"""

import argparse
import dataclasses
import json
import math
import signal
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from jax_single_pool import llama_simple_mlp
from jax_single_pool.attn_patterns_eval import (
    accumulate_attn_patterns,
    attn_pattern_for,
    attn_patterns_log_entries,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from jax_single_pool.checkpoint import (
    init_from_parent,
    make_checkpoint_manager,
    restore_latest,
    save_state,
)
from jax_single_pool.config import (
    DataConfig,
    ExperimentConfig,
    LlamaSimpleMLPTargetConfig,
    TargetConfig,
    load_config,
    load_run_dir_config,
)
from jax_single_pool.data import BatchSchedule, ShardServer, scan_shards
from jax_single_pool.eval import make_eval_step
from jax_single_pool.hf_http import configure_hf_http_retries
from jax_single_pool.llama8b import (
    first_decomposed_layer,
    llama31_8b_config,
    llama_decomposed_lm,
    llama_site_specs,
    load_prefix_from_hf,
    load_target_from_hf,
    prefix_residual,
)
from jax_single_pool.llama8b_sharding import replicate_target
from jax_single_pool.lm import DecomposedModel
from jax_single_pool.recon import build_recon_terms
from jax_single_pool.run_state import build_optimizers, init_train_state
from jax_single_pool.sharding import dp_mesh, init_distributed
from jax_single_pool.target_aliases import AnyFrozenTarget, AnyPrefix
from jax_single_pool.train import TrainState, make_faith_warmup_step, make_train_step
from param_decomp_config.experiment import ResumeProvenance
from param_decomp_config.wandb_config import flatten_typed_lists

_sigterm_received = False


def _install_sigterm_flag() -> None:
    def handler(_signum: int, _frame: FrameType | None) -> None:
        global _sigterm_received
        _sigterm_received = True

    signal.signal(signal.SIGTERM, handler)


def _enable_persistent_compilation_cache(out_dir: Path) -> Path:
    """Cache compiled XLA executables to a shared-FS dir reused across runs/requeues.

    The ~24-min compile of the chunkwise step is keyed by HLO + backend + topology +
    jax/xla version, so a matching re-compile (requeue, or a fresh run at the same
    config+topology) loads from disk in seconds. The dir is a SIBLING of `runs/` (not
    per-run, not inside the immutable per-run workspace) so every run shares it; all 8N
    ranks point at the same shared-FS path. Only process 0 writes (jax gates the write on
    `process_id == 0` to avoid shared-FS write contention); every rank reads. Must run
    after `init_distributed` (the rank gate reads the distributed state) and before the
    first compile."""
    cache_dir = out_dir.parent / "xla_compilation_cache"
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 60.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir


def _ensure_global[T](tree: T, mesh: Mesh) -> T:
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


# wandb keys match the torch trainer's (`train_step.py` emits `loss/<instance_key>`,
# `optimize.py` prefixes `train/`) so a torch-vs-jax run pair overlays on one panel.
# Recon-term keys arrive from the step already shaped (`loss/<instance_key>`) and are
# train/-prefixed by the sink; this table maps only the step's fixed scalar keys.
_METRIC_KEYS = {
    "total": "train/loss/total",
    "faith": "train/loss/FaithfulnessLoss",
    "imp": "train/loss/ImportanceMinimalityLoss",
    "p_imp": "train/schedules/p_imp",
    "src_lr": "train/schedules/lr/src",
    "step_time_s": "train/perf/step_time_s",
    "tok_per_s": "train/perf/tok_per_s",
    "tok_per_s_per_gpu": "train/perf/tok_per_s_per_gpu",
}


class MetricsSink:
    """Process-0 metrics fan-out: jsonl always, wandb when configured."""

    def __init__(self, cfg: ExperimentConfig, wandb_config: dict[str, object], is_main: bool):
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
                id=cfg.run_id,
                group=cfg.wandb.group,
                tags=list(cfg.wandb.tags),
                resume="allow",
                config=wandb_config,
            )
            # Persist the run's pinned config.yaml as a downloadable wandb run file
            # (parity with the torch trainer's init_pd_run -> wandb.save), not just the
            # flattened wandb.config dict. Pinned to run_dir before train() / wandb.init.
            config_yaml = cfg.run_dir / "config.yaml"
            assert config_yaml.exists(), config_yaml
            wandb.save(str(config_yaml), base_path=str(cfg.run_dir), policy="now")
            # slow_eval/* rides a dedicated step axis (torch convention,
            # infra/wandb.py): pd-offline-eval logs those keys retroactively into
            # this run and CANNOT pass step= (wandb silently drops writes behind
            # the live head). The offline job redefines this — idempotent.
            wandb.define_metric("slow_eval/step")
            wandb.define_metric("slow_eval/*", step_metric="slow_eval/step")
            self._wandb = wandb

    def log(self, step: int, record: dict[str, float]) -> None:
        if self._jsonl is None:
            return
        record = {
            _METRIC_KEYS.get(
                k, f"train/{k}" if k.startswith(("grad_norms/", "loss/", "schedules/")) else k
            ): v
            for k, v in record.items()
        }  # keys already starting "train/" or "eval/" pass through verbatim
        self._jsonl.write(json.dumps({"step": step, **record}) + "\n")
        self._jsonl.flush()
        print(
            f"[step {step}] " + " ".join(f"{k}={v:.4g}" for k, v in record.items()),
            flush=True,
        )
        if self._wandb is not None:
            import wandb.errors

            # CommError catches wandb-server hiccups (a transient outage must not kill a
            # multi-day run) while letting genuine misuse (e.g. a non-dict record) raise.
            try:
                self._wandb.log(record, step=step)
            except wandb.errors.CommError as e:
                print(f"wandb communication error, skipping log: {e}", flush=True)


def assert_finetune_structural_compat(cfg: ExperimentConfig, prov: ResumeProvenance) -> None:
    """Fine-tune requires the parent's decomposition STRUCTURE to match the new config's:
    same sites (names + C) and same ci-fn arch. A changed C / layers / target / ci-fn is a
    different-shaped decomposition and is NOT a fine-tune (the parent's V/U + ci_fn would
    not load onto the new reference). Only LR / coeffs / eps / seq / batch / steps may
    change. Read from the parent's pinned `config.yaml` so the failure is a readable config
    diff, not an opaque orbax tree mismatch."""
    parent_cfg = load_run_dir_config(prov.parent_run_dir)
    parent_sites = tuple((s.name, s.C) for s in parent_cfg.target.sites)
    new_sites = tuple((s.name, s.C) for s in cfg.target.sites)
    assert parent_sites == new_sites, (
        f"fine-tune sites mismatch: parent {parent_sites} != new {new_sites}"
    )
    assert parent_cfg.ci_fn == cfg.ci_fn, (
        f"fine-tune ci-fn arch mismatch: parent {parent_cfg.ci_fn} != new {cfg.ci_fn}"
    )


def _init_or_restore_state(
    cfg: ExperimentConfig,
    lm: DecomposedModel,
    frozen: Any,
    opt_vu: optax.GradientTransformation,
    opt_ci: optax.GradientTransformation,
    init_key: PRNGKeyArray,
    src_key: PRNGKeyArray,
    mesh: Mesh,
    checkpoint_manager: ocp.CheckpointManager,
    is_main: bool,
) -> tuple[TrainState, int] | None:
    """The shared init/restore/finetune/faith-warmup phase (SPEC S21/S22/S33).

    Returns `(state, start_step)`, or `None` when a SIGTERM landed mid-warmup (the caller
    must exit cleanly for requeue — no valid checkpoint exists pre-step-0)."""
    state = _ensure_global(init_train_state(cfg, lm, opt_vu, opt_ci, init_key, src_key, mesh), mesh)

    restored = restore_latest(checkpoint_manager, state)
    if restored is not None:
        state, ckpt_step = restored
        assert int(state.step) == ckpt_step, (int(state.step), ckpt_step)
        if is_main:
            print(f"resumed from checkpoint step {ckpt_step}", flush=True)
        return state, ckpt_step

    if cfg.resume_provenance is not None:
        # Fine-tune init (SPEC S33): own ckpts/ is empty, so this is the FIRST entry, not a
        # requeue — load the parent's trained V/U + ci_fn onto the fresh reference, start a
        # clean schedule from step 0 (fresh optimizer / sources, no faith warmup).
        prov = cfg.resume_provenance
        assert_finetune_structural_compat(cfg, prov)
        state = init_from_parent(prov.parent_run_dir / "ckpts", prov.parent_step, state)
        save_state(checkpoint_manager, 0, state)
        if is_main:
            print(
                f"fine-tune: initialized V/U + ci_fn from {prov.parent_run_dir} "
                f"step {prov.parent_step}; training fresh from step 0",
                flush=True,
            )
        return state, 0

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
            if _sigterm_received:
                # No valid checkpoint exists yet (the step-0 save happens only after warmup
                # completes, and resume skips warmup whenever a checkpoint is present — a
                # partially-warmed step-0 save would resume as if fully warmed). Exit
                # cleanly; the SLURM requeue redoes warmup from scratch.
                if is_main:
                    print("SIGTERM during faith warmup: exiting for requeue", flush=True)
                return None
        assert faith_warmup_loss is not None
        jax.block_until_ready(faith_warmup_loss)
        new_opt_vu = _ensure_global(opt_vu.init(eqx.filter(warmed_components, eqx.is_array)), mesh)
        state = dataclasses.replace(
            state, components=warmed_components, components_opt_state=new_opt_vu
        )
        if is_main:
            print(
                f"faith warmup: {cfg.faith_warmup.steps} steps in {time.time() - t0:.0f}s, "
                f"final faith {float(faith_warmup_loss):.3e}",
                flush=True,
            )
    save_state(checkpoint_manager, 0, state)
    return state, 0


def run_decomposition_training(
    cfg: ExperimentConfig,
    raw_cfg: dict[str, object],
    lm: DecomposedModel,
    frozen: Any,
    sample_batch: Callable[[int], jax.Array],
    eval_fn: Callable[[TrainState, int], dict[str, float]] | None,
    eval_every: int,
    perf_tokens_per_step: int | None,
    mesh: Mesh,
) -> None:
    """The generic VPD decomposition-training engine — the ONE train loop every target
    (LM, TMS, ResidMLP, …) runs through.

    The target supplies only its three injectable seams:

    - `sample_batch(step) -> residual [*leading, d]`: the residual entering the decomposed
      model for `step`, already mesh-placed on `P("dp")`. An LM harvests it from the frozen
      prefix over a parquet token batch; a toy generates it synthetically.
    - `eval_fn(state, now_step) -> dict[str, float]`: an in-loop eval pass run every
      `eval_every` completed steps, its record logged under that step. `None` disables it.
    - `eval_every`: the eval cadence. For an LM this is `cfg.eval.every`; a toy folds its
      cheap target-CI eval onto the `log_every` cadence.

    Everything generic — `init_train_state`, fine-tune init, faith warmup, the recon-grid
    step factory, orbax checkpointing, schedules, SIGTERM-save — lives here. The step
    numerics are identical across targets; only the data source and the eval metric differ.

    `perf_tokens_per_step` drives the tok/s perf record; `None` (toys, where a synthetic
    "token" has no meaning) omits the perf keys."""
    is_main = jax.process_index() == 0
    ndev = mesh.devices.size

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    opt_vu, opt_ci, (sched_vu, sched_ci) = build_optimizers(cfg)

    key = random.PRNGKey(cfg.seed)
    init_key, src_key, run_key = random.split(key, 3)

    checkpoint_manager = make_checkpoint_manager(cfg.run_dir / "ckpts", cfg.cadence.keep_last)
    init = _init_or_restore_state(
        cfg, lm, frozen, opt_vu, opt_ci, init_key, src_key, mesh, checkpoint_manager, is_main
    )
    if init is None:
        return  # SIGTERM mid-warmup: clean exit for requeue
    state, start_step = init

    step_fn = make_train_step(
        lm=lm,
        loss_spec=build_recon_terms(
            cfg.loss_metrics, lm.site_names, cfg.n_mask_samples, cfg.sampling
        ),
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=cfg.steps,
        remat_recon_forwards=cfg.remat_recon_forwards,
        mesh=mesh,
    )

    # the raw torch yaml's runtime block describes the UPSTREAM run (e.g. dp: 32);
    # record what this run actually executes on so wandb never lies about topology.
    # flatten the metric lists into the same flat keys torch logs (E14) so cross-impl
    # wandb config queries line up.
    wandb_config = flatten_typed_lists(
        dict(
            raw_cfg,
            jax_runtime={
                "n_devices": ndev,
                "n_processes": jax.process_count(),
                "remat_recon_forwards": cfg.remat_recon_forwards,
                "run_id": cfg.run_id,
                "run_dir": str(cfg.run_dir),
            },
        )
    )
    sink = MetricsSink(cfg, wandb_config, is_main)
    window_t0 = time.time()
    last_logged = start_step

    for step in range(start_step, cfg.steps):
        residual = sample_batch(step)
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
            for loss_name in ("total", *(k for k in record if k.startswith("loss/"))):
                assert math.isfinite(record[loss_name]), (
                    f"non-finite loss {loss_name!r} at step {now_step}: {record[loss_name]}"
                )
            record["step_time_s"] = per_step
            if perf_tokens_per_step is not None:
                record["tok_per_s"] = perf_tokens_per_step / per_step
                record["tok_per_s_per_gpu"] = perf_tokens_per_step / per_step / ndev
            record["train/schedules/lr/components"] = float(jnp.asarray(sched_vu(now_step)))
            record["train/schedules/lr/ci_fn"] = float(jnp.asarray(sched_ci(now_step)))
            mem_stats = jax.local_devices()[0].memory_stats()
            if mem_stats is not None:
                record["train/mem/peak_gb_per_rank"] = mem_stats["peak_bytes_in_use"] / 1e9
            sink.log(now_step, record)
            window_t0 = time.time()

        if eval_fn is not None and now_step % eval_every == 0 and not _sigterm_received:
            eval_record = eval_fn(state, now_step)
            # A SIGTERM raised DURING the eval pass abandons its partial record unlogged and
            # falls through to the save block (synchronous save of the completed `now_step`).
            if not _sigterm_received:
                sink.log(now_step, eval_record)
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


def train(
    cfg: ExperimentConfig,
    raw_cfg: dict[str, object],
    lm: DecomposedModel,
    frozen: AnyFrozenTarget,
    prefix: AnyPrefix,
    prefix_residual_fn: Callable[[Any, Any], jax.Array],
    mesh: Mesh,
) -> None:
    """The LM composition over the generic engine: a parquet `sample_batch` (harvest the
    residual from the frozen prefix) and the CEandKL / CI-L0 / PGD / attn-patterns
    `eval_fn`."""
    data = cfg.data
    assert isinstance(data, DataConfig), "train() is the LM (parquet) data path"
    n_proc = jax.process_count()
    ndev = mesh.devices.size
    assert data.global_batch % ndev == 0, (data.global_batch, ndev)
    is_main = jax.process_index() == 0

    key = random.PRNGKey(cfg.seed)
    _, _, run_key = random.split(key, 3)

    def _harvest(prefix_weights: Any, inputs: Any) -> jax.Array:
        residual = prefix_residual_fn(prefix_weights, inputs)
        return jax.lax.with_sharding_constraint(residual, NamedSharding(mesh, P("dp")))

    harvest = jax.jit(_harvest)

    schedule = BatchSchedule(scan_shards(data.dir), data.global_batch, cfg.seed)
    server = ShardServer(schedule, data.seq_len, jax.process_index(), n_proc)
    assert server.per_process % jax.local_device_count() == 0, (
        server.per_process, jax.local_device_count(),
    )  # fmt: skip

    def sample_batch(step: int) -> jax.Array:
        token_ids = _global_token_batch(server.local_batch(step), mesh, data.global_batch)
        return harvest(prefix, token_ids)

    eval_fn = None
    eval_every = cfg.steps + 1  # unreachable cadence when eval is disabled
    if cfg.eval is not None:
        assert cfg.eval.every % cfg.cadence.log_every == 0, (
            "eval must land on a train-log step: the tok/s window resets after eval, so a "
            "mid-window eval would corrupt the next step-time estimate"
        )
        eval_every = cfg.eval.every
        eval_fn = _make_lm_eval_fn(cfg, lm, frozen, prefix, harvest, run_key, mesh, n_proc, is_main)

    run_decomposition_training(
        cfg=cfg,
        raw_cfg=raw_cfg,
        lm=lm,
        frozen=frozen,
        sample_batch=sample_batch,
        eval_fn=eval_fn,
        eval_every=eval_every,
        perf_tokens_per_step=data.global_batch * data.seq_len,
        mesh=mesh,
    )


def _make_lm_eval_fn(
    cfg: ExperimentConfig,
    lm: DecomposedModel,
    frozen: AnyFrozenTarget,
    prefix: AnyPrefix,
    harvest: Callable[[Any, Any], jax.Array],
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    is_main: bool,
) -> Callable[[TrainState, int], dict[str, float]]:
    """The LM in-loop eval pass closure (CEandKL / CI-L0 / PGD / attn-patterns), keyed
    deterministically off `(run_key, now_step)` so it is bit-identical to the pre-engine
    inline loop. Mirrors the torch `eval_split: train` stream: an independent reader over
    the SAME corpus (own seed), advanced one block of `n_steps` batches per eval pass."""
    assert cfg.eval is not None
    data = cfg.data
    assert isinstance(data, DataConfig)
    eval_schedule = BatchSchedule(scan_shards(data.dir), cfg.eval.batch_size, cfg.seed + 1)
    eval_server = ShardServer(eval_schedule, data.seq_len, jax.process_index(), n_proc)
    assert eval_server.per_process % jax.local_device_count() == 0, (
        eval_server.per_process, jax.local_device_count(),
    )  # fmt: skip
    eval_pgd = (cfg.eval.pgd.n_steps, cfg.eval.pgd.step_size) if cfg.eval.pgd else None
    eval_step_fn = make_eval_step(
        lm,
        cfg.eval.rounding_threshold,
        cfg.eval.ci_alive_threshold,
        cfg.eval.l0_groups,
        eval_pgd,
        mesh,
    )
    attn_steps: dict[str, Any] = {}
    if cfg.eval.attn_patterns is not None:
        pattern_fn = attn_pattern_for(frozen)
        if cfg.eval.attn_patterns.ci_masked:
            attn_steps["CIMaskedAttnPatternsReconLoss"] = make_ci_attn_patterns_step(lm, pattern_fn)
        if cfg.eval.attn_patterns.stochastic:
            attn_steps["StochasticAttnPatternsReconLoss"] = make_stochastic_attn_patterns_step(
                lm, pattern_fn, cfg.n_mask_samples, cfg.sampling
            )

    def eval_fn(state: TrainState, now_step: int) -> dict[str, float]:
        assert cfg.eval is not None
        eval_pass_index = now_step // cfg.eval.every
        # uniform-average of per-batch scalars; mean-safe vs torch's accumulate-then-
        # compute() ONLY because every emitted key is a per-batch reduction that torch also
        # averages across batches AND eval batches are uniform (B, T). See eval.py's module
        # docstring for the per-key parity argument (cites SPEC S8/D2).
        metric_sums: dict[str, jax.Array] = {}
        eval_residuals: list[jax.Array] = []
        for j in range(cfg.eval.n_steps):
            if _sigterm_received:
                break
            eval_tokens = _global_token_batch(
                eval_server.local_batch(eval_pass_index * cfg.eval.n_steps + j),
                mesh,
                cfg.eval.batch_size,
            )
            eval_residual = harvest(prefix, eval_tokens)
            eval_residuals.append(eval_residual)
            # fold values >= cfg.steps never collide with the train step keys
            eval_key = random.fold_in(run_key, cfg.steps + eval_pass_index * cfg.eval.n_steps + j)
            eval_metrics = eval_step_fn(
                state.components, state.ci_fn, frozen, eval_tokens, eval_residual, eval_key
            )
            for k, v in eval_metrics.items():
                metric_sums[k] = metric_sums.get(k, jnp.zeros(())) + v
        eval_record = {f"eval/{k}": float(v) / cfg.eval.n_steps for k, v in metric_sums.items()}
        for class_name, attn_step in attn_steps.items():
            # token-weighted (Σ sum_kl / Σ n), NOT the uniform per-batch average above — KL
            # is summed over distributions, divided by their count.
            attn_key = random.fold_in(run_key, 2 * cfg.steps + eval_pass_index)
            reductions = accumulate_attn_patterns(
                attn_step, state.components, state.ci_fn, frozen, eval_residuals, attn_key
            )
            eval_record |= {
                f"eval/loss/{k}": v
                for k, v in attn_patterns_log_entries(class_name, reductions).items()
            }
        if is_main:
            headline = {
                k: eval_record[f"eval/{k}"]
                for k in ("ce_kl/kl_ci_masked", "ce_kl/ce_unrecovered_ci_masked")
            }
            print(f"[eval @ {now_step}] {headline}", flush=True)
        return eval_record

    return eval_fn


def _pin_config_copy(run_dir: Path, name: str, source: Path) -> None:
    """First run copies `source` into the run dir; resumes byte-compare against it."""
    copy = run_dir / name
    if copy.exists():
        assert copy.read_text() == source.read_text(), (
            f"{copy} differs from {source} — refusing to resume with a changed config"
        )
    else:
        copy.write_text(source.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    _install_sigterm_flag()
    init_distributed()
    # Harden the cold-cache HF weight load against the 8N-rank startup burst before any
    # per-rank Hub call (no-op when huggingface_hub is absent / cache is pre-warmed).
    configure_hf_http_retries()
    mesh = dp_mesh()

    cfg, raw_cfg = load_config(args.config)

    cache_dir = _enable_persistent_compilation_cache(cfg.out_dir)

    is_main = jax.process_index() == 0
    if is_main:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        _pin_config_copy(cfg.run_dir, "config.yaml", args.config)
        print(f"persistent XLA compilation cache: {cache_dir}", flush=True)
        site_summary = " ".join(f"{s.name}:C{s.C}" for s in cfg.target.sites)
        assert isinstance(cfg.data, DataConfig)
        print(
            f"run {cfg.run_name} | {mesh.devices.size} GPU / {jax.process_count()} proc | "
            f"B={cfg.data.global_batch} seq={cfg.data.seq_len} "
            f"sites=[{site_summary}] steps={cfg.steps}",
            flush=True,
        )

    frozen: AnyFrozenTarget
    prefix: AnyPrefix
    prefix_residual_fn: Callable[[Any, Any], jax.Array]
    match cfg.target:
        case TargetConfig():
            llama_cfg = llama31_8b_config()
            lm = llama_decomposed_lm(llama_cfg, llama_site_specs(llama_cfg, cfg.target.sites))
            first_layer = first_decomposed_layer(lm.site_names)
            frozen = replicate_target(
                load_target_from_hf(cfg.target.model_name, llama_cfg, first_layer), mesh
            )
            prefix = jax.device_put(
                load_prefix_from_hf(cfg.target.model_name, llama_cfg, first_layer),
                NamedSharding(mesh, P()),
            )
            prefix_residual_fn = prefix_residual
        case LlamaSimpleMLPTargetConfig():
            cache_dir = llama_simple_mlp.pretrain_cache_dir(cfg.target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            lm = llama_simple_mlp.llama_simple_mlp_decomposed_lm(
                simple_cfg, llama_simple_mlp.site_specs(simple_cfg, cfg.target.sites)
            )
            first_layer = llama_simple_mlp.first_decomposed_layer(lm.site_names)
            frozen = llama_simple_mlp.replicate_frozen(
                llama_simple_mlp.load_target_from_pretrain_cache(
                    cache_dir, simple_cfg, first_layer, jnp.bfloat16
                ),
                mesh,
            )
            prefix = llama_simple_mlp.replicate_frozen(
                llama_simple_mlp.load_prefix_from_pretrain_cache(
                    cache_dir, simple_cfg, first_layer, jnp.bfloat16
                ),
                mesh,
            )
            prefix_residual_fn = llama_simple_mlp.prefix_residual
        case _:
            raise AssertionError(f"jsp-train is LM-only; got target {type(cfg.target).__name__}")

    train(cfg, raw_cfg, lm, frozen, prefix, prefix_residual_fn, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
