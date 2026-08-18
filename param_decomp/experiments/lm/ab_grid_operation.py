"""Binding and execution of the LM `(a, b)`-grid snapshot eval."""

from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.built_run import TargetSites
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import CaptureKeys, DecomposedModel
from param_decomp.core.run import EvalOperation
from param_decomp.core.train import TrainState
from param_decomp.experiments.lm.ab_grid_dataset import (
    ABGridStep,
    ab_grid_payload,
    collect_ab_grid_snapshot,
    make_ab_grid_step,
    read_applet,
    write_ab_grid_snapshot,
)
from param_decomp.experiments.lm.arithmetic_probe import ArithmeticGrid, build_arithmetic_probe
from param_decomp.experiments.lm.eval_config import ABGridDatasetConfig
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.targets.glu_transformer import hf_snapshot_dir


def global_arithmetic_probe(tokens: np.ndarray, mesh: Mesh, n_proc: int) -> jax.Array:
    n, t = tokens.shape
    n_dev = mesh.devices.size
    pad = (-n) % n_dev
    if pad:
        tokens = np.concatenate([tokens, np.zeros((pad, t), tokens.dtype)], axis=0)
    n_pad = tokens.shape[0]
    per_process = n_pad // n_proc
    assert per_process % jax.local_device_count() == 0, (per_process, jax.local_device_count())
    proc = jax.process_index()
    local = tokens[proc * per_process : (proc + 1) * per_process]
    sharding = NamedSharding(mesh, P(("replicate", "fsdp")))
    return jax.make_array_from_process_local_data(sharding, local, (n_pad, t))


def resolve_positions(authored: list[int] | None, seq_len: int) -> tuple[int, ...]:
    """Config positions (negatives allowed, `None` = the answer position) as forward
    indices into the prompt."""
    raw = authored if authored is not None else [-1]
    assert all(-seq_len <= p < seq_len for p in raw), (
        f"positions {raw} out of range for prompt length {seq_len}"
    )
    positions = tuple(p % seq_len for p in raw)
    assert len(set(positions)) == len(positions), (
        f"positions {raw} collapse to duplicates over prompt length {seq_len}"
    )
    return positions


@dataclass(frozen=True)
class ABGridOperation:
    step: ABGridStep
    model: DecomposedModel
    chunks: tuple[tuple[jax.Array, int], ...]
    grid: ArithmeticGrid
    n_prompts: int
    positions: tuple[int, ...]
    seq_len: int
    mean_ci_floor: float
    run_dir: Path
    applet: bytes
    writes_snapshots: bool
    """Process 0 writes the snapshot; every rank joins the collective pass and reports the
    same scalars."""

    def run(self, state: TrainState, now_step: int) -> LogRecord:
        snapshot = collect_ab_grid_snapshot(
            self.step,
            self.model,
            state.decomposition.components,
            state.decomposition.ci_fn,
            self.chunks,
            self.n_prompts,
            self.mean_ci_floor,
        )
        if self.writes_snapshots:
            write_ab_grid_snapshot(
                self.run_dir,
                now_step,
                ab_grid_payload(
                    snapshot,
                    self.grid,
                    self.positions,
                    self.seq_len,
                    now_step,
                    self.mean_ci_floor,
                ),
                self.applet,
            )
        record: LogRecord = {
            f"eval/ab_grids/saved_components/{site}": float(idx.size)
            for site, idx in snapshot.saved.items()
        }
        record["eval/ab_grids/saved_components/total"] = float(
            sum(idx.size for idx in snapshot.saved.values())
        )
        return record


def make_ab_grid_operation(
    config: ABGridDatasetConfig,
    schedule: EvalSchedule,
    target: TargetSites,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
    n_proc: int,
    run_dir: Path,
) -> EvalOperation[LMEvalContext]:
    assert isinstance(target, TargetConfig), (
        f"the ab grid needs an HF tokenizer; {type(target).__name__} has no model_name"
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(hf_snapshot_dir(target.model_name)), local_files_only=True
    )
    probe = build_arithmetic_probe(config.operation, config.a_range, config.b_range, tokenizer)
    n_prompts, seq_len = probe.tokens.shape
    positions = resolve_positions(config.positions, seq_len)
    chunk_prompts = config.chunk_prompts or n_prompts
    chunks = tuple(
        (global_arithmetic_probe(probe.tokens[start : start + chunk_prompts], mesh, n_proc), rows)
        for start in range(0, n_prompts, chunk_prompts)
        if (rows := min(chunk_prompts, n_prompts - start))
    )
    operation = ABGridOperation(
        step=make_ab_grid_step(model, ci_capture_keys, positions),
        model=model,
        chunks=chunks,
        grid=probe.grid,
        n_prompts=n_prompts,
        positions=positions,
        seq_len=seq_len,
        mean_ci_floor=config.mean_ci_floor,
        run_dir=run_dir,
        applet=read_applet(),
        writes_snapshots=jax.process_index() == 0,
    )

    def run(context: LMEvalContext) -> LogRecord:
        return operation.run(context.state, context.now_step)

    return EvalOperation(schedule=schedule, run=run)
