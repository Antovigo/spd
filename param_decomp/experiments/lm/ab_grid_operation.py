"""Binding and execution of the LM `(a, b)`-grid snapshot eval."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from param_decomp.core.built_run import TargetSites
from param_decomp.core.ci_fn.interface import CIFn, CIRole
from param_decomp.core.components import ComponentStacks
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import CaptureKeys, PlacedModel
from param_decomp.core.run import StandaloneOperation, StandaloneOperationPlan
from param_decomp.experiments.lm.ab_grid_dataset import (
    ABGridStep,
    ab_grid_payload,
    ab_grid_split_payload,
    collect_ab_grid_snapshot,
    make_ab_grid_step,
    thumbnail_factor,
    write_ab_grid_snapshot,
)
from param_decomp.experiments.lm.arithmetic_eval import ArithmeticGrid
from param_decomp.experiments.lm.arithmetic_eval_operation import global_arithmetic_probe
from param_decomp.experiments.lm.arithmetic_probe import build_arithmetic_probe
from param_decomp.experiments.lm.eval_config import ABGridDatasetConfig
from param_decomp.experiments.lm.eval_context import LMEvalPass
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.targets.transformer import hf_snapshot_dir


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
    model: PlacedModel[Any, Any, Any, Any, Any]
    chunks: tuple[tuple[LMBatchWithDocuments, int], ...]
    grid: ArithmeticGrid
    n_prompts: int
    positions: tuple[int, ...]
    seq_len: int
    mean_ci_floor: float
    run_dir: Path
    writes_snapshots: bool
    """Process 0 writes the snapshot; every rank joins the collective pass and reports the
    same scalars."""

    def run(self, components: ComponentStacks, ci_fn: CIFn[Any], now_step: int) -> LogRecord:
        snapshot = collect_ab_grid_snapshot(
            self.step,
            self.model,
            components,
            ci_fn,
            self.chunks,
            self.n_prompts,
            self.mean_ci_floor,
        )
        if self.writes_snapshots:
            args = (snapshot, self.grid, self.positions, self.seq_len, now_step, self.mean_ci_floor)
            if thumbnail_factor(self.grid) > 1:
                index, modules = ab_grid_split_payload(*args)
                write_ab_grid_snapshot(self.run_dir, now_step, index, modules)
            else:
                write_ab_grid_snapshot(self.run_dir, now_step, ab_grid_payload(*args))
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
    model: PlacedModel[Any, Any, Any, Any, Any],
    ci_capture_keys: CaptureKeys,
    mesh: Any,
    n_proc: int,
    run_dir: Path,
    roles: tuple[CIRole, ...] = ("output",),
) -> StandaloneOperationPlan[LMEvalPass[Any, Any]]:
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
        step=make_ab_grid_step(model, ci_capture_keys, positions, roles),
        model=model,
        chunks=chunks,
        grid=probe.grid,
        n_prompts=n_prompts,
        positions=positions,
        seq_len=seq_len,
        mean_ci_floor=config.mean_ci_floor,
        run_dir=run_dir,
        writes_snapshots=jax.process_index() == 0,
    )

    def prepare(
        _example_pass: LMEvalPass[Any, Any],
    ) -> StandaloneOperation[LMEvalPass[Any, Any]]:
        def run(eval_pass: LMEvalPass[Any, Any]) -> LogRecord:
            decomposition = eval_pass.decomposition
            return operation.run(decomposition.components, decomposition.ci_fn, eval_pass.now_step)

        return StandaloneOperation(schedule, run)

    return StandaloneOperationPlan(prepare)
