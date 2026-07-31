"""Binding and execution of the fixed-grid LM arithmetic operation."""

from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.core.built_run import TargetSites
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import DecomposedModel
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    EvalOperation,
    MetricsSink,
)
from param_decomp.core.train import TrainState
from param_decomp.experiments.lm.arithmetic_eval import (
    ArithmeticGrid,
    ArithmeticGridStep,
    ArithmeticSelection,
    ComponentActivationModel,
    compute_arithmetic_selection,
    make_arithmetic_grid_step,
    n_alive_scalars,
    render_arithmetic_figures,
)
from param_decomp.experiments.lm.arithmetic_probe import build_arithmetic_probe
from param_decomp.experiments.lm.config import TargetConfig
from param_decomp.experiments.lm.eval import ScalarStep, make_eval_step
from param_decomp.experiments.lm.eval_config import ArithmeticCIGridConfig
from param_decomp.experiments.lm.eval_context import LMEvalContext
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


def _render(
    selection: ArithmeticSelection, grid: ArithmeticGrid, top_k: int, now_step: int
) -> DeferredMediaRecord:
    return DeferredMediaRecord(
        step_key="eval/arithmetic/figure_step",
        step=now_step,
        media={
            f"eval/arithmetic/{key}": value
            for key, value in render_arithmetic_figures(selection, grid, top_k).items()
        },
    )


@dataclass(frozen=True)
class ArithmeticOperation:
    step: ArithmeticGridStep
    probe_eval_step: ScalarStep
    model: ComponentActivationModel
    tokens: jax.Array
    grid: ArithmeticGrid
    n_prompts: int
    thresholds: tuple[float, ...]
    top_k: int
    renderer: BackgroundRenderer

    def run(self, state: TrainState, key: PRNGKeyArray, now_step: int) -> LogRecord:
        selection = compute_arithmetic_selection(
            self.step,
            self.model,
            state.decomposition.components,
            state.decomposition.ci_fn,
            self.tokens,
            self.n_prompts,
            self.thresholds,
            self.top_k,
        )
        scalars = self.probe_eval_step(
            self.model,
            state.decomposition.components,
            state.decomposition.ci_fn,
            self.tokens,
            key,
        )
        self.renderer.submit(partial(_render, selection, self.grid, self.top_k, now_step))
        return {
            **{
                f"eval/arithmetic/{name}": value
                for name, value in n_alive_scalars(selection.active, self.top_k).items()
            },
            **{f"eval/arithmetic/{name}": float(value) for name, value in scalars.items()},
        }


def make_arithmetic_operation(
    config: ArithmeticCIGridConfig,
    schedule: EvalSchedule,
    target: TargetSites,
    model: DecomposedModel,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    assert isinstance(model, ComponentActivationModel), (
        f"arithmetic eval needs masked_component_activations; {type(model).__name__} does not"
    )
    assert isinstance(target, TargetConfig), (
        f"arithmetic eval needs an HF tokenizer; {type(target).__name__} has no model_name"
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(hf_snapshot_dir(target.model_name)), local_files_only=True
    )
    probe = build_arithmetic_probe(config.operation, config.a_range, config.b_range, tokenizer)
    n_prompts = probe.tokens.shape[0]
    ce = config.probe_metrics.ce_kl
    l0 = config.probe_metrics.ci_l0
    pgd = config.probe_metrics.fresh_pgd
    l0_groups = (
        {name: tuple(patterns) for name, patterns in l0.groups.items()}
        if l0.groups is not None
        else None
    )
    fresh_pgd = (
        FreshPGDReconEval(
            name=pgd.name or "PGDReconLoss",
            n_steps=pgd.n_steps,
            step_size=pgd.step_size,
        )
        if pgd is not None
        else None
    )
    operation = ArithmeticOperation(
        step=make_arithmetic_grid_step(model, probe.answer_position, n_prompts),
        probe_eval_step=make_eval_step(
            model,
            ce.rounding_threshold,
            l0.ci_alive_threshold,
            l0_groups,
            fresh_pgd,
            mesh,
            n_valid_rows=n_prompts,
            compiler_options=compiler_options,
        ),
        model=model,
        tokens=global_arithmetic_probe(probe.tokens, mesh, n_proc),
        grid=probe.grid,
        n_prompts=n_prompts,
        thresholds=tuple(config.thresholds),
        top_k=config.top_k,
        renderer=BackgroundRenderer(sink),
    )

    def run(context: LMEvalContext) -> LogRecord:
        key = jax.random.fold_in(run_key, 4 * train_steps + context.pass_index)
        return operation.run(context.state, key, context.now_step)

    return EvalOperation(schedule, run)
