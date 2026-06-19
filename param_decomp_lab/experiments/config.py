"""Shared config schema for in-repo experiment YAMLs, plus the shared validation /
run-identity helpers every experiment reuses.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data` types.
The generic engine reads the pydantic `pd` / `cadence` / `runtime` DIRECTLY, so there is
no flattened mirror to build — `assert_canonical_algorithm_config` only VALIDATES that the
schema lives in the subspace the JAX trainer implements (cosine-to-0.1 LR, plain AdamW,
components-only grad clip, …), and `run_instance` / `*_ci_arch` resolve the run identity and
the CI-fn architecture; each experiment's `run.py` assembles the rest (target + data).
"""

import re
from pathlib import Path
from typing import Any, Self

from pydantic import Field, PositiveInt, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.ci_fn_mlp import GlobalMLPCIArch, MLPCIArch
from param_decomp.config import CIFnArch, RunInstance
from param_decomp.configs import (
    AnyEvalMetricConfig,
    Cadence,
    GlobalSharedMlpCiConfig,
    LayerwiseCiConfig,
    OptimizerConfig,
    PDConfig,
    ResumeProvenance,
    RuntimeConfig,
    WandbConfig,
)
from param_decomp.schedule import ScheduleConfig


class EvalConfig(BaseConfig):
    """Eval-pass settings consumed by `EvalLoop`. `slow_every` must be a multiple of `every`."""

    batch_size: PositiveInt
    n_steps: PositiveInt
    every: PositiveInt
    slow_every: PositiveInt
    slow_on_first_step: bool = True
    metrics: list[AnyEvalMetricConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_slow_every_multiple_of_every(self) -> Self:
        assert self.slow_every % self.every == 0, (
            f"slow_every ({self.slow_every}) must be a multiple of every ({self.every})"
        )
        return self


class ExperimentConfig[T: BaseConfig, D: BaseConfig](BaseConfig):
    """Full YAML schema for an in-repo experiment.

    Subclass with concrete `target` / `data` types per experiment:

        class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
            pass

    Omit the `eval:` block to skip eval entirely; omit `wandb:` to skip wandb (the run
    still writes `config.yaml` + checkpoints locally).

    `run_id` / `out_dir` are minted by `pd-lm` at submit time (both `None` in a
    hand-authored config); the stamped workspace copy carries them, and the trainer
    resumes by byte-comparing that pinned copy.
    """

    run_name: str
    """Human-readable display name (the wandb run NAME)."""
    run_id: str | None = None
    """Canonical `p-<8hex>` id (wandb run ID + run-dir name). `None` in a hand-authored
    config; minted + stamped by `pd-lm` at submit time."""
    out_dir: Path | None = None
    """Run-output root (the run dir is `out_dir / run_id`). `None` lets `pd-lm` mint
    `PARAM_DECOMP_OUT_DIR/runs`; set it to override (the llama8b configs use `jax_runs`)."""

    pd: PDConfig
    runtime: RuntimeConfig
    cadence: Cadence
    target: T
    data: D
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    """Set on resumed runs (parent run dir + step); `None` for fresh runs. Lives on the
    config so it flows into `experiment_config.yaml` and `wandb.config` via `init_pd_run`,
    making a resumed run's lineage visible in the wandb UI."""


_RUN_ID_PATTERN = re.compile(r"^p-[0-9a-f]{8}$")


def layerwise_mlp_ci_arch(cfg: "ExperimentConfig[Any, Any]") -> MLPCIArch:
    """Extract the layerwise per-site MLP CI-fn arch (`fn_type=mlp`) from a schema config.
    Reused by the lab toy providers (which build their own runtime `ExperimentConfig`)."""
    ci = cfg.pd.ci_config
    assert isinstance(ci, LayerwiseCiConfig), ci
    assert ci.fn_type == "mlp", f"layerwise CI fn must be fn_type=mlp, got {ci.fn_type}"
    assert ci.hidden_dims, "layerwise MLP CI fn needs at least one hidden layer"
    return MLPCIArch(hidden_dims=tuple(ci.hidden_dims))


def toy_ci_arch(cfg: "ExperimentConfig[Any, Any]") -> CIFnArch:
    """Extract the MLP CI-fn arch for a toy run: the layerwise per-site MLP
    (`mode=layerwise, fn_type=mlp`) or the global shared MLP (`mode=global,
    fn_type=global_shared_mlp`)."""
    ci = cfg.pd.ci_config
    match ci:
        case LayerwiseCiConfig():
            return layerwise_mlp_ci_arch(cfg)
        case GlobalSharedMlpCiConfig():
            assert ci.hidden_dims, "global_shared_mlp CI fn needs at least one hidden layer"
            return GlobalMLPCIArch(hidden_dims=tuple(ci.hidden_dims))
        case _:
            raise AssertionError(f"toy CI fn must be layerwise mlp or global_shared_mlp, got {ci}")


def _assert_cosine_to_tenth(schedule: ScheduleConfig, who: str) -> None:
    """The trainer hardcodes optax cosine decay to 0.1x with no warmup (SPEC S19/S20)."""
    assert schedule.fn_type == "cosine", f"{who}: only cosine lr supported, got {schedule}"
    assert schedule.warmup_pct == 0.0, f"{who}: lr warmup unsupported, got {schedule}"
    assert schedule.final_val_frac == 0.1, f"{who}: final_val_frac must be 0.1, got {schedule}"


def _assert_plain_adamw(optimizer: OptimizerConfig, who: str) -> None:
    assert optimizer.betas == (0.9, 0.999), f"{who}: betas must be (0.9, 0.999)"
    assert optimizer.weight_decay == 0.0, f"{who}: weight_decay must be 0"


def assert_canonical_algorithm_config(cfg: "ExperimentConfig[Any, Any]") -> None:
    """Assert the schema lives in the subspace the JAX trainer implements (the engine then
    reads `pd` / `cadence` DIRECTLY). The numerics-load-bearing constraints: leaky-hard
    sigmoid, delta component, no tied weights, bf16 compute, cosine-to-0.1 LR with no
    warmup, plain AdamW (betas (0.9, 0.999), no weight decay), components-only grad clip,
    and a fully-specified checkpoint cadence."""
    assert cfg.pd.sigmoid_type == "leaky_hard", cfg.pd.sigmoid_type
    assert cfg.pd.use_delta_component and cfg.pd.tied_weights is None
    assert cfg.runtime.autocast_bf16, "JAX trainer computes in bf16 (autocast analog)"
    assert cfg.pd.faithfulness_warmup_weight_decay == 0.0

    vu_opt = cfg.pd.components_optimizer
    ci_opt = cfg.pd.ci_fn_optimizer
    _assert_cosine_to_tenth(vu_opt.lr_schedule, "components_optimizer")
    _assert_cosine_to_tenth(ci_opt.lr_schedule, "ci_fn_optimizer")
    _assert_plain_adamw(vu_opt, "components_optimizer")
    _assert_plain_adamw(ci_opt, "ci_fn_optimizer")
    assert vu_opt.grad_clip_norm is not None, "components grad clip is part of the method"
    assert ci_opt.grad_clip_norm is None, "CI-fn grad clip unsupported"

    cadence = cfg.cadence
    assert cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None, cadence


def run_instance(cfg: "ExperimentConfig[Any, Any]") -> RunInstance:
    """The resolved run identity + logging lineage. `run_id` / `out_dir` are minted +
    stamped by `pd-lm`; a config reaching the trainer must carry both."""
    assert cfg.run_id is not None and _RUN_ID_PATTERN.match(cfg.run_id), (
        f"run_id must be p-<8hex>, got {cfg.run_id!r} (pd-lm stamps it at submit)"
    )
    assert cfg.out_dir is not None, "out_dir unset (pd-lm mints it at submit)"
    return RunInstance(
        run_name=cfg.run_name,
        run_id=cfg.run_id,
        out_dir=cfg.out_dir,
        wandb=cfg.wandb,
        resume_provenance=cfg.resume_provenance,
    )
