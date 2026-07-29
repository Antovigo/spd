"""Toy-model-redundancy PD experiment: YAML -> `Trainer` glue, plus the `SavedToyModelRedundancyRun` reload class.

Decomposes the `toy_model_redundancy` toy's block MLP matrices with KL recon
over its token distribution. Run via `python -m param_decomp_lab.experiments.toy_model_redundancy.run <config.yaml>`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import fire
import torch
from pydantic import Field
from torch.utils.data import DataLoader, IterableDataset

from param_decomp.base_config import BaseConfig
from param_decomp.component_model import ComponentModel
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl, run_batch_passthrough
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import get_device
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.utils import (
    EXPERIMENT_CONFIG_FILENAME,
    ExperimentConfig,
    init_pd_run,
)
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.seed import set_seed
from param_decomp_lab.toy_models.toy_model_redundancy_copy import ToyModelRedundancyCopyTransformer


class ToyModelRedundancyTargetConfig(BaseConfig):
    run_path: str = Field(..., description="Run dir of a trained toy_model_redundancy.")


class ToyModelRedundancyDataConfig(BaseConfig):
    """Uniform random token sequences; shapes come from the toy's own config."""


class ToyModelRedundancyExperimentConfig(
    ExperimentConfig[ToyModelRedundancyTargetConfig, ToyModelRedundancyDataConfig]
):
    pass


def resolve_toy_run_dir(run_path: str) -> Path:
    path = Path(run_path).expanduser()
    if not path.exists():
        path = PARAM_DECOMP_OUT_DIR / "runs" / run_path
    assert path.exists(), f"toy run not found: {run_path}"
    return path


AnyRedundancyToy = ToyModelRedundancyCopyTransformer


def build_target(target_cfg: ToyModelRedundancyTargetConfig) -> AnyRedundancyToy:
    run_dir = resolve_toy_run_dir(target_cfg.run_path)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg.get("kind") == "copy_attn", f"not a copy-toy run: {run_dir}"
    target_model = ToyModelRedundancyCopyTransformer.from_run_dir(run_dir)
    target_model.eval()
    return target_model


class _TokenDataset(IterableDataset[torch.Tensor]):
    """Infinite uniform input batches from the toy's own sampler, on `device`."""

    def __init__(self, toy: AnyRedundancyToy, batch_size: int, device: str):
        self.toy, self.batch_size, self.device = toy, batch_size, device

    @override
    def __iter__(self):  # noqa: ANN204
        while True:
            yield self.toy.sample_inputs(self.batch_size).to(self.device)


def build_toy_model_redundancy_loader(
    target_cfg: ToyModelRedundancyTargetConfig, *, device: str, batch_size: int
) -> DataLoader[Any]:
    return DataLoader(_TokenDataset(build_target(target_cfg), batch_size, device), batch_size=None)


@dataclass(frozen=True)
class SavedToyModelRedundancyRun:
    """Handle to a completed toy-model-redundancy PD run on disk."""

    cfg: ToyModelRedundancyExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedToyModelRedundancyRun":
        files = resolve_run_files(
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=ToyModelRedundancyExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=run_batch_passthrough,
        )


def _build_eval_loop(cfg: ToyModelRedundancyExperimentConfig, device: str) -> EvalLoop | None:
    if cfg.eval is None:
        return None
    return EvalLoop(
        loader=build_toy_model_redundancy_loader(
            cfg.target, device=device, batch_size=cfg.eval.batch_size
        ),
        metrics=[EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics],
        n_steps=cfg.eval.n_steps,
        every=cfg.eval.every,
        slow_every=cfg.eval.slow_every,
        slow_on_first_step=cfg.eval.slow_on_first_step,
    )


def main(
    config_path: str | Path,
    *,
    group: str | None = None,
    tags: str | None = None,
    run_id: str | None = None,
) -> None:
    """Run a toy-model-redundancy PD experiment end-to-end from a YAML config.

    `run_id` names the run dir (convention: `toy_model_redundancy/<details>`); omitted, a
    random id is allocated.
    """
    cfg = ToyModelRedundancyExperimentConfig.from_file(config_path)
    assert cfg.nontarget is None, "the redundancy toy has no nontarget distribution"

    set_seed(cfg.pd.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    target_model = build_target(cfg.target).to(device)
    train_loader = build_toy_model_redundancy_loader(
        cfg.target, device=device, batch_size=cfg.pd.batch_size
    )
    eval_loop = _build_eval_loop(cfg, device)

    sink = init_pd_run(cfg, group=group, tags=tags, run_id=run_id)
    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=run_batch_passthrough,
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(train_loader, sink, cfg.cadence, eval_loop)
    finally:
        sink.finish()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
