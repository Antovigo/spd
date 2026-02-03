"""Run evaluations on a decomposed SPD model checkpoint."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fire
import torch
import wandb
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import (
    Config,
    EvalConfig,
    LMTaskConfig,
    ResidMLPTaskConfig,
    TMSTaskConfig,
)
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.eval import evaluate
from spd.experiments.lm.prompts_dataset import create_prompts_data_loader
from spd.experiments.resid_mlp.models import ResidMLPTargetRunInfo
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.experiments.tms.models import TMSTargetRunInfo
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.utils.general_utils import set_seed


def prefixed_local_log(data: dict[str, Any], step: int, out_dir: Path, prefix: str) -> None:
    """Save metrics and figures with a configurable prefix."""
    metrics_file = out_dir / f"{prefix}metrics.jsonl"
    metrics_file.touch(exist_ok=True)

    fig_dir = out_dir / f"{prefix}figures"
    fig_dir.mkdir(exist_ok=True)

    metrics_without_images: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, Image.Image):
            filename = f"{k.replace('/', '_')}_{step}.png"
            v.save(fig_dir / filename)
            tqdm.write(f"Saved figure {k} to {fig_dir / filename}")
        elif isinstance(v, wandb.plot.CustomChart):
            json_path = fig_dir / f"{k.replace('/', '_')}_{step}.json"
            payload = {"columns": list(v.table.columns), "data": list(v.table.data), "step": step}
            with open(json_path, "w") as f:
                json.dump(payload, f, default=str)
            tqdm.write(f"Saved custom chart data {k} to {json_path}")
        else:
            metrics_without_images[k] = v

    with open(metrics_file, "a") as f:
        f.write(json.dumps({"step": step, **metrics_without_images}) + "\n")


def create_tms_eval_loader(
    run_config: Config,
    batch_size: int,
    device: str,
) -> DataLoader[Any]:
    """Create TMS eval data loader."""
    task_config = run_config.task_config
    assert isinstance(task_config, TMSTaskConfig)
    assert run_config.pretrained_model_path is not None

    target_run_info = TMSTargetRunInfo.from_path(run_config.pretrained_model_path)
    synced_inputs = target_run_info.config.synced_inputs

    dataset = SparseFeatureDataset(
        n_features=target_run_info.config.tms_model_config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
        active_indices=task_config.active_indices,
    )
    return DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_resid_mlp_eval_loader(
    run_config: Config,
    batch_size: int,
    device: str,
) -> DataLoader[Any]:
    """Create ResidMLP eval data loader."""
    task_config = run_config.task_config
    assert isinstance(task_config, ResidMLPTaskConfig)
    assert run_config.pretrained_model_path is not None

    target_run_info = ResidMLPTargetRunInfo.from_path(run_config.pretrained_model_path)
    synced_inputs = target_run_info.config.synced_inputs

    dataset = ResidMLPDataset(
        n_features=target_run_info.config.resid_mlp_model_config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=task_config.data_generation_type,
        synced_inputs=synced_inputs,
        active_indices=task_config.active_indices,
    )
    return DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_lm_eval_loader(
    run_config: Config,
    batch_size: int,
) -> DataLoader[Any] | Iterable[dict[str, Tensor]]:
    """Create LM eval data loader."""
    task_config = run_config.task_config
    assert isinstance(task_config, LMTaskConfig)

    if task_config.prompts_file is not None:
        assert run_config.tokenizer_name is not None
        loader, _ = create_prompts_data_loader(
            prompts_file=Path(task_config.prompts_file),
            tokenizer_name=run_config.tokenizer_name,
            max_seq_len=task_config.max_seq_len,
            batch_size=batch_size,
            dist_state=None,
            seed=0,
        )
        return loader

    assert task_config.dataset_name is not None, (
        "task_config must have either prompts_file or dataset_name set"
    )
    eval_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=run_config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=None,
    )
    loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=0,
        dist_state=None,
    )
    return loader


def create_eval_loader(
    run_config: Config,
    batch_size: int,
    device: str,
) -> DataLoader[Any] | Iterable[dict[str, Tensor]]:
    """Dispatch data loader creation by task type."""
    match run_config.task_config:
        case TMSTaskConfig():
            return create_tms_eval_loader(run_config, batch_size, device)
        case ResidMLPTaskConfig():
            return create_resid_mlp_eval_loader(run_config, batch_size, device)
        case LMTaskConfig():
            return create_lm_eval_loader(run_config, batch_size)
        case _:
            raise ValueError(f"Unsupported task config type: {type(run_config.task_config)}")


def main(
    checkpoint_path: str,
    eval_config_path: str,
    prefix: str = "additional_",
) -> None:
    """Run evaluations on a decomposed SPD model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (e.g., /path/to/run_dir/model_5000.pth)
            or wandb path (e.g., wandb:entity/project/runs/run_id)
        eval_config_path: Path to YAML file specifying evals to run
        prefix: Prefix for output files (default: "additional_")
    """
    # Load run info and eval config
    run_info = SPDRunInfo.from_path(checkpoint_path)
    run_config = run_info.config
    eval_config = EvalConfig.from_file(eval_config_path)

    logger.info(f"Loaded run config from {checkpoint_path}")
    logger.info(f"Loaded eval config from {eval_config_path}")

    # Determine output directory (same as checkpoint directory)
    out_dir = run_info.checkpoint_path.parent
    logger.info(f"Output directory: {out_dir}")

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(run_config.seed)

    # Load the component model
    logger.info("Loading component model...")
    model = ComponentModel.from_run_info(run_info)
    model = model.to(device)
    model.eval()

    # Create eval data loader
    logger.info("Creating eval data loader...")
    eval_loader = create_eval_loader(
        run_config=run_config,
        batch_size=eval_config.eval_batch_size,
        device=device,
    )
    eval_iterator = loop_dataloader(eval_loader)

    # Run evaluation
    logger.info(f"Running evaluation with {eval_config.n_eval_steps} steps...")
    with torch.no_grad():
        metrics = evaluate(
            eval_metric_configs=eval_config.eval_metric_configs,
            model=model,
            eval_iterator=eval_iterator,
            device=device,
            run_config=run_config,
            slow_step=True,
            n_eval_steps=eval_config.n_eval_steps,
            current_frac_of_training=1.0,
        )

    # Log results
    prefixed_local_log(data=metrics, step=0, out_dir=out_dir, prefix=prefix)
    logger.info(f"Saved metrics to {out_dir / f'{prefix}metrics.jsonl'}")

    # Print summary of numeric metrics
    logger.info("Evaluation results:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, int | float):
            logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
