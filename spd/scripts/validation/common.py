"""Shared helpers for the LM validation scripts."""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import Config, LMTaskConfig, TaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.prompts_dataset import (
    StaticBatchLoader,
    create_prompts_data_loader,
)
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import ModelPath


def load_spd_run(path: ModelPath) -> tuple[ComponentModel, Config, Path]:
    """Load the ComponentModel, its config, and the directory the checkpoint lives in."""
    run_info = SPDRunInfo.from_path(path)
    spd_model = ComponentModel.from_run_info(run_info)
    spd_model.eval()
    return spd_model, run_info.config, run_info.checkpoint_path.parent


def resolve_task_config(config: Config, use_nontarget: bool) -> LMTaskConfig:
    """Return the target or nontarget LM task config, erroring if nontarget is missing."""
    task_config: TaskConfig | None
    if use_nontarget:
        assert config.nontarget_task_config is not None, (
            "--nontarget was passed but the config has no nontarget_task_config"
        )
        task_config = config.nontarget_task_config
    else:
        task_config = config.task_config

    assert isinstance(task_config, LMTaskConfig), (
        f"Validation scripts only support LM tasks, got {type(task_config).__name__}"
    )
    return task_config


def is_prompt_task(task_config: LMTaskConfig) -> bool:
    """Prompt-based LM tasks run exactly one batch containing every prompt."""
    return task_config.prompts_file is not None


def build_lm_loader(
    task_config: LMTaskConfig, config: Config
) -> DataLoader[Any] | StaticBatchLoader:
    """Build a DataLoader for the given LMTaskConfig using the decomposition's batch size."""
    assert config.tokenizer_name is not None, "LM tasks need config.tokenizer_name"

    if task_config.prompts_file is not None:
        prompts_file = Path(task_config.prompts_file).expanduser()
        n_prompts = sum(1 for ln in prompts_file.read_text().splitlines() if ln.strip())
        loader, _ = create_prompts_data_loader(
            prompts_file=prompts_file,
            tokenizer_name=config.tokenizer_name,
            max_seq_len=task_config.max_seq_len,
            batch_size=n_prompts,
        )
        return loader

    assert task_config.dataset_name is not None
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=task_config.dataset_seed,
    )
    loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=config.batch_size,
        buffer_size=task_config.buffer_size,
    )
    return loader


def iterate_input_ids(
    loader: DataLoader[Any] | StaticBatchLoader, device: torch.device
) -> Iterator[Tensor]:
    for batch in loader:
        assert isinstance(batch, dict) and "input_ids" in batch, (
            f"Expected dict batch with 'input_ids', got {type(batch)}"
        )
        yield batch["input_ids"].to(device)


_LAYER_IN_NAME = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def parse_module_name(module_name: str) -> tuple[int, str]:
    """Extract (layer, matrix) from a module name. Falls back to (-1, module_name)."""
    match = _LAYER_IN_NAME.search(module_name)
    if match is None:
        return -1, module_name
    layer = int(match.group(1))
    matrix = module_name[match.end() :]
    return layer, matrix


def build_module_lookup(module_paths: list[str]) -> dict[tuple[int, str], str]:
    """Map (layer, matrix) back to the full module path."""
    lookup: dict[tuple[int, str], str] = {}
    for path in module_paths:
        key = parse_module_name(path)
        assert key not in lookup, (
            f"Duplicate (layer, matrix) key {key} from modules {lookup.get(key)} and {path}"
        )
        lookup[key] = path
    return lookup
