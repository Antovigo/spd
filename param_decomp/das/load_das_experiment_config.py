"""The DAS experiment's settings file (`notes/das/das_experiment_config.yaml`)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

type Task = Literal["addition", "subtraction"]
type Variable = Literal["first_operand", "second_operand", "result"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Frozen):
    tokenized_prompts: Path
    correctness_references: Path
    pruning_run_config: Path
    llama_weights: Path
    prompt_pairs: Path
    pilot: Path


class PromptPairSettings(_Frozen):
    seed: int
    train_prompt_fraction: float
    validation_prompt_fraction: float
    train_pairs: int
    validation_pairs: int
    test_pairs: int


class TrainingSettings(_Frozen):
    learning_rate: float
    train_pairs: int
    batch_size: int
    epochs: int


class PruningSettings(_Frozen):
    position_margin: float
    prune_score_pairs: int
    fresh_fit_ladder: tuple[int, ...]


class PilotSite(_Frozen):
    task: Task
    variable: Variable
    block: int

    @property
    def stream_point(self) -> int:
        """`t = 2 * block + 1`, the stream before this block's MLP."""
        return 2 * self.block + 1

    @property
    def name(self) -> str:
        return f"{self.task}_{self.variable}_block_{self.block}_before_mlp"


class PilotSettings(_Frozen):
    learning_rates: tuple[float, ...]
    train_pairs: tuple[int, ...]
    finetune_epochs: tuple[int, ...]
    sites: tuple[PilotSite, ...]


class DasExperimentConfig(_Frozen):
    paths: Paths
    prompt_pairs: PromptPairSettings
    patched_positions: dict[Variable, tuple[int, ...]]
    training: TrainingSettings
    pruning: PruningSettings
    pilot: PilotSettings


def _expand(paths: dict[str, str]) -> dict[str, Path]:
    return {name: Path(path).expanduser() for name, path in paths.items()}


def load_das_experiment_config(path: Path) -> DasExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    raw["paths"] = _expand(raw["paths"])
    return DasExperimentConfig.model_validate(raw)
