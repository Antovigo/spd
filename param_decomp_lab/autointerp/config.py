"""Autointerp configuration."""

from typing import Annotated, Literal

from pydantic import Field

from param_decomp_config.autointerp import LLMConfig, OpenRouterLLMConfig, StrategyConfig
from param_decomp_config.base import BaseConfig
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME


class AutointerpConfig(BaseConfig):
    llm: LLMConfig = OpenRouterLLMConfig()
    limit: int | None = None
    component_keys_path: str | None = None
    cost_limit_usd: float | None = None
    template_strategy: Annotated[StrategyConfig, Field(discriminator="type")]


class DetectionEvalConfig(BaseConfig):
    type: Literal["detection"] = "detection"
    n_activating: int = 5
    n_non_activating: int = 5
    n_trials: int = 5


class FuzzingEvalConfig(BaseConfig):
    type: Literal["fuzzing"] = "fuzzing"
    n_correct: int = 5
    n_incorrect: int = 2
    n_trials: int = 5


class AutointerpEvalConfig(BaseConfig):
    """Config for label-based autointerp evals (detection, fuzzing)."""

    llm: LLMConfig = OpenRouterLLMConfig(reasoning_effort="none")
    detection_config: DetectionEvalConfig
    fuzzing_config: FuzzingEvalConfig
    limit: int | None = None
    component_keys_path: str | None = None
    seed: int = 0
    cost_limit_usd: float | None = None


class AutointerpSlurmConfig(BaseConfig):
    """Config for the autointerp functional unit (interpret + evals).

    Dependency graph within autointerp:
        interpret         (depends on harvest merge)
        ├── detection     (depends on interpret)
        └── fuzzing       (depends on interpret)
    """

    config: AutointerpConfig
    partition: str | None = DEFAULT_PARTITION_NAME
    time: str = "12:00:00"
    evals: AutointerpEvalConfig | None
    evals_time: str = "12:00:00"
