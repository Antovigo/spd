"""Decomposition-target config: an fnmatch module pattern paired with a component count."""

from pydantic import Field, PositiveInt

from param_decomp_config.base import BaseConfig


class DecompositionTargetConfig(BaseConfig):
    module_pattern: str = Field(..., description="fnmatch-style pattern to match module names")
    C: PositiveInt = Field(
        ..., description="Number of components for modules matching this pattern"
    )
