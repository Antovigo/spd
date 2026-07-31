"""Runtime objects resolved from an authored LM config."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
from jax.typing import DTypeLike

from param_decomp.core.built_run import BuiltRun, TargetSites

WeightsDtype = Literal["float32", "bfloat16"]


def weights_jnp_dtype(dtype: WeightsDtype) -> DTypeLike:
    """The authored frozen-target dtype as the array dtype the target loaders cast to."""
    match dtype:
        case "float32":
            return jnp.float32
        case "bfloat16":
            return jnp.bfloat16


@dataclass(frozen=True)
class ResolvedLMData:
    """Pre-tokenized parquet shard directories: `dir` trains; `eval_dir` is the held-out
    split the eval pass reads."""

    dir: Path
    eval_dir: Path


LMRun = BuiltRun[ResolvedLMData, TargetSites]
