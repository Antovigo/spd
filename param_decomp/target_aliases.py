"""Frozen-target / prefix unions for the LM targets `run.py::main` and
`load_run.py::build_target` dispatch over.

The toy targets (TMS, ResidMLP) live in the lab and are NOT members of these unions —
the generic engine (`run_decomposition_training`) takes the frozen target as `Any`, so
the core never names a toy."""

from param_decomp.llama8b import Prefix, Target
from param_decomp.llama_simple_mlp import SimpleMLPPrefix, SimpleMLPTarget

AnyFrozenTarget = Target | SimpleMLPTarget
AnyPrefix = Prefix | SimpleMLPPrefix
