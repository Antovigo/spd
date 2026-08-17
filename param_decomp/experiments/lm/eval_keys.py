"""Named, disjoint RNG stream indices for language-model evaluation."""

from enum import IntEnum


class EvalKeyStream(IntEnum):
    """Blocks off `run_key`, each with `pd.steps` indices; append rather than renumber."""

    SCALARS = 1
    ATTENTION_PATTERNS = 2
    HIDDEN_ACTS = 3
    WELL_TEMPEREDNESS = 5
