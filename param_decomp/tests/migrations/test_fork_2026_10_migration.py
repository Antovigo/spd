"""`migrations.fork_2026_10`: a pre-2026-10 fork run config reads under the current schema."""

import copy
from pathlib import Path
from typing import Any

import yaml

from param_decomp.experiments.lm.config import LMTargetedExperimentConfig
from param_decomp.migrations.fork_2026_10 import migrate_raw

_REPO = Path(__file__).resolve().parents[3]
_CURRENT = _REPO / "notes/dual_objective/addsub-all-layers-4xh100-08.yaml"


def _pre_merge_spelling(current: dict[str, Any]) -> dict[str, Any]:
    """The same config as a pre-2026-10 run pinned it: no output edge, the CI attention's
    `attention_implementation` with no mask, flat mesh axes, `tuned-v1`."""
    old = copy.deepcopy(current)
    del old["target"]["output_edge"]
    attention = old["decomposition"]["ci"]["attention"]
    attention["attention_implementation"] = attention.pop("implementation")
    del attention["mask"]
    old["runtime"].update(old["runtime"].pop("mesh"))
    old["runtime"]["compiler_options"] = "tuned-v1"
    return old


def test_pre_merge_spelling_migrates_to_the_current_config() -> None:
    current = yaml.safe_load(_CURRENT.read_text())
    assert current["target"]["output_edge"] == {"kind": "materialized"}
    assert current["decomposition"]["ci"]["attention"]["mask"] == "bidirectional"
    old = _pre_merge_spelling(current)

    migrated = migrate_raw(old)

    assert migrated == current
    LMTargetedExperimentConfig.model_validate(migrated)


def test_migration_is_idempotent_and_does_not_mutate() -> None:
    current = yaml.safe_load(_CURRENT.read_text())
    old = _pre_merge_spelling(current)
    snapshot = copy.deepcopy(old)

    once = migrate_raw(old)

    assert old == snapshot
    assert migrate_raw(once) == once
    assert migrate_raw(current) == current
