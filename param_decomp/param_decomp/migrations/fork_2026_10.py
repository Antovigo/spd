"""Read the fork's pre-2026-10 stored LM run configs under the 017a17af4 schema.

The upstream merge of 2026-10 (017a17af4) changed four config spellings that every fork run
pinned before it carries. ``migrate_raw`` rewrites them in memory so the immutable
``launch_config.yaml`` / deliverable of a pre-merge run (e.g. p-ba5a0c05, p-ba8b0c08) still
loads for restoring, analysis and fine-tuning. It is idempotent: a current config passes
through unchanged.

- ``target.output_edge`` became required. Pre-merge runs formed full logits on every
  forward, which is ``{kind: materialized}``.
- The CI transformer's ``attention.attention_implementation`` was renamed
  ``implementation``, and ``attention.mask`` became required. The pre-merge CI attention was
  bidirectional.
- ``runtime`` mesh axes (``replicate``/``fsdp``/``data``/``tp``) moved under
  ``runtime.mesh``.
- ``runtime.compiler_options: tuned-v1`` was retired (jaxlib 0.11 removed the options it
  froze); ``tuned-v2`` is its replacement.

NOT migrated: a persistent adversary's batch-shared ``source_shape: c``, which upstream
removed. Its replacements (``bc``, ``bsc``) change the threat model, so such a config still
refuses at parse.

To convert a standalone config without mutating the source::

    python -m param_decomp.migrations.fork_2026_10 OLD.yaml > NEW.yaml
"""

import copy
import sys
from pathlib import Path
from typing import Any

import yaml

from param_decomp.migrations.schedule_knots import migrate_raw as migrate_schedule_knots

_MESH_AXES = ("replicate", "fsdp", "data", "tp")


def _migrate_ci_attention(node: object) -> None:
    """Rename every `attention_implementation` under `decomposition.ci` in place, adding the
    bidirectional mask the pre-merge CI attention had implicitly."""
    match node:
        case dict():
            if "attention_implementation" in node:
                node["implementation"] = node.pop("attention_implementation")
                node.setdefault("mask", "bidirectional")
            for value in node.values():
                _migrate_ci_attention(value)
        case list():
            for value in node:
                _migrate_ci_attention(value)
        case _:
            pass


def migrate_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """The pre-2026-10 spellings of one complete stored config, rewritten to the current ones
    (schedule knots included). Returns a new dict; `raw` is not mutated."""
    out = migrate_schedule_knots(copy.deepcopy(raw))
    target = out.get("target")
    if isinstance(target, dict):
        target.setdefault("output_edge", {"kind": "materialized"})
    decomposition = out.get("decomposition")
    if isinstance(decomposition, dict):
        _migrate_ci_attention(decomposition.get("ci"))
    runtime = out.get("runtime")
    if isinstance(runtime, dict):
        flat = {axis: runtime.pop(axis) for axis in _MESH_AXES if axis in runtime}
        if flat:
            assert "mesh" not in runtime, f"runtime has both a mesh and flat axes {list(flat)}"
            runtime["mesh"] = flat
        if runtime.get("compiler_options") == "tuned-v1":
            runtime["compiler_options"] = "tuned-v2"
    return out


def read_stored_config(path: Path) -> dict[str, Any]:
    """A stored run config (`launch_config.yaml` or deliverable), migrated to the current
    schema."""
    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, dict), f"stored run config must be a mapping: {path}"
    return migrate_raw(raw)


if __name__ == "__main__":
    (src,) = sys.argv[1:]
    yaml.safe_dump(read_stored_config(Path(src)), sys.stdout, sort_keys=False)
