"""Generate a TSV index of all SPD runs.

Scans SPD_OUT_DIR/spd for run directories and produces a runs_index.tsv with metadata columns:
run_id, git_commit, uncommitted_changes, label, notes, hyperparameters, date, completed.

The hyperparameters column shows only config values that differ between runs sharing the same label.

Usage:
    uv run spd/scripts/index_spd_runs.py                       # default paths
    uv run spd/scripts/index_spd_runs.py -i /path/to/runs      # override input dir
    uv run spd/scripts/index_spd_runs.py -o /path/to/out.tsv   # override output path
"""

import argparse
import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from spd.settings import SPD_OUT_DIR
from spd.utils.run_utils import _DISCRIMINATED_LIST_FIELDS

COLUMNS = [
    "run_id",
    "date",
    "git_commit",
    "uncommitted_changes",
    "label",
    "completed",
    "hyperparameters",
    "notes",
]

NA = "NA"

_RECHECK_WINDOW = timedelta(days=7)


def _is_recent(row: dict[str, str]) -> bool:
    """Return True if the run's date is within the recheck window."""
    date_str = row.get("date", NA)
    if date_str == NA:
        return False
    try:
        run_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    except ValueError:
        return False
    return datetime.now(UTC) - run_date < _RECHECK_WINDOW


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Recursively flatten a nested dict with dot-separated keys.

    For discriminated list fields, uses the discriminator value as key instead of index.
    For other lists, uses index-based keys.
    """
    flat: dict[str, str] = {}
    for key, value in d.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        elif isinstance(value, list) and key in _DISCRIMINATED_LIST_FIELDS:
            disc_field = _DISCRIMINATED_LIST_FIELDS[key]
            for item in value:
                assert isinstance(item, dict)
                disc_value = item[disc_field]
                sub = {k: v for k, v in item.items() if k != disc_field}
                flat.update(_flatten_dict(sub, f"{full_key}.{disc_value}"))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flat.update(_flatten_dict(item, f"{full_key}.{i}"))
                else:
                    flat[f"{full_key}.{i}"] = str(item)
        else:
            flat[full_key] = str(value)
    return flat


def _read_metadata(run_dir: Path) -> dict[str, str]:
    """Read metadata from run_metadata.json, or return NAs for legacy runs."""
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        # Handle both old field name (git_dirty) and new (uncommitted_changes)
        uncommitted = meta.get("uncommitted_changes", meta.get("git_dirty", NA))
        return {
            "run_id": str(meta.get("run_id", run_dir.name)),
            "git_commit": str(meta.get("git_commit", NA)),
            "uncommitted_changes": str(uncommitted),
            "label": str(meta.get("label", "")),
            "notes": str(meta.get("notes", "")),
            "date": str(meta.get("date", NA)),
            "completed": str(meta.get("completed", NA)),
        }

    # Legacy run without metadata — try to determine completion from config + metrics
    completed = _check_legacy_completion(run_dir)
    return {
        "run_id": run_dir.name,
        "git_commit": NA,
        "uncommitted_changes": NA,
        "label": "",
        "notes": "",
        "date": NA,
        "completed": str(completed),
    }


def _check_legacy_completion(run_dir: Path) -> bool:
    """Determine if a legacy run (no run_metadata.json) completed."""
    config_path = run_dir / "final_config.yaml"
    metrics_path = run_dir / "metrics.jsonl"

    if not config_path.exists() or not metrics_path.exists():
        return False

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    total_steps = config_dict.get("steps")
    if total_steps is None:
        return False

    # Read last line of metrics.jsonl
    last_line = ""
    with open(metrics_path, "rb") as f:
        # Seek backwards to find last newline
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return False
        pos = size - 1
        while pos > 0:
            f.seek(pos)
            char = f.read(1)
            if char == b"\n" and pos < size - 1:
                break
            pos -= 1
        last_line = f.readline().decode().strip()

    if not last_line:
        return False

    last_step = json.loads(last_line).get("step", -1)
    return last_step >= total_steps


def _load_existing_index(index_path: Path) -> dict[str, dict[str, str]]:
    """Load existing TSV index into dict[run_id → row]."""
    if not index_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with open(index_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run_id = row["run_id"]
            rows[run_id] = dict(row)
    return rows


def _compute_hyperparameters(
    label_groups: dict[str, list[str]],
    run_dirs: dict[str, Path],
) -> dict[str, str]:
    """Compute hyperparameters column for all runs.

    For each label group with >=2 runs, loads final_config.yaml, flattens, and diffs.
    """
    hyperparams: dict[str, str] = {}

    for _label, run_ids in label_groups.items():
        if len(run_ids) < 2:
            for rid in run_ids:
                hyperparams[rid] = ""
            continue

        # Load and flatten configs for each run in the group
        flattened_configs: dict[str, dict[str, str]] = {}
        for rid in run_ids:
            config_path = run_dirs[rid] / "final_config.yaml"
            if not config_path.exists():
                flattened_configs[rid] = {}
                continue
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            flattened_configs[rid] = _flatten_dict(config_dict)

        # Find keys that differ across runs (ignore notes and label)
        all_keys = set()
        for fc in flattened_configs.values():
            all_keys.update(fc.keys())

        ignore_keys = {"notes", "label"}
        differing_keys: list[str] = []
        for key in sorted(all_keys):
            if key in ignore_keys:
                continue
            values = {fc.get(key) for fc in flattened_configs.values()}
            if len(values) > 1:
                differing_keys.append(key)

        # Format hyperparameters string for each run
        for rid in run_ids:
            fc = flattened_configs.get(rid, {})
            parts = [f"{k}={fc.get(k, NA)}" for k in differing_keys]
            hyperparams[rid] = " ".join(parts)

    return hyperparams


def build_index(runs_dir: Path, index_path: Path) -> None:
    existing = _load_existing_index(index_path)

    # Discover all run directories
    run_dirs: dict[str, Path] = {}
    for entry in sorted(runs_dir.iterdir()):
        if entry.is_dir():
            run_dirs[entry.name] = entry

    # Phase 1: collect per-run metadata (using cache where possible)
    rows: dict[str, dict[str, str]] = {}
    new_run_ids: set[str] = set()
    for run_id, run_dir in tqdm(run_dirs.items(), desc="Reading runs"):
        cached = existing.get(run_id)
        if cached and (cached.get("completed") == "True" or not _is_recent(cached)):
            rows[run_id] = cached
        else:
            rows[run_id] = _read_metadata(run_dir)
            if not cached:
                new_run_ids.add(run_id)

    # Phase 2: compute hyperparameters
    # Group runs by label
    label_groups: dict[str, list[str]] = {}
    for run_id, row in rows.items():
        label = row.get("label", "")
        if label and label != NA:
            label_groups.setdefault(label, []).append(run_id)

    # Determine which label groups need recomputation
    groups_to_recompute: set[str] = set()
    groups_cached: set[str] = set()
    for label, run_ids in label_groups.items():
        if any(rid in new_run_ids for rid in run_ids):
            groups_to_recompute.add(label)
        else:
            groups_cached.add(label)

    # Recompute hyperparameters for groups with new runs
    recompute_groups = {
        lbl: rids for lbl, rids in label_groups.items() if lbl in groups_to_recompute
    }
    fresh_hyperparams = _compute_hyperparameters(recompute_groups, run_dirs)

    # Assign hyperparameters to all runs
    for run_id, row in rows.items():
        if run_id in fresh_hyperparams:
            row["hyperparameters"] = fresh_hyperparams[run_id]
        elif run_id not in existing:
            # New run not in any label group (or solo label)
            row.setdefault("hyperparameters", "")

    # Phase 3: write TSV
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for run_id in sorted(
            rows,
            key=lambda rid: rows[rid].get("date", "") if rows[rid].get("date", "") != NA else "",
            reverse=True,
        ):
            writer.writerow(rows[run_id])

    print(f"Wrote {len(rows)} runs to {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TSV index of SPD runs")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=SPD_OUT_DIR / "spd",
        help="Directory containing run subdirectories",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output TSV path (default: <input-dir>/runs_index.tsv)",
    )
    args = parser.parse_args()

    runs_dir: Path = args.input_dir
    assert runs_dir.is_dir(), f"Runs directory not found: {runs_dir}"

    index_path: Path = args.output if args.output else runs_dir / "runs_index.tsv"

    build_index(runs_dir, index_path)


if __name__ == "__main__":
    main()
