"""Compare SPD runs by extracting varying hyperparameters and final metrics.

Outputs a TSV file suitable for R analysis.

Usage:
    python scripts/compare_runs.py ~/Documents/MATS/pythia
    python scripts/compare_runs.py ~/Documents/MATS/pythia --output results.tsv
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

BRACKET_PATTERN = re.compile(r"^(\w+)\[(.+)\]$")


def flatten_config(data: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config into dot-notation keys.

    Handles lists using bracket notation for classname/module_pattern matching,
    or numeric indices as fallback.
    """
    result: dict[str, Any] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            result.update(flatten_config(value, new_prefix))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                if "classname" in item:
                    bracket_key = f"{prefix}[{item['classname']}]"
                    result.update(flatten_config(item, bracket_key))
                elif "module_pattern" in item:
                    bracket_key = f"{prefix}[{item['module_pattern']}]"
                    result.update(flatten_config(item, bracket_key))
                else:
                    result.update(flatten_config(item, f"{prefix}.{i}"))
            else:
                result[f"{prefix}.{i}"] = item
    else:
        result[prefix] = data

    return result


def find_varying_keys(configs: list[dict[str, Any]]) -> set[str]:
    """Find keys that have different values across configs."""
    if not configs:
        return set()

    all_keys = set()
    for config in configs:
        all_keys.update(config.keys())

    varying_keys = set()
    for key in all_keys:
        values = []
        for config in configs:
            values.append(config.get(key))
        if len(set(repr(v) for v in values)) > 1:
            varying_keys.add(key)

    return varying_keys


def load_final_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load final metrics from metrics.jsonl (last line)."""
    with open(metrics_path) as f:
        lines = f.readlines()

    if not lines:
        return {}

    last_line = lines[-1].strip()
    if not last_line:
        return {}

    metrics = json.loads(last_line)
    metrics.pop("step", None)
    return metrics


def sanitize_column_name(name: str) -> str:
    """Convert column name to R-compatible camelCase.

    Replaces special characters with underscores, then converts to camelCase.
    """
    cleaned = re.sub(r"[/.\s\-\[\]]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")

    parts = cleaned.split("_")
    if not parts:
        return name

    result = parts[0].lower()
    for part in parts[1:]:
        if part:
            result += part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper()

    return result


def discover_runs(target_folder: Path) -> list[Path]:
    """Find all run directories containing metrics.jsonl."""
    metrics_files = list(target_folder.rglob("metrics.jsonl"))
    return [f.parent for f in metrics_files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SPD runs and output TSV for R analysis")
    parser.add_argument("target_folder", type=Path, help="Folder containing run subdirectories")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output TSV path (default: comparison.tsv in target folder)",
    )
    args = parser.parse_args()

    assert args.target_folder.exists(), f"Target folder not found: {args.target_folder}"

    output_path = args.output or (args.target_folder / "comparison.tsv")

    runs = discover_runs(args.target_folder)
    assert runs, f"No runs found in {args.target_folder}"

    print(f"Found {len(runs)} run(s)")

    flattened_configs: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    run_folders: list[str] = []

    for run_dir in sorted(runs):
        config_path = run_dir / "final_config.yaml"
        metrics_path = run_dir / "metrics.jsonl"

        if not config_path.exists():
            print(f"Warning: {config_path} not found, skipping")
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        flattened = flatten_config(config)
        flattened_configs.append(flattened)

        metrics = load_final_metrics(metrics_path)
        all_metrics.append(metrics)

        run_folders.append(run_dir.relative_to(args.target_folder).as_posix())

    varying_keys = sorted(find_varying_keys(flattened_configs))
    print(f"Found {len(varying_keys)} varying hyperparameter(s)")

    all_metric_keys: set[str] = set()
    for metrics in all_metrics:
        all_metric_keys.update(metrics.keys())
    metric_keys = sorted(all_metric_keys)

    columns = ["runFolder"] + varying_keys + metric_keys
    sanitized_columns = [sanitize_column_name(c) for c in columns]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(sanitized_columns)

        for i, run_folder in enumerate(run_folders):
            row = [run_folder]

            for key in varying_keys:
                value = flattened_configs[i].get(key, "")
                row.append(value)

            for key in metric_keys:
                value = all_metrics[i].get(key, "")
                row.append(value)

            writer.writerow(row)

    print(f"Wrote {len(run_folders)} row(s) to {output_path}")


if __name__ == "__main__":
    main()
