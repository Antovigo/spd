"""Scan the runs directory into a TSV index.

One row per `runs/<run_id>/` containing an `experiment_config.yaml`. The `hyperparameters`
column shows only config values that differ *between runs sharing the same `label`*, so a
sweep collapses to just its swept axes. Final metric values are pulled by key from each
run's `metrics.jsonl` (last line that carries the key).

    python -m param_decomp_lab.scripts.index_runs --metrics train/loss/total,eval/...
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from param_decomp.log import logger
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.infra.run_files import RUN_METADATA_FILENAME
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.infra.wandb import flatten_typed_lists

NA = "NA"


def _flatten_plain(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists into dot-separated scalar keys."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(_flatten_plain(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            flat.update(_flatten_plain(item, f"{prefix}.{i}"))
    else:
        flat[prefix] = obj
    return flat


def _flatten_config(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten a config dict, keying loss/eval metric lists by metric short-name.

    `flatten_typed_lists` handles the discriminated lists (and mutates `cfg_dict`,
    removing them); `_flatten_plain` handles everything left.
    """
    flat = flatten_typed_lists(cfg_dict)
    flat.update(_flatten_plain(cfg_dict))
    return flat


def _format_started(iso: str | None) -> str:
    """Concise `YYYY-MM-DD HH:MM` from an ISO timestamp; NA when absent."""
    if iso is None:
        return NA
    return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")


def _duration_hours(started_at: str | None, finished_at: str | None) -> str:
    if started_at is None or finished_at is None:
        return NA
    delta = datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
    return f"{delta.total_seconds() / 3600:.2f}"


def _format_metric_value(v: Any) -> str:
    """Scientific notation for numeric metric values (losses); raw str otherwise."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int | float):
        return f"{v:.3e}"
    return str(v)


def _final_metrics(metrics_path: Path, keys: list[str]) -> dict[str, str]:
    """Last value seen for each requested key across `metrics.jsonl` lines.

    Skips unparseable lines so a run killed mid-write (truncated final line) doesn't
    abort the whole index.
    """
    latest: dict[str, str] = {k: NA for k in keys}
    if not keys or not metrics_path.exists():
        return latest
    for line in metrics_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in keys:
            if key in record:
                latest[key] = _format_metric_value(record[key])
    return latest


class RunRow:
    def __init__(self, run_dir: Path, metric_keys: list[str]):
        self.run_id = run_dir.name
        cfg_dict: dict[str, Any] = yaml.safe_load(
            (run_dir / EXPERIMENT_CONFIG_FILENAME).read_text()
        )
        self.label: str | None = cfg_dict.get("label")
        self.flat_config = _flatten_config(cfg_dict)

        metadata: dict[str, Any] = {}
        meta_path = run_dir / RUN_METADATA_FILENAME
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        self.started_at: str | None = metadata.get("started_at")
        self.duration_hours = _duration_hours(self.started_at, metadata.get("finished_at"))

        self.metrics = _final_metrics(run_dir / "metrics.jsonl", metric_keys)

    def group_key(self) -> str:
        """Runs sharing a label group together; unlabeled runs each stand alone."""
        return self.label if self.label is not None else f"__nolabel__:{self.run_id}"


def _differing_hyperparameters(rows: list[RunRow]) -> dict[str, str]:
    """For each run, the `k=v` pairs whose value differs within its label group."""
    by_group: dict[str, list[RunRow]] = defaultdict(list)
    for row in rows:
        by_group[row.group_key()].append(row)

    out: dict[str, str] = {}
    for group in by_group.values():
        all_keys = {k for row in group for k in row.flat_config}
        differing = {k for k in all_keys if len({str(row.flat_config.get(k)) for row in group}) > 1}
        for row in group:
            pairs = [f"{k}={row.flat_config.get(k)}" for k in sorted(differing)]
            out[row.run_id] = ", ".join(pairs)
    return out


def build_index(runs_dir: Path, out_path: Path, metric_keys: list[str]) -> None:
    run_dirs = sorted(d for d in runs_dir.iterdir() if (d / EXPERIMENT_CONFIG_FILENAME).exists())
    assert run_dirs, f"no runs with {EXPERIMENT_CONFIG_FILENAME} under {runs_dir}"
    rows = [RunRow(d, metric_keys) for d in run_dirs]
    hyperparameters = _differing_hyperparameters(rows)

    columns = ["run_id", "label", "started_at", "duration_hours", *metric_keys, "hyperparameters"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.run_id,
                    row.label if row.label is not None else NA,
                    _format_started(row.started_at),
                    row.duration_hours,
                    *(row.metrics[k] for k in metric_keys),
                    hyperparameters[row.run_id],
                ]
            )
    logger.info(f"Wrote index of {len(rows)} runs to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--runs-dir", type=Path, default=PARAM_DECOMP_OUT_DIR / "runs")
    parser.add_argument("-o", "--out", type=Path, default=None)
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated metric keys to pull final values for (exact keys in metrics.jsonl).",
    )
    args = parser.parse_args()

    metric_keys = [s.strip() for s in args.metrics.split(",") if s.strip()]
    out_path: Path = args.out if args.out is not None else args.runs_dir / "runs_index.tsv"
    build_index(args.runs_dir, out_path, metric_keys)


if __name__ == "__main__":
    main()
