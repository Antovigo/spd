"""Run experiment variations by overriding config parameters.

Usage:
    python -m spd.scripts.run_variations <script_path> <default_config> <overrides_file>

The overrides file contains multiple experiment configs separated by "---".
Each block specifies parameters to override from the default config.

Example overrides file:
    seed: 0
    ---
    seed: 1
    ---
    seed: 2
    lr_schedule.start_val: 0.01

Nested parameters use dot notation:
    - lr_schedule.start_val: 0.01
    - loss_metric_configs.0.coeff: 1e-5  (list index access)
    - task_config.feature_probability: 0.1

List items can be matched by classname using bracket notation:
    - loss_metric_configs[ImportanceMinimalityLoss].coeff: 1e-5
    - loss_metric_configs[StochasticReconLoss].coeff: 0.5
"""

import argparse
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

YamlDict = dict[str, Any]
YamlValue = Any

BRACKET_PATTERN = re.compile(r"^(\w+)\[(\w+)\]$")


def find_list_item_by_classname(items: list[YamlValue], classname: str) -> int:
    """Find the index of a list item with matching classname."""
    for i, item in enumerate(items):
        if isinstance(item, dict) and item.get("classname") == classname:
            return i
    raise ValueError(f"No item with classname '{classname}' found")


def set_nested_value(data: YamlDict | list[YamlValue], key_path: str, value: YamlValue) -> None:
    """Set a value in a nested dict/list structure using dot notation.

    Examples:
        set_nested_value(data, "seed", 42)
        set_nested_value(data, "lr_schedule.start_val", 0.01)
        set_nested_value(data, "loss_metric_configs.0.coeff", 1e-5)
        set_nested_value(data, "loss_metric_configs[ImportanceMinimalityLoss].coeff", 1e-5)
    """
    keys = key_path.split(".")
    current = data

    for i, key in enumerate(keys[:-1]):
        if isinstance(current, list):
            idx = int(key)
            assert 0 <= idx < len(current), (
                f"Index {idx} out of range for list at {'.'.join(keys[:i])}"
            )
            current = current[idx]
        elif match := BRACKET_PATTERN.match(key):
            list_key, classname = match.groups()
            assert list_key in current, (
                f"Key '{list_key}' not found at {'.'.join(keys[:i]) or 'root'}"
            )
            items = current[list_key]
            assert isinstance(items, list), f"Expected list at '{list_key}'"
            idx = find_list_item_by_classname(items, classname)
            current = items[idx]
        else:
            assert key in current, f"Key '{key}' not found at {'.'.join(keys[:i]) or 'root'}"
            current = current[key]

    final_key = keys[-1]
    if isinstance(current, list):
        idx = int(final_key)
        assert 0 <= idx < len(current), f"Index {idx} out of range for list"
        current[idx] = value
    elif match := BRACKET_PATTERN.match(final_key):
        list_key, classname = match.groups()
        assert list_key in current, f"Key '{list_key}' not found"
        items = current[list_key]
        assert isinstance(items, list), f"Expected list at '{list_key}'"
        idx = find_list_item_by_classname(items, classname)
        items[idx] = value
    else:
        current[final_key] = value


def merge_overrides(base_config: YamlDict, overrides: YamlDict) -> YamlDict:
    """Merge overrides into base config, handling dot notation for nested keys."""
    result = deepcopy(base_config)

    for key, value in overrides.items():
        if "." in key or "[" in key:
            set_nested_value(result, key, value)
        else:
            result[key] = value

    return result


def parse_overrides_file(path: Path) -> list[YamlDict]:
    """Parse an overrides file with multiple blocks separated by '---'."""
    content = path.read_text()
    blocks = content.split("---")
    experiments: list[YamlDict] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        parsed = yaml.safe_load(block)
        if parsed:
            experiments.append(parsed)

    return experiments


def run_experiment(script_path: Path, config_path: Path) -> int:
    """Run the decomposition script with the given config."""
    cmd = ["uv", "run", str(script_path), str(config_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment variations with config overrides")
    parser.add_argument("script_path", type=Path, help="Path to the decomposition script")
    parser.add_argument("default_config", type=Path, help="Path to the default config YAML")
    parser.add_argument("overrides_file", type=Path, help="Path to the overrides file")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    args = parser.parse_args()

    assert args.script_path.exists(), f"Script not found: {args.script_path}"
    assert args.default_config.exists(), f"Config not found: {args.default_config}"
    assert args.overrides_file.exists(), f"Overrides file not found: {args.overrides_file}"

    base_config = yaml.safe_load(args.default_config.read_text())
    experiments = parse_overrides_file(args.overrides_file)

    print(f"Found {len(experiments)} experiment variation(s)")

    for i, overrides in enumerate(experiments):
        name = overrides.get("out_dir_name", f"Experiment {i}")
        print(f"\n{'=' * 60}")
        print(f"Variation {i + 1}/{len(experiments)}: {name}")
        print(f"{'=' * 60}")
        print(f"Overrides: {overrides}")

        merged_config = merge_overrides(base_config, overrides)

        if args.dry_run:
            print("Merged config:")
            print(yaml.dump(merged_config, default_flow_style=False))
            continue

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"variation_{i}_", delete=False
        ) as tmp:
            yaml.dump(merged_config, tmp)
            tmp_path = Path(tmp.name)

        try:
            returncode = run_experiment(args.script_path, tmp_path)
            if returncode != 0:
                print(f"Experiment failed with return code {returncode}")
                sys.exit(returncode)
        finally:
            tmp_path.unlink()

    print(f"\nAll {len(experiments)} experiment(s) completed successfully")


if __name__ == "__main__":
    main()
