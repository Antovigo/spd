"""Delete unfinished WandB runs based on runs_index.tsv."""

import argparse
import csv
from pathlib import Path

import wandb

from spd.settings import SPD_OUT_DIR


def get_unfinished_runs(tsv_path: Path) -> list[dict[str, str]]:
    assert tsv_path.exists(), f"TSV file not found: {tsv_path}"
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [
            row
            for row in reader
            if row["completed"] != "True" and row["run_id"].startswith("s-")
        ]


def main() -> None:
    default_runs_dir = SPD_OUT_DIR / "spd"

    parser = argparse.ArgumentParser(description="Delete unfinished WandB runs from runs_index.tsv")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir,
        help="Directory containing runs_index.tsv and run folders",
    )
    parser.add_argument("--entity", required=True, help="WandB entity (username or team)")
    parser.add_argument(
        "--project", help="WandB project (default: inferred from runs dir name)"
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    tsv_path = runs_dir / "runs_index.tsv"
    project = args.project or runs_dir.name

    unfinished = get_unfinished_runs(tsv_path)
    if not unfinished:
        print("No unfinished runs found.")
        return

    print(f"Found {len(unfinished)} unfinished runs in {args.entity}/{project}:\n")
    for row in unfinished:
        label = row.get("label", "")
        print(f"  {row['run_id']}  |  {row['date']}  |  {label}")

    response = input(f"\nDelete these {len(unfinished)} runs from WandB? [y/N] ")
    if response.strip().lower() != "y":
        print("Aborted.")
        return

    api = wandb.Api()
    for row in unfinished:
        run_id = row["run_id"]
        try:
            run = api.run(f"{args.entity}/{project}/{run_id}")
            run.delete()
            print(f"  Deleted {run_id}")
        except wandb.errors.CommError:  # pyright: ignore[reportAttributeAccessIssue]
            print(f"  Not found on WandB: {run_id}")

    print(f"\nDone. Deleted {len(unfinished)} runs.")


if __name__ == "__main__":
    main()
