"""Remove local folders for unfinished runs based on runs_index.tsv."""

import argparse
import csv
import shutil
from pathlib import Path

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

    parser = argparse.ArgumentParser(
        description="Remove local folders for unfinished runs from runs_index.tsv"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir,
        help="Directory containing runs_index.tsv and run folders",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    tsv_path = runs_dir / "runs_index.tsv"

    unfinished = get_unfinished_runs(tsv_path)
    if not unfinished:
        print("No unfinished runs found.")
        return

    existing_folders: list[tuple[dict[str, str], Path]] = []
    missing_folders: list[str] = []

    for row in unfinished:
        folder = runs_dir / row["run_id"]
        if folder.exists():
            existing_folders.append((row, folder))
        else:
            missing_folders.append(row["run_id"])

    print(f"Found {len(unfinished)} unfinished runs:\n")
    for row, folder in existing_folders:
        label = row.get("label", "")
        print(f"  {row['run_id']}  |  {row['date']}  |  {label}  |  {folder}")
    if missing_folders:
        print(f"\n  ({len(missing_folders)} runs have no local folder)")

    if not existing_folders:
        print("\nNo folders to remove.")
        return

    response = input(f"\nDelete {len(existing_folders)} folders? [y/N] ")
    if response.strip().lower() != "y":
        print("Aborted.")
        return

    for _, folder in existing_folders:
        shutil.rmtree(folder)
        print(f"  Removed {folder}")

    print(f"\nDone. Removed {len(existing_folders)} folders.")


if __name__ == "__main__":
    main()
