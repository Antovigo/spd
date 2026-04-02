"""Interactive WandB run cleanup with terminal UI for selecting runs to delete."""

import argparse
import csv
import curses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb

from spd.settings import SPD_OUT_DIR


@dataclass
class RunInfo:
    run_id: str
    state: str
    created_at: str
    tags: list[str]
    # From TSV (if available)
    label: str
    completed: str
    duration_hours: str
    notes: str

    wandb_run: Any

    def display_line(self, max_width: int) -> str:
        tags_str = ",".join(self.tags) if self.tags else ""
        parts = [
            f"{self.run_id:<14}",
            f"{self.state:<10}",
            f"{self.created_at[:16]:<18}",
            f"{tags_str:<16}",
            f"{self.label:<30}",
            f"{self.completed:<6}",
        ]
        if self.notes:
            parts.append(self.notes)
        line = " │ ".join(parts)
        return line[:max_width]


def fetch_runs(entity: str, project: str) -> list[Any]:
    api = wandb.Api()
    return list(api.runs(f"{entity}/{project}"))


def load_tsv_index(runs_dir: Path) -> dict[str, dict[str, str]]:
    tsv_path = runs_dir / "runs_index.tsv"
    if not tsv_path.exists():
        return {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["run_id"]: row for row in reader}


def build_run_infos(wandb_runs: list[Any], tsv_index: dict[str, dict[str, str]]) -> list[RunInfo]:
    infos = []
    for run in wandb_runs:
        tsv_row = tsv_index.get(run.id, {})
        infos.append(
            RunInfo(
                run_id=run.id,
                state=run.state,
                created_at=run.created_at,
                tags=run.tags,
                label=tsv_row.get("label", ""),
                completed=tsv_row.get("completed", ""),
                duration_hours=tsv_row.get("duration_hours", ""),
                notes=tsv_row.get("notes", ""),
                wandb_run=run,
            )
        )
    return infos


HEADER_PARTS = [
    f"{'run_id':<14}",
    f"{'state':<10}",
    f"{'created':<18}",
    f"{'tags':<16}",
    f"{'label':<30}",
    f"{'done':<6}",
    "notes",
]
HEADER = " │ ".join(HEADER_PARTS)


def run_picker(stdscr: curses.window, run_infos: list[RunInfo]) -> list[RunInfo]:
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)

    selected: set[int] = set()
    cursor = 0
    scroll_offset = 0

    while True:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()
        header_lines = 3
        available_rows = max_y - header_lines - 2

        # Header
        stdscr.attron(curses.A_BOLD)
        stdscr.addnstr(
            0, 0, f" {len(run_infos)} runs | {len(selected)} selected for deletion", max_x - 1
        )
        stdscr.addnstr(1, 0, f" {HEADER}", max_x - 1)
        stdscr.attroff(curses.A_BOLD)
        stdscr.addnstr(2, 0, "─" * (max_x - 1), max_x - 1)

        # Scroll
        if cursor < scroll_offset:
            scroll_offset = cursor
        elif cursor >= scroll_offset + available_rows:
            scroll_offset = cursor - available_rows + 1

        for i in range(available_rows):
            idx = scroll_offset + i
            if idx >= len(run_infos):
                break
            row_y = header_lines + i
            info = run_infos[idx]
            marker = "[x]" if idx in selected else "[ ]"
            line = f" {marker} {info.display_line(max_x - 6)}"

            attr = curses.A_NORMAL
            if idx == cursor:
                attr |= curses.A_REVERSE
            if idx in selected:
                attr |= curses.color_pair(2)
            elif info.state == "failed" or info.state == "killed":
                attr |= curses.color_pair(3)

            stdscr.addnstr(row_y, 0, line, max_x - 1, attr)

        # Footer
        footer_y = max_y - 1
        footer = " ↑↓:move  SPACE:toggle  a:all  n:none  f:failed/killed  ENTER:confirm  q:quit"
        stdscr.addnstr(footer_y, 0, footer, max_x - 1, curses.A_DIM)

        stdscr.refresh()
        key = stdscr.getch()

        match key:
            case curses.KEY_UP | 107:  # up or k
                cursor = max(0, cursor - 1)
            case curses.KEY_DOWN | 106:  # down or j
                cursor = min(len(run_infos) - 1, cursor + 1)
            case 32:  # space
                if cursor in selected:
                    selected.discard(cursor)
                else:
                    selected.add(cursor)
                cursor = min(len(run_infos) - 1, cursor + 1)
            case 97:  # a - select all
                selected = set(range(len(run_infos)))
            case 110:  # n - select none
                selected.clear()
            case 102:  # f - select failed/killed
                for idx, info in enumerate(run_infos):
                    if info.state in ("failed", "killed"):
                        selected.add(idx)
            case 10 | 13:  # enter
                return [run_infos[i] for i in sorted(selected)]
            case 113 | 27:  # q or escape
                return []
            case curses.KEY_PPAGE:  # page up
                cursor = max(0, cursor - available_rows)
            case curses.KEY_NPAGE:  # page down
                cursor = min(len(run_infos) - 1, cursor + available_rows)
            case _:
                pass


def main() -> None:
    default_runs_dir = SPD_OUT_DIR / "spd"

    parser = argparse.ArgumentParser(description="Interactive WandB run cleanup")
    parser.add_argument("--entity", required=True, help="WandB entity (username or team)")
    parser.add_argument("--project", default="spd", help="WandB project")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir,
        help="Directory containing runs_index.tsv (for extra metadata)",
    )
    args = parser.parse_args()

    print(f"Fetching runs from {args.entity}/{args.project}...")
    wandb_runs = fetch_runs(args.entity, args.project)
    if not wandb_runs:
        print("No runs found.")
        return

    tsv_index = load_tsv_index(args.runs_dir)
    if tsv_index:
        print(f"Loaded {len(tsv_index)} entries from runs_index.tsv")

    run_infos = build_run_infos(wandb_runs, tsv_index)

    to_delete = curses.wrapper(run_picker, run_infos)

    if not to_delete:
        print("No runs selected. Aborted.")
        return

    print(f"\nAbout to delete {len(to_delete)} runs:")
    for info in to_delete:
        print(f"  {info.run_id}  |  {info.state}  |  {info.created_at[:16]}  |  {info.label}")

    response = input(f"\nConfirm deletion of {len(to_delete)} runs? [y/N] ")
    if response.strip().lower() != "y":
        print("Aborted.")
        return

    for info in to_delete:
        info.wandb_run.delete()
        print(f"  Deleted {info.run_id}")

    print(f"\nDone. Deleted {len(to_delete)} runs.")


if __name__ == "__main__":
    main()
