"""Interactive local folder cleanup with terminal UI for selecting run folders to delete."""

import argparse
import csv
import curses
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from spd.settings import SPD_OUT_DIR

RUN_ID_PATTERN = re.compile(r"^[a-z]+-[0-9a-f]+$")


def _folder_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


@dataclass
class FolderInfo:
    run_id: str
    folder: Path
    date: str
    completed: bool
    duration_hours: str
    size_bytes: int
    label: str
    notes: str

    def display_line(self, max_width: int) -> str:
        done_str = "yes" if self.completed else "no"
        size_str = _human_size(self.size_bytes)
        parts = [
            f"{self.run_id:<14}",
            f"{self.date:<18}",
            f"{done_str:<6}",
            f"{self.duration_hours:<8}",
            f"{size_str:<10}",
            f"{self.label:<30}",
        ]
        if self.notes:
            parts.append(self.notes)
        line = " │ ".join(parts)
        return line[:max_width]


def _load_tsv_index(runs_dir: Path) -> dict[str, dict[str, str]]:
    tsv_path = runs_dir / "runs_index.tsv"
    if not tsv_path.exists():
        return {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["run_id"]: row for row in reader}


def scan_run_folders(runs_dir: Path) -> list[FolderInfo]:
    tsv_index = _load_tsv_index(runs_dir)
    infos: list[FolderInfo] = []

    for child in runs_dir.iterdir():
        if not child.is_dir() or not RUN_ID_PATTERN.match(child.name):
            continue

        run_id = child.name
        metadata: dict[str, str | bool | float] = {}
        metadata_path = child / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        tsv_row = tsv_index.get(run_id, {})

        date = str(metadata.get("date", tsv_row.get("date", "")))
        completed_raw = metadata.get("completed", tsv_row.get("completed", ""))
        completed = completed_raw is True or completed_raw == "True"
        duration = str(metadata.get("duration", tsv_row.get("duration_hours", "")))
        label = str(metadata.get("label", tsv_row.get("label", "")))
        notes = str(metadata.get("notes", tsv_row.get("notes", "")))

        infos.append(
            FolderInfo(
                run_id=run_id,
                folder=child,
                date=date,
                completed=completed,
                duration_hours=duration,
                size_bytes=_folder_size_bytes(child),
                label=label,
                notes=notes,
            )
        )

    infos.sort(key=lambda x: x.date, reverse=True)
    return infos


HEADER_PARTS = [
    f"{'run_id':<14}",
    f"{'date':<18}",
    f"{'done':<6}",
    f"{'hours':<8}",
    f"{'size':<10}",
    f"{'label':<30}",
    "notes",
]
HEADER = " │ ".join(HEADER_PARTS)


def folder_picker(stdscr: curses.window, folder_infos: list[FolderInfo]) -> list[FolderInfo]:
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

        total_size = sum(folder_infos[i].size_bytes for i in selected)
        stdscr.attron(curses.A_BOLD)
        stdscr.addnstr(
            0,
            0,
            f" {len(folder_infos)} folders | {len(selected)} selected ({_human_size(total_size)})",
            max_x - 1,
        )
        stdscr.addnstr(1, 0, f" {HEADER}", max_x - 1)
        stdscr.attroff(curses.A_BOLD)
        stdscr.addnstr(2, 0, "─" * (max_x - 1), max_x - 1)

        if cursor < scroll_offset:
            scroll_offset = cursor
        elif cursor >= scroll_offset + available_rows:
            scroll_offset = cursor - available_rows + 1

        for i in range(available_rows):
            idx = scroll_offset + i
            if idx >= len(folder_infos):
                break
            row_y = header_lines + i
            info = folder_infos[idx]
            marker = "[x]" if idx in selected else "[ ]"
            line = f" {marker} {info.display_line(max_x - 6)}"

            attr = curses.A_NORMAL
            if idx == cursor:
                attr |= curses.A_REVERSE
            if idx in selected:
                attr |= curses.color_pair(2)
            elif not info.completed:
                attr |= curses.color_pair(3)

            stdscr.addnstr(row_y, 0, line, max_x - 1, attr)

        footer_y = max_y - 1
        footer = " ↑↓:move  SPACE:toggle  a:all  n:none  f:incomplete  ENTER:confirm  q:quit"
        stdscr.addnstr(footer_y, 0, footer, max_x - 1, curses.A_DIM)

        stdscr.refresh()
        key = stdscr.getch()

        match key:
            case curses.KEY_UP | 107:  # up or k
                cursor = max(0, cursor - 1)
            case curses.KEY_DOWN | 106:  # down or j
                cursor = min(len(folder_infos) - 1, cursor + 1)
            case 32:  # space
                if cursor in selected:
                    selected.discard(cursor)
                else:
                    selected.add(cursor)
                cursor = min(len(folder_infos) - 1, cursor + 1)
            case 97:  # a - select all
                selected = set(range(len(folder_infos)))
            case 110:  # n - select none
                selected.clear()
            case 102:  # f - select incomplete
                for idx, info in enumerate(folder_infos):
                    if not info.completed:
                        selected.add(idx)
            case 10 | 13:  # enter
                return [folder_infos[i] for i in sorted(selected)]
            case 113 | 27:  # q or escape
                return []
            case curses.KEY_PPAGE:  # page up
                cursor = max(0, cursor - available_rows)
            case curses.KEY_NPAGE:  # page down
                cursor = min(len(folder_infos) - 1, cursor + available_rows)
            case _:
                pass


def main() -> None:
    default_runs_dir = SPD_OUT_DIR / "spd"

    parser = argparse.ArgumentParser(description="Interactive local run folder cleanup")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir,
        help="Directory containing run folders",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    assert runs_dir.exists(), f"Runs directory not found: {runs_dir}"

    print(f"Scanning folders in {runs_dir}...")
    folder_infos = scan_run_folders(runs_dir)
    if not folder_infos:
        print("No run folders found.")
        return

    print(f"Found {len(folder_infos)} run folders.")
    to_delete = curses.wrapper(folder_picker, folder_infos)

    if not to_delete:
        print("No folders selected. Aborted.")
        return

    total_size = sum(info.size_bytes for info in to_delete)
    print(f"\nAbout to delete {len(to_delete)} folders ({_human_size(total_size)}):")
    for info in to_delete:
        print(
            f"  {info.run_id}  |  {info.date}  |  {_human_size(info.size_bytes)}  |  {info.label}"
        )

    response = input(f"\nConfirm deletion of {len(to_delete)} folders? [y/N] ")
    if response.strip().lower() != "y":
        print("Aborted.")
        return

    for info in to_delete:
        shutil.rmtree(info.folder)
        print(f"  Removed {info.folder}")

    print(f"\nDone. Removed {len(to_delete)} folders.")


if __name__ == "__main__":
    main()
