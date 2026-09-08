#!/usr/bin/env python
"""Read the probe ladder's outputs and write the one summary table.

    python summarize_ladder.py --write $DATA_ROOT/ladder/summary.md     # the table
    python summarize_ladder.py --choose-mesh                             # prints c | b | none

Reads, per trial in trials/manifest.tsv: `$DATA_ROOT/ladder/<trial>.rc` (exit code and
duration, written by run_ladder.sh), `$DATA_ROOT/ladder/<trial>.log`, and the run dir's
`metrics.jsonl` (`train/mem/peak_gb_per_rank`, `train/perf/step_time_s`). Steady-state step
time is the mean of the step-20 and step-30 records (the step-10 window contains the
compile). Projected wall-clock for 40k steps multiplies by 1.06: the 1-block reference run's
measured eval + checkpoint overhead at every 500 / slow 4000 (19.35 h wall vs 18.34 h of
pure steps). Stdlib only, so it runs with any python on the pod."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "trials" / "manifest.tsv"
TRIAL_STEPS = 30
STEPS_REAL = 40000
EVAL_OVERHEAD = 1.06


def read_manifest() -> list[dict[str, str]]:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, run_id, config, kind, timeout = line.split("\t")
        rows.append(dict(name=name, run_id=run_id, config=config, kind=kind, timeout=timeout))
    return rows


def metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            out.append(json.loads(line))
    return out


def trial_row(ladder: Path, data_root: Path, t: dict[str, str]) -> dict:
    rc_file = ladder / f"{t['name']}.rc"
    log = ladder / f"{t['name']}.log"
    run_dir = data_root / "runs" / t["run_id"]
    row = dict(
        t,
        ran=rc_file.exists(),
        rc=None,
        duration_min=None,
        peak_gb=None,
        s_per_step=None,
        last_step=None,
        fits=None,
        projected_h=None,
        notes=[],
    )
    if not rc_file.exists():
        row["notes"].append("not run")
        return row
    rc_s, dur_s = rc_file.read_text().split()
    row["rc"], row["duration_min"] = int(rc_s), round(int(dur_s) / 60, 1)
    text = log.read_text(errors="replace") if log.exists() else ""
    if "Traceback" in text:
        row["notes"].append("traceback in log")
    if "RESOURCE_EXHAUSTED" in text or "Out of memory" in text or "OOM" in text:
        row["notes"].append("OOM message")
    if "WATCHDOG" in text:
        row["notes"].append("watchdog killed (hang)")
    if row["rc"] == 124:
        row["notes"].append("timeout/hang")
    recs = metrics(run_dir)
    train = [r for r in recs if "train/perf/step_time_s" in r]
    if train:
        row["last_step"] = max(r["step"] for r in train)
        row["peak_gb"] = round(max(r.get("train/mem/peak_gb_per_rank", 0.0) for r in train), 1)
        steady = [r["train/perf/step_time_s"] for r in train if r["step"] >= 20]
        if steady:
            row["s_per_step"] = round(sum(steady) / len(steady), 2)
            row["projected_h"] = round(row["s_per_step"] * STEPS_REAL * EVAL_OVERHEAD / 3600, 1)
    finished = (
        row["rc"] == 0
        and row["last_step"] == TRIAL_STEPS
        and "traceback in log" not in row["notes"]
    )
    if t["kind"] == "smoke-abgrid":
        grid = run_dir / "ab_grids" / "step_20.js"
        saved = next(
            (
                r.get("eval/ab_grids/saved_components/total")
                for r in recs
                if r.get("step") == 20 and "eval/ab_grids/saved_components/total" in r
            ),
            None,
        )
        row["notes"].append(f"step_20.js {'written' if grid.exists() else 'MISSING'}")
        row["notes"].append(f"saved_components/total={saved}")
        finished = finished and grid.exists() and (saved or 0) > 0
    if t["kind"] == "smoke-resume":
        resumed_at = [int(m) for m in re.findall(r"resumed from checkpoint step (\d+)", text)]
        row["notes"].append(
            "SIGTERM save seen" if "SIGTERM: checkpoint saved" in text else "NO SIGTERM save"
        )
        row["notes"].append(f"resumed from {resumed_at}" if resumed_at else "NO resume line")
        # the resume must start strictly before the end (a leg-1 that ran to 30 proves nothing:
        # the trainer refuses "already trained to step 30" on leg 2)
        finished = (
            finished
            and "SIGTERM: checkpoint saved" in text
            and bool(resumed_at)
            and all(20 <= n < TRIAL_STEPS for n in resumed_at)
        )
    row["fits"] = finished
    return row


def table(rows: list[dict]) -> str:
    head = (
        "| trial | run id | fits? | peak GB/rank | s/step | projected h (40k, x1.06) | wall min | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        fits = "—" if not r["ran"] else ("YES" if r["fits"] else "NO")
        body += (
            f"| {r['name']} | {r['run_id']} | {fits} | {r['peak_gb'] if r['peak_gb'] is not None else '—'} | "
            f"{r['s_per_step'] if r['s_per_step'] is not None else '—'} | "
            f"{r['projected_h'] if r['projected_h'] is not None else '—'} | "
            f"{r['duration_min'] if r['duration_min'] is not None else '—'} | "
            f"rc={r['rc']}; last step {r['last_step']}; {'; '.join(r['notes'])} |\n"
        )
    return head + body


def choose_mesh(rows: list[dict]) -> str:
    fitting = [r for r in rows if r["kind"] == "mesh" and r["fits"] and r["s_per_step"]]
    if not fitting:
        return "none"
    best = min(fitting, key=lambda r: r["s_per_step"])
    return best["name"].split("-", 1)[1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "/workspace/data"))
    )
    ap.add_argument(
        "--write", type=Path, default=None, help="write the markdown table here (also printed)"
    )
    ap.add_argument(
        "--choose-mesh", action="store_true", help="print the fastest fitting mesh letter, or none"
    )
    args = ap.parse_args()
    ladder = args.data_root / "ladder"
    rows = [trial_row(ladder, args.data_root, t) for t in read_manifest()]
    if args.choose_mesh:
        print(choose_mesh(rows))
        return 0
    md = (
        "# addsub-all-layers probe ladder\n\n"
        f"data_root `{args.data_root}`; trials of {TRIAL_STEPS} steps with the step-0 slow eval on; "
        f"projected hours = s/step x {STEPS_REAL} x {EVAL_OVERHEAD} (measured eval+ckpt overhead of the 1-block run).\n\n"
        + table(rows)
        + f"\nchosen mesh: **{choose_mesh(rows)}**\n"
    )
    print(md)
    if args.write:
        args.write.write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
