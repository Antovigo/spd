#!/usr/bin/env python
"""Generate the probe-ladder configs from `../addsub-all-layers-sota.yaml` (never hand-copy).

    python make_trials.py                # regenerate trials/*.yaml + trials/manifest.tsv
    python make_trials.py --check        # also parse every output under the run schema
    python make_trials.py --check --data-root /workspace/data   # + full build (resolves the
                                         # target arch, datasets and the neuron-ranks artifact)

Every trial is the production config with a SMALL, enumerated delta on top; the deltas are
the ladder's whole content, so they are listed here and nowhere else:

  common      pd.steps 30, cadence.train_log_every 10 (step_time_s lands at steps 10/20/30;
              the step-10 window still contains the compile, read 20 and 30), no wandb
              (metrics.jsonl is the record), the trial's own fixed run id and run_name.
  mesh-<m>    cadence.checkpointing none (a 32-block checkpoint is ~20 GB; the mesh probe
              measures memory and step time, not the save path) + the mesh under test.
  smoke-abgrid-<m>
              the AB-grid smoke aimed at the grid's schedule: eval.every 10, slow_every 20
              (EveryAfterFirst fires at step 20; report §8.7), mean_ci_floor 0.0 (only a
              non-empty `saved` set exercises the failing gather; §8.8), a_range/b_range
              [1, 10] (~100 prompts). Checkpointing ON at the production cadence (saves
              step 0 and the final step 30) so the save path is exercised once.
  smoke-resume-<m>
              save_every 20; the driver SIGTERMs the trainer after the step-20 log, expects
              a SIGTERM save, relaunches on the same run id, expects "resumed from
              checkpoint step N" and a clean finish at step 30.
  fallback-128x96-<m>
              the reduced batch (target 128 / nontarget 96, eval 96) — generated, NOT in the
              default ladder (Antoine 2026-09-08: hard time cap); run by hand if both meshes
              fail: `run_ladder.sh --extra fallback-128x96-<m>`.

`pd.steps` is 30 in EVERY trial so the compiled step is shared through the XLA cache across
trials at the same mesh (schedules are traced against `steps`).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
SOTA = HERE.parent / "addsub-all-layers-sota.yaml"
OUT = HERE / "trials"

TRIAL_STEPS = 30
MESHES: dict[str, dict[str, Any]] = {
    # (c): half the frozen-target gathers of (b); target replicated 2x (~4 GB/rank).
    "c": {"replicate": 2, "fsdp": 4, "tp": 1, "sharding": "zero1"},
    # (b): the H100-validated HSDP seat's layout.
    "b": {"replicate": 1, "fsdp": 8, "tp": 1, "sharding": "zero1"},
}
# Fixed ids (`p-<8hex>` is enforced by the trainer) so a rerun of the driver finds and skips
# finished trials, and so the summary can name run dirs before anything ran.
RUN_IDS = {
    "mesh-c": "p-1adde401",
    "mesh-b": "p-1adde402",
    "smoke-abgrid-c": "p-1adde403",
    "smoke-resume-c": "p-1adde404",
    "smoke-abgrid-b": "p-1adde413",
    "smoke-resume-b": "p-1adde414",
    "fallback-128x96-c": "p-1adde405",
    "fallback-128x96-b": "p-1adde406",
}
# Per-trial `timeout` (seconds) — a timeout counts as "does not fit" (a dp>1 OOM hangs).
TIMEOUTS = {"mesh": 4500, "smoke-abgrid": 2700, "smoke-resume": 2700, "fallback": 4500}
# Mesh trials get 75 min: a COLD 32-block compile is silent for tens of minutes before the
# step-0 slow eval even starts (the 4-block run needed 18 min to step 100 with a WARM cache).
# The smokes reuse the mesh trial's compiled step from the cache (same mesh, batch, steps).


def _common(raw: dict[str, Any], name: str) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    cfg["run_name"] = f"addsub-all-layers-trial-{name}"
    cfg["pd"]["steps"] = TRIAL_STEPS
    cfg["cadence"]["train_log_every"] = 10
    cfg.pop("wandb", None)
    return cfg


def _with_mesh(cfg: dict[str, Any], mesh: str) -> dict[str, Any]:
    cfg["runtime"].update(MESHES[mesh])
    return cfg


def mesh_trial(raw: dict[str, Any], mesh: str) -> dict[str, Any]:
    cfg = _with_mesh(_common(raw, f"mesh-{mesh}"), mesh)
    cfg["cadence"]["checkpointing"] = {"kind": "none"}
    return cfg


def smoke_abgrid(raw: dict[str, Any], mesh: str) -> dict[str, Any]:
    cfg = _with_mesh(_common(raw, f"smoke-abgrid-{mesh}"), mesh)
    cfg["eval"]["every"] = 10
    cfg["eval"]["slow_every"] = 20
    grids = [m for m in cfg["eval"]["metrics"] if m["type"] == "ABGridDataset"]
    assert len(grids) == 1, grids
    grids[0].update({"mean_ci_floor": 0.0, "a_range": [1, 10], "b_range": [1, 10]})
    return cfg


def smoke_resume(raw: dict[str, Any], mesh: str) -> dict[str, Any]:
    cfg = _with_mesh(_common(raw, f"smoke-resume-{mesh}"), mesh)
    cfg["cadence"]["checkpointing"]["save_every"] = 20
    return cfg


def fallback(raw: dict[str, Any], mesh: str) -> dict[str, Any]:
    cfg = _with_mesh(_common(raw, f"fallback-128x96-{mesh}"), mesh)
    cfg["cadence"]["checkpointing"] = {"kind": "none"}
    cfg["pd"]["batch_size"] = 128
    cfg["nontarget"]["batch_size"] = 96
    cfg["eval"]["batch_size"] = 96
    return cfg


def _header(name: str, run_id: str) -> str:
    return (
        f"# GENERATED by make_trials.py from ../addsub-all-layers-sota.yaml — do not edit.\n"
        f"# Probe-ladder trial `{name}`, run id {run_id}; the delta from the production config\n"
        f"# is documented in make_trials.py. Comments of the source are not carried over.\n"
    )


def generate() -> list[tuple[str, str, Path, str, int]]:
    raw = yaml.safe_load(SOTA.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, Path, str, int]] = []  # name, run_id, path, kind, timeout
    for mesh in MESHES:
        rows.append(
            (
                f"mesh-{mesh}",
                RUN_IDS[f"mesh-{mesh}"],
                mesh_trial(raw, mesh),
                "mesh",
                TIMEOUTS["mesh"],
            )
        )  # type: ignore[arg-type]
    for mesh in MESHES:
        rows.append(
            (
                f"smoke-abgrid-{mesh}",
                RUN_IDS[f"smoke-abgrid-{mesh}"],
                smoke_abgrid(raw, mesh),
                "smoke-abgrid",
                TIMEOUTS["smoke-abgrid"],
            )
        )  # type: ignore[arg-type]
        rows.append(
            (
                f"smoke-resume-{mesh}",
                RUN_IDS[f"smoke-resume-{mesh}"],
                smoke_resume(raw, mesh),
                "smoke-resume",
                TIMEOUTS["smoke-resume"],
            )
        )  # type: ignore[arg-type]
        rows.append(
            (
                f"fallback-128x96-{mesh}",
                RUN_IDS[f"fallback-128x96-{mesh}"],
                fallback(raw, mesh),
                "fallback",
                TIMEOUTS["fallback"],
            )
        )  # type: ignore[arg-type]
    written: list[tuple[str, str, Path, str, int]] = []
    for name, run_id, cfg, kind, timeout in rows:
        path = OUT / f"{name}.yaml"
        path.write_text(_header(name, run_id) + yaml.safe_dump(cfg, sort_keys=False, width=100))
        written.append((name, run_id, path, kind, timeout))
    manifest = OUT / "manifest.tsv"
    manifest.write_text(
        "# name\trun_id\tconfig\tkind\ttimeout_s\n"
        + "".join(f"{n}\t{r}\t{p.name}\t{k}\t{t}\n" for n, r, p, k, t in written)
    )
    return written


def check(paths: list[Path], data_root: Path | None) -> None:
    from param_decomp.experiments.lm.config import (
        LMTargetedExperimentConfig,
        build_targeted_experiment_config,
    )

    for path in paths:
        cfg = LMTargetedExperimentConfig.model_validate(yaml.safe_load(path.read_text()))
        line = f"parsed  {path.name}: steps={cfg.pd.steps} mesh={cfg.runtime.replicate}x{cfg.runtime.fsdp}x{cfg.runtime.tp}/{cfg.runtime.sharding}"
        if data_root is not None:
            built = build_targeted_experiment_config(cfg, "p-0000abcd", data_root)
            line += f" sites={len(built.target.sites)} chunks={len(built.ci_fn.chunks)}"  # type: ignore[attr-defined]
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="parse every output (and the source) under the run schema",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="with --check: fully build against this data root",
    )
    args = ap.parse_args()
    written = generate()
    for _name, run_id, path, kind, timeout in written:
        print(f"wrote {path.relative_to(HERE)}  ({kind}, {run_id}, timeout {timeout}s)")
    if args.check:
        check([SOTA] + [p for _, _, p, _, _ in written], args.data_root)


if __name__ == "__main__":
    sys.exit(main())
