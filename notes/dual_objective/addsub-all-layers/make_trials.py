#!/usr/bin/env python
"""Generate a probe ladder's trial configs from a production config (never hand-copy).

    python make_trials.py                                   # the 8x H100 ladder
    python make_trials.py --profile 4xh100                  # the 4x H100 ladder
    python make_trials.py --profile 4xh100 --check --data-root $DATA_ROOT

A PROFILE is one pod shape: which production config it derives from, how many GPUs, which
meshes to A/B, and where its trials and run ids live. Run ids are disjoint between profiles,
so the two ladders can share a `$DATA_ROOT` without colliding.

Trial names are `<kind>-<mesh>-b<target>x<nontarget>`, uniformly — the batch is part of the
identity because the ladder may have to fall back to a smaller one, and the smoke that
follows must run at whatever shape actually won.

Every trial is the production config with a SMALL, enumerated delta; the deltas are the
ladder's whole content, so they are listed here and nowhere else:

  common    pd.steps 30, cadence.train_log_every 10 (step_time_s lands at steps 10/20/30;
            the step-10 window still contains the compile, read 20 and 30), no wandb
            (metrics.jsonl is the record), the trial's own fixed run id and run_name.
  mesh      cadence.checkpointing none (a 32-block checkpoint is ~20 GB; a mesh probe
            measures memory and step time, not the save path) + the mesh + the batch.
  smoke-abgrid
            aimed at the AB grid's own schedule: eval.every 10, slow_every 20
            (EveryAfterFirst fires at step 20; report §8.7), mean_ci_floor 0.0 (only a
            non-empty `saved` set exercises the failing gather; §8.8), a_range/b_range
            [1, 10] (~100 prompts). Checkpointing ON at the production cadence, so the save
            path is exercised once.
  smoke-resume
            save_every 20; the driver SIGTERMs the trainer after the step-20 log, expects a
            SIGTERM save, relaunches on the same run id, expects "resumed from checkpoint
            step N" and a clean finish at step 30.

`pd.steps` is 30 in EVERY trial so the compiled step is shared through the XLA cache across
trials at the same mesh and batch (schedules are traced against `steps`).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
TRIAL_STEPS = 30

# (replicate, fsdp, tp, sharding). `zero1` shards the optimizer state over the FULL data
# mesh under every entry here, so these differ in how the frozen target is sharded and how
# often it is gathered, not in optimizer-state footprint. Never `zero1` at `fsdp: 1`
# (report §8.5: 26x all-reduce) — the config builder accepts it, so the guard is lore.
PROFILES: dict[str, dict[str, Any]] = {
    "8xh100": {
        "sota": "addsub-all-layers-8xh100-sota.yaml",
        "out": "trials",
        "ladder": "ladder",
        "run_id": "p-a1132b01",
        "devices": 8,
        # (a) `replicate 8 / fsdp 1 / ddp` is deliberately absent: it replicates the whole
        # trainable state for ~46 GB of static per rank on an 80 GB card.
        "meshes": {"c": (2, 4, 1, "zero1"), "b": (1, 8, 1, "zero1")},
        "extra_meshes": {},
    },
    "4xh100": {
        "sota": "addsub-all-layers-4xh100-sota.yaml",
        "out": "trials-4xh100",
        "ladder": "ladder-4xh100",
        # p-b4132c01 was the PRE-BALANCE run (batch normalization, diluted hidden
        # coefficients, LR 3.2e-04); it reached step 14804 and is kept for comparison. The
        # balanced sota gets its own id because resume byte-compares the pinned config.
        "run_id": "p-ba1a0c01",
        "devices": 4,
        "meshes": {"f4": (1, 4, 1, "zero1"), "r2f2": (2, 2, 1, "zero1")},
        # generated but never in the default order: 45.7 GB static leaves ~29 GB for
        # activations, which the estimate does not support. `run_ladder.sh --only` it.
        "extra_meshes": {"ddp": (4, 1, 1, "ddp")},
    },
}

TIMEOUTS = {"mesh": 21600, "smoke-abgrid": 7200, "smoke-resume": 10800}
# MEASURED 2026-09-09, and the reason these are hours rather than minutes: a COLD 32-block
# compile runs well past 45 min with no output at all. The first 4xh100 ladder killed FOUR
# trials at the 45-min watchdog fuse while XLA was still compiling at 200% CPU — two
# different meshes died at an identical 48 min, which is a timer, not a memory limit. Sizing
# from the 4-block run (18 min to step 100 on a WARM cache) was simply wrong at 224 sites and
# four passes. A compile reaches the XLA cache only when it COMPLETES, so a trial killed
# mid-compile caches nothing and its successor starts over. The smokes reuse the mesh trial's
# compiled step from the cache (same mesh, batch and steps), so they are the short ones.


def batch_tag(target: int, nontarget: int) -> str:
    return f"b{target}x{nontarget}"


def trial_run_id(profile: str, name: str) -> str:
    """A trial's fixed run id, DERIVED FROM ITS NAME rather than its position in the list.

    Run ids must be stable under edits to the trial set. The ladder skips a finished trial
    by its `.rc` file, but the trainer keys the run DIRECTORY by id, and that directory
    holds a byte-compared `launch_config.yaml`. Allocated sequentially, adding a mesh or
    reordering the list would hand an id that already has a run dir to a DIFFERENT trial,
    which then either resumes an unrelated trajectory or refuses on the pinned config.
    Hashing the (profile, name) pair makes an id depend on nothing but the identity of the
    trial, so the set can grow and shrink freely and the profiles cannot collide."""
    digest = hashlib.blake2b(f"{profile}/{name}".encode(), digest_size=4).hexdigest()
    return f"p-{digest}"


def _common(raw: dict[str, Any], name: str) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    cfg["run_name"] = f"{raw['run_name']}-trial-{name}"
    cfg["pd"]["steps"] = TRIAL_STEPS
    cfg["cadence"]["train_log_every"] = 10
    cfg.pop("wandb", None)
    return cfg


def _apply(
    cfg: dict[str, Any], mesh: tuple[int, int, int, str], batch: tuple[int, int, int]
) -> dict[str, Any]:
    replicate, fsdp, tp, sharding = mesh
    cfg["runtime"].update(replicate=replicate, fsdp=fsdp, tp=tp, sharding=sharding)
    target, nontarget, eval_batch = batch
    cfg["pd"]["batch_size"] = target
    cfg["nontarget"]["batch_size"] = nontarget
    cfg["eval"]["batch_size"] = eval_batch
    return cfg


def mesh_trial(raw, name, mesh, batch):
    cfg = _apply(_common(raw, name), mesh, batch)
    cfg["cadence"]["checkpointing"] = {"kind": "none"}
    return cfg


def smoke_abgrid(raw, name, mesh, batch):
    cfg = _apply(_common(raw, name), mesh, batch)
    cfg["eval"]["every"] = 10
    cfg["eval"]["slow_every"] = 20
    grids = [m for m in cfg["eval"]["metrics"] if m["type"] == "ABGridDataset"]
    assert len(grids) == 1, grids
    grids[0].update({"mean_ci_floor": 0.0, "a_range": [1, 10], "b_range": [1, 10]})
    return cfg


def smoke_resume(raw, name, mesh, batch):
    cfg = _apply(_common(raw, name), mesh, batch)
    cfg["cadence"]["checkpointing"]["save_every"] = 20
    return cfg


BUILDERS = {"mesh": mesh_trial, "smoke-abgrid": smoke_abgrid, "smoke-resume": smoke_resume}


def generate(profile_name: str) -> tuple[Path, list[tuple[str, str, Path, str, int]]]:
    profile = PROFILES[profile_name]
    sota = HERE.parent / profile["sota"]
    raw = yaml.safe_load(sota.read_text())
    world = raw["runtime"]["replicate"] * raw["runtime"]["fsdp"] * raw["runtime"]["tp"]
    assert world == profile["devices"], (
        f"{sota.name} authors a {world}-GPU mesh but profile {profile_name!r} is "
        f"{profile['devices']} GPUs"
    )
    authored = (raw["pd"]["batch_size"], raw["nontarget"]["batch_size"], raw["eval"]["batch_size"])
    fallback = (128, 96, 96)
    batches = [authored] if authored == fallback else [authored, fallback]

    out = HERE / profile["out"]
    out.mkdir(parents=True, exist_ok=True)
    meshes = {**profile["meshes"], **profile["extra_meshes"]}
    rows: list[tuple[str, str, Path, str, int]] = []
    for target, nontarget, eval_batch in batches:
        tag = batch_tag(target, nontarget)
        for mesh_name, mesh in meshes.items():
            for kind, builder in BUILDERS.items():
                # the smokes exist only for the meshes the driver may choose
                if kind != "mesh" and mesh_name in profile["extra_meshes"]:
                    continue
                name = f"{kind}-{mesh_name}-{tag}"
                run_id = trial_run_id(profile_name, name)
                cfg = builder(raw, name, mesh, (target, nontarget, eval_batch))
                path = out / f"{name}.yaml"
                path.write_text(
                    f"# GENERATED by make_trials.py --profile {profile_name} from "
                    f"../{profile['sota']} — do not edit.\n"
                    f"# Trial {name!r}, run id {run_id}; the delta from the production config is\n"
                    f"# documented in make_trials.py. Source comments are not carried over.\n"
                    + yaml.safe_dump(cfg, sort_keys=False, width=100)
                )
                rows.append((name, run_id, path, kind, TIMEOUTS[kind]))
    ids = [r[1] for r in rows]
    assert len(set(ids)) == len(ids), f"run id collision in profile {profile_name}: {ids}"
    manifest = out / "manifest.tsv"
    manifest.write_text(
        f"# profile\t{profile_name}\n# devices\t{profile['devices']}\n"
        f"# authored_batch\t{batch_tag(authored[0], authored[1])}\n"
        f"# fallback_batch\t{batch_tag(fallback[0], fallback[1])}\n"
        f"# default_meshes\t{','.join(profile['meshes'])}\n"
        "# name\trun_id\tconfig\tkind\ttimeout_s\n"
        + "".join(f"{n}\t{r}\t{p.name}\t{k}\t{t}\n" for n, r, p, k, t in rows)
    )
    return out, rows


def check(paths: list[Path], data_root: Path | None) -> None:
    from param_decomp.experiments.lm.config import (
        LMTargetedExperimentConfig,
        build_targeted_experiment_config,
    )

    for path in paths:
        cfg = LMTargetedExperimentConfig.model_validate(yaml.safe_load(path.read_text()))
        rt = cfg.runtime
        line = (
            f"parsed  {path.name:34s} steps={cfg.pd.steps:<6d} "
            f"mesh={rt.replicate}x{rt.fsdp}x{rt.tp}/{rt.sharding} "
            f"batch={cfg.pd.batch_size}/{cfg.nontarget.batch_size}"
        )
        if data_root is not None:
            built = build_targeted_experiment_config(cfg, "p-0000abcd", data_root)
            line += f" sites={len(built.target.sites)}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--profile", default="8xh100", choices=sorted(PROFILES))
    ap.add_argument("--check", action="store_true", help="parse every output (and the source)")
    ap.add_argument("--data-root", type=Path, default=None, help="with --check: also build")
    args = ap.parse_args()
    out, rows = generate(args.profile)
    for _name, run_id, path, kind, timeout in rows:
        print(f"wrote {path.relative_to(HERE)}  ({kind}, {run_id}, timeout {timeout}s)")
    if args.check:
        sota = HERE.parent / PROFILES[args.profile]["sota"]
        check([sota] + [p for _, _, p, _, _ in rows], args.data_root)


if __name__ == "__main__":
    sys.exit(main())
