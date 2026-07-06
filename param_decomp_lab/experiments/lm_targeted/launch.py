"""Launch a targeted-PD (tPD) LM decomposition run
(`python -m param_decomp_lab.experiments.lm_targeted.run`).

CONFIG-DRIVEN via `runtime.dp`, exactly like `pd-lm` (`experiments.lm.launch`): `dp is None`
runs the trainer INLINE; `dp = N` (a multiple of 8) snapshots the tree to an immutable
shared-FS workspace, installs the `[cuda]` extra there, and sbatches across `N // 8` nodes.
The only tPD difference is the module run (`...lm_targeted.run`) and validating the config
against `LMTargetedExperimentConfig`; everything else reuses `experiments.lm.launch`.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import fire
import yaml

from param_decomp.log import logger
from param_decomp_lab.experiments.lm.launch import (
    GPUS_PER_NODE,
    WORKSPACES_DIR,
    _build_workspace,
    _config_path_relative_to_repo,
    _render_rank_env,
    _stamp_config,
    _wandb_url,
)
from param_decomp_lab.experiments.lm_targeted.config import LMTargetedExperimentConfig
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job

_SRUN_FLAGS = "--kill-on-bad-exit=1 --ntasks-per-node=1"

_MODULE = "param_decomp_lab.experiments.lm_targeted.run"


def _validate_config(config_path: Path) -> tuple[LMTargetedExperimentConfig, str]:
    raw = yaml.safe_load(config_path.read_text())
    assert "run_id" not in raw, f"{config_path}: run_id is minted at submit, omit it"
    cfg = LMTargetedExperimentConfig(**raw)
    return cfg, cfg.run_name


def _rank_command(config_rel: Path, run_id: str, rank_env: str) -> str:
    return (
        f"source .venv/bin/activate\n{rank_env}\n"
        f"exec python -m {_MODULE} "
        f"{shlex.quote(str(config_rel))} --run-id {shlex.quote(run_id)}"
    )


def _run_local(config_rel: Path, run_name: str, group: str | None, tags: list[str]) -> None:
    """Mint a run id, stamp the live config in place, and run the trainer inline."""
    run_id = generate_run_id("param_decomp")
    _stamp_config(REPO_ROOT / config_rel, group, tags)
    logger.section(f"pd-lm-targeted local: {run_name} ({run_id})")
    subprocess.run(
        [sys.executable, "-m", _MODULE, str(config_rel), "--run-id", run_id],
        cwd=REPO_ROOT,
        check=True,
        env=os.environ.copy(),
    )


def main(
    config_path: str,
    *,
    time: str = "72:00:00",
    qos: str | None = None,
    run_id: str | None = None,
    group: str | None = None,
    tags: str | None = None,
    comment: str | None = None,
) -> None:
    """Launch a targeted-PD LM trainer run. The mode (inline vs SLURM) is a pure function of
    the config's `runtime.dp`; see `experiments.lm.launch.main` for the shared arg semantics.
    """
    config_rel = _config_path_relative_to_repo(config_path)
    cfg, run_name = _validate_config(REPO_ROOT / config_rel)
    tag_list = [s.strip() for s in tags.split(",")] if tags is not None else []

    dp = cfg.runtime.dp
    if dp is None:
        _run_local(config_rel, run_name, group, tag_list)
        return
    assert dp % GPUS_PER_NODE == 0, f"runtime.dp={dp} must be a multiple of {GPUS_PER_NODE}"
    nodes = dp // GPUS_PER_NODE

    if run_id is None:
        run_id = generate_run_id("param_decomp")
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")
        workspace = WORKSPACES_DIR / run_id
        _build_workspace(workspace, snapshot_ref, config_rel, group, tag_list)
    else:
        snapshot_ref = f"refs/runs/snapshot/{run_id}"
        workspace = WORKSPACES_DIR / run_id
        assert workspace.exists(), f"no workspace to resubmit: {workspace}"

    wandb_url = _wandb_url(cfg, run_id)
    slurm_config = SlurmConfig(
        job_name=f"pd-{run_name}",
        partition=None,
        qos=qos,
        n_gpus=GPUS_PER_NODE,
        n_nodes=nodes,
        ntasks_per_node=1,
        time=time,
        signal="TERM@300",
        requeue=True,
        comment=comment if comment is not None else (wandb_url or run_id),
    )
    rank_env = _render_rank_env(cfg.runtime.launch_env)
    srun = f"srun --nodes={nodes} --ntasks={nodes} {_SRUN_FLAGS}"
    command = f"{srun} bash -c {shlex.quote(_rank_command(config_rel, run_id, rank_env))}"
    script = generate_script(slurm_config, command, setup=f'cd "{workspace}"')
    result = submit_slurm_job(script, "pd-lm-targeted")

    logger.section("pd-lm-targeted job submitted!")
    summary: dict[str, str | None] = {
        "Run ID": run_id,
        "Run name": run_name,
        "Job ID": result.job_id,
        "Log file": result.log_pattern,
        "Script": str(result.script_path),
        "Snapshot": snapshot_ref,
        "Workspace": str(workspace),
    }
    if wandb_url is not None:
        summary["WandB run URL"] = wandb_url
    logger.values(summary)


def cli() -> None:
    fire.Fire(main)
