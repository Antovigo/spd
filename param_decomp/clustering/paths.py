"""Canonical output paths and ID generation for clustering artifacts."""

from pathlib import Path

from param_decomp.infra.run_files import generate_run_id


def clustering_run_dir(data_root: Path, run_id: str) -> Path:
    return data_root / "clustering" / "runs" / run_id


def clustering_harvest_dir(data_root: Path, harvest_id: str) -> Path:
    return data_root / "clustering" / "harvests" / harvest_id


def clustering_ensemble_dir(data_root: Path, ensemble_id: str) -> Path:
    return data_root / "clustering" / "ensembles" / ensemble_id


def new_run_id() -> str:
    return generate_run_id("clustering/runs")


def new_harvest_id() -> str:
    return generate_run_id("clustering/harvests")


def new_ensemble_id() -> str:
    return generate_run_id("clustering/ensembles")
