"""Output layout and ids of CI filter runs and of the ablation studies built on them.

Everything lives with the decomposition it reads:

    <run_dir>/<cf-id>-<step>/       one filtering run of checkpoint <step> (below)
    <run_dir>/Trash/                retired filtering runs, kept for the record; nothing
                                    reads from it
    <run_dir>/analysis/ablations/step_<step>/  anything that measures the effect of ABLATING
                                    components, and the tables comparing several sources:
        decomposition_alive.npz     {site: (C,)} bool, the decomposition's own alive set on the
                                    pool (the row set of the tables and sweeps)
        components.tsv              one row per alive component, every measurement joined
        <source>/                   screen (`components.npz`, `summary.json`, `verified.json`)
                                    and full sweep (`ablation_sweep.tsv`) for one CI source:
                                    `run` (the decomposition's own CI) or a filter id
        model_ablation.tsv          candidates subtracted from the MODEL (weight delta on)
        nontarget_probe/            the same subtraction scored on general text, plus the
                                    browsable `app/index.html`
        mask_ablations/<source>.json  global-mask ablations (all-on, alive-union, dead-only, ...)
        l0_by_position.tsv          per-position L0 of the decomposition and every filter

One filtering run:

    <run_dir>/<cf-id>-<step>/
        config.yaml                 the pinned CIFilterConfig
        logs/                       the launcher's stage logs (pod runs: `run_pipeline.sh`)
        training/metrics.jsonl      per-step optimization log
        training/ci_fn/             the fine-tuned CI fn (orbax; a later filter's `init`)
        eval/pool_evals.jsonl       pool evaluations (before the first step, periodic, final)
        eval/summary.json           the final PoolEval (its `pgd_recon` is the final PGD eval)
        eval/pgd_recon.json         the PGD recon eval of the starting and the final CI fn
        alive/alive.json            {site: [component ids]} with max CI > alive_threshold
        alive/max_ci.npz            {site: (C,)} max output CI over every prompt and position
        alive/kept.npz              {site: (C,)} bool, the components NOT removed by pruning
        ab_grids/index.html         the applet (open over file://)
        ab_grids/manifest.js        slice index
        ab_grids/<op>_pos<p>.js     one (operation, position) slice (split ones: full grids
                                    in `<op>_pos<p>/<site>.js`)
        ab_grids/alive.html, alive/ the alive-only model's inner grids (`scripts/alive_grids.py`)
        dataset/                    activations of the original and the rounded-CI decomposed
                                    model over the pool (`scripts/collect_dataset.py`, its own
                                    README.md)
"""

import secrets
from dataclasses import dataclass
from pathlib import Path

TRASH = "Trash"
"""Subfolder of `<run_dir>/` holding retired filtering runs."""


def filter_dir(run_dir: Path, step: int, filter_id: str) -> Path:
    return run_dir / f"{filter_id}-{step}"


def filter_dirs(run_dir: Path, step: int) -> list[Path]:
    """Every ACTIVE filtering run of this checkpoint (not `Trash/`), sorted."""
    return sorted(
        d
        for d in run_dir.glob(f"*-{step}")
        if d.is_dir() and (d / "config.yaml").exists() and (d / "alive").is_dir()
    )


def filter_id(directory: Path, step: int) -> str:
    return directory.name.removesuffix(f"-{step}")


def ablations_dir(run_dir: Path, step: int) -> Path:
    """Where every ablation study of this run and step writes (layout in the module docstring)."""
    return run_dir / "analysis" / "ablations" / f"step_{step}"


def ablation_source_dir(run_dir: Path, step: int, source: str) -> Path:
    """One CI source's screen and sweep: `run` or a filter id."""
    return ablations_dir(run_dir, step) / source


@dataclass(frozen=True)
class CIFilterOutputs:
    root: Path

    @staticmethod
    def for_run(run_dir: Path, step: int, filter_id: str) -> "CIFilterOutputs":
        return CIFilterOutputs(filter_dir(run_dir, step, filter_id))

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def metrics(self) -> Path:
        return self.root / "training" / "metrics.jsonl"

    @property
    def ci_fn(self) -> Path:
        return self.root / "training" / "ci_fn"

    @property
    def pool_evals(self) -> Path:
        return self.root / "eval" / "pool_evals.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / "eval" / "summary.json"

    @property
    def pgd(self) -> Path:
        return self.root / "eval" / "pgd_recon.json"

    @property
    def alive(self) -> Path:
        return self.root / "alive" / "alive.json"

    @property
    def max_ci(self) -> Path:
        return self.root / "alive" / "max_ci.npz"

    @property
    def kept(self) -> Path:
        return self.root / "alive" / "kept.npz"

    @property
    def grids(self) -> Path:
        return self.root / "ab_grids"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    def create(self) -> None:
        """Fail closed on an existing filter dir."""
        self.root.mkdir(parents=True, exist_ok=False)
        for sub in ("training", "eval", "alive", "ab_grids", "logs"):
            (self.root / sub).mkdir()


def new_ci_filter_id() -> str:
    """A fresh `cf-<8 hex>` id; any explicit name is equally valid (it names the output dir and
    the wandb run)."""
    return f"cf-{secrets.token_hex(4)}"
