"""Output layout and ids of CI filter runs.

A CI filter's outputs live with the decomposition it reads:

    <run_dir>/analysis/ci_filter/step_<step>/<cf-id>/
        config.yaml                 the pinned CIFilterConfig
        training/metrics.jsonl      per-step optimization log
        training/ci_fn/             the fine-tuned CI fn (orbax; a later filter's `init`)
        eval/pool_evals.jsonl       pool evaluations (before the first step, periodic, final)
        eval/summary.json           the final PoolEval (its `pgd_recon` is the final PGD eval)
        eval/pgd_recon.json         the PGD recon eval of the starting and the final CI fn
        alive/alive.json            {site: [component ids]} with max CI > alive_threshold
        alive/max_ci.npz            {site: (C,)} max output CI over every prompt and position
        ab_grids/index.html         the applet (open over file://)
        ab_grids/manifest.js        slice index
        ab_grids/<op>_pos<p>.js     one (operation, position) slice
"""

import secrets
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CIFilterOutputs:
    root: Path

    @staticmethod
    def for_run(run_dir: Path, step: int, filter_id: str) -> "CIFilterOutputs":
        return CIFilterOutputs(run_dir / "analysis" / "ci_filter" / f"step_{step}" / filter_id)

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
    def grids(self) -> Path:
        return self.root / "ab_grids"

    def create(self) -> None:
        """Fail closed on an existing filter dir."""
        self.root.mkdir(parents=True, exist_ok=False)
        for sub in ("training", "eval", "alive", "ab_grids"):
            (self.root / sub).mkdir()


def new_ci_filter_id() -> str:
    """A fresh `cf-<8 hex>` id; any explicit name is equally valid (it names the output dir and
    the wandb run)."""
    return f"cf-{secrets.token_hex(4)}"
