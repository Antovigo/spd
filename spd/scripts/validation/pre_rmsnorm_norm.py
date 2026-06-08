"""Compare the pre-RMSnorm residual-stream norm of the original model vs. the circuit only.

For each active input feature (a single feature set to `--magnitude`, all others zero), this runs:
- the **original** target model and reads its pre-RMSnorm residual stream (`return_residual=True`,
  i.e. the residual after the MLP layers, before the final RMSnorm / `W_U` readout); and
- the **circuit only**: the decomposed model masked to that feature's active subcomponents (CI >
  `--ci-thr`, delta OFF), read at the same pre-RMSnorm point.

It plots the original norm on the x-axis against the circuit-only norm on the y-axis, one point per
input feature. Points on the `y = x` line mean the active subcomponents alone reproduce the
original residual-stream magnitude that feeds the final RMSnorm.

Which features are probed: every feature for a full-data run, or only `task_config.active_indices`
for a targeted run.

Resid-MLP tasks only. `return_residual=True` is valid for both RMSnorm and no-norm targets (it
returns the residual before the optional final norm in either case).

Usage:
    python -m spd.scripts.validation.pre_rmsnorm_norm <model_path> \
        [--ci-thr=0.1] [--magnitude=1.0] [--output=PATH] [--output-fig=PATH]

Output files (default in the decomposed model's folder):
- `pre_rmsnorm_norm.tsv` — one row per probed input feature.
- `pre_rmsnorm_norm.png` — original vs circuit-only pre-RMSnorm norm scatter.
"""

import csv
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from spd.configs import ResidMLPTaskConfig
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.scripts.validation.common import load_spd_run
from spd.spd_types import ModelPath

FIELDS = ["feature", "n_active", "orig_norm", "circuit_norm"]


def _compute(
    spd_model: ComponentModel,
    features: list[int],
    n_features: int,
    magnitude: float,
    ci_thr: float,
    sampling: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    """One single-active-feature input per probed feature; return per-feature norms + active count."""
    target_model = spd_model.target_model
    n_sel = len(features)
    x = torch.zeros(n_sel, n_features, device=device)
    x[torch.arange(n_sel, device=device), torch.tensor(features, device=device)] = magnitude

    with torch.no_grad():
        orig_resid = target_model(x, return_residual=True)
        assert isinstance(orig_resid, Tensor)
        orig_norm = orig_resid.float().norm(dim=-1)  # (n_sel,)

        cache = spd_model(x, cache_type="input").cache
        ci = spd_model.calc_causal_importances(pre_weight_acts=cache, sampling=sampling)
        active_masks = {
            name: (ci.lower_leaky[name] > ci_thr).to(orig_resid.dtype)
            for name in spd_model.module_to_c
        }
        n_active = torch.zeros(n_sel, device=device)
        for m in active_masks.values():
            n_active += m.sum(dim=-1)

        mask_infos = make_mask_infos(active_masks, weight_deltas_and_masks=None)
        circuit_resid = spd_model(x, return_residual=True, mask_infos=mask_infos)
        assert isinstance(circuit_resid, Tensor)
        circuit_norm = circuit_resid.float().norm(dim=-1)  # (n_sel,)

    return [
        {
            "feature": feat,
            "n_active": int(n_active[i].item()),
            "orig_norm": orig_norm[i].item(),
            "circuit_norm": circuit_norm[i].item(),
        }
        for i, feat in enumerate(features)
    ]


def pre_rmsnorm_norm(
    model_path: ModelPath,
    ci_thr: float = 0.1,
    magnitude: float = 1.0,
    output: str | None = None,
    output_fig: str | None = None,
) -> tuple[Path, Path]:
    """Write the per-feature original-vs-circuit pre-RMSnorm norm TSV + scatter figure."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    task_config = config.task_config
    assert isinstance(task_config, ResidMLPTaskConfig), (
        f"pre_rmsnorm_norm only supports resid_mlp tasks, got {type(task_config).__name__}"
    )
    n_features = spd_model.target_model.config.n_features  # pyright: ignore[reportAttributeAccessIssue]

    # Targeted run: probe only the target features. Full-data run: probe every feature.
    features = (
        list(task_config.active_indices)
        if task_config.active_indices is not None
        else list(range(n_features))
    )
    logger.info(f"Probing {len(features)} input features (magnitude={magnitude}, ci_thr={ci_thr})")

    rows = _compute(
        spd_model=spd_model,
        features=features,
        n_features=n_features,
        magnitude=magnitude,
        ci_thr=ci_thr,
        sampling=config.sampling,
        device=device,
    )

    out_path = Path(output).expanduser() if output else run_dir / "pre_rmsnorm_norm.tsv"
    fig_path = Path(output_fig).expanduser() if output_fig else run_dir / "pre_rmsnorm_norm.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows to {out_path}")

    _plot(rows, run_dir.name, fig_path)
    logger.info(f"Saved figure to {fig_path}")
    return out_path, fig_path


def _plot(rows: list[dict[str, Any]], title: str, fig_path: Path) -> None:
    """Scatter original vs circuit-only pre-RMSnorm norm, one point per feature, with a y=x line."""
    orig = np.array([r["orig_norm"] for r in rows])
    circuit = np.array([r["circuit_norm"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(orig, circuit, c=[r["n_active"] for r in rows], cmap="viridis", s=30, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="# active subcomponents")

    lo = min(orig.min(), circuit.min())
    hi = max(orig.max(), circuit.max())
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, color="gray", ls="--", lw=1, label="circuit = original")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    # Few points (targeted): label each with its feature index.
    if len(rows) <= 20:
        for r in rows:
            ax.annotate(
                str(r["feature"]),
                (r["orig_norm"], r["circuit_norm"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("original model pre-RMSnorm norm")
    ax.set_ylabel("circuit-only pre-RMSnorm norm")
    ax.set_title(f"Pre-RMSnorm residual norm: original vs circuit\n{title}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire(pre_rmsnorm_norm)
