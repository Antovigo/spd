"""Decompose the pre-norm representation into the active feature's readout direction and the null space.

For each active input feature (a single feature set to `--magnitude`, all others zero), this runs the
**original** target model and the **circuit only** (the decomposed model masked to that feature's
active subcomponents: CI > `--ci-thr`, delta OFF), and at every representation position (the
activation a final RMSnorm would normalise, just before the readout) measures two norms relative to
the readout matrix `R` (row i = feature i's readout direction, `R[i] = u_i`):

- **logit-lens** component ALONG `û_i = u_i/||u_i||`: `|a . û_i|` — how much the representation writes
  to the active feature's output logit (the part a final RMSnorm would rescale).
- **null-space** component: `||(I - P) a||`, where `P = R⁺R` projects onto the row space of `R` (the
  output-relevant subspace) — the magnitude orthogonal to *every* readout direction.

Supported models: resid_mlp (representation = residual stream after each layer, readout = `W_Uᵀ`) and
tms (representation = the `linear1` hidden bottleneck, readout = `linear2`). For tms the readout is
overcomplete (40 rows in 10-dim), so the null space is empty and that row is ~0 by construction.

The figure is two rows (logit-lens / null space) by one column per position, with the original model
on the x-axis and the circuit on the y-axis, sharing square axes from 0 to the global maximum.

Usage:
    python -m spd.scripts.validation.pre_rmsnorm_norm <model_path> \
        [--ci-thr=0.1] [--magnitude=1.0] [--output=PATH] [--output-fig=PATH]
"""

import csv
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from spd.configs import ResidMLPTaskConfig, TMSTaskConfig
from spd.experiments.resid_mlp.models import ResidMLP
from spd.experiments.tms.models import TMSModel
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.scripts.validation.common import load_spd_run
from spd.spd_types import ModelPath

_KINDS = ("along_orig", "along_circ", "null_orig", "null_circ")


def _reprs_and_readout(
    spd_model: ComponentModel, x: Tensor, ci_thr: float, sampling: Any
) -> tuple[list[str], dict[str, Tensor], dict[str, Tensor], Tensor, Tensor]:
    """Return (positions, orig_reprs, circuit_reprs, readout, n_active).

    `readout` is (n_features, d_repr) with row i the readout direction of feature i. Each repr dict
    maps a position name to an (n_sel, d_repr) tensor (the pre-norm representation there).
    """
    target_model = spd_model.target_model
    with torch.no_grad():
        ci_cache = spd_model(x, cache_type="input").cache
        ci = spd_model.calc_causal_importances(pre_weight_acts=ci_cache, sampling=sampling)
        active = {
            name: (ci.lower_leaky[name] > ci_thr).to(x.dtype) for name in spd_model.module_to_c
        }
        n_active = torch.zeros(x.shape[0], device=x.device)
        for m in active.values():
            n_active += m.sum(dim=-1)
        orig_cache = spd_model(x, cache_type="output").cache
        mask_infos = make_mask_infos(active, weight_deltas_and_masks=None)
        circ_cache = spd_model(x, mask_infos=mask_infos, cache_type="output").cache

    if isinstance(target_model, ResidMLP):
        embedding = x.float() @ target_model.W_E.float()
        n_layers = target_model.config.n_layers
        positions = [f"layer{k}_resid" for k in range(n_layers)]

        def reprs(cache: dict[str, Tensor]) -> dict[str, Tensor]:
            out: dict[str, Tensor] = {}
            resid = embedding
            for k in range(n_layers):
                resid = resid + cache[f"layers.{k}.mlp_out"].float()
                out[f"layer{k}_resid"] = resid
            return out

        readout = target_model.W_U.float().T  # (n_features, d_embed)
        return positions, reprs(orig_cache), reprs(circ_cache), readout, n_active

    if isinstance(target_model, TMSModel):
        positions = ["hidden"]
        orig = {"hidden": orig_cache["linear1"].float()}
        circ = {"hidden": circ_cache["linear1"].float()}
        readout = target_model.linear2.weight.float()  # (n_features, n_hidden)
        return positions, orig, circ, readout, n_active

    raise NotImplementedError(f"pre_rmsnorm_norm does not support {type(target_model).__name__}")


def _decompose(a: Tensor, u_hat: Tensor, p_out: Tensor) -> tuple[Tensor, Tensor]:
    """Magnitude of each row of `a` along the active readout direction `u_hat`, and in the readout
    null space (`p_out` is the orthogonal projector onto the readout's row space)."""
    along = (a * u_hat).sum(dim=-1).abs()
    null = (a - a @ p_out).norm(dim=-1)
    return along, null


def _compute(
    spd_model: ComponentModel,
    features: list[int],
    n_features: int,
    magnitude: float,
    ci_thr: float,
    sampling: Any,
    device: torch.device,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Probe one single-active-feature input per feature; split each representation along/null."""
    n_sel = len(features)
    feat_idx = torch.tensor(features, device=device)
    x = torch.zeros(n_sel, n_features, device=device)
    x[torch.arange(n_sel, device=device), feat_idx] = magnitude

    positions, orig_reprs, circ_reprs, readout, n_active = _reprs_and_readout(
        spd_model, x, ci_thr, sampling
    )
    u = readout[feat_idx]  # (n_sel, d_repr) active feature readout directions
    u_hat = u / u.norm(dim=-1, keepdim=True)
    p_out = torch.linalg.pinv(readout) @ readout  # projector onto row space of readout

    decomposed: dict[str, dict[str, Tensor]] = {}
    for p in positions:
        ao_along, ao_null = _decompose(orig_reprs[p], u_hat, p_out)
        ac_along, ac_null = _decompose(circ_reprs[p], u_hat, p_out)
        decomposed[p] = {
            "along_orig": ao_along,
            "along_circ": ac_along,
            "null_orig": ao_null,
            "null_circ": ac_null,
        }

    rows: list[dict[str, Any]] = []
    for i, feat in enumerate(features):
        row: dict[str, Any] = {"feature": feat, "n_active": int(n_active[i].item())}
        for p in positions:
            for kind in _KINDS:
                row[f"{p}_{kind}"] = decomposed[p][kind][i].item()
        rows.append(row)
    return positions, rows


def pre_rmsnorm_norm(
    model_path: ModelPath,
    ci_thr: float = 0.1,
    magnitude: float = 1.0,
    output: str | None = None,
    output_fig: str | None = None,
) -> tuple[Path, Path]:
    """Write the per-feature logit-lens / null-space TSV + the 2-row scatter grid."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    task_config = config.task_config
    assert isinstance(task_config, (ResidMLPTaskConfig, TMSTaskConfig)), (
        f"pre_rmsnorm_norm only supports resid_mlp / tms tasks, got {type(task_config).__name__}"
    )
    target_model = spd_model.target_model
    assert isinstance(target_model, (ResidMLP, TMSModel))
    n_features = target_model.config.n_features

    features = (
        list(task_config.active_indices)
        if task_config.active_indices is not None
        else list(range(n_features))
    )
    logger.info(f"Probing {len(features)} input features (magnitude={magnitude}, ci_thr={ci_thr})")

    positions, rows = _compute(
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

    fields = ["feature", "n_active"] + [f"{p}_{k}" for p in positions for k in _KINDS]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows to {out_path}")

    _plot(positions, rows, run_dir.name, fig_path)
    logger.info(f"Saved figure to {fig_path}")
    return out_path, fig_path


def _pretty(position: str) -> str:
    if position == "hidden":
        return "hidden (pre-readout)"
    k = position.removeprefix("layer").removesuffix("_resid")
    return f"layer {k} residual"


def _plot(positions: list[str], rows: list[dict[str, Any]], title: str, fig_path: Path) -> None:
    """Top row: logit-lens (along active readout dir). Bottom row: readout null space. One column per
    position. Original on x, circuit on y, shared square axes [0, max]."""
    n_active = [r["n_active"] for r in rows]
    cmin, cmax = min(n_active), max(n_active)
    annotate = len(rows) <= 20

    vmax = max(r[f"{p}_{k}"] for p in positions for k in _KINDS for r in rows) * 1.05 or 1.0

    rowspec = [
        ("along", "logit-lens norm\n(‖ active readout dir)"),
        ("null", "null-space norm\n(⊥ all readout dirs)"),
    ]
    n = len(positions)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10), squeeze=False)
    sc = None
    for col, p in enumerate(positions):
        for r_i, (kind, _) in enumerate(rowspec):
            ax = axes[r_i][col]
            xs = np.array([row[f"{p}_{kind}_orig"] for row in rows])
            ys = np.array([row[f"{p}_{kind}_circ"] for row in rows])
            sc = ax.scatter(
                xs, ys, c=n_active, cmap="viridis", vmin=cmin, vmax=cmax, s=30, alpha=0.85
            )
            ax.plot([0, vmax], [0, vmax], color="gray", ls="--", lw=1, label="circuit = original")
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            if annotate:
                for i, row in enumerate(rows):
                    ax.annotate(
                        str(row["feature"]),
                        (xs[i], ys[i]),
                        fontsize=8,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )
            ax.set_xlabel("original model")
        axes[0][col].set_title(_pretty(p))
    for r_i, (_, label) in enumerate(rowspec):
        axes[r_i][0].set_ylabel(f"circuit — {label}")
    axes[0][0].legend(fontsize=8)

    assert sc is not None
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="# active subcomponents")
    fig.suptitle(
        f"Representation: active readout direction vs null space — {title}\n"
        "top: logit-lens (along active readout) · bottom: readout null space"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire(pre_rmsnorm_norm)
