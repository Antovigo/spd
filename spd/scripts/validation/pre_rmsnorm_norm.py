"""Decompose the residual stream into the active feature's logit direction and the unembed null space.

For each active input feature (a single feature set to `--magnitude`, all others zero), this runs the
**original** target model and the **circuit only** (the decomposed model masked to that feature's
active subcomponents: CI > `--ci-thr`, delta OFF), and at every residual-stream position measures two
norms of the residual `a`:

- **logit-lens** component ALONG the active unembedding direction `u_i = W_U[:, i] / ||W_U[:, i]||`:
  `|a . u_i|` — how much the residual writes to the active feature's output logit (the part a final
  RMSnorm would rescale). Question 1: does the circuit produce amplified / deflated logits along the
  active unembedding direction?
- **null-space** component: `||(I - P) a||`, where `P = W_U W_U^+` projects onto the column space of
  `W_U` — the residual magnitude orthogonal to *every* unembedding column (output-irrelevant; it
  touches no logit). Question 2: does the circuit write more to the unembedding null space than the
  original model does?

The figure is two rows (logit-lens / null space) by one column per residual position (resid_mlp1 has
one), with the original model on the x-axis and the circuit on the y-axis. All plots share square axes
from 0 to the global maximum.

Which features are probed: every feature for a full-data run, or only `task_config.active_indices`
for a targeted run. Resid-MLP tasks only.

Usage:
    python -m spd.scripts.validation.pre_rmsnorm_norm <model_path> \
        [--ci-thr=0.1] [--magnitude=1.0] [--output=PATH] [--output-fig=PATH]

Output files (default in the decomposed model's folder):
- `pre_rmsnorm_norm.tsv` — one row per probed feature; per position the along/null norm for the
  original model and the circuit.
- `pre_rmsnorm_norm.png` — 2 rows (logit-lens / null space) x one column per position.
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
from spd.experiments.resid_mlp.models import ResidMLP
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.scripts.validation.common import load_spd_run
from spd.spd_types import ModelPath

_KINDS = ("along_orig", "along_circ", "null_orig", "null_circ")


def _residual_vectors(
    cache: dict[str, Tensor], embedding: Tensor, n_layers: int
) -> dict[str, Tensor]:
    """Residual stream after each layer writes: `embedding + sum_{j<=k} mlp_out_j`, one per layer."""
    vectors: dict[str, Tensor] = {}
    resid = embedding
    for k in range(n_layers):
        resid = resid + cache[f"layers.{k}.mlp_out"].float()
        vectors[f"layer{k}_resid"] = resid
    return vectors


def _decompose(a: Tensor, u_hat: Tensor, p_out: Tensor) -> tuple[Tensor, Tensor]:
    """Magnitude of each row of `a` along the active unembed direction `u_hat`, and in the W_U null
    space (`p_out` is the orthogonal projector onto the column space of W_U)."""
    along = (a * u_hat).sum(dim=-1).abs()  # |a . u_i|
    null = (a - a @ p_out).norm(dim=-1)  # ||(I - P) a||
    return along, null


def _compute(
    spd_model: ComponentModel,
    features: list[int],
    n_features: int,
    n_layers: int,
    magnitude: float,
    ci_thr: float,
    sampling: Any,
    device: torch.device,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Probe one single-active-feature input per feature; split each residual into logit + null-space."""
    target_model = spd_model.target_model
    assert isinstance(target_model, ResidMLP)
    n_sel = len(features)
    feat_idx = torch.tensor(features, device=device)
    x = torch.zeros(n_sel, n_features, device=device)
    x[torch.arange(n_sel, device=device), feat_idx] = magnitude
    embedding = x.float() @ target_model.W_E.float()

    w_u = target_model.W_U.float()  # (d_embed, n_features)
    # Unit unembedding direction of each probed feature: column W_U[:, i], normalized. (n_sel, d_embed)
    u = w_u[:, feat_idx].T
    u_hat = u / u.norm(dim=-1, keepdim=True)
    # Orthogonal projector onto col(W_U) (the output-relevant subspace of the residual stream).
    p_out = w_u @ torch.linalg.pinv(w_u)  # (d_embed, d_embed)

    with torch.no_grad():
        ci_cache = spd_model(x, cache_type="input").cache
        ci = spd_model.calc_causal_importances(pre_weight_acts=ci_cache, sampling=sampling)
        active_masks = {
            name: (ci.lower_leaky[name] > ci_thr).to(embedding.dtype)
            for name in spd_model.module_to_c
        }
        n_active = torch.zeros(n_sel, device=device)
        for m in active_masks.values():
            n_active += m.sum(dim=-1)

        orig_cache = spd_model(x, cache_type="output").cache
        mask_infos = make_mask_infos(active_masks, weight_deltas_and_masks=None)
        circuit_cache = spd_model(x, mask_infos=mask_infos, cache_type="output").cache

    orig_resids = _residual_vectors(orig_cache, embedding, n_layers)
    circuit_resids = _residual_vectors(circuit_cache, embedding, n_layers)
    positions = [f"layer{k}_resid" for k in range(n_layers)]

    decomposed: dict[str, dict[str, Tensor]] = {}
    for p in positions:
        ao_along, ao_null = _decompose(orig_resids[p], u_hat, p_out)
        ac_along, ac_null = _decompose(circuit_resids[p], u_hat, p_out)
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
    assert isinstance(task_config, ResidMLPTaskConfig), (
        f"pre_rmsnorm_norm only supports resid_mlp tasks, got {type(task_config).__name__}"
    )
    target_model = spd_model.target_model
    assert isinstance(target_model, ResidMLP)
    n_features = target_model.config.n_features
    n_layers = target_model.config.n_layers

    # Targeted run: probe only the target features. Full-data run: probe every feature.
    features = (
        list(task_config.active_indices)
        if task_config.active_indices is not None
        else list(range(n_features))
    )
    logger.info(
        f"Probing {len(features)} input features over {n_layers} residual positions "
        f"(magnitude={magnitude}, ci_thr={ci_thr})"
    )

    positions, rows = _compute(
        spd_model=spd_model,
        features=features,
        n_features=n_features,
        n_layers=n_layers,
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
    k = position.removeprefix("layer").removesuffix("_resid")
    return f"layer {k} residual"


def _plot(positions: list[str], rows: list[dict[str, Any]], title: str, fig_path: Path) -> None:
    """Top row: logit-lens (along active unembed dir). Bottom row: W_U null space. One column per
    residual position. Original on x, circuit on y, shared square axes [0, max]."""
    n_active = [r["n_active"] for r in rows]
    cmin, cmax = min(n_active), max(n_active)
    annotate = len(rows) <= 20

    vmax = max(r[f"{p}_{k}"] for p in positions for k in _KINDS for r in rows) * 1.05

    rowspec = [
        ("along", "logit-lens norm\n(‖ active unembed dir)"),
        ("null", "null-space norm\n(⊥ all unembed dirs)"),
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
        f"Residual: active logit direction vs W_U null space — {title}\n"
        "top: logit-lens (along active unembed) · bottom: unembedding null space"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire(pre_rmsnorm_norm)
