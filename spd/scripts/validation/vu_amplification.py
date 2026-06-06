"""Compare each component's read→write (V→U) gain against the original weight's gain there.

Each `LinearComponents` component `c` reads along `V[:, c]` and writes `(V[:, c]ᵀ x) · U[c, :]`, so its
rank-1 contribution to the weight is `U[c, :] ⊗ V[:, c]`. Along its own (unit) read direction
`V̂_c` and (unit) write direction `Û_c`, the component's gain is exactly the `(Û_c, V̂_c)` matrix
element of that rank-1 weight, `‖V_c‖·‖U_c‖`. The original target weight's gain along the SAME
direction pair is `Û_cᵀ W_orig V̂_c`. Their ratio is how much the component amplifies its own V→U
direction relative to the original model:

    amplification_c = (‖V_c‖ · ‖U_c‖) / (Û_cᵀ · W_orig · V̂_c)

`> 1` means the component pushes harder in its read→write direction than the whole original weight
does there (the other components + delta must cancel the excess, since `Σ_c V_c U_cᵀ + delta =
W_orig`). The ratio is invariant to the component's internal sign convention (flipping both `V_c`
and `U_c` leaves numerator and denominator unchanged). A negative ratio means the original weight
maps the read direction onto the *opposite* write direction. This is a weight-only analysis — no
data is run.

Only `LinearComponents` modules are supported, restricted to the alive components in the input TSV
(from `find_alive_components.py`).

Usage:
    python -m spd.scripts.validation.vu_amplification <model_path> <alive_components_tsv> \
        [--output=PATH] [--output-fig=PATH]

Output files (default in the decomposed model's folder):
- `vu_amplification.tsv` — one row per alive component.
- `vu_amplification.png` — component gain vs original gain, and the amplification distribution.
"""

import csv
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from spd.log import logger
from spd.models.components import LinearComponents
from spd.scripts.validation.common import (
    build_module_lookup,
    load_spd_run,
)
from spd.spd_types import ModelPath

FIELDS = [
    "layer",
    "matrix",
    "component",
    "v_norm",
    "u_norm",
    "component_gain",
    "orig_gain",
    "amplification",
]


def _load_alive(
    path: Path, module_lookup: dict[tuple[int, str], str]
) -> dict[str, list[tuple[int, str, int]]]:
    """Group alive components by module name. Returns {module_name: [(layer, matrix, component)]}."""
    by_module: dict[str, list[tuple[int, str, int]]] = {}
    with path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            layer = int(record["layer"])
            matrix = record["matrix"]
            component = int(record["component"])
            key = (layer, matrix)
            assert key in module_lookup, (
                f"No decomposed module matches layer={layer}, matrix={matrix}. "
                f"Available: {sorted(module_lookup.keys())}"
            )
            by_module.setdefault(module_lookup[key], []).append((layer, matrix, component))
    return by_module


def _module_rows(
    components: LinearComponents,
    orig_weight: Tensor,  # (d_out, d_in)
    alive: list[tuple[int, str, int]],
) -> list[dict[str, Any]]:
    """Compute the directional gain ratio for every alive component of one module (vectorised)."""
    idx = torch.tensor([c for _, _, c in alive], device=components.V.device)
    v = components.V[:, idx].float()  # (d_in, n)
    u = components.U[idx, :].float()  # (n, d_out)

    v_norm = v.norm(dim=0)  # (n,)
    u_norm = u.norm(dim=1)  # (n,)
    assert (v_norm > 0).all() and (u_norm > 0).all(), "alive component has a zero V or U vector"

    component_gain = v_norm * u_norm  # = Û_cᵀ (V_c U_cᵀ) V̂_c
    v_hat = v / v_norm  # (d_in, n)
    u_hat = u / u_norm[:, None]  # (n, d_out)
    w_v_hat = orig_weight.float() @ v_hat  # (d_out, n)
    orig_gain = (u_hat * w_v_hat.T).sum(dim=1)  # (n,) = Û_cᵀ W_orig V̂_c
    amplification = component_gain / orig_gain

    rows: list[dict[str, Any]] = []
    for i, (layer, matrix, component) in enumerate(alive):
        rows.append(
            {
                "layer": layer,
                "matrix": matrix,
                "component": component,
                "v_norm": v_norm[i].item(),
                "u_norm": u_norm[i].item(),
                "component_gain": component_gain[i].item(),
                "orig_gain": orig_gain[i].item(),
                "amplification": amplification[i].item(),
            }
        )
    return rows


def vu_amplification(
    model_path: ModelPath,
    alive_components_path: str,
    output: str | None = None,
    output_fig: str | None = None,
) -> tuple[Path, Path]:
    """Write the per-component V→U gain ratio TSV + a figure, for the alive components."""
    spd_model, _config, run_dir = load_spd_run(model_path)

    module_lookup = build_module_lookup(spd_model.target_module_paths)
    alive_path = Path(alive_components_path).expanduser()
    alive_by_module = _load_alive(alive_path, module_lookup)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for module_name, alive in alive_by_module.items():
            components = spd_model.components[module_name]
            assert isinstance(components, LinearComponents), (
                f"{module_name} is {type(components).__name__}, only LinearComponents is supported"
            )
            orig_weight = spd_model.target_weight(module_name)
            rows.extend(_module_rows(components, orig_weight, alive))

    rows.sort(key=lambda r: (r["layer"], r["matrix"], r["component"]))

    out_path = Path(output).expanduser() if output else run_dir / "vu_amplification.tsv"
    fig_path = Path(output_fig).expanduser() if output_fig else run_dir / "vu_amplification.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    amps = np.array([r["amplification"] for r in rows])
    n_amplify = int((amps > 1).sum())
    logger.info(
        f"{len(rows)} alive components: {n_amplify} amplify (ratio > 1), "
        f"median amplification {np.median(amps):.3g}, max {amps.max():.3g}"
    )
    logger.info(f"Saved {out_path}")

    _plot(rows, fig_path)
    logger.info(f"Saved figure to {fig_path}")
    return out_path, fig_path


def _plot(rows: list[dict[str, Any]], fig_path: Path) -> None:
    """Left: component gain vs original gain (above the y=x line = amplifies). Right: the sorted
    amplification ratio per component on a symlog axis, with a reference line at 1."""
    matrices = sorted({r["matrix"] for r in rows})
    cmap = plt.get_cmap("tab10")
    color_of = {m: cmap(i % 10) for i, m in enumerate(matrices)}
    colors = [color_of[r["matrix"]] for r in rows]

    orig_gain = np.array([r["orig_gain"] for r in rows])
    component_gain = np.array([r["component_gain"] for r in rows])
    amps = np.array([r["amplification"] for r in rows])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    ax0.scatter(orig_gain, component_gain, c=colors, s=25, alpha=0.8)
    lim = max(np.abs(orig_gain).max(), component_gain.max()) * 1.05
    ax0.plot([-lim, lim], [-lim, lim], color="gray", ls="--", lw=1, label="component = original")
    ax0.axvline(0, color="gray", lw=0.5)
    ax0.set_xlim(-lim, lim)
    ax0.set_ylim(0, lim)
    ax0.set_xlabel("original-weight gain  Û_cᵀ W_orig V̂_c")
    ax0.set_ylabel("component gain  ‖V_c‖·‖U_c‖")
    ax0.set_title("Component vs. original gain (above line ⇒ amplifies)")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8)

    order = np.argsort(-amps)
    ranks = np.arange(len(amps))
    ax1.scatter(ranks, amps[order], c=[colors[i] for i in order], s=25, alpha=0.8)
    ax1.axhline(1.0, color="C3", ls="--", lw=1, label="amplification = 1")
    ax1.axhline(0.0, color="gray", lw=0.5)
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_xlabel("component (sorted by amplification)")
    ax1.set_ylabel("amplification  (component gain / original gain)")
    ax1.set_title("Amplification per alive component")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    handles = [plt.Line2D([], [], marker="o", ls="", color=color_of[m], label=m) for m in matrices]
    fig.legend(handles=handles, loc="lower center", ncol=len(matrices), fontsize=8)
    fig.suptitle("V→U directional amplification vs. the original model")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire(vu_amplification)
