"""Geometric interaction strength vs coactivation analysis.

For each module in an SPD model, this script computes:
1. Geometric interaction strength (GIS): how much each component's activation
   geometrically interferes with others, based on the absolute left singular
   vectors (U matrices). GIS(i→j) = |U_i|^T|U_j| / ||U_i||^2
2. Coactivation fraction (from harvest data): how often component i is active
   given that component j is active. P(i active | j active).

Outputs per-module:
- Scatter plots of GIS vs coactivation fraction
- GIS heatmaps (sorted by activation density)
- GIS × coactivation product heatmaps

Usage:
    python spd/scripts/geometric_interaction/geometric_interaction.py spd/scripts/geometric_interaction/config.yaml
    python spd/scripts/geometric_interaction/geometric_interaction.py --model_path="wandb:..."
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.gridspec import GridSpec
from pydantic import Field
from torch import Tensor

from spd.base_config import BaseConfig
from spd.log import logger
from spd.settings import SPD_OUT_DIR
from spd.utils.run_utils import save_file


class GeometricInteractionConfig(BaseConfig):
    """Configuration for geometric interaction analysis."""

    model_path: str = Field(..., description="Path to SPD model (wandb: or local)")
    harvest_id: str | None = Field(
        None,
        description="Harvest ID (e.g. 'h-20260319_121635'). Uses latest if None.",
    )
    alive_density_threshold: float = Field(
        default=0.0001,
        description="Minimum activation density to consider a component alive",
    )
    output_dir: str | None = Field(
        None,
        description="Directory to save results (defaults to 'out/<run_id>' relative to script)",
    )


# ── Data loading ──────────────────────────────────────────────────────────────


def extract_run_id(model_path: str) -> str:
    """Extract the short run id from a wandb or local path."""
    return model_path.rstrip("/").split("/")[-1]


def resolve_run_dir(model_path: str) -> Path:
    from spd.utils.wandb_utils import parse_wandb_run_path

    try:
        _entity, project, run_id = parse_wandb_run_path(str(model_path))
        run_dir = SPD_OUT_DIR / "runs" / f"{project}-{run_id}"
        assert run_dir.exists(), f"Run dir not found: {run_dir}"
        return run_dir
    except ValueError:
        return Path(model_path).parent


def load_component_uv(
    model_path: str,
) -> dict[str, tuple[Float[Tensor, "C d_out"], Float[Tensor, "d_in C"]]]:
    """Load U and V matrices from checkpoint, keyed by module path.

    Parses state dict keys of the form ``_components.<module_key>.U`` where
    ``<module_key>`` uses dashes as separators (e.g. ``h-0-mlp-c_fc``).
    """
    from spd.utils.general_utils import fetch_latest_local_checkpoint

    run_dir = resolve_run_dir(model_path)
    checkpoint_path = fetch_latest_local_checkpoint(run_dir, prefix="model")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    uv: dict[str, dict[str, Tensor]] = defaultdict(dict)
    for key, tensor in state_dict.items():
        if not key.startswith("_components."):
            continue
        # _components.<module_key>.<param>
        _, module_key, param_name = key.split(".", maxsplit=2)
        if param_name in ("U", "V"):
            module_name = module_key.replace("-", ".")
            uv[module_name][param_name] = tensor

    result: dict[str, tuple[Float[Tensor, "C d_out"], Float[Tensor, "d_in C"]]] = {}
    for module_name, params in sorted(uv.items()):
        assert "U" in params and "V" in params, f"Missing U or V for {module_name}"
        result[module_name] = (params["U"], params["V"])

    return result


def resolve_harvest_dir(run_id: str, harvest_id: str | None) -> Path:
    harvest_root = SPD_OUT_DIR / "harvest" / run_id
    assert harvest_root.exists(), f"No harvest data for run {run_id} at {harvest_root}"

    if harvest_id is not None:
        harvest_dir = harvest_root / harvest_id
        assert harvest_dir.exists(), f"Harvest dir not found: {harvest_dir}"
        return harvest_dir

    # Pick the most recent (sorted lexicographically — dirs are timestamped)
    candidates = sorted(d for d in harvest_root.iterdir() if d.is_dir())
    assert candidates, f"No harvest subdirectories in {harvest_root}"
    logger.info(f"Auto-selected harvest: {candidates[-1].name}")
    return candidates[-1]


def load_harvest_coactivation(
    run_id: str, harvest_id: str | None
) -> tuple[list[str], Float[Tensor, " N"], Float[Tensor, "N N"], int]:
    """Load coactivation counts from harvest ``component_correlations.pt``.

    Returns:
        component_keys: e.g. ["h.0.mlp.c_fc:0", "h.0.mlp.c_fc:1", ...]
        count_i: per-component activation count (tokens where component was active)
        count_ij: pairwise coactivation count (tokens where both were active)
        count_total: total tokens processed during harvest
    """
    harvest_dir = resolve_harvest_dir(run_id, harvest_id)
    corr_path = harvest_dir / "component_correlations.pt"
    assert corr_path.exists(), f"No correlation data at {corr_path}"

    data = torch.load(corr_path, map_location="cpu", weights_only=True)
    return (
        data["component_keys"],
        data["count_i"],
        data["count_ij"],
        data["count_total"],
    )


# ── Per-module splitting ──────────────────────────────────────────────────────


def parse_component_keys(component_keys: list[str]) -> dict[str, list[int]]:
    """Parse ``"module_name:idx"`` keys into ``{module_name: [global_indices]}``."""
    module_to_global: dict[str, list[int]] = defaultdict(list)
    for global_idx, key in enumerate(component_keys):
        module_name = key.rsplit(":", 1)[0]
        module_to_global[module_name].append(global_idx)
    return dict(module_to_global)


def compute_per_module_coactivation(
    component_keys: list[str],
    count_i: Float[Tensor, " N"],
    count_ij: Float[Tensor, "N N"],
    count_total: int,
) -> tuple[
    dict[str, Float[Tensor, " C"]],
    dict[str, Float[Tensor, "C C"]],
    dict[str, Float[Tensor, "C C"]],
]:
    """Split global harvest counts into per-module activation density, coactivation fractions,
    and raw coactivation counts.

    Coactivation fraction[i, j] = P(i active | j active) = count_ij[i,j] / count_j.
    """
    module_to_inds = parse_component_keys(component_keys)

    activation_density: dict[str, Float[Tensor, " C"]] = {}
    coactivation_fractions: dict[str, Float[Tensor, "C C"]] = {}
    coactivation_counts: dict[str, Float[Tensor, "C C"]] = {}

    for module_name, global_inds in sorted(module_to_inds.items()):
        idx = torch.tensor(global_inds)
        counts = count_i[idx].float()
        activation_density[module_name] = counts / count_total

        coact_counts = count_ij[idx][:, idx].float()
        coactivation_counts[module_name] = coact_counts
        # P(i active | j active) = count_ij[i,j] / count_j
        # denom[i,j] = counts[j] → broadcast counts along rows
        denom = counts.unsqueeze(0).expand_as(coact_counts)
        frac = coact_counts / denom
        coactivation_fractions[module_name] = torch.nan_to_num(frac, nan=0.0)

    return activation_density, coactivation_fractions, coactivation_counts


# ── GIS computation ───────────────────────────────────────────────────────────


def compute_geometric_interaction_strength(
    uv_by_module: dict[str, tuple[Float[Tensor, "C d_out"], Float[Tensor, "d_in C"]]],
) -> dict[str, Float[Tensor, "C C"]]:
    """Compute geometric interaction strength between component pairs.

    GIS(i→j) = |U_i|^T |U_j| / ||U_i||^2

    Measures "what fraction of component i's energy overlaps with component j's
    output direction". Asymmetric: GIS(i→j) != GIS(j→i).
    """
    gis_matrices: dict[str, Float[Tensor, "C C"]] = {}

    for module_name, (U, _V) in sorted(uv_by_module.items()):
        abs_U = U.abs()  # (C, d_out)
        norms_sq = (U * U).sum(dim=1)  # (C,) — L2 norms squared of raw U

        # |U_i|^T |U_j| for all (i, j) pairs
        inner_products = einops.einsum(abs_U, abs_U, "C1 d, C2 d -> C1 C2")

        # Divide each row i by ||U_i||^2
        gis = inner_products / norms_sq.unsqueeze(1)
        gis = torch.nan_to_num(gis, nan=0.0)
        gis_matrices[module_name] = gis

    return gis_matrices


# ── Alive component selection ─────────────────────────────────────────────────


def select_alive_components(
    density: Float[Tensor, " C"],
    alive_threshold: float,
) -> Float[Tensor, " n_alive"]:
    """Return indices of alive components sorted by density (descending)."""
    sorted_inds = torch.argsort(density, descending=True)
    alive_inds = sorted_inds[density[sorted_inds] > alive_threshold]
    return alive_inds


def crop_to_alive(
    matrix: Float[Tensor, "C C"],
    alive_inds: Float[Tensor, " n_alive"],
) -> Float[Tensor, "n_alive n_alive"]:
    return matrix[alive_inds][:, alive_inds]


# ── Plotting ──────────────────────────────────────────────────────────────────


def _count_to_sizes(
    counts_flat: np.ndarray,
    s_min: float = 0.3,
    s_max: float = 8.0,
) -> np.ndarray:
    """Map raw coactivation counts to marker sizes via log-scale normalisation."""
    log_counts = np.log1p(counts_flat)
    lo, hi = log_counts.min(), log_counts.max()
    if hi <= lo:
        return np.full_like(log_counts, (s_min + s_max) / 2)
    t = (log_counts - lo) / (hi - lo)
    return s_min + t * (s_max - s_min)


def plot_scatter_per_module(
    gis_matrices: dict[str, Float[Tensor, "C C"]],
    coactivation_fractions: dict[str, Float[Tensor, "C C"]],
    coactivation_counts: dict[str, Float[Tensor, "C C"]],
    activation_density: dict[str, Float[Tensor, " C"]],
    alive_threshold: float,
    output_dir: Path,
    run_id: str,
) -> None:
    """Per-module scatter plots of GIS vs coactivation fraction (alive components only).

    Each point is a (i, j) pair from the full matrix (both directions), matching the
    original analysis behaviour. Dot size scales with absolute coactivation count.
    Marginal histograms show density along each axis.
    """
    scatter_dir = output_dir / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    for module_name in sorted(gis_matrices.keys()):
        density = activation_density[module_name]
        alive_inds = select_alive_components(density, alive_threshold)
        n_alive = len(alive_inds)

        if n_alive < 2:
            logger.warning(f"{module_name}: {n_alive} alive, skipping")
            continue

        gis = crop_to_alive(gis_matrices[module_name], alive_inds)
        coact = crop_to_alive(coactivation_fractions[module_name], alive_inds)
        raw_counts = crop_to_alive(coactivation_counts[module_name], alive_inds)

        gis_flat = gis.flatten().cpu().numpy()
        coact_flat = coact.flatten().cpu().numpy()
        counts_flat = raw_counts.flatten().cpu().numpy()
        sizes = _count_to_sizes(counts_flat)

        # Layout: main scatter + top histogram + right histogram
        fig = plt.figure(figsize=(9, 9))
        gs = fig.add_gridspec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05
        )
        ax_main = fig.add_subplot(gs[1, 0])
        ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_main)
        # Hide the empty corner
        fig.add_subplot(gs[0, 1]).set_visible(False)

        ax_main.scatter(gis_flat, coact_flat, alpha=0.35, s=sizes, linewidths=0)
        ax_main.set_ylim(-0.01, 1.01)
        ax_main.set_xlabel("Geometric Interaction Strength")
        ax_main.set_ylabel("Coactivation Fraction")

        ax_hist_x.hist(gis_flat, bins=80, color="#0173B2", alpha=0.7, edgecolor="none")
        ax_hist_x.tick_params(labelbottom=False)
        ax_hist_x.set_ylabel("Count")
        ax_hist_x.set_title(f"{module_name}  ({n_alive} alive components)")

        ax_hist_y.hist(
            coact_flat,
            bins=80,
            orientation="horizontal",
            color="#0173B2",
            alpha=0.7,
            edgecolor="none",
        )
        ax_hist_y.tick_params(labelleft=False)
        ax_hist_y.set_xlabel("Count")

        ax_main.text(
            0.98,
            0.98,
            f"pairs: {len(gis_flat)}\nrun: {run_id}",
            transform=ax_main.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        safe_name = module_name.replace(".", "_")
        path = scatter_dir / f"{safe_name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  {path}")


def plot_heatmap_grid(
    matrices: dict[str, Float[Tensor, "C C"]],
    activation_density: dict[str, Float[Tensor, " C"]],
    alive_threshold: float,
    output_path: Path,
    cmap: str,
    cbar_label: str,
    title_suffix: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Plot a grid of heatmaps (one per module) into a single figure, sorted by density."""
    module_names = sorted(matrices.keys())
    n_modules = len(module_names)

    fig = plt.figure(figsize=(8, 8 * n_modules))
    gs: GridSpec = fig.add_gridspec(n_modules, 2, width_ratios=[17, 1], wspace=0.1)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    images = []

    for i, module_name in enumerate(module_names):
        density = activation_density[module_name]
        alive_inds = select_alive_components(density, alive_threshold)
        n_alive = len(alive_inds)

        if n_alive < 2:
            continue

        mat = crop_to_alive(matrices[module_name], alive_inds)

        ax = fig.add_subplot(gs[i, 0])
        im = ax.matshow(mat.cpu().numpy(), aspect="auto", cmap=cmap, norm=norm)
        images.append(im)
        ax.set_title(f"{module_name} ({n_alive} alive) — {title_suffix}")
        ax.set_xlabel("Component index (sorted by density)")
        ax.set_ylabel("Component index (sorted by density)")

    if images:
        cbar_ax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  {output_path}")


def plot_gis_heatmaps(
    gis_matrices: dict[str, Float[Tensor, "C C"]],
    activation_density: dict[str, Float[Tensor, " C"]],
    alive_threshold: float,
    output_dir: Path,
) -> None:
    plot_heatmap_grid(
        matrices=gis_matrices,
        activation_density=activation_density,
        alive_threshold=alive_threshold,
        output_path=output_dir / "heatmaps" / "gis.png",
        cmap="Reds",
        cbar_label="Geometric Interaction Strength",
        title_suffix="GIS",
    )


def plot_coactivation_heatmaps(
    coactivation_fractions: dict[str, Float[Tensor, "C C"]],
    activation_density: dict[str, Float[Tensor, " C"]],
    alive_threshold: float,
    output_dir: Path,
) -> None:
    plot_heatmap_grid(
        matrices=coactivation_fractions,
        activation_density=activation_density,
        alive_threshold=alive_threshold,
        output_path=output_dir / "heatmaps" / "coactivation.png",
        cmap="Blues",
        cbar_label="Coactivation Fraction",
        title_suffix="Coactivation",
    )


def plot_gis_coact_product_heatmaps(
    gis_matrices: dict[str, Float[Tensor, "C C"]],
    coactivation_fractions: dict[str, Float[Tensor, "C C"]],
    activation_density: dict[str, Float[Tensor, " C"]],
    alive_threshold: float,
    output_dir: Path,
) -> None:
    products = {
        name: gis_matrices[name] * coactivation_fractions[name]
        for name in gis_matrices
        if name in coactivation_fractions
    }
    plot_heatmap_grid(
        matrices=products,
        activation_density=activation_density,
        alive_threshold=alive_threshold,
        output_path=output_dir / "heatmaps" / "gis_x_coactivation.png",
        cmap="Purples",
        cbar_label="GIS x Coactivation Fraction",
        title_suffix="GIS x Coact",
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main(config_path: Path | str | None = None, **overrides: Any) -> None:
    if config_path is not None:
        config = GeometricInteractionConfig.from_file(config_path)
    else:
        config = GeometricInteractionConfig(**overrides)

    run_id = extract_run_id(config.model_path)

    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
    else:
        output_dir = Path(__file__).parent / "out" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────

    logger.info(f"Loading component weights from {config.model_path}")
    uv_by_module = load_component_uv(config.model_path)
    for name, (U, V) in sorted(uv_by_module.items()):
        logger.info(f"  {name}: U={tuple(U.shape)}, V={tuple(V.shape)}")

    logger.info(f"Loading harvest coactivation data for run {run_id}")
    component_keys, count_i, count_ij, count_total = load_harvest_coactivation(
        run_id, config.harvest_id
    )
    logger.info(f"  {len(component_keys)} components, {count_total:,} tokens")

    # ── Compute ───────────────────────────────────────────────────────────

    activation_density, coactivation_fractions, coactivation_counts = (
        compute_per_module_coactivation(component_keys, count_i, count_ij, count_total)
    )

    for name, density in sorted(activation_density.items()):
        n_alive = int((density > config.alive_density_threshold).sum().item())
        logger.info(f"  {name}: {n_alive}/{len(density)} alive")

    logger.info("Computing geometric interaction strengths...")
    gis_matrices = compute_geometric_interaction_strength(uv_by_module)

    # Validate that GIS and harvest modules match
    gis_modules = set(gis_matrices.keys())
    harvest_modules = set(coactivation_fractions.keys())
    shared_modules = gis_modules & harvest_modules
    assert shared_modules, (
        f"No overlapping modules between checkpoint and harvest.\n"
        f"  Checkpoint: {sorted(gis_modules)}\n"
        f"  Harvest: {sorted(harvest_modules)}"
    )
    if gis_modules != harvest_modules:
        logger.warning(
            f"Module mismatch — only using {len(shared_modules)} shared modules.\n"
            f"  In checkpoint only: {sorted(gis_modules - harvest_modules)}\n"
            f"  In harvest only: {sorted(harvest_modules - gis_modules)}"
        )
        gis_matrices = {k: v for k, v in gis_matrices.items() if k in shared_modules}
        coactivation_fractions = {
            k: v for k, v in coactivation_fractions.items() if k in shared_modules
        }
        coactivation_counts = {k: v for k, v in coactivation_counts.items() if k in shared_modules}
        activation_density = {k: v for k, v in activation_density.items() if k in shared_modules}

    # ── Plots ─────────────────────────────────────────────────────────────

    logger.info(f"Generating scatter plots → {output_dir / 'scatter'}")
    plot_scatter_per_module(
        gis_matrices=gis_matrices,
        coactivation_fractions=coactivation_fractions,
        coactivation_counts=coactivation_counts,
        activation_density=activation_density,
        alive_threshold=config.alive_density_threshold,
        output_dir=output_dir,
        run_id=run_id,
    )

    logger.info(f"Generating heatmaps → {output_dir / 'heatmaps'}")
    plot_gis_heatmaps(gis_matrices, activation_density, config.alive_density_threshold, output_dir)
    plot_coactivation_heatmaps(
        coactivation_fractions, activation_density, config.alive_density_threshold, output_dir
    )
    plot_gis_coact_product_heatmaps(
        gis_matrices,
        coactivation_fractions,
        activation_density,
        config.alive_density_threshold,
        output_dir,
    )

    # ── Save raw data ─────────────────────────────────────────────────────

    data_path = output_dir / "data.pt"
    save_file(
        {
            "gis_matrices": gis_matrices,
            "coactivation_fractions": coactivation_fractions,
            "coactivation_counts": coactivation_counts,
            "activation_density": activation_density,
            "config": config.model_dump(),
            "run_id": run_id,
        },
        data_path,
    )
    logger.info(f"Saved raw data → {data_path}")


if __name__ == "__main__":
    fire.Fire(main)
