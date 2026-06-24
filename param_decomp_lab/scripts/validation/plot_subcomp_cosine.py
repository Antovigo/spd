"""Cosine-similarity heatmaps between alive L18 MLP subcomponents' V and U vectors.

Two figures, because the SwiGLU boundary splits the vectors into incompatible spaces:
- `cosine_gate_up_<op>.png` — gate + up subcomponents together (they share both spaces):
  left = V vectors (residual space, d_model), right = U vectors (neuron space, d_int).
- `cosine_down_<op>.png` — down subcomponents: left = V (neuron space), right = U (residual
  space). Down can't be compared with gate/up since its U/V live in transposed dimensions.

Within each figure both heatmaps share one component ordering, sorted by the representative
activation period (from `compute_subcomp_periods`); a thick separator divides components of
different periods. RdBu (reversed → positive=red, negative=blue), symmetric in [-1, 1].

CPU-only — reads the checkpoint U/V via mmap, no forward pass.

Usage:
    python -m param_decomp_lab.scripts.validation.plot_subcomp_cosine <model_path> \
        [--op=add] [--output-dir=PATH]

Outputs: `<run_dir>/figures/subcomp_cosine/cosine_{gate_up,down}_<op>.png`.
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.common import (
    MLP_MATRICES,
    load_component_uv,
    read_alive_components,
    read_subcomp_periods,
)


def _cosine(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Row-wise cosine-similarity matrix of a `[n, dim]` stack."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    return unit @ unit.T


def _heatmap(
    ax: Axes, sim: NDArray[np.float32], labels: list[str], boundaries: list[int], title: str
) -> None:
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_yticklabels(labels, fontsize=5)
    for b in boundaries:  # thick separators between period groups
        ax.axhline(b - 0.5, color="black", lw=1.5)
        ax.axvline(b - 0.5, color="black", lw=1.5)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_group(
    projs: tuple[str, ...],
    uv: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]],
    alive_by_proj: dict[str, list[int]],
    periods: dict[tuple[str, int], int],
    op: str,
    title: str,
    out_path: Path,
) -> None:
    # Order: by period, then proj, then component. Each entry carries its V and U vector.
    entries = [(proj, c) for proj in projs for c in alive_by_proj.get(proj, [])]
    entries.sort(key=lambda pc: (periods[pc], pc[0], pc[1]))
    assert entries, f"no alive components for {projs}"

    v_stack = np.stack([uv[proj][0][:, c] for proj, c in entries])
    u_stack = np.stack([uv[proj][1][c, :] for proj, c in entries])
    labels = [f"{proj[0]}{c}·p{periods[(proj, c)]}" for proj, c in entries]
    group_periods = [periods[pc] for pc in entries]
    boundaries = [i for i in range(1, len(entries)) if group_periods[i] != group_periods[i - 1]]

    fig, axes = plt.subplots(1, 2, figsize=(2 * (len(entries) * 0.28 + 2), len(entries) * 0.28 + 2))
    _heatmap(axes[0], _cosine(v_stack), labels, boundaries, "V vectors (input)")
    _heatmap(axes[1], _cosine(u_stack), labels, boundaries, "U vectors (output)")
    fig.suptitle(f"{title} — cosine similarity ({op}, sorted by period)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_path} ({len(entries)} components)")


def plot_subcomp_cosine(
    model_path: ModelPath, op: str = "add", output_dir: str | None = None
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent

    alive = read_alive_components(run_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES)
    periods = read_subcomp_periods(run_dir / f"subcomp_periods_{op}.tsv")
    layer = alive[0].layer
    alive_by_proj: dict[str, list[int]] = {proj: [] for proj in MLP_MATRICES}
    for a in alive:
        alive_by_proj[a.proj].append(a.component)

    uv = load_component_uv(checkpoint, layer, MLP_MATRICES)
    out_dir = (
        Path(output_dir).expanduser() if output_dir else run_dir / "figures" / "subcomp_cosine"
    )
    _plot_group(
        ("gate_proj", "up_proj"),
        uv,
        alive_by_proj,
        periods,
        op,
        "gate + up",
        out_dir / f"cosine_gate_up_{op}.png",
    )
    _plot_group(
        ("down_proj",),
        uv,
        alive_by_proj,
        periods,
        op,
        "down",
        out_dir / f"cosine_down_{op}.png",
    )
    return out_dir


if __name__ == "__main__":
    fire.Fire(plot_subcomp_cosine)
