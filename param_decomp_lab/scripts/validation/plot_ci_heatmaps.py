"""Plot per-position CI heatmaps from a `find_alive_components` per-position JSON.

For each token position, draws a heatmap with prompts on the y-axis (tiny text) and alive
subcomponents on the x-axis, faceted by decomposed matrix, coloured by lower-leaky CI. One
PNG per position is written into a subfolder.

The alive subcomponents shown (x-axis) are the union of components active across the
selected prompts; the colour of a cell is that component's CI at this position for that
prompt (0 where it is inactive). CPU-only — no model is loaded.

Usage:
    python -m param_decomp_lab.scripts.validation.plot_ci_heatmaps <per_position_json> \
        [--n-prompts=50] [--grep=SUBSTRING] [--output-dir=PATH]

Output: `<json_dir>/ci_heatmaps/position_<pos>.png` (one per position).
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from param_decomp.log import logger  # noqa: E402
from param_decomp_lab.scripts.validation.common import parse_module_name  # noqa: E402

# prompt -> position(str) -> module -> [{"component": int, "ci": float}]
PerPosition = dict[str, dict[str, dict[str, list[dict[str, Any]]]]]


def _select_prompts(data: PerPosition, grep: str | None, n_prompts: int) -> list[str]:
    """Prompts in file order, optionally filtered to those containing `grep`, capped at n."""
    prompts = [p for p in data if grep is None or grep in p]
    assert prompts, f"no prompts match grep={grep!r}"
    return prompts[:n_prompts]


def _alive_components(data: PerPosition, prompts: list[str]) -> dict[str, list[int]]:
    """Sorted alive-component indices per module, unioned over the selected prompts."""
    alive: dict[str, set[int]] = defaultdict(set)
    for prompt in prompts:
        for per_module in data[prompt].values():
            for module, comps in per_module.items():
                alive[module].update(entry["component"] for entry in comps)
    return {module: sorted(alive[module]) for module in alive}


def _ci_matrix(
    data: PerPosition, prompts: list[str], pos: str, module: str, components: list[int]
) -> np.ndarray:
    """[n_prompts, n_components] of CI at `pos` for `module`; 0 where a component is inactive."""
    col_of = {comp: j for j, comp in enumerate(components)}
    matrix = np.zeros((len(prompts), len(components)))
    for row, prompt in enumerate(prompts):
        for entry in data[prompt].get(pos, {}).get(module, []):
            matrix[row, col_of[entry["component"]]] = entry["ci"]
    return matrix


def _plot_position(
    data: PerPosition,
    prompts: list[str],
    pos: str,
    modules: list[str],
    alive: dict[str, list[int]],
    out_path: Path,
) -> None:
    widths = [max(1, len(alive[m])) for m in modules]
    fig_w = max(6.0, sum(widths) * 0.07 + 1.5 * len(modules))
    fig_h = max(2.5, len(prompts) * 0.13 + 1.0)
    fig, axes = plt.subplots(
        1, len(modules), sharey=True, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": widths}
    )
    axes = np.atleast_1d(axes)

    im = None
    for ax, module in zip(axes, modules, strict=True):
        components = alive[module]
        matrix = _ci_matrix(data, prompts, pos, module, components)
        im = ax.imshow(matrix, aspect="auto", cmap="RdPu", vmin=0.0, vmax=1.0, interpolation="none")
        _, matrix_name = parse_module_name(module)
        ax.set_title(matrix_name, fontsize=8)
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, fontsize=3, rotation=90)
        ax.set_xlabel("subcomponent", fontsize=7)

    axes[0].set_yticks(range(len(prompts)))
    axes[0].set_yticklabels(prompts, fontsize=3)
    axes[0].set_ylabel("prompt", fontsize=7)

    assert im is not None
    fig.colorbar(im, ax=list(axes), fraction=0.015, pad=0.02, label="CI")
    fig.suptitle(f"position {pos}", fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ci_heatmaps(
    json_path: str,
    n_prompts: int = 50,
    grep: str | None = None,
    output_dir: str | None = None,
) -> Path:
    """Write one CI-heatmap PNG per token position into a subfolder. Returns that folder."""
    json_file = Path(json_path).expanduser()
    data: PerPosition = json.loads(json_file.read_text())

    prompts = _select_prompts(data, grep, n_prompts)
    alive = _alive_components(data, prompts)
    modules = sorted(alive, key=parse_module_name)
    positions = sorted({pos for prompt in prompts for pos in data[prompt]}, key=int)

    out_dir = Path(output_dir).expanduser() if output_dir else json_file.parent / "ci_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos in positions:
        prompts_here = [p for p in prompts if pos in data[p]]
        if not prompts_here:
            continue
        _plot_position(
            data, prompts_here, pos, modules, alive, out_dir / f"position_{int(pos):02d}.png"
        )

    logger.info(f"Wrote {len(positions)} heatmaps for {len(prompts)} prompts to {out_dir}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(plot_ci_heatmaps)
