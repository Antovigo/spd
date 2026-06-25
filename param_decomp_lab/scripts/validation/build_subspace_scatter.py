"""Interactive 3D scatter of L18 MLP activations in a user-picked 3-subcomponent subspace.

A GPU-free HTML applet: the user picks up to 3 subcomponents from a thumbnail list (each
thumbnail is that subcomponent's inner-activation pattern over the (a, b) grid) and sees the
last-token activation projected onto those 3 directions as a rotatable 3D scatter.

- **input** space: the post-RMSNorm MLP input (`mlp_input`) projected onto the unit V
  directions of the up/gate subcomponents (`x · V̂_c`, i.e. the inner activation).
- **output** space: the MLP output (`mlp_output`) projected onto the unit U directions of the
  down subcomponents (`y · Û_c`).

Both come straight from the `collect_hidden_activations` npz + the checkpoint directions; the
available subcomponents are the alive set (`alive_filtered_<op>.tsv`). A dark-grey shadow is
the points flattened onto the floor of the 3D box (it stays on the bottom plane).

CPU-only. Usage:
    python -m param_decomp_lab.scripts.validation.build_subspace_scatter <model_path> \
        [--op=add] [--output-dir=PATH]

Output: `<run_dir>/figures/subspace_scatter_<op>/index.html` (self-contained Plotly applet).
"""

import base64
import io
import json
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.offline as pyo  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from param_decomp.log import logger  # noqa: E402
from param_decomp_lab.infra.paths import ModelPath  # noqa: E402
from param_decomp_lab.scripts.validation.common import (  # noqa: E402
    MLP_MATRICES,
    load_component_uv,
    read_alive_components,
    read_subcomp_periods,
)

_APP_TEMPLATE = Path(__file__).with_name("subspace_scatter_app.html")
# side -> (which vector defines the direction, the matrices, the activation grid key)
_SIDES = {
    "input": ("V", ("gate_proj", "up_proj"), "mlp_input"),
    "output": ("U", ("down_proj",), "mlp_output"),
}


def _thumbnail(grid: NDArray[np.float32]) -> str:
    """A small signed-diverging heatmap of the (a, b) pattern as a base64 PNG data URI."""
    lim = float(np.abs(grid).max()) or 1.0
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.imshow(grid.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_subspace_scatter(
    model_path: ModelPath, op: str = "add", output_dir: str | None = None
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent

    alive = read_alive_components(run_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES)
    layer = alive[0].layer
    uv = load_component_uv(checkpoint, layer, MLP_MATRICES)
    periods = read_subcomp_periods(run_dir / f"subcomp_periods_{op}.tsv")
    alive_by_proj: dict[str, list[int]] = {p: [] for p in MLP_MATRICES}
    for a in alive:
        alive_by_proj[a.proj].append(a.component)

    npz_path = run_dir / f"hidden_activations_{op}.npz"
    assert npz_path.exists(), f"missing {npz_path.name}; run collect_hidden_activations first"
    hidden = np.load(npz_path, allow_pickle=True)
    n = int(hidden["a"].shape[0])
    a_per_row = np.repeat(np.arange(1, n + 1), n)
    b_per_row = np.tile(np.arange(1, n + 1), n)

    sides: dict[str, Any] = {}
    for side, (which, projs, grid_key) in _SIDES.items():
        acts = hidden[grid_key].reshape(n * n, hidden[grid_key].shape[-1]).astype(np.float32)
        # Sort the pickable list by period (then matrix, component) so the applet groups it.
        items = sorted(
            ((proj, c, periods[(proj, c)]) for proj in projs for c in alive_by_proj[proj]),
            key=lambda t: (t[2], t[0], t[1]),
        )
        comps: list[dict[str, Any]] = []
        dirs: list[NDArray[np.float32]] = []
        for proj, c, period in items:
            v, u = uv[proj]
            d = v[:, c] if which == "V" else u[c, :]
            d = d / max(float(np.linalg.norm(d)), 1e-12)
            dirs.append(d)
            coords = acts @ d  # [N] activation projected onto the unit direction
            comps.append(
                {
                    "label": f"{proj[0]}{c}",
                    "period": period,
                    "proj": [round(float(x), 4) for x in coords],
                    "thumb": _thumbnail(coords.reshape(n, n)),
                }
            )
        # Pairwise cosine of the unit directions, so the applet can place the picked axes at
        # their true mutual angles (Cholesky embedding) instead of forcing them orthogonal.
        gram = np.stack(dirs) @ np.stack(dirs).T
        sides[side] = {
            "comps": comps,
            "gram": [[round(float(g), 4) for g in row] for row in gram],
        }
        logger.info(f"{side}: {len(comps)} pickable subcomponents")

    payload = {
        "meta": {
            "op": op,
            "a": a_per_row.tolist(),
            "b": b_per_row.tolist(),
            "sum": (a_per_row + b_per_row).tolist(),
        },
        "sides": sides,
    }
    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else run_dir / "figures" / f"subspace_scatter_{op}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    template = _APP_TEMPLATE.read_text()
    for marker in ("/*__PLOTLY_JS__*/", "/*__PD_DATA__*/"):
        assert marker in template, f"template missing injection marker {marker}"
    html = template.replace("/*__PLOTLY_JS__*/", pyo.get_plotlyjs()).replace(
        "/*__PD_DATA__*/", json.dumps(payload, separators=(",", ":"))
    )
    (out_dir / "index.html").write_text(html)
    logger.info(f"wrote subspace-scatter applet for {op} → {out_dir}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_subspace_scatter)
