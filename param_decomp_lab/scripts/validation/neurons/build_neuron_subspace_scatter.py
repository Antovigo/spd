"""Interactive 3D scatter of MLP input/output activations in a ≤3-neuron weight subspace.

The neuron analogue of `build_subspace_scatter`: each axis is one L18 neuron's **weight
direction**, and the cloud is the MLP's activation on the 201×201 `a<op>b=` prompt grid
projected onto the picked directions — kept at their true mutual angles (Cholesky of the
unit-direction Gram), exactly like the subcomponent applet. A **side** selector chooses the
activation/direction pairing:

- **input · gate read**: the post-RMSNorm MLP input projected onto the unit gate rows (`x · ĝ_n`).
- **input · up read**: the same input onto the unit up rows (`x · û_n`).
- **output · down write**: the MLP output onto the unit down columns (`y · d̂_n`), with
  `y = W_down · silu(gate)·up` recomputed from the cached preactivations.

Neurons are picked from a thumbnail list (grouped per task by their dominant pure
translation-invariance period, sorted by ablation KL) or by typing an index into a form.
Included neurons: census candidates whose individual-ablation `max_kl` on either op exceeds
`--kl-thr` (default 0.005 — comfortably above the null-patch floor). Activations come from the
cached `activations_<op>.npz`; the frozen L18 MLP weights are mmap-read from `model_path` (any
run over the base model). Projections ship uint8-quantized per (side, op, neuron).

CPU-only. Usage:
    python -m param_decomp_lab.scripts.validation.neurons.build_neuron_subspace_scatter \
        <model_path> [--census-dir=PATH] [--kl-thr=0.005] [--output-dir=PATH]

Output: `<census_dir>/subspace_applet/{index.html,data.js}`.
"""

import base64
import csv
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
from param_decomp_lab.scripts.validation.common import load_target_mlp_weights  # noqa: E402
from param_decomp_lab.scripts.validation.neurons.common import (  # noqa: E402
    NEURON_OPS,
    NEURONS_DIR,
    PERIODS,
    silu_combine,
)

_APP_TEMPLATE = Path(__file__).with_name("neuron_subspace_scatter_app.html")
_SIDES = ("input-gate", "input-up", "output")
_PERIOD_SCORE_FLOOR = 0.5  # below this on every pure lag → "no period" group
_CHUNK_ROWS = 20  # grid rows per silu/matmul chunk on the output side (memory control)


def _q8(grid: NDArray[np.float32]) -> dict[str, Any]:
    lo, hi = float(grid.min()), float(grid.max())
    scale = (hi - lo) or 1.0
    q = np.clip(np.round((grid - lo) / scale * 255.0), 0, 255).astype(np.uint8)
    return {"b64": base64.b64encode(q.tobytes()).decode("ascii"), "lo": lo, "hi": hi}


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


def _dominant_period(root: Path, op: str, neuron_ids: list[int]) -> dict[int, int]:
    """Neuron → dominant pure period from the combined-channel translation-invariance scores
    (0 = none above the floor). Missing periodicity npz → everything ungrouped."""
    path = root / f"periodicity_{op}.npz"
    if not path.exists():
        logger.info(f"no {path.name}; cards ungrouped for {op}")
        return dict.fromkeys(neuron_ids, 0)
    d = np.load(path)
    score, lags = d["score"], d["lags"]  # [14336, 3, n_lags], [n_lags, 2]
    comb_ci = list(d["channels"]).index("combined")
    out: dict[int, int] = {}
    for nid in neuron_ids:
        best_p, best_s = 0, _PERIOD_SCORE_FLOOR
        for p in PERIODS:
            for la, lb in ((p, 0), (0, p)):
                (li,) = np.where((lags[:, 0] == la) & (lags[:, 1] == lb))[0]
                s = float(score[nid, comb_ci, li])
                if s > best_s:
                    best_p, best_s = p, s
        out[nid] = best_p
    return out


def _unit_dirs(
    weights: dict[str, NDArray[np.float32]], kept: list[int]
) -> dict[str, NDArray[np.float32]]:
    """Per side, the kept neurons' unit weight directions `[K, d]` — gate/up rows in the
    residual space (d_model), down columns likewise."""
    gate = weights["gate_proj"][kept, :].astype(np.float32)  # (K, d_model) rows = reads
    up = weights["up_proj"][kept, :].astype(np.float32)
    down = weights["down_proj"][:, kept].T.astype(np.float32)  # columns = writes -> (K, d_model)
    return {
        s: d / np.linalg.norm(d, axis=1, keepdims=True)
        for s, d in {"input-gate": gate, "input-up": up, "output": down}.items()
    }


def _output_projection(
    d: Any, weights: dict[str, NDArray[np.float32]], down_unit: NDArray[np.float32]
) -> NDArray[np.float32]:
    """`y · d̂_n` over the grid, `y = W_down · silu(gate)·up` from the cached preactivations.

    Chunked over grid rows: `y · d̂ = h · (W_downᵀ d̂)` needs only the 14336-d `h` per chunk."""
    m = weights["down_proj"].astype(np.float32).T @ down_unit.T  # (14336, K)
    gate, up = d["gate_preact"], d["up_preact"]  # (n, n, 14336) fp16
    n = gate.shape[0]
    out = np.empty((n * n, m.shape[1]), np.float32)
    for r0 in range(0, n, _CHUNK_ROWS):
        r1 = min(r0 + _CHUNK_ROWS, n)
        h = silu_combine(gate[r0:r1], up[r0:r1]).reshape(-1, m.shape[0])
        out[r0 * n : r1 * n] = h @ m
    return out


def build_neuron_subspace_scatter(
    model_path: ModelPath,
    census_dir: str | None = None,
    kl_thr: float = 0.005,
    output_dir: str | None = None,
) -> Path:
    root = Path(census_dir).expanduser() if census_dir else NEURONS_DIR
    cand_path = root / "candidates.tsv"
    assert cand_path.exists(), f"missing candidates: {cand_path} (run select_candidate_neurons)"
    with cand_path.open() as f:
        cand = list(csv.DictReader(f, delimiter="\t"))
    kl = {int(r["neuron"]): {op: float(r[f"max_kl_{op}"]) for op in NEURON_OPS} for r in cand}
    kept = sorted(
        (n for n, k in kl.items() if max(k.values()) > kl_thr),
        key=lambda n: -max(kl[n].values()),
    )
    assert kept, f"no neuron clears max_kl > {kl_thr}"
    logger.info(f"{len(kept)} of {len(cand)} candidates clear max_kl > {kl_thr}")

    first = np.load(root / f"activations_{NEURON_OPS[0]}.npz")
    layer = int(first["layer"])
    weights = load_target_mlp_weights(
        Path(model_path).expanduser(), layer, ("gate_proj", "up_proj", "down_proj")
    )
    dirs = _unit_dirs(weights, kept)
    grams = {s: np.round(d @ d.T, 4).tolist() for s, d in dirs.items()}

    sides: dict[str, dict[str, Any]] = {
        s: {"gram": grams[s], "grids": {}, "thumbs": {}} for s in _SIDES
    }
    period_by_op: dict[str, dict[int, int]] = {}
    n = 0
    for op in NEURON_OPS:
        acts_path = root / f"activations_{op}.npz"
        assert acts_path.exists(), f"missing activations: {acts_path}"
        d = np.load(acts_path)
        n = int(d["gate_preact"].shape[0])
        x = d["mlp_input"].reshape(n * n, -1).astype(np.float32)
        proj = {
            "input-gate": x @ dirs["input-gate"].T,  # (n², K) = x · ĝ_n
            "input-up": x @ dirs["input-up"].T,
            "output": _output_projection(d, weights, dirs["output"]),
        }
        del x
        for s in _SIDES:
            sides[s]["grids"][op] = {str(nid): _q8(proj[s][:, i]) for i, nid in enumerate(kept)}
            sides[s]["thumbs"][op] = {
                str(nid): _thumbnail(proj[s][:, i].reshape(n, n)) for i, nid in enumerate(kept)
            }
        period_by_op[op] = _dominant_period(root, op, kept)
        logger.info(f"{op}: {len(kept)} neurons × {len(_SIDES)} sides on the {n}×{n} grid")
        del proj

    payload = {
        "meta": {
            "ops": list(NEURON_OPS),
            "sides": list(_SIDES),
            "n": n,
            "layer": layer,
            "kl_thr": kl_thr,
            "mods": list(PERIODS),
        },
        "neurons": [
            {
                "id": nid,
                "kl": {op: round(kl[nid][op], 4) for op in NEURON_OPS},
                "period": {op: period_by_op[op][nid] for op in NEURON_OPS},
            }
            for nid in kept
        ],
        "sides": sides,
    }

    out_dir = Path(output_dir).expanduser() if output_dir else root / "subspace_applet"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        "window.PD_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    template = _APP_TEMPLATE.read_text()
    assert "/*__PLOTLY_JS__*/" in template, "template missing the plotly injection marker"
    (out_dir / "index.html").write_text(template.replace("/*__PLOTLY_JS__*/", pyo.get_plotlyjs()))
    size_mb = (out_dir / "data.js").stat().st_size / 1e6
    logger.info(f"wrote neuron subspace applet ({size_mb:.1f} MB data.js) → {out_dir}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_neuron_subspace_scatter)
