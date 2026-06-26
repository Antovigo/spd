"""Reduce and estimate the real dimensionality of the L18 MLP subcomponent representation.

The decomposition replaces L18's MLP with a few rank-1 subcomponents, so the relevant
last-token activation content lives in the directions those subcomponents span. We measure
the dimensionality of that content, separately for two sides:
- **input**: what the up/gate subcomponents read — the post-RMSNorm MLP input, projected
  onto the span of their unit V directions.
- **output**: what the down subcomponents write — the MLP output, projected onto the span
  of their unit U directions.

Both use the real activations stored by `collect_hidden_activations` (`mlp_input` /
`mlp_output`). For each side we build an orthonormal basis Q of the subcomponent span from
the Gram matrix G of the unit directions (G = E Λ Eᵀ; Q = Dᵀ E Λ^(-1/2)), and reduce the
activation to `z = Qᵀ x` — a plain projection of the real vector (no synthesis), so the
collapse of redundant directions (several subcomponents reading the same plane) is exact.

Reports three dimensionality reads per side — geometric rank (non-negligible G eigenvalues),
linear effective dimension (participation ratio of cov(z)'s spectrum), and intrinsic
dimension (TwoNN, on z) — plus the completeness fraction ‖z‖²/‖x‖² (centered), i.e. how much
of the activation variance lives in the subcomponent subspace. Saves `z` (raw + PCA-rotated)
and the bases for Objective 7, and writes a self-contained Plotly applet.

CPU-only. Usage:
    python -m param_decomp_lab.scripts.validation.reduce_dimensionality <model_path> \
        [--op=add] [--rank-eps=1e-6] [--output-dir=PATH]

Outputs: `analysis/datasets/dimensionality_<op>.{npz,json}` (the scalar summary) and
`analysis/dimensionality_<op>/index.html`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np
import plotly.offline as pyo
import skdim
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.common import (
    MLP_MATRICES,
    analysis_datasets_dir,
    analysis_dir,
    load_component_uv,
    read_alive_components,
)

_APP_TEMPLATE = Path(__file__).with_name("dimensionality_explorer_app.html")
# Which side reads/writes which matrices, and which vector (V=read, U=write) is its direction.
_SIDES = {"input": ("V", ("gate_proj", "up_proj")), "output": ("U", ("down_proj",))}


def _unit_directions(
    uv: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]],
    which: str,
    projs: tuple[str, ...],
    alive_by_proj: dict[str, list[int]],
) -> NDArray[np.float32]:
    """Stack the unit read (V column) or write (U row) directions of the alive components."""
    rows: list[NDArray[np.float32]] = []
    for proj in projs:
        v, u = uv[proj]
        for c in alive_by_proj.get(proj, []):
            d = v[:, c] if which == "V" else u[c, :]
            rows.append(d / max(float(np.linalg.norm(d)), 1e-12))
    assert rows, f"no alive components for {projs}"
    return np.stack(rows)


@dataclass
class _SideResult:
    z: NDArray[np.float32]  # [N, r] orthonormal projection
    pca: NDArray[np.float32]  # [N, r] PCA-rotated (centered)
    g_eigs: NDArray[np.float64]  # G spectrum (descending)
    pca_eigs: NDArray[np.float64]  # cov(z) spectrum (descending)
    rank: int
    participation_ratio: float
    twonn: float
    var_fraction: float


def _analyse_side(
    directions: NDArray[np.float32], activations: NDArray[np.float32], rank_eps: float
) -> _SideResult:
    g = directions @ directions.T
    evals, evecs = np.linalg.eigh(g)  # ascending
    evals, evecs = evals[::-1], evecs[:, ::-1]
    rank = int((evals > rank_eps * evals[0]).sum())
    e_r, l_r = evecs[:, :rank], evals[:rank]
    q = directions.T @ e_r / np.sqrt(l_r)  # [d, rank], orthonormal columns

    x = activations.astype(np.float64)
    z = x @ q  # [N, rank]
    xc = x - x.mean(0)
    zc = z - z.mean(0)
    var_fraction = float((zc**2).sum() / (xc**2).sum())

    # PCA of z (= the linear effective-dimension spectrum and the display ordering).
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    pca_eigs = s**2 / z.shape[0]
    pr = float(pca_eigs.sum() ** 2 / (pca_eigs**2).sum())
    pca = zc @ vt.T

    twonn = float(skdim.id.TwoNN().fit(z).dimension_)
    return _SideResult(
        z=z.astype(np.float32),
        pca=pca.astype(np.float32),
        g_eigs=evals,
        pca_eigs=pca_eigs,
        rank=rank,
        participation_ratio=round(pr, 3),
        twonn=round(twonn, 3),
        var_fraction=round(var_fraction, 4),
    )


def _round_cols(mat: NDArray[np.float32]) -> list[list[float]]:
    """Column-major, rounded — one list per axis (what the applet indexes by axis)."""
    return [[round(float(v), 4) for v in mat[:, j]] for j in range(mat.shape[1])]


def reduce_dimensionality(
    model_path: ModelPath,
    op: str = "add",
    rank_eps: float = 1e-6,
    output_dir: str | None = None,
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent
    data_dir = analysis_datasets_dir(run_dir)

    alive = read_alive_components(data_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES)
    layer = alive[0].layer
    alive_by_proj: dict[str, list[int]] = {p: [] for p in MLP_MATRICES}
    for a in alive:
        alive_by_proj[a.proj].append(a.component)
    uv = load_component_uv(checkpoint, layer, MLP_MATRICES)

    npz_path = data_dir / f"hidden_activations_{op}.npz"
    assert npz_path.exists(), f"missing {npz_path.name}; run collect_hidden_activations first"
    hidden = np.load(npz_path, allow_pickle=True)
    n = int(hidden["a"].shape[0])
    a_per_row = np.repeat(np.arange(1, n + 1), n)
    b_per_row = np.tile(np.arange(1, n + 1), n)

    results: dict[str, _SideResult] = {}
    for side, (which, projs) in _SIDES.items():
        directions = _unit_directions(uv, which, projs, alive_by_proj)
        grid = hidden["mlp_input" if side == "input" else "mlp_output"]
        activations = grid.reshape(n * n, grid.shape[-1])
        results[side] = _analyse_side(directions, activations, rank_eps)
        r = results[side]
        logger.info(
            f"{side}: {directions.shape[0]} dirs → rank {r.rank}, PR {r.participation_ratio}, "
            f"TwoNN {r.twonn}, var captured {r.var_fraction:.3f}"
        )

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run_dir) / f"dimensionality_{op}"
    )
    _write_outputs(run_dir, out_dir, op, n, a_per_row, b_per_row, results)
    logger.info(f"wrote dimensionality analysis + applet for {op} → {out_dir}")
    return out_dir


def _write_outputs(
    run_dir: Path,
    out_dir: Path,
    op: str,
    n: int,
    a_per_row: NDArray[np.int64],
    b_per_row: NDArray[np.int64],
    results: dict[str, _SideResult],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    npz: dict[str, Any] = {"a": a_per_row, "b": b_per_row, "op": op, "n": n}
    summary: dict[str, Any] = {"op": op, "run_id": run_dir.name}
    for side, r in results.items():
        npz[f"z_{side}"] = r.z
        npz[f"pca_{side}"] = r.pca
        summary[side] = {
            "rank": r.rank,
            "participation_ratio": r.participation_ratio,
            "twonn": r.twonn,
            "var_fraction": r.var_fraction,
        }
    data_dir = analysis_datasets_dir(run_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / f"dimensionality_{op}.npz", **npz)
    (data_dir / f"dimensionality_{op}.json").write_text(json.dumps(summary, indent=2))

    payload = {
        "meta": {
            "op": op,
            "run_id": run_dir.name,
            "a": a_per_row.tolist(),
            "b": b_per_row.tolist(),
            "sum": (a_per_row + b_per_row).tolist(),
        },
        "sides": {
            side: {
                "z": _round_cols(r.z),
                "pca": _round_cols(r.pca),
                "pca_eigs": [round(float(e), 5) for e in r.pca_eigs],
                "rank": r.rank,
                "participation_ratio": r.participation_ratio,
                "twonn": r.twonn,
                "var_fraction": r.var_fraction,
            }
            for side, r in results.items()
        },
    }
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    template = _APP_TEMPLATE.read_text()
    for marker in ("/*__PLOTLY_JS__*/", "/*__PD_DATA__*/"):
        assert marker in template, f"template missing injection marker {marker}"
    html = template.replace("/*__PLOTLY_JS__*/", pyo.get_plotlyjs()).replace(
        "/*__PD_DATA__*/", json.dumps(payload, separators=(",", ":"))
    )
    (out_dir / "index.html").write_text(html)


if __name__ == "__main__":
    fire.Fire(reduce_dimensionality)
