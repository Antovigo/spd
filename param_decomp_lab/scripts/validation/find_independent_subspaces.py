"""Independent Subspace Analysis of the reduced L18 MLP representation (Objective 7).

Takes the orthonormalised representation `z` from `reduce_dimensionality` and discovers
mutually independent *blocks* — subspaces that store different information independently —
without assuming what they encode. For each side (input / output):

1. reduce z to the PCs capturing `--var-keep` of the variance (drop noise directions),
2. run ICA (scikit-learn FastICA) to get independent 1-D components,
3. group those components into subspaces by their **magnitude (energy) correlation** — two
   components belong together when |corr(|s_i|, |s_j|)| is high, so a circular feature whose
   components are linearly uncorrelated but jointly dependent (cos²+sin²=const) is recovered
   as one subspace rather than split.

Each block is then characterised post-hoc (its components, and — for interpretation only —
how it varies over the (a, b) grid), and the blocks are checked for near-orthogonality via
the principal angles between their z-space directions. Discovery itself uses no (a, b) labels.

CPU-only. Usage:
    python -m param_decomp_lab.scripts.validation.find_independent_subspaces <model_path> \
        [--op=add] [--var-keep=0.9] [--group-distance=0.75] [--seed=0] [--max-iter=20000] \
        [--output-dir=PATH]

Outputs: `analysis/datasets/independent_subspaces_<op>.json` (the block summary) and
`analysis/independent_subspaces_<op>/index.html` (a self-contained Plotly applet).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import fire
import numpy as np
import plotly.offline as pyo
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import subspace_angles
from scipy.spatial.distance import squareform
from sklearn.decomposition import FastICA

from param_decomp.log import logger
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.common import analysis_datasets_dir, analysis_dir

_APP_TEMPLATE = Path(__file__).with_name("isa_explorer_app.html")
_SIDES = ("input", "output")


@dataclass
class _SideISA:
    sources: NDArray[np.float64]  # [N, k] independent components
    labels: NDArray[np.int64]  # [k] subspace id per component
    energy_corr: NDArray[np.float64]  # [k, k] |corr| of component magnitudes
    principal_angles: NDArray[np.float64]  # [n_sub, n_sub] min principal angle (deg)
    subspaces: list[list[int]]  # component indices per subspace


def _isa_side(
    z: NDArray[np.float32], var_keep: float, group_distance: float, seed: int, max_iter: int
) -> _SideISA:
    zc = (z - z.mean(0)).astype(np.float64)
    sv = np.linalg.svd(zc, full_matrices=False, compute_uv=False)
    var = sv**2
    k = int(np.searchsorted(np.cumsum(var) / var.sum(), var_keep) + 1)
    k = max(2, min(k, zc.shape[1]))

    ica = FastICA(n_components=k, whiten="unit-variance", max_iter=max_iter, random_state=seed)
    sources = np.asarray(ica.fit_transform(zc), dtype=np.float64)  # [N, k]
    if ica.n_iter_ >= max_iter:
        logger.warning(
            f"FastICA did not converge in {max_iter} iters ({k} comps) — result may be unstable"
        )
    mixing = np.asarray(ica.mixing_, dtype=np.float64)  # [r, k] — z-space direction per component

    energy = np.abs(sources)
    energy_corr = np.abs(np.corrcoef(energy.T))  # [k, k]
    dist = 1.0 - energy_corr
    np.fill_diagonal(dist, 0.0)
    labels = np.asarray(
        fcluster(
            linkage(squareform(dist, checks=False), method="average"),
            t=group_distance,
            criterion="distance",
        ),
        dtype=np.int64,
    )
    subspaces = [
        [int(c) for c in np.nonzero(labels == lab)[0]] for lab in sorted(set(labels.tolist()))
    ]

    n_sub = len(subspaces)
    angles = np.full((n_sub, n_sub), 90.0)
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            ang = np.degrees(
                subspace_angles(
                    np.take(mixing, subspaces[i], axis=1), np.take(mixing, subspaces[j], axis=1)
                )
            )
            angles[i, j] = angles[j, i] = float(ang.min())
    np.fill_diagonal(angles, 0.0)  # a subspace's principal angle with itself is 0
    return _SideISA(
        sources=sources,
        labels=np.asarray(labels, dtype=np.int64),
        energy_corr=energy_corr,
        principal_angles=angles,
        subspaces=subspaces,
    )


def _round_cols(mat: NDArray[np.floating]) -> list[list[float]]:
    return [[round(float(v), 4) for v in mat[:, j]] for j in range(mat.shape[1])]


def _round_mat(mat: NDArray[np.floating]) -> list[list[float]]:
    return [[round(float(v), 3) for v in row] for row in mat]


def find_independent_subspaces(
    model_path: ModelPath,
    op: str = "add",
    var_keep: float = 0.9,
    group_distance: float = 0.75,
    seed: int = 0,
    max_iter: int = 20000,
    output_dir: str | None = None,
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent
    dim_npz = analysis_datasets_dir(run_dir) / f"dimensionality_{op}.npz"
    assert dim_npz.exists(), f"missing {dim_npz.name}; run reduce_dimensionality first"
    data = np.load(dim_npz, allow_pickle=True)
    a, b = data["a"].astype(int), data["b"].astype(int)

    results = {
        side: _isa_side(data[f"z_{side}"], var_keep, group_distance, seed, max_iter)
        for side in _SIDES
    }
    for side, r in results.items():
        dims = [len(s) for s in r.subspaces]
        logger.info(f"{side}: {r.sources.shape[1]} ICs → {len(r.subspaces)} subspaces, dims {dims}")

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run_dir) / f"independent_subspaces_{op}"
    )
    _write_outputs(run_dir, out_dir, op, a, b, results)
    logger.info(f"wrote ISA decomposition + applet for {op} → {out_dir}")
    return out_dir


def _write_outputs(
    run_dir: Path,
    out_dir: Path,
    op: str,
    a: NDArray[np.int64],
    b: NDArray[np.int64],
    results: dict[str, _SideISA],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "op": op,
        "run_id": run_dir.name,
        "sides": {
            side: {
                "n_components": r.sources.shape[1],
                "subspaces": [list(map(int, s)) for s in r.subspaces],
            }
            for side, r in results.items()
        },
    }
    data_dir = analysis_datasets_dir(run_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"independent_subspaces_{op}.json").write_text(json.dumps(summary, indent=2))

    payload = {
        "meta": {"op": op, "a": a.tolist(), "b": b.tolist(), "sum": (a + b).tolist()},
        "sides": {
            side: {
                "sources": _round_cols(r.sources),
                "subspaces": [list(map(int, s)) for s in r.subspaces],
                "energy_corr": _round_mat(r.energy_corr),
                "principal_angles": _round_mat(r.principal_angles),
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
    fire.Fire(find_independent_subspaces)
