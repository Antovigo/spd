"""Build a GPU-free HTML explorer over the ablation-KL data.

Consumes `collect_ablation_kl`'s `data.npz` and emits a self-contained applet (`index.html`
+ `data.js`, `file://`-openable, no server/CDN/GPU): a top-half gallery of components and a
bottom-half detail panel with three zoomed (a, b) heatmaps for the selected component —
causal importance, ablation KL, and normalized inner activation — with axis ticks and a
hover readout. Each component's periodicity (axis + base) is detected by autocorrelation of
the ablation-KL marginal (spiky, non-sinusoidal); it colors the gallery borders and drives
the period filter. The gallery is filtered by mean causal importance and sorted by KL.

Usage:
    python -m param_decomp_lab.scripts.validation.build_arith_ablation_explorer <run_dir_or_npz> \
        [--output-dir=PATH]

Output: `<run_dir>/analysis/arith_ablation_explorer/{index.html,data.js}`.
"""

import base64
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.common import analysis_datasets_dir, analysis_dir

_APP_TEMPLATE = Path(__file__).with_name("arith_ablation_explorer_app.html")


def _autocorr(sig: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized autocorrelation of a *linearly-detrended* signal (r[0] = 1).

    Detrending matters: the autocorrelation of a monotonic ramp stays high at every small
    lag, which a naive peak-finder reads as a short period. Removing the linear trend leaves
    only genuine periodic structure.
    """
    t = np.arange(len(sig), dtype=np.float64)
    coef = np.polyfit(t, sig, 1)
    s = sig - (coef[0] * t + coef[1])
    denom = float((s * s).sum())
    if denom < 1e-12:
        return np.zeros_like(s)
    return np.correlate(s, s, mode="full")[len(s) - 1 :] / denom


def _fundamental_period(r: NDArray[np.float64]) -> tuple[int, float]:
    """Smallest strict local-max lag reaching the autocorrelation plateau (the fundamental).

    `(0, 0.0)` when there is no oscillatory peak (e.g. a detrended-flat signal).
    """
    max_lag = len(r) // 2
    if max_lag < 3:
        return 0, 0.0
    peak = float(r[2 : max_lag + 1].max())
    if peak < 0.1:
        return 0, 0.0
    for lag in range(2, max_lag + 1):
        right_ok = lag == max_lag or r[lag] >= r[lag + 1]
        if r[lag] > r[lag - 1] and right_ok and r[lag] >= 0.8 * peak:
            return lag, round(peak, 4)
    return 0, 0.0


def _detect_period(kl: NDArray[np.float64], n: int) -> dict[str, Any]:
    """Autocorrelation-based period of the ablation-KL grid; pick the strongest axis.

    Grid axis order is [a-1, b-1]: f(a) averages over b (axis 1), f(b) over a (axis 0).
    """
    idx = np.arange(1, n + 1)
    ai, bj = np.meshgrid(idx, idx, indexing="ij")
    flat = kl.ravel()
    s_key, d_key = (ai + bj).ravel(), (ai - bj).ravel()
    marg = {
        "a": kl.mean(axis=1),
        "b": kl.mean(axis=0),
        "sum": np.array([flat[s_key == s].mean() for s in range(2, 2 * n + 1)]),
        "diff": np.array([flat[d_key == d].mean() for d in range(-(n - 1), n)]),
    }
    per_axis = {ax: _fundamental_period(_autocorr(m.astype(np.float64))) for ax, m in marg.items()}
    best_axis = max(per_axis, key=lambda ax: per_axis[ax][1])
    period, strength = per_axis[best_axis]
    return {"axis": best_axis, "period": period, "strength": strength}


def _b64_i8(arr: NDArray[np.float64]) -> str:
    scale = float(np.abs(arr).max()) or 1.0
    q = np.clip(np.rint(arr / scale * 127), -127, 127).astype(np.int8)
    return base64.b64encode(np.ascontiguousarray(q).tobytes()).decode("ascii")


def _b64_u8(arr: NDArray[np.float64], scale: float) -> str:
    q = np.clip(np.rint(arr / scale * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(np.ascontiguousarray(q).tobytes()).decode("ascii")


def build_arith_ablation_explorer(source: str, output_dir: str | None = None) -> Path:
    """Write the ablation-KL explorer (`index.html` + `data.js`). Returns the output folder."""
    src = Path(source).expanduser()
    if src.suffix == ".npz":
        npz_path = src
        run_dir = npz_path.parents[3]  # <run>/analysis/datasets/ablation_kl/data.npz
    else:
        run_dir = src
        npz_path = analysis_datasets_dir(run_dir) / "ablation_kl" / "data.npz"
    assert npz_path.exists(), f"npz not found: {npz_path} (run collect_ablation_kl first)"
    meta = json.loads((npz_path.parent / "meta.json").read_text())
    n = int(meta["n"])

    # ||U||·||V|| per component, keyed by full module path (the short matrix name collides
    # across layers).
    norms: dict[tuple[str, int], float] = {}
    comp_tsv = npz_path.parent / "components.tsv"
    if comp_tsv.exists():
        with comp_tsv.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                norms[(f"model.layers.{row['layer']}.{row['matrix']}", int(row["component"]))] = (
                    float(row["norm"])
                )

    d = np.load(npz_path, allow_pickle=True)
    kl, ci, inner = d["kl"], d["ci"], d["inner_act"]  # [A, n, n]
    modules, shorts, comps = d["module"], d["short"], d["component"]
    n_alive = kl.shape[0]

    kl_scale = float(np.percentile(kl[kl > 0], 99)) if (kl > 0).any() else 1.0
    inner_scale = float(np.percentile(np.abs(inner), 99)) or 1.0

    components: list[dict[str, Any]] = []
    for i in range(n_alive):
        components.append(
            {
                "matrix": str(shorts[i]),
                "module": str(modules[i]),
                "c": int(comps[i]),
                "mean_kl": round(float(kl[i].mean()), 6),
                "max_kl": round(float(kl[i].max()), 6),
                "mean_ci": round(float(ci[i].mean()), 5),
                "norm": round(norms.get((str(modules[i]), int(comps[i])), 0.0), 4),
                **_detect_period(kl[i].astype(np.float64), n),
                "grids": {
                    "ci": _b64_u8(np.clip(ci[i], 0, 1), 1.0),
                    "kl": _b64_u8(kl[i], kl_scale),
                    "inner": _b64_i8(inner[i]),
                },
            }
        )

    payload = {
        "meta": {
            "run_id": meta["run_id"],
            "op": meta["op"],
            "n": n,
            "position": meta["position"],
            "n_alive": n_alive,
            "matrices": sorted({str(s) for s in shorts}),
            "kl_scale": kl_scale,
            "inner_scale": inner_scale,
        },
        "components": components,
    }

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run_dir) / "arith_ablation_explorer"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    data_js = out_dir / "data.js"
    data_js.write_text("window.PD_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    shutil.copyfile(_APP_TEMPLATE, out_dir / "index.html")

    logger.info(
        f"wrote explorer to {out_dir} (data.js {data_js.stat().st_size / 1e6:.1f} MB, "
        f"{n_alive} components) — open index.html"
    )
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_arith_ablation_explorer)
