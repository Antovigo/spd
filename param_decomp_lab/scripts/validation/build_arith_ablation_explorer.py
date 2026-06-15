"""Build a GPU-free interactive HTML explorer over the ablation-KL data (Objective 2).

Consumes `collect_ablation_kl`'s `data.npz` and turns it into a self-contained applet
(`index.html` + `data.js`, `file://`-openable, no server/CDN/GPU). For every alive
component it:
- detects periodicity from the ablation-KL signal by *autocorrelation* (the periodic
  pattern is spiky, not sinusoidal, so autocorrelation finds the fundamental lag where
  FFT smears across harmonics) along each axis a / b / a+b / a-b, and
- packs the per-(a, b) grids of all five selectable color metrics (causal importance,
  ablation KL, normalized inner activation, original predicted token, ablated predicted
  token), quantized, into `data.js`.

The applet shows a grid of per-component (a, b) heatmaps filtered by detected period, with
the color metric switchable live, plus a detail panel with the autocorrelation curves.

Usage:
    python -m param_decomp_lab.scripts.validation.build_arith_ablation_explorer <run_dir_or_npz> \
        [--min-strength=0.4] [--output-dir=PATH]

Output: `<run_dir>/figures/arith_ablation_explorer/{index.html,data.js}`.
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

_APP_TEMPLATE = Path(__file__).with_name("arith_ablation_explorer_app.html")


def _autocorr(sig: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized autocorrelation r[0..len-1] of a *linearly-detrended* signal (r[0] = 1).

    Detrending matters: the autocorrelation of a monotonic ramp decays slowly and stays
    high at every small lag, which a naive peak-finder reads as a short period. Removing the
    linear trend leaves only genuine periodic structure (spikes every k).
    """
    t = np.arange(len(sig), dtype=np.float64)
    coef = np.polyfit(t, sig, 1)
    s = sig - (coef[0] * t + coef[1])
    denom = float((s * s).sum())
    if denom < 1e-12:
        return np.zeros_like(s)
    full = np.correlate(s, s, mode="full")[len(s) - 1 :]
    return full / denom


def _fundamental_period(r: NDArray[np.float64]) -> tuple[int, float]:
    """Smallest *strict* local-max lag reaching the autocorrelation plateau.

    Returns `(period, strength)` where strength is the peak autocorrelation over lags >= 2;
    a multiple of the true period also peaks, so the smallest qualifying lag is the
    fundamental. `(0, 0.0)` when there is no oscillatory peak (e.g. a detrended-flat signal):
    no fallback, since a non-oscillating signal has no period.
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


def _diag_marginals(
    grid: NDArray[np.float64], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean over constant-(a+b) and constant-(a-b) lines. grid axis order is [a-1, b-1]."""
    idx = np.arange(1, n + 1)
    ai, bj = np.meshgrid(idx, idx, indexing="ij")  # ai along axis 0 (a), bj axis 1 (b)
    s_key, d_key = (ai + bj).ravel(), (ai - bj).ravel()
    flat = grid.ravel()
    sum_m = np.array([flat[s_key == s].mean() for s in range(2, 2 * n + 1)])
    diff_m = np.array([flat[d_key == d].mean() for d in range(-(n - 1), n)])
    return sum_m, diff_m


def _analyse_periodicity(kl: NDArray[np.float64], n: int) -> dict[str, Any]:
    """Autocorrelation-based period per axis on the KL grid; pick the strongest axis.

    Grid axis order is [a-1, b-1]: f(a) averages over b (axis 1), f(b) over a (axis 0).
    """
    marg = {"a": kl.mean(axis=1), "b": kl.mean(axis=0)}
    marg["sum"], marg["diff"] = _diag_marginals(kl, n)
    autocorr = {ax: _autocorr(m.astype(np.float64)) for ax, m in marg.items()}
    per_axis = {ax: _fundamental_period(r) for ax, r in autocorr.items()}
    best_axis = max(per_axis, key=lambda ax: per_axis[ax][1])
    period, strength = per_axis[best_axis]
    return {
        "axis": best_axis,
        "period": period,
        "strength": strength,
        "autocorr": {
            ax: [round(float(x), 3) for x in autocorr[ax][: len(marg[ax]) // 2 + 1]] for ax in marg
        },
        "marg_kl": {ax: [round(float(x), 5) for x in marg[ax]] for ax in marg},
    }


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _token_to_int(decode: dict[str, str]) -> dict[int, int]:
    """Map token id -> the integer it decodes to, or -32768 (sentinel) for non-numeric."""
    out: dict[int, int] = {}
    for sid, s in decode.items():
        t = s.strip()
        out[int(sid)] = int(t) if t.lstrip("-").isdigit() and -32000 < int(t) < 32000 else -32768
    return out


def build_arith_ablation_explorer(
    source: str,
    min_strength: float = 0.4,
    output_dir: str | None = None,
) -> Path:
    """Write the ablation-KL explorer (`index.html` + `data.js`). Returns the output folder."""
    src = Path(source).expanduser()
    npz_path = src if src.suffix == ".npz" else src / "ablation_kl" / "data.npz"
    assert npz_path.exists(), f"npz not found: {npz_path} (run collect_ablation_kl first)"
    meta = json.loads((npz_path.parent / "meta.json").read_text())
    run_dir = npz_path.parent.parent

    # Keyed by full module path (layer + matrix), since the short matrix name alone collides
    # across layers.
    norms: dict[tuple[str, int], float] = {}
    comp_tsv = npz_path.parent / "components.tsv"
    if comp_tsv.exists():
        with comp_tsv.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                module = f"model.layers.{row['layer']}.{row['matrix']}"
                norms[(module, int(row["component"]))] = float(row["norm"])

    d = np.load(npz_path, allow_pickle=True)
    kl, ci, inner = d["kl"], d["ci"], d["inner_act"]  # [A, n, n]
    abl_token, orig_token = d["abl_token"], d["orig_token"]
    modules, shorts, comps = d["module"], d["short"], d["component"]
    n = int(meta["n"])
    n_alive = kl.shape[0]

    tok2int = _token_to_int(meta["token_decode"])
    vec_map = np.vectorize(lambda t: tok2int.get(int(t), -32768))
    orig_int = vec_map(orig_token).astype(np.int16)  # [n, n]
    abl_int = vec_map(abl_token).astype(np.int16)  # [A, n, n]

    kl_scale = float(np.percentile(kl[kl > 0], 99)) if (kl > 0).any() else 1.0
    inner_scale = float(np.percentile(np.abs(inner), 99)) or 1.0
    # Range spans both original and ablated predictions, so the ablated-token view (whose
    # whole point is predictions that differ from the original) isn't clamped to one color.
    all_int = np.concatenate([orig_int.ravel(), abl_int.ravel()])
    valid_tok = all_int[all_int != -32768]
    tok_lo, tok_hi = (int(valid_tok.min()), int(valid_tok.max())) if valid_tok.size else (0, 2 * n)

    components: list[dict[str, Any]] = []
    for i in range(n_alive):
        analysis = _analyse_periodicity(kl[i].astype(np.float64), n)
        components.append(
            {
                "matrix": str(shorts[i]),
                "module": str(modules[i]),
                "c": int(comps[i]),
                "mean_kl": round(float(kl[i].mean()), 6),
                "max_kl": round(float(kl[i].max()), 6),
                "norm": round(norms.get((str(modules[i]), int(comps[i])), 0.0), 4),
                **analysis,
                "grids": {
                    "ci": _b64(np.clip(np.rint(ci[i] * 255), 0, 255).astype(np.uint8)),
                    "kl": _b64(np.clip(np.rint(kl[i] / kl_scale * 255), 0, 255).astype(np.uint8)),
                    "inner": _b64(
                        np.clip(np.rint(inner[i] / inner_scale * 127), -127, 127).astype(np.int8)
                    ),
                    "abl": _b64(abl_int[i]),
                },
            }
        )

    payload = {
        "meta": {
            "run_id": meta["run_id"],
            "op": meta["op"],
            "n": n,
            "kl_direction": meta["kl_direction"],
            "position": meta["position"],
            "n_alive": n_alive,
            "matrices": sorted({str(s) for s in shorts}),
            "kl_scale": kl_scale,
            "inner_scale": inner_scale,
            "token_range": [tok_lo, tok_hi],
            "min_strength": min_strength,
        },
        "components": components,
        "orig": {"int": _b64(orig_int)},
    }

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else run_dir / "figures" / "arith_ablation_explorer"
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
