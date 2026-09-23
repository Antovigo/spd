"""Per read point, the geometry the applet's 3-D view needs: the class centroids (in read
coordinates) of every kept hypothesis at every position and operation set, and the read point's
alive reads. One lazily loaded file per read point.

    python -m param_decomp.arith_repr.scripts.export_geometry --resid <raw residual dir>
        --analysis <dir> --out <app dir>/geom --keys attn_in.18 mlp_in.18

Writes `<out>/<key>.js` = `window.registerArithGeom("<key>", {...})` with float16 arrays as
base64 (`{"shape": [...], "f16": "..."}`). Centroids are raw means of the read-basis
coordinates (no centring), so the applet's origin is the zero of the post-norm residual."""

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from param_decomp.arith_repr.hypotheses import (
    QUANTITIES_BY_POSITION,
    Labels,
    build_hypotheses,
    pooled_quantities,
    supported,
)
from param_decomp.arith_repr.scripts.analyse import load_labels, load_position


def f16(a: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(a, dtype=np.float16)
    return {"shape": list(arr.shape), "f16": base64.b64encode(arr.tobytes()).decode()}


def classes_of(fit: dict[str, Any], labels: Labels) -> tuple[np.ndarray, int]:
    values = labels.quantity(fit["quantity"])
    ok = supported(values)
    if fit["kind"] == "periodic":
        tau = int(fit["period"])
        return np.where(ok, values % tau, -1), tau
    lo = int(values[ok].min())
    return np.where(ok, values - lo, -1), int(values[ok].max() - lo + 1)


def export_key(key: str, resid: Path, analysis: Path, labels_all: Labels, out: Path) -> None:
    record = json.loads((analysis / f"{key}.json").read_text())
    z = np.load(analysis / f"{key}.npz")
    B = z["B"].astype(np.float64)
    keep_prompt = (labels_all.op == 0) | (labels_all.a >= labels_all.b)
    op_masks = {
        "both": np.ones(labels_all.n, bool),
        "add": labels_all.op == 0,
        "sub": labels_all.op == 1,
    }
    payload: dict[str, Any] = {
        "key": key,
        "reads": {"v": f16(z["v"]), "kinds": z["kinds"].tolist(), "ids": z["ids"].tolist()},
        "positions": {},
    }
    for position, pos in record["positions"].items():
        Y_all = load_position(resid, key, int(position)).astype(np.float64) @ B
        per_op: dict[str, Any] = {}
        for op_name, op in pos["ops"].items():
            mask = op_masks[op_name] & keep_prompt
            labels = labels_all.subset(mask)
            Y = Y_all[mask]  # NOT centred: the origin is the zero of the post-norm residual
            # The pure part's fitted directions (row space of `Phi^T Y`, centred), ordered by
            # energy: what "this code" means elsewhere in the applet, unlike the full class
            # centroids whose span contains every coarser period and the mean.
            quantities = QUANTITIES_BY_POSITION[int(position)]
            if op_name == "both":
                quantities = pooled_quantities(quantities)
            Yc = Y - Y.mean(axis=0)
            pure: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for h in build_hypotheses(labels, quantities):
                fitted = h.Phi.T @ Yc  # (m, k)
                _, sv, Wt = np.linalg.svd(fitted, full_matrices=False)
                # Pure-part class centroids IN FUNCTION SPACE: the class means of the pure
                # part's fitted signal `Phi Phi^T Y`; exactly orthogonal to every coarser period
                # (projecting raw centroids onto the fitted directions is not — the noise
                # directions of the fit overlap the coarser periods' planes).
                cls = h.classes(labels)
                valid = cls >= 0
                counts = np.bincount(cls[valid], minlength=h.n_classes)
                ind = np.zeros((labels.n, h.n_classes))
                ind[np.flatnonzero(valid), cls[valid]] = 1.0
                weights = (h.Phi.T @ ind) / np.maximum(counts, 1)[None, :]  # (m, n_classes)
                pure[h.name] = (Wt.T, sv**2 / max(float(np.sum(Yc**2)), 1e-300), weights.T @ fitted)
            hyps: dict[str, Any] = {}
            for fit in op["fits"]:
                if not fit["dim_S"]:
                    continue
                W, energy, pure_centroids = pure[fit["name"]]
                cls, n = classes_of(fit, labels)
                valid = cls >= 0
                counts = np.bincount(cls[valid], minlength=n)
                sums = np.zeros((n, Y.shape[1]))
                np.add.at(sums, cls[valid], Y[valid])
                hyps[fit["name"]] = {
                    "counts": counts.tolist(),
                    "centroids": f16(sums / np.maximum(counts, 1)[:, None]),
                    "dirs": f16(W),
                    "pure_centroids": f16(pure_centroids),
                    "dir_energy": [round(float(e), 6) for e in energy],
                    "kept": fit["kept"],
                }
            per_op[op_name] = hyps
        payload["positions"][position] = per_op
        print(f"{key} p{position}: {sum(len(h) for h in per_op.values())} hypotheses", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}.js").write_text(
        f"window.registerArithGeom({json.dumps(key)}, {json.dumps(payload)});"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    args = parser.parse_args()
    labels = load_labels(json.loads((args.resid / "pool.json").read_text()))
    for key in args.keys:
        export_key(key, args.resid, args.analysis, labels, args.out)


if __name__ == "__main__":
    main()
