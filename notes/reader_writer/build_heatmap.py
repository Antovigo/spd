"""Data for the reader x writer heatmap applet (`heatmap.html`), from `attribute.py` outputs.

For every block l and position t it takes the writers w (down components) that have a held-out
R^2 (on in at least MIN_EVAL held-out prompts) and the readers r (gate/up components), and
stores two W x R matrices:

    frac_mean[w, r] = E[(y_w(s_r -> mean) - y_w)^2 | w on] / Var(y_w | w on)
    frac_zero[w, r] = E[(y_w(s_r -> 0)    - y_w)^2 | w on] / E[y_w^2 | w on]

(y_w the writer's inner activation, s_r the reader's masked activation, both from `attribute.py`
on the fit half) plus the offset flag (frac_mean < 1 % and frac_zero > 10 %).

Rows and columns are ordered by bipartite spectral co-clustering (Dhillon 2001): with
A = clip(frac_mean, 0, 1), the normalised matrix D_w^-1/2 A D_r^-1/2 (D_w, D_r the row and column
sums) is decomposed by SVD; writers and readers are embedded together with singular vectors
2..n+1 (each scaled by D^-1/2), and the joint point set is ordered by Ward clustering with
optimal leaf ordering. n is taken from N_VECS, keeping the order with the largest diagonal mass
(share of A within 10 % of the axis length of the diagonal). Writers and readers that connect to each other then sit at the same place
along their axes, so connected blocks lie near the diagonal.

Each layer gets the positions among --positions that `attribute.py` has analysed.

Writes <root>/heatmap/index.html and <root>/heatmap/data/L<l>.js (window.hmLayer(...)).

    python build_heatmap.py
"""

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering

ROOT = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer")
POSITION_LABELS = ("<BOS>", "a", "op", "b", "=")
LOG_MIN = -4  # codes: 0 = below 10^LOG_MIN, 1..255 log-linear on [10^LOG_MIN, 1], 255 = >= 1
READER_MIN = 1e-3  # readers whose largest frac_mean over the kept writers is below this are dropped
N_VECS = (4, 8, 16)


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def encode(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        lg = np.log10(np.maximum(x, 1e-30))
    code = 1 + np.round((lg - LOG_MIN) / -LOG_MIN * 254)
    return np.where(x < 10.0**LOG_MIN, 0, np.clip(code, 1, 255)).astype(np.uint8)


def co_order(A: np.ndarray, n_vec: int) -> tuple[np.ndarray, np.ndarray]:
    """Joint ordering of rows and columns of a nonnegative A (W, R)."""
    W, R = A.shape
    dw, dr = A.sum(1), A.sum(0)
    live_w, live_r = np.flatnonzero(dw > 0), np.flatnonzero(dr > 0)
    B = A[np.ix_(live_w, live_r)]
    sw, sr = 1 / np.sqrt(dw[live_w]), 1 / np.sqrt(dr[live_r])
    U, _, Vt = np.linalg.svd(sw[:, None] * B * sr[None, :], full_matrices=False)
    n = min(n_vec, U.shape[1] - 1)
    Z = np.concatenate([sw[:, None] * U[:, 1 : n + 1], sr[:, None] * Vt[1 : n + 1].T])
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    link = linkage(Z, "ward")
    leaves = leaves_list(optimal_leaf_ordering(link, Z))
    nw = live_w.size
    rows = [live_w[i] for i in leaves if i < nw]
    cols = [live_r[i - nw] for i in leaves if i >= nw]
    rows += [w for w in range(W) if dw[w] == 0]
    cols += [r for r in range(R) if dr[r] == 0]
    return np.asarray(rows), np.asarray(cols)


def diag_mass(A: np.ndarray, band: float) -> float:
    """Share of A's mass within `band` (fraction of the axis length) of the diagonal."""
    W, R = A.shape
    i, j = np.meshgrid(np.arange(W) / max(W - 1, 1), np.arange(R) / max(R - 1, 1), indexing="ij")
    return float(A[np.abs(i - j) <= band].sum() / max(A.sum(), 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--positions", type=int, nargs="+", default=[4, 3, 2, 1, 0])
    args = ap.parse_args()
    out = args.root / "heatmap"
    (out / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("heatmap.html"), out / "index.html")
    rng = np.random.default_rng(0)
    layers = []
    for ldir in sorted(args.root.glob("L*"), key=lambda p: int(p.name[1:])):
        layer = int(ldir.name[1:])
        positions = [t for t in args.positions if (ldir / f"attr_pos{t}.npz").exists()]
        if not positions:
            continue
        doc: dict = {"layer": layer, "positions": {}}
        for t in positions:
            at = np.load(ldir / f"attr_pos{t}.npz")
            rg = int(at["n_gate"])
            evaluated = ~np.isnan(at["r2_abl"]).all(1)
            fm = np.nan_to_num(at["frac_mean"][evaluated], nan=0.0)
            fz = np.nan_to_num(at["frac_zero"][evaluated], nan=0.0)
            off = at["offset"][evaluated]
            keep_r = np.flatnonzero((fm.max(0) >= READER_MIN) | off.any(0))
            A = np.clip(fm[:, keep_r], 0, 1)
            orders = [co_order(A, n) for n in N_VECS]
            best = int(np.argmax([diag_mass(A[np.ix_(r, c)], 0.1) for r, c in orders]))
            rows, cols = orders[best]
            Ao = A[np.ix_(rows, cols)]
            perm_w = rng.permutation(Ao.shape[0])
            perm_r = rng.permutation(Ao.shape[1])
            w_ids = np.flatnonzero(evaluated)[rows]
            r_ids = keep_r[cols]
            k99 = at["k_r2_99"][evaluated][rows]
            doc["positions"][str(t)] = {
                "label": POSITION_LABELS[t],
                "n_writers_all": int(evaluated.size), "n_readers_all": int(fm.shape[1]),
                "writers": [{"key": f"down{int(at['writer_index'][w])}",
                             "p_on": round(float(at["p_on"][w]), 6),
                             "write": float(at["write"][w]), "k99": int(k)}
                            for w, k in zip(w_ids, k99, strict=True)],
                "readers": [{"key": ("gate" if r < rg else "up") + str(int(at["reader_index"][r]))}
                            for r in r_ids],
                "frac_mean": b64(encode(fm[np.ix_(rows, keep_r[cols])])),
                "frac_zero": b64(encode(fz[np.ix_(rows, keep_r[cols])])),
                "offset": b64(off[np.ix_(rows, keep_r[cols])].astype(np.uint8)),
                "n_vec": N_VECS[best], "diag10": diag_mass(Ao, 0.1),
                "diag10_random": diag_mass(Ao[np.ix_(perm_w, perm_r)], 0.1),
            }  # fmt: skip
            p = doc["positions"][str(t)]
            print(f"L{layer} t={t}: {len(p['writers'])}/{p['n_writers_all']} writers, "
                  f"{len(p['readers'])}/{p['n_readers_all']} readers; mass within 10 % of the "
                  f"diagonal {p['diag10']:.2f} (random order {p['diag10_random']:.2f})", flush=True)  # fmt: skip
        (out / "data" / f"L{layer}.js").write_text(f"window.hmLayer({json.dumps(doc)});")
        layers.append(layer)
    meta = {"layers": layers, "positions": args.positions, "log_min": LOG_MIN,
            "reader_min": READER_MIN, "position_labels": list(POSITION_LABELS)}  # fmt: skip
    (out / "data" / "meta.js").write_text(f"window.HM_META = {json.dumps(meta)};")
    print(f"-> {out / 'index.html'}")


if __name__ == "__main__":
    main()
