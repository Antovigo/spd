"""Data for the virtual-weight heatmap applet (`heatmap.html` -> OUT/heatmap/index.html).

Rows are writers, columns readers, one pixel per virtual weight M[w, r] (README.md next to the
outputs). Both axes are grouped by residual-stream position (writers: m = 0 embedding, 2b+1 o,
2b+2 down; readers: l = 2b q/k/v, 2b+1 gate/up) in stream order; within a group, components are
ordered by average-linkage clustering (cosine distance, optimal leaf ordering) of their |M|
profiles (a writer's row over all readers, a reader's column over all writers), restricted to
co-active pairs. For each token position p the applet can restrict both axes to the components
active at p; those subsets are re-clustered on their pairs among themselves. A second ordering
("direction") clusters each group on the vectors themselves instead: writers on U_hat, readers on
gamma_l * V_hat, distance 1 - |cos| (`orders[metric][position]`, indices into the PNG order).

Activity: a component is active at token position p (0..4: <BOS>, a, op, b, =) if its output
CI on the original model exceeds 0.01 somewhere on the (a, b) grid at p; an embedding token is
active where it occurs. A pair (w, r) is co-active if they share an active position (the
residual stream only carries a write to reads at the same position). Per-position max / mean CI
also go to VW/ci_positions.npz.

Files written to OUT/heatmap:
* `index.html`  - the applet (copied from heatmap.html).
* `data.js`     - component tables, group boundaries, and the matrix as an 8-bit grayscale PNG
  (base64): q = 0 where the writer is not upstream, else q = 128 + round(127 s) with
  s = sign(M) sqrt(min(|M|, 1)), so M = sign(q - 128) ((q - 128) / 127)^2.
* `comps/c<col>.js` - per alive component (dataset column col): over the (a, b) grid, both ops
  and the 5 positions, the alive-only model's inner activation in the ||V|| = 1 gauge (int8,
  scale per position: value = q / 127 * scale[pos]) and the filter's output CI on the original
  model (uint8, CI = q / 255).
"""

import base64
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist

from param_decomp.arith_repr.vectors.common import DATASET, RUN

VW = RUN / "analysis/virtual_weights"
OUT = VW / "heatmap"
HERE = Path(__file__).parent
N, T = 20000, 5
ALIVE = 0.01
POS_SHORT = ("B", "a", "o", "b", "=")  # <BOS>, a, op, b, =


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def cluster_order(x: np.ndarray, metric: str) -> np.ndarray:
    """Leaf order of average-linkage clustering of the rows of `x` (optimal leaf ordering).
    metric "profile": cosine distance; "direction": 1 - |cos| (a component's sign is a gauge)."""
    if len(x) < 3:
        return np.arange(len(x))
    if metric == "profile":
        d = pdist(x + 1e-12, "cosine")  # all-zero rows have undefined cosine distance
    else:
        y = x / np.linalg.norm(x, axis=1, keepdims=True)
        d = np.clip(1 - np.abs(y @ y.T), 0, None)[np.triu_indices(len(y), 1)]
    z = optimal_leaf_ordering(linkage(d, "average"), d)
    return leaves_list(z)


def grouped_order(
    pos: np.ndarray, x: np.ndarray, metric: str = "profile"
) -> tuple[np.ndarray, list[int]]:
    """Order: by stream position, clustered within; and the start index of every group."""
    order, starts = [], []
    for g in np.unique(pos):
        idx = np.flatnonzero(pos == g)
        starts.append(len(order))
        order.extend(idx[cluster_order(x[idx], metric)])
    return np.asarray(order), starts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comps").mkdir(exist_ok=True)
    ix = np.load(VW / "index.npz")
    M = np.load(VW / "M.npy")
    tokens = np.load(DATASET / "index.npz")["tokens"]
    emb_ids = [int(t) for k, t in zip(ix["w_kind"], ix["w_cidx"], strict=True) if k == "embed"]
    tok_idx = np.searchsorted(np.asarray(emb_ids), tokens).astype(np.uint8)  # (N, T) -> embed index

    # per-position CI summaries and activity: a component is active at token position p if its
    # CI exceeds ALIVE somewhere on the grid at p; an embedding token is active where it occurs
    ci = np.load(DATASET / "original/ci.npy")  # (N, T, A) float16
    ci_max = ci.max(0).astype(np.float32).T  # (A, T)
    ci_mean = ci.mean(0, dtype=np.float32).T  # (A, T)
    np.savez(VW / "ci_positions.npz", ci_max=ci_max, ci_mean=ci_mean)
    bits = 1 << np.arange(T)
    comp_act = ((ci_max > ALIVE) * bits).sum(1)  # (A,) bitmask over positions
    emb_act = np.array([((tok_idx == e).any(0) * bits).sum() for e in range(len(emb_ids))])
    w_act = np.where(ix["w_kind"] == "embed", emb_act[np.clip(np.searchsorted(emb_ids, ix["w_cidx"]), 0, len(emb_ids) - 1)],
                     comp_act[ix["w_col"]])  # fmt: skip
    r_act = comp_act[ix["r_col"]]
    co = (w_act[:, None] & r_act[None, :]) != 0
    up = np.isfinite(M)
    print(f"upstream pairs {up.sum()}, co-active {(up & co).sum()}", flush=True)
    for name_, act in (("writers", w_act), ("readers", r_act)):
        vals, cnt = np.unique(act, return_counts=True)
        print(name_, {"".join(POS_SHORT[p] for p in range(T) if v >> p & 1) or "none": int(c)
                      for v, c in zip(vals, cnt, strict=True)}, flush=True)  # fmt: skip

    absM = np.abs(np.nan_to_num(M)) * co
    w_ord, w_starts = grouped_order(ix["w_pos"], absM)
    r_ord, r_starts = grouped_order(ix["r_pos"], absM.T)
    # per token position p: the writers / readers active at p (indices into the canonical order
    # above), re-clustered within each stream-position group on their pairs among themselves
    # "direction" orders cluster on the vectors themselves: writers on U_hat, readers on
    # gamma_l * V_hat (1 - |cos|), for all components ("any") and per token position
    absC = np.abs(np.nan_to_num(M[np.ix_(w_ord, r_ord)]))
    vec = np.load(VW / "vectors.npz")
    Uc, Vc = vec["U_hat"][w_ord], vec["V_til"][r_ord]
    wp, rp = ix["w_pos"][w_ord], ix["r_pos"][r_ord]
    wa, ra = w_act[w_ord], r_act[r_ord]
    orders: dict[str, dict[str, dict[str, list[int]]]] = {"profile": {}, "direction": {}}
    orders["profile"]["any"] = {"w": list(range(len(w_ord))), "r": list(range(len(r_ord)))}
    orders["direction"]["any"] = {"w": grouped_order(wp, Uc, "direction")[0].tolist(),
                                  "r": grouped_order(rp, Vc, "direction")[0].tolist()}  # fmt: skip
    for t in range(T):
        sw, sr = np.flatnonzero(wa >> t & 1), np.flatnonzero(ra >> t & 1)
        sub = absC[np.ix_(sw, sr)]
        orders["profile"][str(t)] = {"w": sw[grouped_order(wp[sw], sub)[0]].tolist(),
                                     "r": sr[grouped_order(rp[sr], sub.T)[0]].tolist()}  # fmt: skip
        orders["direction"][str(t)] = {"w": sw[grouped_order(wp[sw], Uc[sw], "direction")[0]].tolist(),
                                       "r": sr[grouped_order(rp[sr], Vc[sr], "direction")[0]].tolist()}  # fmt: skip
        print(f"position {t}: {len(sw)} writers, {len(sr)} readers", flush=True)
    print("clustered", flush=True)

    Ms = M[np.ix_(w_ord, r_ord)]
    s = np.sign(Ms) * np.sqrt(np.minimum(np.abs(Ms), 1.0))
    q = np.where(np.isnan(Ms), 0, np.clip(np.round(128 + 127 * np.nan_to_num(s)), 1, 255))
    buf = io.BytesIO()
    Image.fromarray(q.astype(np.uint8), mode="L").save(buf, "PNG", optimize=True)
    print(f"png {buf.tell() / 1e6:.1f} MB", flush=True)

    def name(kind: str, layer: int, cidx: int, token: str = "") -> str:
        return f"tok {token!r}" if kind == "embed" else f"L{layer}.{kind}.c{cidx}"

    def ci_lists(cols: np.ndarray) -> dict[str, list]:
        ok = cols >= 0
        mx = np.where(ok[:, None], ci_max[np.maximum(cols, 0)], np.nan)
        mn = np.where(ok[:, None], ci_mean[np.maximum(cols, 0)], np.nan)
        r = lambda a: [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in a]  # noqa: E731
        return {"ci_max": r(mx), "ci_mean": r(mn)}

    data = {
        "point_names": [str(p) for p in ix["point_names"]],
        "writers": {
            "name": [name(str(ix["w_kind"][j]), int(ix["w_layer"][j]), int(ix["w_cidx"][j]), str(ix["w_token"][j])) for j in w_ord],
            "kind": [str(ix["w_kind"][j]) for j in w_ord],
            "pos": [int(ix["w_pos"][j]) for j in w_ord],
            "col": [int(ix["w_col"][j]) for j in w_ord],
            "emb": [emb_ids.index(int(ix["w_cidx"][j])) if ix["w_kind"][j] == "embed" else -1 for j in w_ord],
            "act": [int(w_act[j]) for j in w_ord],
            **ci_lists(ix["w_col"][w_ord]),
            "starts": w_starts,
        },
        "readers": {
            "name": [name(str(ix["r_kind"][j]), int(ix["r_layer"][j]), int(ix["r_cidx"][j])) for j in r_ord],
            "kind": [str(ix["r_kind"][j]) for j in r_ord],
            "pos": [int(ix["r_pos"][j]) for j in r_ord],
            "col": [int(ix["r_col"][j]) for j in r_ord],
            "act": [int(r_act[j]) for j in r_ord],
            **ci_lists(ix["r_col"][r_ord]),
            "starts": r_starts,
        },
        "orders": orders,
        "tokens": b64(tok_idx),
        "png": base64.b64encode(buf.getvalue()).decode(),
    }  # fmt: skip
    (OUT / "data.js").write_text("window.VW = " + json.dumps(data) + ";\n")
    shutil.copy(HERE / "heatmap.html", OUT / "index.html")
    print("data.js written", flush=True)
    if "--no-comps" in sys.argv:
        return

    vnorm = np.load(DATASET / "index.npz")["comp_v_norm"]
    inner = np.load(VW / "alive_only/inner.npy")  # (N, T, A)
    print("activations loaded", flush=True)
    for c in range(inner.shape[2]):
        x = inner[:, :, c].T / vnorm[c]  # (T, N): prompt i = op * 10000 + (a-1) * 100 + (b-1)
        scale = np.abs(x).max(1) + 1e-12
        qi = np.round(x / scale[:, None] * 127).astype(np.int8)
        qc = np.round(ci[:, :, c].T.astype(np.float32) * 255).astype(np.uint8)
        (OUT / "comps" / f"c{c}.js").write_text(
            f'window.vwComp({c},{{"scale":{json.dumps([float(v) for v in scale])},'
            f'"inner":"{b64(qi)}","ci":"{b64(qc)}"}});\n'
        )
        if c % 1000 == 0:
            print(f"comp {c}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
