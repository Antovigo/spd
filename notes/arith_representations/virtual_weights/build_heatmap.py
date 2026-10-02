"""Data for the virtual-weight heatmap applet (`heatmap.html` -> OUT/heatmap/index.html).

Rows are writers, columns readers, one pixel per virtual weight M[w, r] (README.md next to the
outputs). Both axes are grouped by residual-stream position (writers: m = 0 embedding, 2b+1 o,
2b+2 down; readers: l = 2b q/k/v, 2b+1 gate/up) in stream order; within a group, components are
ordered by average-linkage clustering (cosine distance, optimal leaf ordering) of their |M|
profiles (a writer's row over all readers, a reader's column over all writers).

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


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def cluster_order(profiles: np.ndarray) -> np.ndarray:
    """Leaf order of average-linkage clustering of the rows of `profiles` (cosine distance)."""
    if len(profiles) < 3:
        return np.arange(len(profiles))
    p = profiles + 1e-12  # all-zero rows have undefined cosine distance
    d = pdist(p, "cosine")
    z = optimal_leaf_ordering(linkage(d, "average"), d)
    return leaves_list(z)


def grouped_order(pos: np.ndarray, profiles: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Order: by stream position, clustered within; and the start index of every group."""
    order, starts = [], []
    for g in np.unique(pos):
        idx = np.flatnonzero(pos == g)
        starts.append(len(order))
        order.extend(idx[cluster_order(profiles[idx])])
    return np.asarray(order), starts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comps").mkdir(exist_ok=True)
    ix = np.load(VW / "index.npz")
    absM = np.abs(np.nan_to_num(np.load(VW / "M.npy")))
    M = np.load(VW / "M.npy")

    w_ord, w_starts = grouped_order(ix["w_pos"], absM)
    r_ord, r_starts = grouped_order(ix["r_pos"], absM.T)
    print("clustered", flush=True)

    Ms = M[np.ix_(w_ord, r_ord)]
    s = np.sign(Ms) * np.sqrt(np.minimum(np.abs(Ms), 1.0))
    q = np.where(np.isnan(Ms), 0, np.clip(np.round(128 + 127 * np.nan_to_num(s)), 1, 255))
    buf = io.BytesIO()
    Image.fromarray(q.astype(np.uint8), mode="L").save(buf, "PNG", optimize=True)
    print(f"png {buf.tell() / 1e6:.1f} MB", flush=True)

    def name(kind: str, layer: int, cidx: int, token: str = "") -> str:
        return f"tok {token!r}" if kind == "embed" else f"L{layer}.{kind}.c{cidx}"

    tokens = np.load(DATASET / "index.npz")["tokens"]
    emb_ids = [int(t) for k, t in zip(ix["w_kind"], ix["w_cidx"], strict=True) if k == "embed"]
    tok_idx = np.searchsorted(np.asarray(emb_ids), tokens).astype(np.uint8)  # (N, T) -> embed index

    data = {
        "point_names": [str(p) for p in ix["point_names"]],
        "writers": {
            "name": [name(str(ix["w_kind"][j]), int(ix["w_layer"][j]), int(ix["w_cidx"][j]), str(ix["w_token"][j])) for j in w_ord],
            "kind": [str(ix["w_kind"][j]) for j in w_ord],
            "pos": [int(ix["w_pos"][j]) for j in w_ord],
            "col": [int(ix["w_col"][j]) for j in w_ord],
            "emb": [emb_ids.index(int(ix["w_cidx"][j])) if ix["w_kind"][j] == "embed" else -1 for j in w_ord],
            "starts": w_starts,
        },
        "readers": {
            "name": [name(str(ix["r_kind"][j]), int(ix["r_layer"][j]), int(ix["r_cidx"][j])) for j in r_ord],
            "kind": [str(ix["r_kind"][j]) for j in r_ord],
            "pos": [int(ix["r_pos"][j]) for j in r_ord],
            "col": [int(ix["r_col"][j]) for j in r_ord],
            "starts": r_starts,
        },
        "tokens": b64(tok_idx),
        "png": base64.b64encode(buf.getvalue()).decode(),
    }  # fmt: skip
    (OUT / "data.js").write_text("window.VW = " + json.dumps(data) + ";\n")
    shutil.copy(HERE / "heatmap.html", OUT / "index.html")
    print("data.js written", flush=True)

    vnorm = np.load(DATASET / "index.npz")["comp_v_norm"]
    inner = np.load(VW / "alive_only/inner.npy")  # (N, T, A)
    ci = np.load(DATASET / "original/ci.npy")  # (N, T, A) float16
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
