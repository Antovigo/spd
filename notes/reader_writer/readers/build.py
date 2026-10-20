"""Data for the MLP readers applet (`index.html`): AB-grids of every reader's inner activation.

A reader is a gate or up component of block l. Its inner activation on a prompt at position t is
t_c / ||V_c||, with t_c = x . V_c the alive-only model's value harvested by
`../variance/extract_alive_only.py` (`<SRC>/alive_only/L<l>/inner.npy`, shape (N, 4, K_l), positions
1..4; position 0, BOS, is the same on every prompt) and V_c the component's read vector (column of
`V_<kind>` in `comps.npz`). Readers pruned by the learned mask MASK (`ab_grids/pruned.js`, default
"kl_int, sampled mask, λ=5e-05") are left out.

The pool is a, b in 1..400 for ops "+" then "-" (prompt index = op * 160000 + (a - 1) * 400 + b - 1).
Every grid is stored as uint8, q = round(127.5 * (1 + v / m)) with m the grid's max |v| over both ops:

    OUT/meta.js                         layers, positions, kept readers per layer
    OUT/L<l>/p<t>/thumbs.js             every kept reader's 5x5-cell-mean grids (80 x 80 per op)
    OUT/L<l>/p<t>/<kind><id>.js         one reader's full grids (400 x 400 per op)

    python build.py [--src SRC] [--out OUT] [--mask NAME] [--layers 0 1 ...]
"""

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np

RUN = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
SRC = RUN / "analysis/reader_writer/variance"
FILTER = RUN / "addsub-08-last-pos-40000"
OUT = RUN / "analysis/reader_writer/readers_app/data"
N_A = N_B = 400
OPS = ("+", "-")
POSITIONS = (1, 2, 3, 4)
POSITION_LABELS = {1: "a", 2: "op", 3: "b", 4: "="}
POOL = 5
KINDS = ("gate", "up")


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def quantize(grids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """grids (K, ...) float32 -> (uint8 of the same shape, per-grid max |v| (K,))."""
    m = np.abs(grids.reshape(grids.shape[0], -1)).max(axis=1)
    m = np.where(m > 0, m, 1.0).astype(np.float32)
    scaled = grids / m.reshape(-1, *([1] * (grids.ndim - 1)))
    return np.clip(np.rint(127.5 * (1.0 + scaled)), 0, 255).astype(np.uint8), m


def load_pruned(name: str) -> tuple[str, dict[str, list[int]]]:
    text = (FILTER / "ab_grids/pruned.js").read_text().strip()
    prefix = "window.PRUNED = "
    assert text.startswith(prefix), text[:40]
    doc = json.loads(text[len(prefix) :].rstrip(";"))
    (mask,) = [m for m in doc["masks"] if m["name"] == name]
    return mask["label"], mask["pruned"]


def build_layer(layer: int, pruned: dict[str, list[int]], alive: dict[str, list[int]]) -> dict:
    comps = np.load(SRC / f"alive_only/L{layer}/comps.npz")
    inner = np.load(SRC / f"alive_only/L{layer}/inner.npy", mmap_mode="r")
    assert inner.shape == (2 * N_A * N_B, len(POSITIONS), comps["kind"].size), inner.shape
    cols, kinds, ids, norms = [], [], [], []
    for kind in KINDS:
        site = f"layers.{layer}.mlp.{kind}_proj"
        sel = np.nonzero(comps["kind"] == kind)[0]
        idx = comps["index"][sel].astype(int)
        assert sorted(idx.tolist()) == sorted(alive[site]), site
        v_norm = np.linalg.norm(comps[f"V_{kind}"], axis=0)
        drop = set(pruned.get(site, []))
        for j, (col, cid) in enumerate(zip(sel, idx, strict=True)):
            if cid not in drop:
                cols.append(int(col))
                kinds.append(kind)
                ids.append(int(cid))
                norms.append(float(v_norm[j]))
    cols_a = np.asarray(cols)
    norm_a = np.asarray(norms, np.float32)
    readers = [{"kind": k, "id": i} for k, i in zip(kinds, ids, strict=True)]
    for pi, t in enumerate(POSITIONS):
        out = OUT / f"L{layer}" / f"p{t}"
        out.mkdir(parents=True, exist_ok=True)
        vals = np.asarray(inner[:, pi, :][:, cols_a], np.float32) / norm_a  # (N, K)
        grids = vals.T.reshape(len(cols), len(OPS), N_A, N_B)
        q, m = quantize(grids)
        mean = grids.reshape(len(cols), len(OPS), -1).mean(axis=2)
        std = grids.reshape(len(cols), len(OPS), -1).std(axis=2)
        for r, reader in enumerate(readers):
            key = f"L{layer}/p{t}/{reader['kind']}{reader['id']}"
            doc = {
                "key": key,
                "absmax": float(m[r]),
                "mean": mean[r].tolist(),
                "std": std[r].tolist(),
                "grid": b64(q[r]),
            }
            (out / f"{reader['kind']}{reader['id']}.js").write_text(
                f"window.regFull({json.dumps(doc)});"
            )
        pooled = grids.reshape(len(cols), len(OPS), N_A // POOL, POOL, N_B // POOL, POOL).mean(
            axis=(3, 5)
        )
        qt, mt = quantize(pooled)
        doc = {
            "key": f"L{layer}/p{t}",
            "n": N_A // POOL,
            "absmax": mt.tolist(),
            "full_absmax": m.tolist(),
            "grid": b64(qt),
        }
        (out / "thumbs.js").write_text(f"window.regThumbs({json.dumps(doc)});")
    return {"readers": readers, "v_norm": norms}


def main() -> None:
    global SRC, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--mask", default="kl_sampled/lam_5e-05")
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(32)))
    args = ap.parse_args()
    SRC, OUT = args.src, args.out
    label, pruned = load_pruned(args.mask)
    alive = json.loads((FILTER / "alive/alive.json").read_text())["components"]
    layers = {}
    for layer in args.layers:
        t0 = time.time()
        layers[str(layer)] = build_layer(layer, pruned, alive)
        print(
            f"L{layer}: {len(layers[str(layer)]['readers'])} readers, {time.time() - t0:.0f} s",
            flush=True,
        )
    meta = {
        "mask": args.mask,
        "mask_label": label,
        "src": str(SRC),
        "n_a": N_A,
        "n_b": N_B,
        "a_min": 1,
        "b_min": 1,
        "ops": list(OPS),
        "positions": list(POSITIONS),
        "position_labels": POSITION_LABELS,
        "pool": POOL,
        "layers": layers,
    }
    (OUT / "meta.js").write_text(f"window.META = {json.dumps(meta)};")


if __name__ == "__main__":
    main()
