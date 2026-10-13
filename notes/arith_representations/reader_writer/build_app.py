"""Data for the reader/writer applet (`app.html`), from `extract_layer.py` and `attribute.py` outputs.

Writes OUT/app/ (open OUT/app/index.html over file://):

    index.html                 copy of app.html
    data/meta.js               window.RW_META: per position, writers and readers with their
                               supports (R^2 >= 0.9 and 0.99), scores, R^2-vs-k curves, greedy paths
    data/pos<t>/grid_<key>.js  the component's masked activation s = m a over the (a, b) grid of
                               each operation: uint8 codes 0..254 linear in [lo, hi], 255 = off
    data/vec/<key>.js          U_r (gate/up reader) or V_w (down writer), float32, d_mlp
    data/pos<t>/w_<key>.js     writer w: the pre-activations with EVERY reader at its mean over
                               w's on prompts (G0, H0, float32 d_mlp), those means (support readers),
                               and a sample of on prompts (prompt id, support readers' s, y_w)

<key> is <kind><component index>, e.g. gate123, down7.

The browser predicts y_w(s_r, s_o) = V_w . (silu(G) * H) with G = G0 + sum over the two plotted
readers of (s - mean) U (gate ones; up ones go into H the same way), every other reader at its
mean. This check is reproduced here in float32 (`meta.js` stores the max deviation).

    python build_app.py --layer 18 --positions 4 3
"""

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np

RUN_DIR = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
N_SCATTER = 2000
N_A = N_B = 400
OPS = ("add", "sub")
POSITION_LABELS = ("<BOS>", "a", "op", "b", "=")


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def jsonp(path: Path, fn: str, key: str, doc: dict) -> None:
    path.write_text(f"window.{fn}({json.dumps(key)}, {json.dumps(doc)});")


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def grid_doc(s: np.ndarray, on: np.ndarray) -> dict:
    """s, on (N,) over the pool (add block then sub block, row-major (a, b))."""
    vals = s[on]
    lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 0.0)
    span = hi - lo if hi > lo else 1.0
    code = np.clip(np.round((s - lo) / span * 254), 0, 254).astype(np.uint8)
    code[~on] = 255
    n = N_A * N_B
    return {"lo": lo, "hi": hi, "n_a": N_A, "n_b": N_B,
            **{op: b64(code[i * n : (i + 1) * n]) for i, op in enumerate(OPS)}}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--positions", type=int, nargs="+", default=[4, 3])
    args = ap.parse_args()
    out = RUN_DIR / "analysis" / "reader_writer" / f"L{args.layer}"
    app = out / "app"
    (app / "data" / "vec").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("app.html"), app / "index.html")
    comps = np.load(out / "comps.npz")
    kind, index = comps["kind"], comps["index"]
    n_gate, n_up = int((kind == "gate").sum()), int((kind == "up").sum())
    keys = [f"{k}{i}" for k, i in zip(kind, index, strict=True)]
    inner = np.load(out / "inner.npy", mmap_mode="r")
    masks = np.load(out / "masks.npy", mmap_mode="r")
    assert inner.shape[0] == len(OPS) * N_A * N_B, inner.shape

    def u_of(col: int) -> np.ndarray:
        return comps["U_gate"][col] if col < n_gate else comps["U_up"][col - n_gate]

    written_vec: set[str] = set()
    meta: dict = {"layer": args.layer, "run": RUN_DIR.name, "ops": list(OPS),
                  "position_labels": list(POSITION_LABELS), "positions": {}}  # fmt: skip
    rng = np.random.default_rng(0)
    for t in args.positions:
        at = np.load(out / f"attr_pos{t}.npz")
        greedy = {g["writer"]: g for g in json.loads((out / f"greedy_pos{t}.json").read_text())}
        pdir = app / "data" / f"pos{t}"
        pdir.mkdir(exist_ok=True)
        a = np.asarray(inner[:, t], np.float32)
        m = np.asarray(masks[:, t])
        rcols, wcols, rg = at["reader_cols"], at["writer_cols"], int(at["n_gate"])
        R, W = rcols.size, wcols.size
        s = a[:, rcols] * m[:, rcols]
        for col in np.concatenate([rcols, wcols]):
            jsonp(pdir / f"grid_{keys[col]}.js", "rwGrid", f"{t}:{keys[col]}",
                  grid_doc(a[:, col] * m[:, col], m[:, col]))  # fmt: skip
        score = at["I"]
        share = score / np.maximum(score.sum(1, keepdims=True), 1e-30)
        # k at the target, or (target never reached) the k of the best held-out R^2
        ks_grid, r2_ig = at["ks"], np.nan_to_num(at["r2_ig"], nan=-np.inf)
        k_of = {tag: np.where(at[f"k_{tag}"] > 0, at[f"k_{tag}"], ks_grid[np.argmax(r2_ig, 1)])
                for tag in ("r2_90", "r2_99")}  # fmt: skip
        supports = {tag: [list(map(int, at["rank_ig"][w, : int(k_of[tag][w])])) for w in range(W)]
                    for tag in k_of}  # fmt: skip
        Ug = np.stack([u_of(c) for c in rcols[:rg]])
        Uu = np.stack([u_of(c) for c in rcols[rg:]])
        Vd = comps["V_down"][:, wcols - n_gate - n_up]
        writers, max_dev = [], 0.0
        for w in range(W):
            col = int(wcols[w])
            sup = sorted(set(supports["r2_99"][w]) | set(supports["r2_90"][w]))
            mean = at["mean_s"][w]
            G0, H0 = mean[:rg] @ Ug, mean[rg:] @ Uu
            v = Vd[:, w]
            key = keys[col]
            if key not in written_vec:
                jsonp(
                    app / "data" / "vec" / f"{key}.js",
                    "rwVec",
                    key,
                    {"v": b64(v.astype(np.float32))},
                )
                written_vec.add(key)
            on = np.flatnonzero(m[:, col])
            pick = (
                np.sort(rng.choice(on, min(N_SCATTER, on.size), replace=False)) if on.size else on
            )
            y = a[pick, col]
            # float32 check of the browser's predictor: all support readers at their true values
            G = (
                G0
                + (s[pick][:, [r for r in sup if r < rg]] - mean[[r for r in sup if r < rg]])
                @ Ug[[r for r in sup if r < rg]]
            )
            H = (
                H0
                + (s[pick][:, [r for r in sup if r >= rg]] - mean[[r for r in sup if r >= rg]])
                @ Uu[[r - rg for r in sup if r >= rg]]
            )
            yhat = (silu(G) * H) @ v
            r2_sample = (
                float(1 - np.mean((y - yhat) ** 2) / max(np.var(y), 1e-30)) if y.size > 1 else None
            )
            jsonp(pdir / f"w_{key}.js", "rwWriter", f"{t}:{key}", {
                "G0": b64(G0.astype(np.float32)), "H0": b64(H0.astype(np.float32)),
                "readers": sup, "mean": [float(mean[r]) for r in sup],
                "prompt": pick.astype(int).tolist(), "y": y.astype(float).round(5).tolist(),
                "s": s[pick][:, sup].T.astype(float).round(5).tolist(),
                "yhat_support99": yhat.astype(float).round(5).tolist(),
            })  # fmt: skip
            g = greedy.get(w)
            writers.append({
                "key": key, "comp": int(index[col]), "p_on": float(at["p_on"][w]),
                "write": float(at["write"][w]), "k90": int(k_of["r2_90"][w]),
                "k99": int(k_of["r2_99"][w]), "reached90": bool(at["k_r2_90"][w] > 0),
                "reached99": bool(at["k_r2_99"][w] > 0), "sup90": supports["r2_90"][w],
                "sup99": supports["r2_99"][w],
                "share": {int(r): float(share[w, r]) for r in sup},
                "r2_ig": np.nan_to_num(at["r2_ig"][w], nan=-1).round(4).tolist(),
                "r2_wt": np.nan_to_num(at["r2_wt"][w], nan=-1).round(4).tolist(),
                "greedy": None if g is None else {"chosen": g["chosen"], "r2": [round(x, 4) for x in g["r2_held"]]},
                "r2_sample_support": r2_sample,
            })  # fmt: skip
            max_dev = max(max_dev, 0.0 if r2_sample is None else 1 - r2_sample)
        readers = []
        for r in range(R):
            col = int(rcols[r])
            key = keys[col]
            if key not in written_vec:
                jsonp(
                    app / "data" / "vec" / f"{key}.js",
                    "rwVec",
                    key,
                    {"v": b64(u_of(col).astype(np.float32))},
                )
                written_vec.add(key)
            readers.append({
                "key": key, "kind": str(kind[col]), "comp": int(index[col]),
                "p_on": float(m[:, col].mean()),
                "mean_abs": float(np.abs(s[:, r]).mean()), "std": float(s[:, r].std()),
                "writers90": [w for w in range(W) if r in supports["r2_90"][w]],
                "writers99": [w for w in range(W) if r in supports["r2_99"][w]],
            })  # fmt: skip
        tot = sum(x["write"] for x in writers) or 1.0
        for x in writers:
            x["write_share"] = x["write"] / tot
        meta["positions"][str(t)] = {"label": POSITION_LABELS[t], "ks": at["ks"].tolist(),
                                     "writers": writers, "readers": readers}  # fmt: skip
        print(f"t={t}: {W} writers, {R} readers; min sample R^2 with the R^2>=0.99 support "
              f"(train+test prompts) {1 - max_dev:.4f}", flush=True)  # fmt: skip
    (app / "data" / "meta.js").write_text(f"window.RW_META = {json.dumps(meta)};")
    print(f"-> {app / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
