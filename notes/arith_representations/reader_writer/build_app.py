"""Data for the reader/writer applet (`app.html`), from `extract_layer.py` and `attribute.py` outputs.

Writes OUT/app/ (open OUT/app/index.html over file://):

    index.html                 copy of app.html
    data/meta.js               window.RW_META: per position, writers and readers with their
                               supports (R^2 >= 0.9 and 0.99), scores, R^2-vs-k curves, greedy paths
    data/pos<t>/grid_<key>.js  the component's masked activation s = m a over the (a, b) grid of
                               each operation: uint8 codes 0..254 linear in [lo, hi], 255 = off
    data/pos<t>/full_<key>.js  the same for the full inner activation a (before the mask), every
                               prompt coded
    data/vec/<key>.js          U_r (gate/up reader) or V_w (down writer), float32, d_mlp
    data/pos<t>/w_<key>.js     writer w: the pre-activations with EVERY reader at its mean over
                               w's on prompts (G0, H0, float32 d_mlp), those means (support readers),
                               and each support reader's quantiles over w's on prompts (of s
                               and of the full inner activation a)

<key> is <kind><component index>, e.g. gate123, down7.

meta.js also carries each component's mean CI at t: the CI filter's output CI averaged over the
320,000 prompts, read from the filter's ab_grids slices (`mean_ci` of add_pos<t> and sub_pos<t>,
each a mean over its 160,000 prompts, averaged).

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


def mean_ci_of(ab_grids: Path, layer: int, t: int) -> dict[str, np.ndarray]:
    """kind -> mean CI (float32, every component index of the site) at position t, both ops."""
    out: dict[str, np.ndarray] = {}
    for op in OPS:
        text = (ab_grids / f"{op}_pos{t}.js").read_text()
        doc = json.loads(text[text.index("(") + 1 : text.rindex(")")])
        for mod in doc["modules"]:
            for kind in ("gate", "up", "down"):
                if mod["name"] == f"layers.{layer}.mlp.{kind}_proj":
                    x = np.frombuffer(base64.b64decode(mod["mean_ci"]), np.float32) / len(OPS)
                    out[kind] = out[kind] + x if kind in out else x
    assert set(out) == {"gate", "up", "down"}, (layer, t, sorted(out))
    return out


def write_ci_grids(
    ab_grids: Path, layer: int, t: int, pdir: Path, comps: list[tuple[str, int]]
) -> int:
    """data/pos<t>/ci_<key>.js: the component's output CI on every prompt, uint8 (CI * 255) per op,
    copied from the CI filter's ab_grids. ab_grids keeps per-prompt CI only for components whose
    mean CI is at least its mean_ci_floor (0.05); the others get no file. Returns files written."""
    saved: dict[tuple[str, str], list[int]] = {}
    for op in OPS:
        text = (ab_grids / f"{op}_pos{t}.js").read_text()
        doc = json.loads(text[text.index("(") + 1 : text.rindex(")")])
        for mod in doc["modules"]:
            for kind in ("gate", "up", "down"):
                if mod["name"] == f"layers.{layer}.mlp.{kind}_proj":
                    saved[(op, kind)] = list(mod["saved"])
    ci: dict[tuple[str, str], np.ndarray] = {}
    for (op, kind), idx in saved.items():
        path = ab_grids / f"{op}_pos{t}" / f"layers.{layer}.mlp.{kind}_proj.js"
        if not idx or not path.exists():
            continue
        text = path.read_text()
        doc = json.loads(text[text.index("(") + 1 : text.rindex(")")])
        ci[(op, kind)] = np.frombuffer(base64.b64decode(doc["ci"]), np.uint8).reshape(
            len(idx), N_A * N_B
        )
    n = 0
    for kind, comp in comps:
        if not all((op, kind) in ci and comp in saved[(op, kind)] for op in OPS):
            continue
        rows = {op: ci[(op, kind)][saved[(op, kind)].index(comp)] for op in OPS}
        jsonp(
            pdir / f"ci_{kind}{comp}.js",
            "rwCI",
            f"{t}:{kind}{comp}",
            {op: b64(rows[op]) for op in OPS},
        )
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--positions", type=int, nargs="+", default=[4, 3])
    ap.add_argument("--run_dir", type=Path, default=RUN_DIR)
    ap.add_argument(
        "--ab_grids",
        type=Path,
        default=None,
        help="CI filter ab_grids folder (default <run_dir>/addsub-08-last-pos-40000/ab_grids)",
    )
    ap.add_argument("--ci_only", action="store_true",
                    help="only (re)write the per-prompt CI files and index.html, keep everything else")  # fmt: skip
    args = ap.parse_args()
    ab_grids = args.ab_grids or args.run_dir / "addsub-08-last-pos-40000" / "ab_grids"
    out = args.run_dir / "analysis" / "reader_writer" / f"L{args.layer}"
    app = out / "app"
    if args.ci_only:
        shutil.copy(Path(__file__).with_name("app.html"), app / "index.html")
        comps = np.load(out / "comps.npz")
        for t in args.positions:
            at = np.load(out / f"attr_pos{t}.npz")
            cols = np.concatenate([at["reader_cols"], at["writer_cols"]])
            pairs = [(str(comps["kind"][c]), int(comps["index"][c])) for c in cols]
            n = write_ci_grids(ab_grids, args.layer, t, app / "data" / f"pos{t}", pairs)
            print(f"t={t}: CI files for {n} of {len(pairs)} components", flush=True)
        return
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
    meta: dict = {"layer": args.layer, "run": args.run_dir.name, "ops": list(OPS),
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
        all_on = np.ones(a.shape[0], bool)
        for col in np.concatenate([rcols, wcols]):
            jsonp(pdir / f"grid_{keys[col]}.js", "rwGrid", f"{t}:{keys[col]}",
                  grid_doc(a[:, col] * m[:, col], m[:, col]))  # fmt: skip
            jsonp(pdir / f"full_{keys[col]}.js", "rwGrid", f"{t}:full:{keys[col]}",
                  grid_doc(a[:, col], all_on))  # fmt: skip
        mci = mean_ci_of(ab_grids, args.layer, t)
        n_ci = write_ci_grids(ab_grids, args.layer, t, pdir,
                              [(str(kind[c]), int(index[c])) for c in np.concatenate([rcols, wcols])])  # fmt: skip
        print(f"t={t}: CI files for {n_ci} of {rcols.size + wcols.size} components", flush=True)
        score = at["score"]
        share = score / np.maximum(score.sum(1, keepdims=True), 1e-30)
        # k at the target, or (target never reached) the k of the best held-out R^2
        ks_grid, r2_abl = at["ks"], np.nan_to_num(at["r2_abl"], nan=-np.inf)
        k_of = {tag: np.where(at[f"k_{tag}"] > 0, at[f"k_{tag}"], ks_grid[np.argmax(r2_abl, 1)])
                for tag in ("r2_90", "r2_99")}  # fmt: skip
        supports = {tag: [list(map(int, at["rank_abl"][w, : int(k_of[tag][w])])) for w in range(W)]
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
                # per support reader, over ALL prompts where w is on: min, 10/30/50/70/90 %, max of
                # its masked activation s (q) and of its full inner activation a (qa)
                "q": np.quantile(s[on][:, sup], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], axis=0).T.astype(float).round(5).tolist(),
                "qa": np.quantile(a[on][:, rcols[sup]], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], axis=0).T.astype(float).round(5).tolist(),
            })  # fmt: skip
            g = greedy.get(w)
            writers.append({
                "key": key, "comp": int(index[col]), "p_on": float(at["p_on"][w]),
                "mean_ci": float(mci["down"][index[col]]),
                "write": float(at["write"][w]), "k90": int(k_of["r2_90"][w]),
                "k99": int(k_of["r2_99"][w]), "reached90": bool(at["k_r2_90"][w] > 0),
                "reached99": bool(at["k_r2_99"][w] > 0), "sup90": supports["r2_90"][w],
                "sup99": supports["r2_99"][w],
                "share": {int(r): float(share[w, r]) for r in sup},
                "r2_abl": np.nan_to_num(at["r2_abl"][w], nan=-1).round(4).tolist(),
                "r2_wt": np.nan_to_num(at["r2_wt"][w], nan=-1).round(4).tolist(),
                "greedy": None if g is None else {"chosen": g["chosen"], "r2": [round(x, 4) for x in g["r2_held"]]},
                "r2_sample_support": r2_sample,
                "offsets": [{"r": int(r), "frac_zero": float(at["frac_zero"][w, r]),
                             "frac_mean": float(at["frac_mean"][w, r]),
                             "mean": float(at["mean_s"][w, r])}
                            for r in np.flatnonzero(at["offset"][w])],
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
                "mean_ci": float(mci[str(kind[col])][index[col]]),
                "writers90": [w for w in range(W) if r in supports["r2_90"][w]],
                "writers99": [w for w in range(W) if r in supports["r2_99"][w]],
                "offset_for": [int(w) for w in np.flatnonzero(at["offset"][:, r])],
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
