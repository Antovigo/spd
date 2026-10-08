"""Data for the quantity explorer applet (qapplet.html), written to OUT/applet (local, static files).

Per position t (meta.js, window.QA.pos[t]): the domain (prompt index, op, a, b per domain row), the
quantity map (sites in stream order, quantities, and per accepted (quantity, site): drop-one
excess, dependence, stability), each quantity's natural value per domain row (qfeat.Cand.shown),
and a fixed random subsample of 4000 domain rows (t >= 3).
Per site (sites/t<t>_<site>.js): for every accepted quantity q, its basis vectors v (orthonormal,
in the readers' span, ranked by the variance of q's term along them; dirs_t<t>.npz) with the
standard deviation along each; the projections of the stream divided by its RMS, x(i) / rho(i),
on every basis vector (int16 with a per-vector scale: what the readers multiply by their read
vectors); every reader's read vector projected on every basis vector (the arrows; reader r's
inner activation is (x / rho) . V~_r); the Gram matrix of the basis vectors (for the applet's
Gram-Schmidt of the three chosen axes); and q's readers with their share of read variance.
Reader grids and CI come from the virtual-weights heatmap applet (virtual_weights/heatmap/comps).
Plotting uses plotly.js: a local copy OUT/plotly.min.js if present (download it once from
cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js), else the CDN.

    python qapplet.py [--tag name] [--positions 1,2,3,4]
"""

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np
from qdata import DATA, POS_NAMES, load
from qfeat import pool
from qprep import VW

OUT = DATA.parent / "applet"
SUBSAMPLE = 4000


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--positions", default="1,2,3,4")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    (OUT / "sites").mkdir(parents=True, exist_ok=True)
    vix = np.load(VW / "index.npz")
    names = {int(c): f"L{int(li)}.{k}.c{int(ci)}" for c, k, li, ci in zip(vix["r_col"], vix["r_kind"], vix["r_layer"], vix["r_cidx"], strict=True)}  # fmt: skip
    meta = {"pos": {}, "reader_names": {}, "pos_names": POS_NAMES}
    for t in map(int, args.positions.split(",")):
        run = json.loads((DATA / f"qloop_t{t}{tag}.json").read_text())
        dirs = json.loads((DATA / f"dirs_t{t}{tag}.json").read_text())
        dz = np.load(DATA / f"dirs_t{t}{tag}.npz")
        pos = load(t)
        cands = {c.name: c for c in pool(pos)}
        sites = [s["site"] for s in run["sites"] if not s.get("constant")]
        qnames = sorted({a["name"] for s in run["sites"] for a in s["accepted"]})
        qi = {n: i for i, n in enumerate(qnames)}
        dep = {
            (s["site"], q["name"]): q["dependence"] for s in dirs["sites"] for q in s["quantities"]
        }
        cells = [[qi[a["name"]], sites.index(s["site"]), round(a["drop_one_loss"] - a["chance"], 6),
                  dep.get((s["site"], a["name"]), 0.0), a.get("stability", 1.0)]
                 for s in run["sites"] if not s.get("constant") for a in s["accepted"]]  # fmt: skip
        sub = np.sort(np.random.default_rng(0).choice(pos.D, min(SUBSAMPLE, pos.D), replace=False))
        meta["pos"][t] = {
            "D": pos.D, "rows": b64(pos.rows.astype(np.int32)), "op": b64(pos.op.astype(np.uint8)),
            "a": b64(pos.a.astype(np.uint8)), "b": b64(pos.b.astype(np.uint8)),
            "sites": sites, "quantities": qnames, "cells": cells,
            "values": {i: b64(cands[n].shown().astype(np.float32)) for n, i in qi.items()},
            "subsample": b64(sub.astype(np.int32)) if t >= 3 else None,
        }  # fmt: skip
        for drec in dirs["sites"]:
            lab = drec["site"]
            s = pos.sites[lab]
            X = s.Z / s.rho[:, None]  # stream / rho in span coordinates (D x k)
            vecs, quants = [], []
            for q in drec["quantities"]:
                B = (
                    dz[f"{lab}/{q['index']}/basis"].astype(np.float64) @ s.Q
                )  # r x k, span coordinates
                quants.append({"q": qi[q["name"]], "offset": len(vecs), "r": len(B),
                               "sd": [round(float(x), 5) for x in dz[f"{lab}/{q['index']}/sd"]],
                               "readers": [[int(np.flatnonzero(s.cols == int(c))[0]), sh] for c, sh in q["readers"].items()]})  # fmt: skip
                vecs.extend(B)
            V = np.array(vecs).T  # k x R
            P = X @ V  # D x R
            scale = np.abs(P).max(0) / 32767 + 1e-30
            data = {
                "k": int(s.W.shape[0]), "cols": [int(c) for c in s.cols], "R": V.shape[1], "quantities": quants,
                "proj": b64(np.round(P / scale).astype(np.int16)), "pscale": [float(x) for x in scale],
                "arrows": b64((s.W.T @ V).astype(np.float32)), "gram": b64((V.T @ V).astype(np.float32)),
            }  # fmt: skip
            (OUT / "sites" / f"t{t}_{lab}.js").write_text(
                f"window.QAsite({json.dumps(f'{t}|{lab}')},{json.dumps(data)});\n"
            )
            meta["reader_names"].update({int(c): names[int(c)] for c in s.cols})
        print(f"t={t}: {len(dirs['sites'])} sites written", flush=True)
    (OUT / "meta.js").write_text("window.QA = " + json.dumps(meta) + ";\n")
    shutil.copy(Path(__file__).with_name("qapplet.html"), OUT / "index.html")
    print(f"applet: {OUT / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
