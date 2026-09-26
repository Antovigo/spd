"""Atlas of the quantities read from the residual stream, at every layer, reader site and position.

    python -m param_decomp.arith_repr.isa.atlas prep <model>          # split inner.npy by layer
    python -m param_decomp.arith_repr.isa.atlas layer <model> <layer> # stages A-B at 8 read points
    python -m param_decomp.arith_repr.isa.atlas link <model>          # stage C, then labels (D)

`model` is `decomposed` (the components-only model: CI-active components, no weight delta) or
`original`. Both operations are pooled (20,000 prompts), so the operation is one of the variables.

A read point is (layer, site, position): site `attn` = the q, k and v readers of the block's
attention, `mlp` = its gate and up readers; position a, op, b or `=` (BOS is constant).

Stage A, quantities at one read point (label-free).
    1. Readers' inner activations times the stream's RMS: `h_c = x . (g * V_c)`, linear in the
       raw stream x. Constant readers are dropped.
    2. Rank r: dimensions of the reader correlation above the same rank after shuffling each
       reader over the prompts (parallel analysis). Whiten those r dimensions, with the readers
       weighted by their variance.
    3. Sphere pursuit (`pursuit.pursue`): frames are fitted on half of the distinct prompts and
       accepted and scored on the other half (at positions a and op many prompts share one stream,
       and few distinct points can be fitted to look like a sphere). Seeded by the readers from the
       largest variance down,
       fit the k-dim frame (k <= 4) with the most constant squared norm,
       J = Var(|z|^2) / (2k) (0 for a binary variable, a circle, a simplex; 1 for noise); keep it if
       J < 0.5 on held-out prompts, split it into independent spheres, remove it, next reader.
    4. Each quantity: its coordinates z on every prompt, its stream pattern E[(x - mean) z] (where
       it is written), and each reader's share of loading energy in it.
Stage B, relations at one read point: R2(i | j), how much quantity i is a function of quantity j
    (nearest-neighbour regression, `variables.cond_r2`).
Stage C, codes across the network: two quantities at different read points carry the same code
    when one is a linear image of the other. Their coordinates are unit-variance and uncorrelated
    within each quantity, so the canonical correlations are the singular values of their cross-
    covariance block; two quantities are linked when they have the same dimension and all their
    canonical correlations are >= LINK, and the connected components are the codes. A smaller
    quantity inside a larger one (all its canonical correlations >= LINK) is recorded as contained.
Stage D, labels after the fact: each quantity's share of variance explained by the operation and by
    a, b, a + b, a - b and the result (a + b on addition, a - b on subtraction), and its dominant
    harmonic; a code is named after its best-explained member."""

import json
import sys
from typing import Any

import numpy as np
from scipy.sparse.csgraph import connected_components

from param_decomp.arith_repr.isa.pursuit import assign, parallel_rank, pursue
from param_decomp.arith_repr.isa.variables import r2_matrix
from param_decomp.arith_repr.vectors.common import DATASET, RUN, comp_table

ATLAS = RUN / "analysis/arith_repr/atlas"
POSITIONS = {1: "a", 2: "op", 3: "b", 4: "="}
SITES = {"attn": ("q", "k", "v"), "mlp": ("gate", "up")}
TOL = 0.5
LINK = 0.8
N_FIT = 4000


def prep(model: str) -> None:
    """Split the (N, 5, A) inner activations into one (N, 5, n_layer) array per layer."""
    comps = comp_table()
    inner = np.load(DATASET / model / "inner.npy", mmap_mode="r")
    out = ATLAS / model / "inner"
    out.mkdir(parents=True, exist_ok=True)
    layers = [np.flatnonzero(comps["layer"] == li) for li in range(32)]
    arrays = [np.lib.format.open_memmap(out / f"L{li}.npy", "w+", np.float32, (inner.shape[0], 5, c.size))
              for li, c in enumerate(layers)]  # fmt: skip
    for s in range(0, inner.shape[0], 1000):
        block = np.asarray(inner[s : s + 1000])
        for li, c in enumerate(layers):
            arrays[li][s : s + 1000] = block[:, :, c]
        print("prep", s, flush=True)
    for a in arrays:
        a.flush()


def read_point_index(layer: int, site: str) -> int:
    """Index of the raw stream point the site's readers read (before the attn / mlp add)."""
    return 2 * layer + (0 if site == "attn" else 1)


def analyse_point(H: np.ndarray, x: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Stages A and B at one read point. H: (N, n) readers; x: (N, 4096) raw stream."""
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(1) + 1e-5)
    Hr = H.astype(np.float64) * rms[:, None]
    Hc = Hr - Hr.mean(0)
    var = Hc.var(0)
    keep = var > 1e-8 * max(float(var.max()), 1e-30)
    if keep.sum() < 2:
        return {"r": 0, "quantities": [], "n_readers": int(keep.sum())}
    Hc, var = Hc[:, keep], var[keep]
    kept = [n for n, k in zip(names, keep, strict=True) if k]
    N = len(Hc)
    Hz = Hc / np.sqrt(var)
    r = parallel_rank(Hz, np.linalg.svd(Hz, compute_uv=False))
    if r == 0:
        return {"r": 0, "quantities": [], "n_readers": len(kept)}
    U, sv, Vt = np.linalg.svd(Hc / Hc.std(), full_matrices=False)
    Y = U[:, :r] * np.sqrt(N)
    L = Vt[:r].T * (sv[:r] / np.sqrt(N))
    # distinct prompts (at positions a and op many prompts share one stream): fit frames on half of
    # them, accept and score on the other half
    _, uniq = np.unique(np.round(Hc, 6), axis=0, return_index=True)
    order = np.random.default_rng(0).permutation(uniq)
    half = len(order) // 2
    fit, test = order[:half][:N_FIT], order[half:][:N_FIT]
    found = pursue(Y[fit], L, np.argsort(-var), tol=TOL, Y_test=Y[test])
    E = assign(L, found)
    xc = x.astype(np.float32) - x.astype(np.float32).mean(0)
    Zs = [Y @ q.frame for q in found]
    quantities = []
    for gi, (q, z) in enumerate(zip(found, Zs, strict=True)):
        readers = np.flatnonzero(E[:, gi] >= 0.25)
        readers = readers[np.argsort(-(E[readers, gi] * var[readers]))]
        quantities.append({
            "k": int(z.shape[1]), "J": round(float(q.J), 3), "seed": kept[q.seed],
            "readers": [[kept[i], round(float(E[i, gi]), 2), round(float(np.sqrt(var[i])), 3)] for i in readers],
            "z": z.astype(np.float32), "pattern": (xc.T @ z.astype(np.float32) / N).astype(np.float16),
        })  # fmt: skip
    R2 = r2_matrix(Zs) if len(Zs) > 1 else np.eye(len(Zs))
    return {"r": int(r), "n_readers": len(kept), "n_distinct": int(len(uniq)), "quantities": quantities, "r2": np.round(R2, 3).tolist(),
            "explained": float(1 - E[:, -1].mean()), "unexplained": int((E[:, -1] > 0.5).sum())}  # fmt: skip


def run_layer(model: str, layer: int) -> None:
    comps = comp_table()
    cols = np.flatnonzero(comps["layer"] == layer)
    inner = np.load(ATLAS / model / "inner" / f"L{layer}.npy", mmap_mode="r")
    resid = np.load(DATASET / model / "resid.npy", mmap_mode="r")
    out = ATLAS / model / "points"
    out.mkdir(parents=True, exist_ok=True)
    for site, kinds in SITES.items():
        sel = np.flatnonzero(np.isin(comps["kind"][cols], kinds))
        names = [f"L{layer}.{comps['kind'][cols[i]]}.c{comps['cidx'][cols[i]]}" for i in sel]
        t = read_point_index(layer, site)
        for pos, pname in POSITIONS.items():
            key = f"L{layer}_{site}_{pname}".replace("=", "eq")
            H = np.asarray(inner[:, pos][:, sel])
            x = np.asarray(resid[t, :, pos], np.float32)
            res: dict[str, Any] = (
                analyse_point(H, x, names)
                if sel.size >= 2
                else {"r": 0, "quantities": [], "n_readers": int(sel.size)}
            )
            arrays = {}
            for qi, q in enumerate(res["quantities"]):
                arrays[f"z{qi}"] = q.pop("z")
                arrays[f"p{qi}"] = q.pop("pattern")
            np.savez(out / f"{key}.npz", **arrays)
            res.update({"layer": layer, "site": site, "pos": pname, "key": key})
            (out / f"{key}.json").write_text(json.dumps(res))
            print(
                f"{key}: n {res['n_readers']} r {res['r']} quantities {len(res['quantities'])}",
                flush=True,
            )


def labels_pooled() -> dict[str, np.ndarray]:
    ix = np.load(DATASET / "index.npz")
    a, b, op = ix["a"].astype(int), ix["b"].astype(int), ix["op"].astype(int)
    return {
        "op": op,
        "a": a,
        "b": b,
        "a+b": a + b,
        "a-b": a - b,
        "result": np.where(op == 0, a + b, a - b),
    }


def profile(z: np.ndarray, lab: dict[str, np.ndarray]) -> dict[str, Any]:
    """Share of z's variance explained by each label, and the dominant harmonic of each line."""
    tot = float(z.var(0).sum())
    out: dict[str, Any] = {}
    for nm, v in lab.items():
        inv = np.unique(v, return_inverse=True)[1]
        cnt = np.bincount(inv)
        m = np.stack([np.bincount(inv, weights=z[:, j]) / cnt for j in range(z.shape[1])], 1)
        eta = float((m[inv] - z.mean(0)).var(0).sum() / max(tot, 1e-12))
        entry: dict[str, Any] = {"eta2": round(eta, 3)}
        if nm != "op":
            res = np.mod(v, 100)
            rc = np.bincount(res, minlength=100)
            mm = np.stack(
                [np.bincount(res, weights=z[:, j], minlength=100) / rc for j in range(z.shape[1])],
                1,
            )
            power = (np.abs(np.fft.rfft(mm - mm.mean(0), axis=0)) ** 2).sum(1)[1:]
            power = power / max(power.sum(), 1e-12)
            k = int(np.argmax(power)) + 1
            entry.update({"k": k, "share": round(float(power[k - 1]), 3)})
        out[nm] = entry
    return out


def name_of(prof: dict[str, Any], dims: int, J: float) -> str:
    best = max(prof, key=lambda k: prof[k]["eta2"])
    shape = "binary" if dims == 1 and J < 0.25 else "circle" if dims == 2 else f"{dims}-dim"
    if best == "op":
        return f"op, {shape}"
    k, share = prof[best]["k"], prof[best]["share"]
    period = 100 // int(np.gcd(k, 100))
    tag = f"{best} mod {period}" if share >= 0.5 else f"{best} (mixed periods)"
    return f"{tag}, {shape}"


def link(model: str) -> None:
    """Stage C (codes across read points) and stage D (labels)."""
    pts = ATLAS / model / "points"
    lab = labels_pooled()
    items: list[dict[str, Any]] = []
    Zs: list[np.ndarray] = []
    points = []
    for jp in sorted(pts.glob("*.json"), key=lambda p: (int(p.stem.split("_")[0][1:]), p.stem)):
        d = json.loads(jp.read_text())
        arr = np.load(jp.with_suffix(".npz"))
        points.append({k: d[k] for k in ("key", "layer", "site", "pos", "r", "n_readers")} |
                      {"explained": d.get("explained", 0.0), "unexplained": d.get("unexplained", 0),
                       "r2": d.get("r2", []), "n_q": len(d["quantities"])})  # fmt: skip
        for qi, q in enumerate(d["quantities"]):
            z = arr[f"z{qi}"].astype(np.float64)
            prof = profile(z, lab)
            items.append({"point": d["key"], "qi": qi, "layer": d["layer"], "site": d["site"], "pos": d["pos"],
                          "k": q["k"], "J": q["J"], "readers": q["readers"][:6], "n": len(q["readers"]),
                          "name": name_of(prof, q["k"], q["J"]), "profile": prof})  # fmt: skip
            Zs.append(z / np.maximum(z.std(0), 1e-12))
    ks = np.array([z.shape[1] for z in Zs])
    offs = np.concatenate([[0], np.cumsum(ks)])
    Zall = np.concatenate(Zs, 1).astype(np.float32)
    Zall -= Zall.mean(0)
    C = (Zall.T @ Zall) / len(Zall)
    G = len(Zs)
    # same code: equal dimension and all canonical correlations >= LINK (one quantity is a linear
    # image of the other, they carry the same information). A k-dim quantity inside a larger one
    # carries less information and is only recorded as contained: grouping by containment, or by
    # all-but-one shared dimensions, chains codes through axes many frames share (the op flag). A block can only qualify if its
    # squared Frobenius norm is >= (max(k, k') - 1) * LINK^2, which is tested first. Looser
    # containment (a 1-dim magnitude inside a 4-dim frame) is only recorded: grouping by it chains
    # everything that contains, say, a's magnitude.
    F = np.add.reduceat(np.add.reduceat(C**2, offs[:-1], axis=0), offs[:-1], axis=1)
    kmin = np.minimum(ks[:, None], ks[None, :])
    edge = np.zeros((G, G), bool)
    contains: list[list[int]] = [[] for _ in range(G)]
    for i, j in zip(*np.nonzero(np.triu(kmin * LINK**2 - 1e-9 <= F, 1)), strict=True):
        s = np.linalg.svd(C[offs[i] : offs[i + 1], offs[j] : offs[j + 1]], compute_uv=False)
        shared = int((s >= LINK).sum())
        if shared == ks[i] == ks[j]:
            edge[i, j] = edge[j, i] = True
        if shared == min(ks[i], ks[j]) and ks[i] != ks[j]:
            (contains[j] if ks[i] < ks[j] else contains[i]).append(int(i if ks[i] < ks[j] else j))
    _, code = connected_components(edge, directed=False)
    for gi, (it, c) in enumerate(zip(items, code, strict=True)):
        it["code"] = int(c)
        it["contains"] = [f"{items[j]['point']}#{items[j]['qi']}" for j in contains[gi]]
    codes = []
    for c in range(int(code.max()) + 1):
        members = [it for it in items if it["code"] == c]
        best = min(members, key=lambda it: it["J"])
        codes.append({"code": c, "name": best["name"], "size": len(members),
                      "members": [f"{m['point']}#{m['qi']}" for m in members]})  # fmt: skip
    out = {
        "model": model,
        "points": points,
        "quantities": items,
        "codes": codes,
        "link": LINK,
        "tol": TOL,
    }
    (ATLAS / model / "atlas.json").write_text(json.dumps(out))
    print(f"{model}: {len(points)} points, {G} quantities, {len(codes)} codes", flush=True)


if __name__ == "__main__":
    cmd, model = sys.argv[1], sys.argv[2]
    if cmd == "prep":
        prep(model)
    elif cmd == "layer":
        run_layer(model, int(sys.argv[3]))
    elif cmd == "link":
        link(model)
    else:
        raise SystemExit(f"unknown command {cmd}")
