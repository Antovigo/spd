"""Which components write each atlas quantity, in the components-only model.

    python -m param_decomp.arith_repr.isa.writers masks            # split original/ci.npy by layer
    python -m param_decomp.arith_repr.isa.writers run <position>   # a | op | b | =
        # -> ATLAS/decomposed/writers_<position>.json
    python -m param_decomp.arith_repr.isa.writers provenance       # -> ATLAS/decomposed/provenance.json

In the components-only model the stream at a position is exactly the token embedding plus the masked
writes of the o and down components before the read point: `x = e + sum_c h_c m_c U_c`, with `h_c` the
inner activation, `m_c = 1[CI_c > 0.01 on the original model]` the mask and `U_c` the write vector
(checked: the sum of masked writes reproduces each sublayer's update to within 1 %, the fp16 storage).

Each atlas quantity is a fixed linear read of the raw stream, `z = (x - mean x) . F`: the readers are
`H = x . (g * V)` once multiplied by the RMS, and the quantity's coordinates are a linear combination
of the kept readers, `z = H W` (W by least squares, exact). So `F = (g * V) W`, and writer c adds
`(h_c m_c - mean) (U_c . F)` to z. Its share of the quantity's variance is
`sum_j Cov(h_c m_c, z_j) (U_c . F_j) / sum_j Var(z_j)`; the embedding's share is computed the same
way, and the shares of the embedding and of all earlier writers add up to 1."""

import json
import sys
from collections import defaultdict
from typing import Any

import numpy as np

from param_decomp.arith_repr.isa.atlas import ATLAS, POSITIONS, SITES, read_point_index
from param_decomp.arith_repr.vectors.common import DATASET, RESID, comp_table, load_uv

MODEL = "decomposed"
MASK_CI = 0.01
TOP = 25


def prep_masks() -> None:
    comps = comp_table()
    ci = np.load(DATASET / "original/ci.npy", mmap_mode="r")
    out = ATLAS / "masks"
    out.mkdir(parents=True, exist_ok=True)
    layers = [np.flatnonzero(comps["layer"] == li) for li in range(32)]
    arrays = [np.lib.format.open_memmap(out / f"L{li}.npy", "w+", np.uint8, (ci.shape[0], 5, c.size))
              for li, c in enumerate(layers)]  # fmt: skip
    for s in range(0, ci.shape[0], 1000):
        block = np.asarray(ci[s : s + 1000]) > MASK_CI
        for li, c in enumerate(layers):
            arrays[li][s : s + 1000] = block[:, :, c]
        print("masks", s, flush=True)
    for a in arrays:
        a.flush()


def filters(layer: int, site: str, pos: int, Z: np.ndarray, comps: dict[str, np.ndarray],
            gains: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:  # fmt: skip
    """F (4096, k) with z = (x - mean) . F for the read point's quantities Z (N, k), and the share of
    z's variance that F reproduces (1 up to the kept-reader filter)."""
    cols = np.flatnonzero(comps["layer"] == layer)
    sel = np.flatnonzero(np.isin(comps["kind"][cols], SITES[site]))
    inner = np.load(ATLAS / MODEL / "inner" / f"L{layer}.npy", mmap_mode="r")
    x = np.asarray(
        np.load(DATASET / MODEL / "resid.npy", mmap_mode="r")[
            read_point_index(layer, site), :, pos
        ],
        np.float64,
    )
    rms = np.sqrt((x**2).mean(1) + 1e-5)
    Hc = np.asarray(inner[:, pos][:, sel], np.float64) * rms[:, None]
    Hc -= Hc.mean(0)
    var = Hc.var(0)
    keep = var > 1e-8 * max(float(var.max()), 1e-30)
    W = np.linalg.lstsq(Hc[:, keep], Z, rcond=None)[0]
    V, _ = load_uv(comps, cols[sel[keep]])
    g = gains["ln1" if site == "attn" else "ln2"][layer]
    F = (np.stack(V, 1) * g[:, None]) @ W  # (4096, k)
    xc = x - x.mean(0)
    fit = 1 - float(((xc @ F - Z) ** 2).sum() / max((Z**2).sum(), 1e-12))
    return F, fit


def run(pname: str) -> None:
    pos = {v: k for k, v in POSITIONS.items()}[pname]
    comps = comp_table()
    gains = dict(np.load(RESID / "norms.npz"))
    at = json.loads((ATLAS / MODEL / "atlas.json").read_text())
    qs = [q for q in at["quantities"] if q["pos"] == pname]
    order = {(li, s): 2 * li + (0 if s == "attn" else 1) for li in range(32) for s in SITES}
    qs.sort(key=lambda q: (order[(q["layer"], q["site"])], q["qi"]))
    Zs, Fs, fits = [], [], []
    by_point: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qs):
        by_point[q["point"]].append(i)
    for key, idx in by_point.items():
        arr = np.load(ATLAS / MODEL / "points" / f"{key}.npz")
        q0 = qs[idx[0]]
        Z = np.concatenate([arr[f"z{qs[i]['qi']}"].astype(np.float64) for i in idx], 1)
        F, fit = filters(q0["layer"], q0["site"], pos, Z - Z.mean(0), comps, gains)
        off = 0
        for i in idx:
            k = qs[i]["k"]
            Zs.append(Z[:, off : off + k] - Z[:, off : off + k].mean(0))
            Fs.append(F[:, off : off + k])
            fits.append(fit)
            off += k
        print("filters", key, round(fit, 4), flush=True)
    ks = np.array([z.shape[1] for z in Zs])
    offs = np.concatenate([[0], np.cumsum(ks)])
    Zall, Fall = np.concatenate(Zs, 1), np.concatenate(Fs, 1)
    N = len(Zall)
    vq = np.array([float(z.var(0).sum()) for z in Zs])
    rp = np.array([order[(q["layer"], q["site"])] for q in qs])

    def per_quantity(cols_contrib: np.ndarray) -> np.ndarray:
        """(n_w, sum k) contributions -> (n_w, Q) shares."""
        return np.add.reduceat(cols_contrib, offs[:-1], axis=1) / vq[None]

    x0 = np.asarray(np.load(DATASET / MODEL / "resid.npy", mmap_mode="r")[0, :, pos], np.float64)
    emb = per_quantity((((x0 - x0.mean(0)) @ Fall) * Zall).mean(0)[None])[0]
    top: list[list[tuple[float, str]]] = [[] for _ in qs]
    agg: list[dict[str, float]] = [defaultdict(float) for _ in qs]
    total = emb.copy()
    for li in range(32):
        cols = np.flatnonzero(comps["layer"] == li)
        inner = np.load(ATLAS / MODEL / "inner" / f"L{li}.npy", mmap_mode="r")
        masks = np.load(ATLAS / "masks" / f"L{li}.npy", mmap_mode="r")
        for kind in ("o", "down"):
            w = np.flatnonzero(comps["kind"][cols] == kind)
            if not w.size:
                continue
            hm = np.asarray(inner[:, pos][:, w], np.float64) * np.asarray(
                masks[:, pos][:, w], np.float64
            )
            hm -= hm.mean(0)
            _, U = load_uv(comps, cols[w])
            S = np.stack(U) @ Fall  # (n_w, sum k)
            C = hm.T @ Zall / N
            share = per_quantity(S * C)  # (n_w, Q)
            writes_at = 2 * li + (
                0 if kind == "o" else 1
            )  # o writes before the MLP read point of li
            valid = rp > writes_at
            share[:, ~valid] = 0.0
            total += share.sum(0)
            names = [f"L{li}.{kind}.c{int(comps['cidx'][cols[j]])}" for j in w]
            for qi in np.flatnonzero(valid):
                agg[qi][f"L{li}.{kind}"] += float(share[:, qi].sum())
                for j in np.argsort(-np.abs(share[:, qi]))[:TOP]:
                    top[qi].append((float(share[j, qi]), names[j]))
        print("writers", li, flush=True)
    out = []
    for qi, q in enumerate(qs):
        best = sorted(top[qi], key=lambda t: -abs(t[0]))[:TOP]
        out.append({"point": q["point"], "qi": q["qi"], "code": q["code"], "name": q["name"], "k": q["k"],
                    "J": q["J"], "filter_fit": round(fits[qi], 4), "embedding": round(float(emb[qi]), 4),
                    "total": round(float(total[qi]), 4),
                    "by_sublayer": {k: round(v, 4) for k, v in agg[qi].items() if abs(v) >= 0.005},
                    "top": [[n, round(s, 4)] for s, n in best]})  # fmt: skip
    res: dict[str, Any] = {"position": pname, "model": MODEL, "quantities": out}
    (ATLAS / MODEL / f"writers_{pname.replace('=', 'eq')}.json").write_text(json.dumps(res))
    print("saved", len(out), flush=True)


def poly_r2(y: np.ndarray, Z: np.ndarray, degree: int, n: int = 6000, seed: int = 0) -> float:
    """Held-out R2 of y regressed on Z's coordinates (degree 1), or on them plus all their squares
    and pairwise products (degree 2). Fitted on half of a random subsample, scored on the other."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    Zs = (Z[idx] - Z[idx].mean(0)) / np.maximum(Z[idx].std(0), 1e-12)
    feats = [np.ones((len(Zs), 1)), Zs]
    if degree == 2:
        iu = np.triu_indices(Zs.shape[1])
        feats.append(Zs[:, iu[0]] * Zs[:, iu[1]])
    X = np.concatenate(feats, 1)
    yy = y[idx]
    half = len(idx) // 2
    coef = np.linalg.lstsq(X[:half], yy[:half], rcond=None)[0]
    res = yy[half:] - X[half:] @ coef
    return float(1 - (res**2).sum() / max(((yy[half:] - yy[half:].mean(0)) ** 2).sum(), 1e-12))


def provenance(min_share: float = 0.15, max_j: float = 0.35) -> None:
    """What the main writers compute from, restricted to what the components can compute.

    For every writer with at least `min_share` of a quantity (J < max_j), the held-out R2 of its
    masked activation given the atlas quantities it could read:
    - an o component at (layer l, position p) outputs a weighted sum of values, linear in the source
      streams: linear R2 given each quantity read at the attention input (l, attn, p') of every
      position p' <= p (what the heads can copy), and the component's dominant head;
    - a down component reads neurons silu(g) * u, with g and u linear in the MLP input: degree-2 R2
      (linear terms, squares, pairwise products) given each quantity read at (l, mlp, p), and given
      the best pair of them, whose cross terms carry products such as an a-code times a b-code.
    A flexible regression would be uninformative: at positions a and op every prompt with the same a
    shares one stream, so any code that identifies a predicts everything."""
    comps = comp_table()
    at = json.loads((ATLAS / MODEL / "atlas.json").read_text())
    qmeta = {f"{q['point']}#{q['qi']}": q for q in at["quantities"]}
    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in at["quantities"]:
        by_point[q["point"]].append(q)
    zcache: dict[str, list[tuple[str, np.ndarray]]] = {}

    def zs(point: str) -> list[tuple[str, np.ndarray]]:
        if point not in zcache:
            f = ATLAS / MODEL / "points" / f"{point}.npz"
            arr = np.load(f) if f.exists() else None
            zcache[point] = [] if arr is None else [
                (f"{point}#{q['qi']}", arr[f"z{q['qi']}"].astype(np.float64)) for q in by_point[point] if q["J"] < 0.5]  # fmt: skip
        return zcache[point]

    wanted: dict[str, dict[str, Any]] = {}
    for pname in POSITIONS.values():
        d = json.loads((ATLAS / MODEL / f"writers_{pname.replace('=', 'eq')}.json").read_text())
        for q in d["quantities"]:
            if q["J"] >= max_j:
                continue
            for name, share in q["top"]:
                if share >= min_share:
                    w = wanted.setdefault(
                        f"{name}@{pname}", {"writer": name, "pos": pname, "writes": []}
                    )
                    w["writes"].append([f"{q['point']}#{q['qi']}", q["name"], share])
    posnum = {v: k for k, v in POSITIONS.items()}
    out = []
    for key, w in wanted.items():
        lname, kind, cname = w["writer"].split(".")
        li, cid = int(lname[1:]), int(cname[1:])
        cols = np.flatnonzero(comps["layer"] == li)
        j = int(np.flatnonzero((comps["kind"][cols] == kind) & (comps["cidx"][cols] == cid))[0])
        p = posnum[w["pos"]]
        inner = np.load(ATLAS / MODEL / "inner" / f"L{li}.npy", mmap_mode="r")
        masks = np.load(ATLAS / "masks" / f"L{li}.npy", mmap_mode="r")
        hm = np.asarray(inner[:, p, j], np.float64) * np.asarray(masks[:, p, j], np.float64)
        entry: dict[str, Any] = {"writer": w["writer"], "pos": w["pos"], "writes": w["writes"],
                                 "active": round(float((np.asarray(masks[:, p, j]) > 0).mean()), 3)}  # fmt: skip
        if kind == "o":
            cands = [
                c
                for pp, pn in POSITIONS.items()
                if pp <= p
                for c in zs(f"L{li}_attn_{pn}".replace("=", "eq"))
            ]
            V, _ = load_uv(comps, cols[[j]])
            per_head = (V[0].reshape(32, 128) ** 2).sum(1)
            entry["head"] = int(np.argmax(per_head))
            entry["head_share"] = round(float(per_head.max() / per_head.sum()), 3)
            degree = 1
        else:
            cands = zs(f"L{li}_mlp_{w['pos']}".replace("=", "eq"))
            degree = 2
        single = sorted(((poly_r2(hm, z, degree), k) for k, z in cands), reverse=True)
        entry["degree"] = degree
        entry["single"] = [[k, qmeta[k]["name"], round(v, 3)] for v, k in single[:5]]
        top = [k for _, k in single[:6]]
        zd = dict(cands)
        pairs = sorted(((poly_r2(hm, np.hstack([zd[x], zd[y]]), degree), x, y)
                        for a_, x in enumerate(top) for y in top[a_ + 1 :]), reverse=True)  # fmt: skip
        entry["pair"] = [
            [x, qmeta[x]["name"], y, qmeta[y]["name"], round(v, 3)] for v, x, y in pairs[:3]
        ]
        out.append(entry)
        print(key, entry.get("head"), entry["single"][:2], entry["pair"][:1], flush=True)
    (ATLAS / MODEL / "provenance.json").write_text(json.dumps(out))
    print("saved", len(out), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "masks":
        prep_masks()
    elif sys.argv[1] == "provenance":
        provenance()
    else:
        run(sys.argv[2])
