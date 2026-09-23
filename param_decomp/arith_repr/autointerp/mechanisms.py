"""Cluster residual writers into mechanisms, each acting on the arrangement of one variable.

    python -m param_decomp.arith_repr.autointerp.mechanisms --run <run_dir>

Inputs: `mech/prof_<p>_<o>.npz`, `mech/u_gram.npy` (mech_prep.py), `wiring.npz`, `uv_alive.npz`
and the filter dataset (per-prompt gates / inner activations, raw residual).

Variable of a writer (o/down component) at a position and op: the categorical variable of
`mech_prep.variables` with the fewest classes whose adjusted R^2 on the gated write
w = inner * [CI > 0.01] is within 90 % of the best one, the best being >= 0.5. Its contribution
to that variable's arrangement is C_c = m_c (x) U_c (m_c: class means of w, centred).

1. Code: all writers of the variable in one block (a layer's MLP, or one attention head) — one
   step of the computation, typically a lookup table whose members handle different classes —
   then blocks of the same or adjacent layers merged when their supports tile the classes
   (overlap <= DISJOINT of the smaller support union).
2. Mechanism: codes of the same (position, op, variable) whose arrangements have the same shape,
   CKA of their class-mean Grams >= MERGE_CKA (average linkage) — the same representation
   written again at several layers. The CKA of every code pair and of the same pair with the
   classes of one code permuted is saved (`code_pairs.parquet`) to show the threshold sits far
   in the tail of the null.
3. Checks, per mechanism (and the class-level ones per code): `coverage` (classes in some
   member's support), `groups` (classes the joint write tells apart) vs the best member and the
   best code, `loo_needed` (members whose removal merges classes), `overlap` of the supports vs
   random writer sets of the same variable (`tiling`), per-prompt `purity` and balanced
   nearest-centroid `acc` of the joint write vs the best member / code, `kappa` = joint
   between-class energy / sum over members (1 = orthogonal writes, > 1 constructive, < 1
   cancelling), `consumers` (readers drawing >= CONSUMER of their class-mean profile of the
   variable from the joint write; `joint_consumers`: >= 2 members give >= CO_READ each).
4. Arrangement transformation: the variable's arrangement in the raw residual before the first
   member (X_in) vs the joint write: `cka_in` (same shape?), spectra in / out and `new_freqs`
   (cyclic variables), `in_span` (share of the write inside span(X_in): reshaping what is
   there vs adding directions), `cos_in` (> 0 amplify, < 0 erase); for mechanisms with a head,
   `src_pos` / `src_cka`: the earlier position whose arrangement the write copies best.

Writes `mech/mechanisms.parquet`, `mech/codes.parquet`, `mech/members.parquet`,
`mech/code_pairs.parquet`."""

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THR = 0.01
LABEL_MIN = 0.5
READ_MIN = 0.3
CO_READ = 0.1
CONSUMER = 0.2
SAME = 0.8  # |corr| of write profiles: duplicates inside a code
MERGE_CKA = 0.7  # CKA of two codes' arrangements: the same content, re-written
DISJOINT = 0.2  # max support overlap (share of the smaller support) inside a code
FILTER = "analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling/dataset"
AI = "analysis/arith_repr/autointerp"
RNG = np.random.default_rng(0)


@dataclass
class PosOp:
    """Everything about one (position, op) the clustering needs."""

    p: int
    o: int
    prof: dict[str, np.ndarray]
    cols: np.ndarray
    names: list[str]
    label: dict[int, str]  # prof row of a writer -> its variable
    w: np.ndarray = field(repr=False)  # (n, A) gated write per prompt, this op
    classes: dict[str, np.ndarray] = field(repr=False)  # variable -> class index per prompt


def coarsest(prof: dict[str, np.ndarray], names: list[str], key: str, j: int) -> str | None:
    r2 = prof[key][:, j]
    best = float(r2.max())
    if best < LABEL_MIN:
        return None
    ncls = [prof[f"n.{n}"].size for n in names]
    for k in np.argsort(ncls, kind="stable"):
        if r2[k] >= 0.9 * best:
            return names[k]
    return None


def cyclic_tau(var: str) -> int | None:
    if "%" in var and var.split("%")[0] in ("a", "b", "res"):
        return int(var.split("%")[1])
    return None


def support(m: np.ndarray) -> np.ndarray:
    """Classes where the write departs from its typical (median) class by >= half its max."""
    d = np.abs(m - np.median(m))
    return d >= 0.5 * d.max() if d.max() > 0 else np.zeros(m.size, bool)


def spectrum(K: np.ndarray) -> np.ndarray:
    """Power of a class-mean Gram (tau x tau, classes 0..tau-1) per folded frequency 0..tau//2."""
    tau = K.shape[0]
    F = np.exp(-2j * np.pi * np.outer(np.arange(tau), np.arange(tau)) / tau)
    pw = np.real(np.einsum("kv,vw,kw->k", F, K, F.conj())) / tau
    fold = np.zeros(tau // 2 + 1)
    for k in range(tau):
        fold[min(k, tau - k)] += max(pw[k], 0.0)
    fold[0] = 0.0
    return fold / max(fold.sum(), 1e-30)


def geometry(K: np.ndarray, tau: int | None) -> dict[str, float | str]:
    """Shape of an arrangement from its (unweighted, centred) class-mean Gram."""
    ev = np.clip(np.linalg.eigvalsh(K), 0, None)
    pr = float(ev.sum() ** 2 / max((ev**2).sum(), 1e-30))
    out: dict[str, float | str] = {"pr": pr}
    if tau is None:
        return out
    diag = np.array([np.mean([K[v, (v + d) % tau] for v in range(tau)]) for d in range(tau)])
    circ = np.array([[diag[(w - v) % tau] for w in range(tau)] for v in range(tau)])
    out["circulant"] = float((circ**2).sum() / max((K**2).sum(), 1e-30))
    sp = spectrum(K)
    out["spectrum"] = " ".join(
        f"{100 * k // tau}:{sp[k]:.2f}" for k in np.argsort(-sp)[:3] if sp[k] >= 0.05
    )
    ent = -np.sum(sp[sp > 0] * np.log(sp[sp > 0])) / max(np.log(tau // 2), 1e-9)
    out["flatness"] = float(ent)
    if tau >= 5:
        dist = np.diag(K)[:, None] + np.diag(K)[None] - 2 * K
        cyc = np.minimum(
            np.abs(np.subtract.outer(np.arange(tau), np.arange(tau))),
            tau - np.abs(np.subtract.outer(np.arange(tau), np.arange(tau))),
        )
        iu = np.triu_indices(tau, 1)
        out["ordered"] = float(np.corrcoef(dist[iu], cyc[iu])[0, 1]) if dist[iu].std() > 0 else 0.0
    out["shape"] = shape_name(out, sp, tau)
    return out


def shape_name(g: dict[str, float | str], sp: np.ndarray, tau: int) -> str:
    pr, circ, flat = float(g["pr"]), float(g["circulant"]), float(g["flatness"])
    if pr < 1.5:
        return "line"
    top = int(np.argmax(sp))
    if sp[top] >= 0.6 and circ >= 0.6:
        return f"circle period {tau // np.gcd(top, tau)}"
    if circ >= 0.7 and flat >= 0.8 and pr >= 0.5 * (tau - 1):
        return "simplex (one-hot like)"
    if circ >= 0.7:
        return "symmetric mix of circles"
    return "irregular"


class Data:
    def __init__(self, run: Path) -> None:
        ds = run / FILTER
        ix = np.load(ds / "index.npz")
        self.a, self.b, self.op = ix["a"], ix["b"], ix["op"]
        self.kind, self.layer, self.cidx = ix["comp_kind"], ix["comp_layer"], ix["comp_index"]
        self.site = ix["comp_site"]
        wz = np.load(run / AI / "wiring.npz")
        self.writers, self.readers = wz["writers"], wz["readers"]
        self.vw, self.rms, self.head = wz["resid_vw"], wz["rms"], wz["head"]
        self.wrow = {int(c): j for j, c in enumerate(self.writers)}
        self.rrow = {int(c): j for j, c in enumerate(self.readers)}
        self.gram = np.load(run / AI / "mech" / "u_gram.npy")
        uv = np.load(run / AI / "uv_alive.npz")
        self.U = np.zeros((self.writers.size, 4096), np.float32)
        for s in np.unique(self.site[self.writers]):
            ids = {int(i): j for j, i in enumerate(uv[s + ".ids"])}
            for n, c in enumerate(self.writers):
                if self.site[c] == s:
                    self.U[n] = uv[s + ".U"][ids[int(self.cidx[c])]]
        self.ci = np.load(ds / "original" / "ci.npy", mmap_mode="r")
        self.inner = np.load(ds / "original" / "inner.npy", mmap_mode="r")
        self.resid = np.load(ds / "original" / "resid.npy", mmap_mode="r")
        self.run = run

    def name(self, c: int) -> str:
        k = str(self.kind[c]).split(".")[-1].replace("_proj", "")
        return f"L{self.layer[c]} {k} c{self.cidx[c]}"

    def block(self, c: int) -> str:
        if str(self.kind[c]) == "self_attn.o_proj":
            return f"H{int(np.argmax(self.head[c]))}"
        return "MLP"

    def read_point(self, c: int) -> int:
        """Index into `rms` (attn_in.l = 2l, mlp_in.l = 2l + 1) of a reader."""
        return 2 * int(self.layer[c]) + (0 if str(self.kind[c]).startswith("self_attn") else 1)

    def posop(self, p: int, o: int) -> PosOp:
        from param_decomp.arith_repr.autointerp.mech_prep import variables

        prof = dict(np.load(self.run / AI / "mech" / f"prof_{p}_{o}.npz"))
        cols, names = prof["cols"], [str(n) for n in prof["var_names"]]
        label = {}
        for j, c in enumerate(cols):
            if int(c) in self.wrow:
                v = coarsest(prof, names, "r2_w", j)
                if v is not None:
                    label[j] = v
        m = self.op == o
        on = np.asarray(self.ci[:, p, :], np.float32)[m] > THR
        w = np.asarray(self.inner[:, p, :], np.float32)[m] * on
        var = variables(self.a[m], self.b[m], o, p)
        classes = {k: np.unique(v, return_inverse=True)[1] for k, v in var.items()}
        return PosOp(p, o, prof, cols, names, label, w, classes)

    @lru_cache(maxsize=8)  # noqa: B019
    def resid_at(self, point: int, p: int, o: int) -> np.ndarray:
        x = np.asarray(self.resid[point, :, p, :], np.float32)[self.op == o]
        return x - x.mean(0)


def arrangement(d: Data, po: PosOp, var: str, point: int, p: int | None = None) -> np.ndarray:
    """Class means (n_classes, d) of the raw residual at `point`, position p (default po.p)."""
    x = d.resid_at(point, po.p if p is None else p, po.o)
    g = po.classes[var]
    n = np.bincount(g)
    oh = np.zeros((n.size, g.size), np.float32)
    oh[g, np.arange(g.size)] = 1.0 / n[g]
    return oh @ x


def reader_shares(d: Data, po: PosOp, var: str, js: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(len(js), n_readers) share of each reader's class-mean profile of `var` from each writer."""
    prof, kv = po.prof, po.names.index(var)
    rj = [j for j, c in enumerate(po.cols) if int(c) in d.rrow and prof["r2_i"][kv, j] >= READ_MIN]
    if not rj:
        return np.zeros((len(js), 0)), np.array([], int)
    wts = prof[f"n.{var}"] / prof[f"n.{var}"].sum()
    Mw, Ir = prof[f"w.{var}"][js], prof[f"i.{var}"][rj]
    pts = np.array([d.read_point(int(po.cols[j])) for j in rj])
    v_rw = d.vw[
        np.ix_([d.wrow[int(po.cols[j])] for j in js], [d.rrow[int(po.cols[j])] for j in rj])
    ]
    v_rw = v_rw / d.rms[pts, po.p][None]
    den = np.maximum(((Ir**2) * wts).sum(1), 1e-30)
    return ((Mw * wts) @ Ir.T) * v_rw / den[None], np.array(rj)


def joint_metrics(d: Data, po: PosOp, var: str, js: list[int]) -> dict[str, float]:
    """Per-prompt checks of the joint write of prof rows `js` on `var`."""
    cols = po.cols[js]
    wr = [d.wrow[int(c)] for c in cols]
    G = d.gram[np.ix_(wr, wr)]
    w_js = po.w[:, cols]
    w_js = w_js - w_js.mean(0)
    g = po.classes[var]
    out = {"purity": purity(w_js, g, G), "acc": nc_acc(w_js, g, G)}
    out["acc_member"] = max(nc_acc(w_js[:, [i]], g, G[np.ix_([i], [i])]) for i in range(len(js)))
    n = np.bincount(g)
    out["chance"] = 1.0 / n.size
    Mw = po.prof[f"w.{var}"][js]
    wts = n / n.sum()
    between = [float((Mw[i] ** 2 * wts).sum() * G[i, i]) for i in range(len(js))]
    K = (Mw * wts) @ Mw.T  # (|S|, |S|) weighted profile inner products
    out["kappa"] = float((K * G).sum() / max(sum(between), 1e-30))
    return out


def purity(W: np.ndarray, g: np.ndarray, G: np.ndarray) -> float:
    n = np.bincount(g)
    oh = np.zeros((n.size, g.size), np.float32)
    oh[g, np.arange(g.size)] = 1.0 / n[g]
    mu = oh @ W
    total = float(np.einsum("ni,ij,nj->", W, G, W) / W.shape[0])
    between = float(np.einsum("v,vi,ij,vj->", n / n.sum(), mu, G, mu))
    return between / max(total, 1e-30)


def nc_acc(W: np.ndarray, g: np.ndarray, G: np.ndarray, n_classes: int | None = None) -> float:
    """Balanced nearest-centroid accuracy of the class from the write W @ U (metric G)."""
    nc = int(g.max()) + 1 if n_classes is None else n_classes
    n = np.bincount(g, minlength=nc)
    oh = np.zeros((nc, g.size), np.float32)
    oh[g, np.arange(g.size)] = 1.0 / np.maximum(n[g], 1)
    mu = oh @ W
    WG = W @ G
    score = -2 * WG @ mu.T + np.einsum("vi,ij,vj->v", mu, G, mu)[None]
    pred = np.argmin(score, 1)
    ok = np.bincount(g, weights=(pred == g).astype(float), minlength=nc)
    present = n > 0
    return float(np.mean(ok[present] / n[present]))


def transformation(d: Data, po: PosOp, var: str, js: list[int]) -> dict[str, float | str]:
    """How the joint write reshapes the variable's arrangement already in the residual."""
    cols = po.cols[js]
    first = min(js, key=lambda j: (d.layer[po.cols[j]], d.block(int(po.cols[j])) == "MLP"))
    c0 = int(po.cols[first])
    pt_in = 2 * int(d.layer[c0]) + (1 if d.block(c0) == "MLP" else 0)
    X = arrangement(d, po, var, pt_in)
    n = np.bincount(po.classes[var])
    wts = n / n.sum()
    wr = [d.wrow[int(c)] for c in cols]
    Us, G = d.U[wr], d.gram[np.ix_(wr, wr)]
    Mw = po.prof[f"w.{var}"][js].T  # (nc, |S|)
    Kc = Mw @ G @ Mw.T
    Kx = X @ X.T
    out: dict[str, float | str] = {"point_in": pt_in}
    Wd = np.diag(wts)
    out["cka_in"] = cka(Kx, Kc, Wd)
    P = X @ Us.T  # (nc, |S|)
    cross = float((wts * (P * Mw).sum(1)).sum())
    nx = float((wts * (X**2).sum(1)).sum())
    nc_ = float((wts * np.diag(Kc)).sum())
    out["cos_in"] = cross / max(np.sqrt(nx * nc_), 1e-30)
    out["write_vs_code"] = nc_ / max(nx, 1e-30)
    q, s, _ = np.linalg.svd(X.T * np.sqrt(wts), full_matrices=False)
    Q = q[:, s > 1e-3 * s.max()]
    inside = (Mw @ (Us @ Q)) ** 2
    out["in_span"] = float((wts * inside.sum(1)).sum() / max(nc_, 1e-30))
    tau = cyclic_tau(var)
    if tau is not None:
        gx, gc = geometry(center(Kx), tau), geometry(center(Kc), tau)
        out |= {f"in_{k}": v for k, v in gx.items()} | {f"out_{k}": v for k, v in gc.items()}
        spx, spc = spectrum(center(Kx)), spectrum(center(Kc))
        new = [k for k in range(1, tau // 2 + 1) if spc[k] >= 0.1 and spx[k] < 0.01]
        out["new_freqs"] = " ".join(str(100 * k // tau) for k in new)
    else:
        out |= {f"out_{k}": v for k, v in geometry(center(Kc), None).items()}
    if any(d.block(int(c)) != "MLP" for c in cols) and po.p > 1:
        best = (0.0, -1)
        for s_ in range(1, po.p):
            Xs = arrangement(d, po, var, pt_in, p=s_)
            best = max(best, (cka(Xs @ Xs.T, Kc, Wd), s_))
        out["src_cka"], out["src_pos"] = best
    return out


def center(K: np.ndarray) -> np.ndarray:
    H = np.eye(K.shape[0]) - 1.0 / K.shape[0]
    return H @ K @ H


def cka(Kx: np.ndarray, Ky: np.ndarray, Wd: np.ndarray) -> float:
    a, b = center(Kx), center(Ky)
    num = np.trace(a @ Wd @ b @ Wd)
    den = np.sqrt(np.trace(a @ Wd @ a @ Wd) * np.trace(b @ Wd @ b @ Wd))
    return float(num / max(den, 1e-30))


def describe_support(po: PosOp, var: str, m: np.ndarray) -> str:
    from param_decomp.arith_repr.autointerp.onsets import ranges

    vals = po.prof[f"vals.{var}"]
    sup = support(m)
    sign = "+" if (m[sup] - np.median(m)).mean() >= 0 else "-"
    if var in ("units(a,b)", "tens(a,b)"):
        base = 10 if var == "units(a,b)" else 11
        xn, yn = ("a%10", "b%10") if base == 10 else ("a//10", "b//10")
        rows: dict[str, list[int]] = {}
        for x in sorted({int(v) // base for v in vals[sup]}):
            ys = [int(v) % base for v in vals[sup] if int(v) // base == x]
            rows.setdefault(ranges(ys), []).append(x)
        cells = "; ".join(f"{xn} {{{ranges(xs)}}} x {yn} {{{ys}}}" for ys, xs in rows.items())
        return f"{cells} ({sign})"
    return f"{var} in {{{ranges([int(v) for v in vals[sup]])}}} ({sign})"


def profile_corr(M: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """|weighted correlation| between the class-mean write profiles (rows of M)."""
    Mc = M - (M * wts).sum(1, keepdims=True)
    Mn = Mc * np.sqrt(wts)
    Mn /= np.maximum(np.linalg.norm(Mn, axis=1, keepdims=True), 1e-30)
    return np.abs(Mn @ Mn.T)


def average_linkage(S: np.ndarray, thr: float) -> list[list[int]]:
    """Agglomerate items while the mean pairwise similarity of the best pair of groups >= thr."""
    groups = [[i] for i in range(S.shape[0])]
    L = S.astype(np.float64).copy()
    np.fill_diagonal(L, -np.inf)
    size = np.ones(S.shape[0])
    alive = np.ones(S.shape[0], bool)
    while alive.sum() > 1:
        i, j = np.unravel_index(np.argmax(L), L.shape)
        if L[i, j] < thr:
            break
        groups[i] += groups[j]
        L[i] = (L[i] * size[i] + L[j] * size[j]) / (size[i] + size[j])
        L[:, i] = L[i]
        L[i, i] = -np.inf
        L[j], L[:, j] = -np.inf, -np.inf
        size[i] += size[j]
        alive[j] = False
    return [groups[i] for i in np.flatnonzero(alive)]


def write_gram(d: Data, po: PosOp, var: str, js: list[int]) -> np.ndarray:
    """Class-mean Gram (n_classes, n_classes) of the joint write of prof rows `js`."""
    wr = [d.wrow[int(po.cols[j])] for j in js]
    M = po.prof[f"w.{var}"][js]
    return M.T @ d.gram[np.ix_(wr, wr)] @ M


def n_groups(K: np.ndarray) -> int:
    """Classes the write tells apart: groups of class means closer than 5 % of the mean distance."""
    D = np.diag(K)[:, None] + np.diag(K)[None] - 2 * K
    off = D[~np.eye(D.shape[0], dtype=bool)]
    if off.size == 0 or off.mean() <= 0:
        return 1
    close = 0.05 * off.mean() > D
    parent = list(range(K.shape[0]))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(close), strict=True):
        parent[find(int(i))] = find(int(j))
    return len({find(i) for i in range(K.shape[0])})


def local_codes(d: Data, po: PosOp, var: str, js: list[int]) -> list[list[int]]:
    """Codes: all writers of `var` in one block (a layer's MLP or one attention head), then
    blocks of the same or adjacent layers merged when their supports tile the classes (the two
    blocks' support unions overlap by <= DISJOINT of the smaller one)."""
    M = po.prof[f"w.{var}"]
    blocks: dict[tuple[int, str], list[int]] = {}
    for j in js:
        c = int(po.cols[j])
        blocks.setdefault((int(d.layer[c]), d.block(c)), []).append(j)
    keys = sorted(blocks)
    sup = {k: np.any([support(M[j]) for j in blocks[k]], 0) for k in keys}
    codes: list[list[tuple[int, str]]] = []
    for k in keys:
        for code in codes:
            lays = [x[0] for x in code] + [k[0]]
            if max(lays) - min(lays) > 1:
                continue
            if all(
                (sup[k] & sup[x]).sum() <= DISJOINT * min(sup[k].sum(), sup[x].sum()) for x in code
            ):
                code.append(k)
                break
        else:
            codes.append([k])
    return [[j for k in code for j in blocks[k]] for code in codes]


def tiling(po: PosOp, var: str, js: list[int], n_draw: int = 200) -> dict[str, float]:
    """How much the members' supports overlap: `overlap` = sum of support sizes / size of their
    union (1 = a perfect tiling, each class handled by one member), and `overlap_p` = the share
    of random sets of as many writers of the same variable at this position (any layer) that
    overlap as little or less (small = the tiling is not what random picks give)."""
    if len(js) < 2:
        return {}
    M = po.prof[f"w.{var}"]
    pool = [j for j, v in po.label.items() if v == var]
    sups = {j: support(M[j]) for j in pool}

    def ratio(sel: list[int]) -> float:
        s = np.array([sups[j] for j in sel])
        return float(s.sum() / max(np.any(s, 0).sum(), 1))

    obs = ratio(js)
    if len(pool) <= len(js):
        return {"overlap": obs}
    null = [ratio(list(RNG.choice(pool, len(js), replace=False))) for _ in range(n_draw)]
    return {
        "overlap": obs,
        "overlap_p": float(np.mean(np.array(null) <= obs)),
        "overlap_null": float(np.median(null)),
    }


def set_metrics(d: Data, po: PosOp, var: str, parts: list[list[int]], full: bool) -> dict[str, Any]:
    """Checks of a set of writers (the union of `parts`) on `var`."""
    members = [j for part in parts for j in part]
    wts = po.prof[f"n.{var}"] / po.prof[f"n.{var}"].sum()
    M = po.prof[f"w.{var}"]
    K = write_gram(d, po, var, members)
    row: dict[str, Any] = tiling(po, var, members) | {
        "n_members": len(members),
        "coverage": float(np.any([support(M[j]) for j in members], 0).mean()),
        "groups": n_groups(K),
        "groups_member": max(n_groups(write_gram(d, po, var, [j])) for j in members),
        "n_classes": int(wts.size),
    }
    if len(parts) > 1:
        row["groups_part"] = max(n_groups(write_gram(d, po, var, part)) for part in parts)
        Wd = np.diag(wts)
        grams = [write_gram(d, po, var, part) for part in parts]
        ck = [
            cka(grams[a], grams[b], Wd) for a in range(len(parts)) for b in range(a + 1, len(parts))
        ]
        row["part_cka"] = float(np.mean(ck))
    if len(members) > 1 and len(members) <= 80:
        base = row["groups"]
        need = [
            n_groups(write_gram(d, po, var, [k for k in members if k != j])) < base for j in members
        ]
        row["loo_needed"] = float(np.mean(need))
    if not full:
        return row
    jm = joint_metrics(d, po, var, members)
    row |= jm
    if len(parts) > 1:
        row["acc_part"] = max(joint_metrics(d, po, var, part)["acc"] for part in parts)
    S, _ = reader_shares(d, po, var, members)
    tot = np.clip(S, 0, None).sum(0)
    row["consumers"] = int((S.sum(0) >= CONSUMER).sum())
    big = (np.clip(S, 0, None) >= CO_READ * np.maximum(tot, 1e-30)).sum(0)
    row["joint_consumers"] = int(((S.sum(0) >= CONSUMER) & (big >= 2)).sum())
    row |= transformation(d, po, var, members)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--positions", default="1,2,3,4")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    d = Data(args.run)
    out = args.run / AI / "mech"
    rows_mech, rows_code, rows_member, rows_pair = [], [], [], []
    for p in map(int, args.positions.split(",")):
        for o in (0, 1):
            po = d.posop(p, o)
            by_var: dict[str, list[int]] = {}
            for j, v in po.label.items():
                by_var.setdefault(v, []).append(j)
            n_code = n_mech = 0
            for var, js in sorted(by_var.items()):
                js = sorted(js, key=lambda j: (d.layer[po.cols[j]], int(po.cols[j])))
                codes = local_codes(d, po, var, js)
                wts = po.prof[f"n.{var}"] / po.prof[f"n.{var}"].sum()
                grams = [write_gram(d, po, var, c) for c in codes]
                Wd = np.diag(wts)
                ck = np.array([[cka(ga, gb, Wd) for gb in grams] for ga in grams])
                for a_ in range(len(codes)):
                    for b_ in range(a_ + 1, len(codes)):
                        perm = RNG.permutation(wts.size)
                        la = [int(d.layer[po.cols[j]]) for j in codes[a_]]
                        lb = [int(d.layer[po.cols[j]]) for j in codes[b_]]
                        rows_pair.append(
                            {
                                "p": p,
                                "o": o,
                                "var": var,
                                "cka": float(ck[a_, b_]),
                                "cka_null": cka(grams[a_], grams[b_][np.ix_(perm, perm)], Wd),
                                "gap": abs(min(la) - min(lb)),
                                "nc": int(wts.size),
                                "n_a": len(codes[a_]),
                                "n_b": len(codes[b_]),
                            }
                        )
                for mech in average_linkage(ck, MERGE_CKA):
                    mid = f"{p}{'as'[o]}-{len(rows_mech)}"
                    mcodes = sorted((codes[i] for i in mech), key=lambda c: d.layer[po.cols[c[0]]])
                    row = {"mech": mid, "p": p, "o": o, "var": var, "n_codes": len(mcodes)}
                    lay = [int(d.layer[po.cols[j]]) for c in mcodes for j in c]
                    row |= {
                        "layer_min": min(lay),
                        "layer_max": max(lay),
                        "layers": " ".join(f"L{x}" for x in sorted(set(lay))),
                    }
                    row |= set_metrics(d, po, var, mcodes, full=True)
                    rows_mech.append(row)
                    for ci, code in enumerate(mcodes):
                        cid = f"{mid}.{ci}"
                        rows_code.append(
                            {
                                "mech": mid,
                                "code": cid,
                                "p": p,
                                "o": o,
                                "var": var,
                                "layers": " ".join(
                                    sorted({f"L{d.layer[po.cols[j]]}" for j in code})
                                ),
                            }
                            | set_metrics(d, po, var, [[j] for j in code], full=False)
                        )
                        for j in code:
                            c = int(po.cols[j])
                            rows_member.append(
                                {
                                    "mech": mid,
                                    "code": cid,
                                    "p": p,
                                    "o": o,
                                    "var": var,
                                    "col": c,
                                    "name": d.name(c),
                                    "layer": int(d.layer[c]),
                                    "block": d.block(c),
                                    "support": describe_support(po, var, po.prof[f"w.{var}"][j]),
                                    "r2": float(po.prof["r2_w"][po.names.index(var), j]),
                                    "on_rate": float(po.prof["on_rate"][j]),
                                }
                            )
                    n_mech += 1
                n_code += len(codes)
            print(p, o, "writers", len(po.label), "codes", n_code, "mechanisms", n_mech, flush=True)
    pd.DataFrame(rows_mech).to_parquet(out / f"mechanisms{args.tag}.parquet")
    pd.DataFrame(rows_code).to_parquet(out / f"codes{args.tag}.parquet")
    pd.DataFrame(rows_member).to_parquet(out / f"members{args.tag}.parquet")
    pd.DataFrame(rows_pair).to_parquet(out / f"code_pairs{args.tag}.parquet")
    print("saved")


if __name__ == "__main__":
    main()
