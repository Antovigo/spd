"""Independent subspace analysis of the inputs of one MLP, from its alive gate and up readers.

    python -m param_decomp.arith_repr.isa.pipeline <layer> <add|both> [original|decomposed]
        # -> ISA_OUT/isa_L<l>_<ops>[_<model>].json

The readers of one MLP all apply a linear read `V_c` to the same normalised stream, so their inner
activations are one linear image of that stream. Steps 0-3 use only those activations (and the
stream's RMS, to undo the norm's per-prompt scale); no task label is read before the post-hoc part.

0. `H` (prompts x readers) at `=`, multiplied back by the stream RMS.
1. PCA; keep the dimensions carrying `VAR_KEEP` of the variance; whiten. This removes the readers'
   angles and redundancy: any basis of the same subspace gives the same whitened cloud up to rotation.
2. ICA (FastICA, negentropy with log cosh) in the whitened space, then pairwise dependence of the
   ICA outputs by RBF-kernel CKA (normalised HSIC) on a subsample, with a permutation null. Average-
   linkage clustering of `1 - CKA`, cut at `DEP_FACTOR` x the null's 99.9th percentile, gives the
   groups (independent subspace analysis = ICA + grouping of dependent outputs).
3. Each group's prompts, projected on the group's top three principal axes, for the 3D view.

Post-hoc (labels revealed only here): correlation ratio of every group and every ICA output on
arithmetic labels."""

import json
import sys
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import FastICA

from param_decomp.arith_repr.vectors.common import DATASET, RUN, comp_table, read_point

ISA_OUT = RUN / "analysis/arith_repr/isa"
POS = 4
OPS = {"both": slice(0, 20000), "add": slice(0, 10000)}
VAR_KEEP = 0.99
M_HSIC = 2000
N_NULL = 400
DEP_FACTOR = 2.0
N_SHOW = 3000
SEED = 0
LOGCOSH_GAUSS = 0.37457  # E[log cosh(nu)], nu ~ N(0, 1)
POSTHOC_OUT = ("op", "a", "b", "a+b", "a-b", "a mod 50", "b mod 50", "a mod 20", "b mod 20",
               "a mod 10", "b mod 10", "a+b mod 10", "a-b mod 10")  # fmt: skip


def load(
    layer: int, ops: str, model: str = "original"
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Gate and up readers at `=`, times the stream's RMS. `model`: "original" or "decomposed" (the
    components-only model: CI-active components, no weight delta)."""
    comps = comp_table()
    cols = np.flatnonzero((comps["layer"] == layer) & np.isin(comps["kind"], ("gate", "up")))
    inner = np.load(DATASET / model / "inner.npy", mmap_mode="r")
    acts = np.asarray(inner[OPS[ops], POS], np.float32)[:, cols].astype(np.float64)
    resid = np.load(DATASET / model / "resid.npy", mmap_mode="r")
    x = np.asarray(resid[read_point(layer, "gate"), OPS[ops], POS], np.float64)
    rms = np.sqrt((x**2).mean(1) + 1e-5)
    names = [f"{comps['kind'][c]}.c{comps['cidx'][c]}" for c in cols]
    return acts * rms[:, None], rms, names, cols


def centered_grams(S: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """RBF Gram (median heuristic) of every column on `sub`, double-centred, unit Frobenius norm."""
    m = sub.size
    out = np.empty((S.shape[1], m * m), np.float32)
    for i in range(S.shape[1]):
        v = S[sub, i]
        d2 = (v[:, None] - v[None]) ** 2
        gram = np.exp(-d2 / (2 * np.median(d2[d2 > 0])))
        gram -= gram.mean(0)[None]
        gram -= gram.mean(1)[:, None]
        out[i] = (gram / np.linalg.norm(gram)).ravel()
    return out


def cka(F: np.ndarray) -> np.ndarray:
    C = F @ F.T
    np.fill_diagonal(C, 1.0)
    return np.clip(C, 0.0, 1.0)


def cka_null(F: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    null = np.empty(N_NULL)
    for t in range(N_NULL):
        i, j = rng.choice(F.shape[0], 2, replace=False)
        p = rng.permutation(m)
        null[t] = float((F[i] * F[j].reshape(m, m)[p][:, p].ravel()).sum())
    return null


def labels(ops: str) -> dict[str, np.ndarray]:
    ix = np.load(DATASET / "index.npz")
    sl = OPS[ops]
    a, b, op = ix["a"][sl].astype(int), ix["b"][sl].astype(int), ix["op"][sl].astype(int)
    res = np.where(op == 0, a + b, a - b)
    out = {"op": op, "a": a, "b": b, "a+b": a + b, "a-b": a - b, "result": res,
           "op,a": op * 1000 + a, "op,b": op * 1000 + b}  # fmt: skip
    for period in (2, 5, 10, 20, 25, 50):
        for nm, v in (("a", a), ("b", b), ("a+b", a + b), ("a-b", a - b), ("result", res)):
            out[f"{nm} mod {period}"] = np.mod(v, period)
    return out


def eta2(X: np.ndarray, lab: np.ndarray) -> float:
    """Share of the total variance of X (N, d) explained by the class means of `lab`."""
    _, inv = np.unique(lab, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    Xc = X - X.mean(0)
    mu = np.stack([np.bincount(inv, weights=Xc[:, j]) / cnt for j in range(X.shape[1])], 1)
    return float((mu[inv] ** 2).sum() / max((Xc**2).sum(), 1e-12))


def rnd(x: np.ndarray, nd: int = 3) -> list[Any]:
    return np.round(x, nd).tolist()


def display(Sg: np.ndarray, show: np.ndarray) -> dict[str, Any]:
    """One group's prompts on its top three principal axes, for the 3D view."""
    Sgc = Sg - Sg.mean(0)
    _, gs, gv = np.linalg.svd(Sgc, full_matrices=False)
    k3 = min(3, Sg.shape[1])
    P = np.pad(Sgc @ gv[:k3].T, ((0, 0), (0, 3 - k3)))
    return {"pca3": float((gs[:k3] ** 2).sum() / (gs**2).sum()), "pts": rnd(P[show])}


def main() -> None:
    layer, ops = int(sys.argv[1]), sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "original"
    tag = "" if model == "original" else f"_{model}"
    rng = np.random.default_rng(SEED)
    Hr, rms, names, cols = load(layer, ops, model)
    N, n = Hr.shape
    Hc = Hr - Hr.mean(0)
    reader_corr = np.corrcoef(Hc.T)
    reader_order = leaves_list(linkage(Hc.T, "average", metric="correlation"))

    # 1. PCA + whitening
    U, sv, Vt = np.linalg.svd(Hc, full_matrices=False)
    ev = sv**2 / (sv**2).sum()
    r = int(np.searchsorted(np.cumsum(ev), VAR_KEEP) + 1)
    Y = U[:, :r] * np.sqrt(N)
    print(f"L{layer} {ops}: {n} readers, r = {r} dims for {VAR_KEEP:.0%}", flush=True)

    # 2. ICA + dependence grouping
    ica = FastICA(algorithm="parallel", whiten=False, fun="logcosh", max_iter=5000, tol=1e-6,  # pyright: ignore[reportArgumentType]
                  random_state=SEED)  # fmt: skip
    s_raw = np.asarray(ica.fit_transform(Y))
    S = (s_raw - s_raw.mean(0)) / s_raw.std(0)
    # S = Hc @ unmix: whitening, then the ICA rotation, then unit variance
    unmix = (Vt[:r].T * (np.sqrt(N) / sv[:r])) @ np.asarray(ica.components_).T / s_raw.std(0)
    negent = (np.log(np.cosh(S)).mean(0) - LOGCOSH_GAUSS) ** 2
    kurt = (S**4).mean(0) - 3
    sub = rng.choice(N, M_HSIC, replace=False)
    F_ica = centered_grams(S, sub)
    dep_ica = cka(F_ica)
    null = cka_null(F_ica, M_HSIC, rng)
    del F_ica
    dep_pca = cka(centered_grams(Y, sub))
    thr = DEP_FACTOR * float(np.quantile(null, 0.999))
    dist = 1.0 - dep_ica
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), "average")
    gid = fcluster(Z, t=1.0 - thr, criterion="distance") - 1
    print(f"CKA null q99.9 {np.quantile(null, 0.999):.4f}, thr {thr:.4f}, groups {gid.max() + 1}",
          flush=True)  # fmt: skip

    # mixing: H ~ S B; variance of each reader from each group
    B = np.linalg.lstsq(S, Hc, rcond=None)[0]
    var_h = Hc.var(0)
    G = int(gid.max() + 1)
    share = np.zeros((G, n))
    for g in range(G):
        m = np.flatnonzero(gid == g)
        share[g] = (S[:, m] @ B[m]).var(0) / var_h
    tot = share @ var_h / var_h.sum()
    gorder = np.argsort(-tot)
    remap = np.empty(G, int)
    remap[gorder] = np.arange(G)
    gid = remap[gid]
    share, tot = share[gorder], tot[gorder]
    leaf = np.empty(r, int)
    leaf[leaves_list(Z)] = np.arange(r)
    out_order = np.lexsort((leaf, gid))

    # 3. per-group display coordinates; post-hoc labels
    lab = labels(ops)
    show = rng.choice(N, N_SHOW, replace=False)
    groups = []
    for g in range(G):
        m = np.flatnonzero(gid == g)
        Sg = S[:, m]
        mm = np.ix_(m, m)
        within = float(dep_ica[mm][~np.eye(m.size, dtype=bool)].mean()) if m.size > 1 else None
        grp: dict[str, Any] = {"dims": int(m.size), "outputs": [int(i) for i in m], "var_share": float(tot[g]),
               "within_dep": within, "mean_kurt": float(kurt[m].mean()),
               "top_readers": [[names[c], float(share[g, c])] for c in np.argsort(-share[g])[:8]]}  # fmt: skip
        grp.update(display(Sg, show))
        grp["eta2"] = {k: eta2(Sg, v) for k, v in lab.items()}
        groups.append(grp)
        print(f"group {g}: dims {m.size} share {tot[g]:.3f}", flush=True)

    outputs = []
    for i in out_order:
        h, e = np.histogram(S[:, i], bins=60)
        outputs.append({"i": int(i), "g": int(gid[i]), "neg": float(negent[i]), "kurt": float(kurt[i]),
                        "h": h.tolist(), "lo": float(e[0]), "hi": float(e[-1]),
                        "eta2": {k: round(eta2(S[:, [i]], lab[k]), 3) for k in POSTHOC_OUT}})  # fmt: skip
    res = {
        "layer": layer, "ops": ops, "pos": "=", "N": N, "n_readers": n, "readers": names,
        "rms_cv": float(rms.std() / rms.mean()), "var_keep": VAR_KEEP, "r": r,
        "ev": rnd(ev, 5), "reader_order": reader_order.tolist(),
        "reader_corr": rnd(reader_corr[np.ix_(reader_order, reader_order)], 2),
        "out_order": out_order.tolist(), "gid": gid.tolist(),
        "ica_corr": rnd(np.abs(np.corrcoef(S.T))[np.ix_(out_order, out_order)], 3),
        "dep_ica": rnd(dep_ica[np.ix_(out_order, out_order)], 4), "dep_pca": rnd(dep_pca, 4),
        "null_q999": float(np.quantile(null, 0.999)), "null_max": float(null.max()), "thr": thr,
        "outputs": outputs, "share": rnd(share, 3), "groups": groups,
        "show_a": lab["a"][show].tolist(), "show_b": lab["b"][show].tolist(),
        "show_op": lab["op"][show].tolist(),
    }  # fmt: skip
    ISA_OUT.mkdir(parents=True, exist_ok=True)
    np.savez(ISA_OUT / f"isa_L{layer}_{ops}{tag}_unmix.npz", unmix=unmix, gid=gid, cols=cols,
             S=S.astype(np.float32))  # fmt: skip
    path = ISA_OUT / f"isa_L{layer}_{ops}{tag}.json"
    path.write_text(json.dumps(res))
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
