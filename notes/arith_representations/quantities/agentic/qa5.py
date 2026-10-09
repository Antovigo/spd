"""V5 tools for the quantity-finding agent (INSTRUCTIONS_v5.md, PLAN_v5.md): per-reader sparse
quantities. Training rows only, as in qa.py (same value hold-out, inner folds, formula language).

    python qa5.py sites <t>                             the read sites of position t, in stream order
    python qa5.py show <t> <site> <outdir>              look at the readers' inner activations
    python qa5.py fit <t> <site> <spec.py> <outdir>     select each reader's quantities, report the score

Model: each reader r (a component reading the stream at the site; alive = CI > 0.01 on at least one
training row) is fit on its own support S_r (a subset of the list Q of n_Q quantities) by ridge
regression of its raw read y_r on [1, Phi_q for q in S_r] (Phi_q: the standardised features of
quantity q, d_q of them), on its important entries only (rows i with CI_{i,r} > 0.01; N_r of them).

Score (lower is better):
  J = sum over quantities used by >= 1 reader of formula_bits(q)
    + sum_r [log2(n_Q + 1) + log2 C(n_Q, |S_r|) + 1/2 log2(N_r) (1 + sum_{q in S_r} d_q)]
    + B_BITS x 100 x (1 - R2_ho),
  R2_ho = 1 - sum_r E_r / sum_r T_r on the inner folds, important entries only (E_r: squared error of
  the prediction from the other training rows; T_r: squared deviation from their mean).
Supports are chosen by the tool: per reader, stepwise (forward, then backward, until stable) on its
share of J; then whole quantities are removed while that lowers J.
"""

import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa  # noqa: E402
from qa import IMPORTANT, RIDGE, Site, design, formula_bits, load_spec, panels  # noqa: E402

B_BITS = float(os.environ.get("QA_B_BITS", "3000"))
N_MIN = 20  # important training entries a reader needs to be scored; also per fold training part
EXACT_MAX = 256  # exhaustive support search when 2^n_Q <= this


# ---------------------------------------------------------------- per-reader sufficient statistics
class ReaderStats:
    """Sums of reader r's important training entries: over all of them and per inner fold (the fold's
    rows; its training rows are the rest). Grams of the full feature matrix, so any support is a slice."""

    def __init__(
        self, P: np.ndarray, y: np.ndarray, m: np.ndarray, folds: list[np.ndarray]
    ) -> None:
        Pm, ym = P[m], y[m]
        self.n = int(m.sum())
        self.PP, self.Py, self.yy = Pm.T @ Pm, Pm.T @ ym, float(ym @ ym)
        self.sP, self.sy = Pm.sum(0), float(ym.sum())
        self.fold = []
        for te in folds:
            tm = te[m[te]]
            Pt, yt = P[tm], y[tm]
            self.fold.append(
                (Pt.T @ Pt, Pt.T @ yt, float(yt @ yt), Pt.sum(0), float(yt.sum()), len(tm))
            )

    def _fold_err(self, c: np.ndarray, f: tuple) -> tuple[float, float]:
        PPt, Pyt, yyt, sPt, syt, nte = f
        ntr = self.n - nte
        if ntr < N_MIN or nte == 0:  # skipped for this reader, for every support
            return 0.0, 0.0
        my = (self.sy - syt) / ntr
        tot = yyt - 2 * my * syt + nte * my * my  # test entries around the training mean
        if len(c) == 0:
            return tot, tot
        mP = (self.sP[c] - sPt[c]) / ntr
        A = self.PP[np.ix_(c, c)] - PPt[np.ix_(c, c)] - ntr * np.outer(mP, mP)
        bvec = self.Py[c] - Pyt[c] - ntr * mP * my
        lam = max(RIDGE * np.trace(A) / len(c), 1e-12)
        g = np.linalg.solve(A + lam * np.eye(len(c)), bvec)
        # test error: sum (y - my - (p - mP) g)^2
        Pyc = Pyt[c] - sPt[c] * my - mP * syt + nte * mP * my
        PPc = (
            PPt[np.ix_(c, c)] - np.outer(sPt[c], mP) - np.outer(mP, sPt[c]) + nte * np.outer(mP, mP)
        )
        return tot - 2 * g @ Pyc + g @ PPc @ g, tot

    def heldout(self, c: np.ndarray) -> tuple[float, float]:
        e = t = 0.0
        for f in self.fold:
            ef, tf = self._fold_err(c, f)
            e, t = e + ef, t + tf
        return e, t

    def fit(self, c: np.ndarray) -> tuple[float, np.ndarray]:
        """Intercept and coefficients on all important training entries."""
        my = self.sy / self.n
        if len(c) == 0:
            return my, np.zeros(0)
        mP = self.sP[c] / self.n
        A = self.PP[np.ix_(c, c)] - self.n * np.outer(mP, mP)
        bvec = self.Py[c] - self.n * mP * my
        lam = max(RIDGE * np.trace(A) / len(c), 1e-12)
        g = np.linalg.solve(A + lam * np.eye(len(c)), bvec)
        return my - mP @ g, g


# ---------------------------------------------------------------- selection (worker globals)
_G: dict = {}


def _cols(S: tuple[int, ...]) -> np.ndarray:
    slc = _G["slc"]
    return np.concatenate([slc[q] for q in S]) if S else np.zeros(0, int)


def _cost(st: ReaderStats, S: tuple[int, ...], allowed: list[int]) -> tuple[float, float]:
    nQ = len(allowed)
    e, _ = st.heldout(_cols(S))
    sup = math.log2(nQ + 1) + math.log2(math.comb(nQ, len(S)))
    coef = 0.5 * math.log2(max(st.n, N_MIN)) * (1 + sum(_G["dims"][q] for q in S))
    return sup + coef + B_BITS * 100 * e / _G["T"], e


def _select(job: tuple[int, tuple[int, ...], list[int]]) -> tuple[int, tuple[int, ...], float]:
    """Reader r's support among `allowed`: exhaustive when 2^n_Q <= EXACT_MAX (or forced by
    _G["exact"]), otherwise stepwise from `start` (forward then backward, until neither lowers the
    cost; ties to the lower index)."""
    r, start, allowed = job
    st = _G["stats"][r]
    exact = _G.get("exact")
    if exact or (exact is None and 2 ** len(allowed) <= EXACT_MAX):
        from itertools import combinations

        best = min((_cost(st, S, allowed)[0], S) for k in range(len(allowed) + 1)
                   for S in combinations(sorted(allowed), k))  # fmt: skip
        return r, best[1], _cost(st, best[1], allowed)[1]
    S = tuple(sorted(start))
    cur, _ = _cost(st, S, allowed)
    while True:
        changed = False
        while True:  # forward
            best = min(((_cost(st, tuple(sorted(S + (q,))), allowed)[0], q) for q in allowed if q not in S),
                       default=None)  # fmt: skip
            if best is None or best[0] >= cur - 1e-9:
                break
            cur, S, changed = best[0], tuple(sorted(S + (best[1],))), True
        while S:  # backward
            best = min((_cost(st, tuple(x for x in S if x != q), allowed)[0], q) for q in S)
            if best[0] >= cur - 1e-9:
                break
            cur, S, changed = best[0], tuple(x for x in S if x != best[1]), True
        if not changed:
            break
    return r, S, _cost(st, S, allowed)[1]


class Scorer5:
    def __init__(
        self, s: Site, quantities: list[tuple[str, str]], procs: int | None = None
    ) -> None:
        self.s, self.Q = s, quantities
        P, blocks, dims = design(s, quantities)
        self.P, self.dims = P, dims
        self.fbits = [formula_bits(src) for _, src in quantities]
        self.slc, c = [], 0
        for Bk in blocks:
            self.slc.append(np.arange(c, c + Bk.shape[1]))
            c += Bk.shape[1]
        self.Y = s.Y
        self.imp = s.CI > IMPORTANT
        n_imp = self.imp.sum(0)
        self.readers = [r for r in range(self.Y.shape[1]) if n_imp[r] >= N_MIN]
        self.unscored = [r for r in range(self.Y.shape[1]) if 0 < n_imp[r] < N_MIN]
        self.stats = {
            r: ReaderStats(P, self.Y[:, r], self.imp[:, r], s.folds) for r in self.readers
        }
        self.T = sum(self.stats[r].heldout(np.zeros(0, int))[1] for r in self.readers)
        _G.update(stats=self.stats, slc=self.slc, dims=self.dims, T=self.T)
        self.procs = procs or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    def select(self, allowed: list[int], start: dict[int, tuple] | None = None,
               readers: list[int] | None = None) -> dict[int, tuple[tuple[int, ...], float]]:  # fmt: skip
        _G.update(stats=self.stats, slc=self.slc, dims=self.dims, T=self.T)
        rs = self.readers if readers is None else readers
        jobs = [
            (r, tuple(q for q in (start or {}).get(r, ()) if q in allowed), allowed) for r in rs
        ]
        if self.procs > 1 and len(jobs) > 8:
            with Pool(self.procs) as pool:
                out = pool.map(_select, jobs, chunksize=max(1, len(jobs) // (4 * self.procs)))
        else:
            out = [_select(j) for j in jobs]
        return {r: (S, e) for r, S, e in out}

    def J(self, allowed: list[int], sel: dict[int, tuple[tuple[int, ...], float]]) -> dict:
        nQ = len(allowed)
        used = sorted({q for S, _ in sel.values() for q in S})
        fb = sum(self.fbits[q] for q in used)
        sup = sum(math.log2(nQ + 1) + math.log2(math.comb(nQ, len(S))) for S, _ in sel.values())
        coef = sum(0.5 * math.log2(max(self.stats[r].n, N_MIN)) * (1 + sum(self.dims[q] for q in S))
                   for r, (S, _) in sel.items())  # fmt: skip
        r2 = 1 - sum(e for _, e in sel.values()) / self.T
        mb = fb + sup + coef
        return {"J": mb + B_BITS * 100 * (1 - r2), "model_bits": mb, "formula_bits": fb,
                "support_bits": sup, "coefficient_bits": coef, "r2": r2, "used": used}  # fmt: skip

    def run(self) -> tuple[list[int], dict, dict]:
        """Select supports over the whole list, then drop whole quantities while J falls."""
        allowed = list(range(len(self.Q)))
        sel = self.select(allowed)
        cur = self.J(allowed, sel)
        while True:
            best = None
            for q in allowed:
                if len(allowed) == 1:
                    break
                al = [x for x in allowed if x != q]
                hit = [r for r, (S, _) in sel.items() if q in S]
                new = dict(sel)
                new.update(self.select(al, {r: sel[r][0] for r in hit}, hit))
                if not hit:  # support code still changes with n_Q
                    new = {r: (S, e) for r, (S, e) in sel.items()}
                j = self.J(al, new)
                if best is None or j["J"] < best[0]["J"]:
                    best = (j, al, new)
            if best is None or best[0]["J"] >= cur["J"] - 1e-9:
                break
            cur, allowed, sel = best
        return allowed, sel, cur

    def dense(self) -> tuple[dict, dict]:
        """Every scored reader reads every quantity (S_r = Q), scored with the V5 metric."""
        allowed = list(range(len(self.Q)))
        S = tuple(allowed)
        sel = {r: (S, self.stats[r].heldout(_cols(S))[0]) for r in self.readers}
        return self.J(allowed, sel), sel

    def drop_one(self, allowed: list[int], sel: dict) -> dict[int, dict]:
        base = self.J(allowed, sel)["J"]
        out = {}
        for q in allowed:
            al = [x for x in allowed if x != q]
            hit = [r for r, (S, _) in sel.items() if q in S]
            new = dict(sel)
            if hit:
                new.update(self.select(al, {r: sel[r][0] for r in hit}, hit))
            j = self.J(al, new)
            out[q] = {
                "drop_one_delta": round(j["J"] - base, 1),
                "heldout_r2_without_it": round(j["r2"], 4),
            }
        return out

    def predict(self, sel: dict) -> np.ndarray:
        """In-sample prediction of every alive reader's raw read on all training rows (fit on its
        important entries); dead readers keep their mean."""
        Yh = np.tile(self.Y.mean(0), (self.s.N, 1))
        for r in self.unscored:
            Yh[:, r] = self.Y[self.imp[:, r], r].mean()
        for r, (S, _) in sel.items():
            c = _cols(S)
            b0, g = self.stats[r].fit(c)
            Yh[:, r] = b0 + self.P[:, c] @ g
        return Yh

    def shares(self, sel: dict) -> np.ndarray:
        """readers x quantities: increase of reader r's in-sample important-entry error when q is
        removed from its support (refit), as a share of its explained sum of squares."""
        H = np.zeros((self.Y.shape[1], len(self.Q)))
        for r, (S, _) in sel.items():
            st, m = self.stats[r], self.imp[:, r]
            y = self.Y[m, r]

            def err(SS: tuple[int, ...], st=st, m=m, y=y) -> float:  # noqa: ANN001
                c = _cols(SS)
                b0, g = st.fit(c)
                return float(((y - b0 - self.P[m][:, c] @ g) ** 2).sum())

            e0, eS = float(((y - y.mean()) ** 2).sum()), err(S)
            for q in S:
                H[r, q] = (err(tuple(x for x in S if x != q)) - eS) / max(e0 - eS, 1e-300)
        return H


# ---------------------------------------------------------------- figures
def incidence(H: np.ndarray, rows: list[int], s: Site, names: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2 + 0.45 * H.shape[1], 1.5 + 0.13 * len(rows)))
    im = ax.imshow(
        np.clip(H[rows], 0, 1), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows)), [f"c{int(s.cols[r])}" for r in rows], fontsize=5)
    ax.set_title("share of each reader's explained variance lost without the quantity", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- commands
def cmd_fit(t: int, label: str, spec: Path, out: Path, top: int = 6) -> dict:
    s = Site(t, label)
    Q = load_spec(spec)
    sc = Scorer5(s, Q)
    allowed, sel, res = sc.run()
    dro = sc.drop_one(allowed, sel)
    H = sc.shares(sel)
    ci_mass = {r: float(s.CI[:, r][sc.imp[:, r]].sum()) for r in sel}
    sizes = np.array([len(S) for S, _ in sel.values()])
    wts = np.array([ci_mass[r] for r in sel])
    Yh = sc.predict(sel)
    E = s.Y - Yh
    Yc = s.Y - s.Y.mean(0)
    alive = np.zeros(s.Y.shape[1], bool)
    alive[list(sel)] = True
    impA = sc.imp & alive[None, :]
    rows = []
    for q, (nm, src) in enumerate(Q):
        users = [r for r, (S, _) in sel.items() if q in S]
        row = {"name": nm, "formula": src, "dims": sc.dims[q], "formula_bits": round(sc.fbits[q], 1),
               "kept": q in allowed, "n_readers": len(users),
               "ci_weighted_share_of_explained": round(float(sum(H[r, q] * ci_mass[r] for r in users) / max(wts.sum(), 1e-300)), 3)}  # fmt: skip
        if q in dro:
            row.update(dro[q])
        rows.append(row)
    rep = {
        "site": label, "t": t, "N_train": s.N, "n_readers": int(s.Y.shape[1]), "n_alive_readers": len(sel),
        "B_bits_per_point": B_BITS, "score_J": round(res["J"], 1), "model_bits": round(res["model_bits"], 1),
        "formula_bits": round(res["formula_bits"], 1), "support_bits": round(res["support_bits"], 1),
        "coefficient_bits": round(res["coefficient_bits"], 1),
        "r2_reads_heldout_inner_important": round(res["r2"], 4),
        "r2_reads_in_sample_important": round(1 - float((E**2 * impA).sum() / max((Yc**2 * impA).sum(), 1e-300)), 4),
        "r2_reads_in_sample_all_entries": round(1 - float((E[:, alive] ** 2).sum() / max((Yc[:, alive] ** 2).sum(), 1e-300)), 4),
        "unscored_readers_below_N_min": [f"c{int(s.cols[r])}" for r in sc.unscored],
        "quantities_per_reader_mean": round(float(sizes.mean()), 2),
        "quantities_per_reader_ci_weighted_mean": round(float((sizes * wts).sum() / max(wts.sum(), 1e-300)), 2),
        "quantities_per_reader_histogram": {int(k): int((sizes == k).sum()) for k in np.unique(sizes)},
        "quantities": rows,
    }  # fmt: skip
    order = sorted(sel, key=lambda r: -ci_mass[r])
    rep["readers_by_ci_mass"] = [
        {"reader": f"c{int(s.cols[r])}", "important_rows": int(sc.imp[:, r].sum()),
         "quantities": [Q[q][0] for q in sel[r][0]]} for r in order[:40]]  # fmt: skip
    act_res = E / s.rho[:, None] * impA
    w = (s.CI * act_res**2).sum(0)
    worst = np.argsort(-w)[:top]
    rep["worst_readers"] = [
        {"reader": f"c{int(s.cols[r])}", "share_of_ci_weighted_residual": round(float(w[r] / max(w.sum(), 1e-300)), 3),
         "quantities": [Q[q][0] for q in sel.get(int(r), ((), 0))[0]]} for r in worst]  # fmt: skip
    Ei = E * impA
    Ec = Ei - Ei.mean(0)
    var = {"op": s.op, "a": s.a, "b": s.b, "a+b": s.a + s.b, "a-b": s.a - s.b, "a%10": s.a % 10,
           "b%10": s.b % 10, "a//10": s.a // 10, "b//10": s.b // 10}  # fmt: skip
    if t <= 2:
        var = {k: v for k, v in var.items() if "b" not in k and (t == 2 or k != "op")}
    tot, left = (Ec**2).sum(), []
    for k, x in var.items():
        vals, inv = np.unique(x, return_inverse=True)
        means = np.zeros((len(vals), Ec.shape[1]))
        np.add.at(means, inv, Ec)
        means /= np.bincount(inv)[:, None]
        sh = float((means[inv] ** 2).sum() / max(tot, 1e-300))
        left.append((round(sh - (len(vals) - 1) / (s.N - 1), 4), k))
    rep["residual_explained_by_variable_values"] = sorted(left, reverse=True)
    out.mkdir(parents=True, exist_ok=True)
    incidence(H, order, s, [nm for nm, _ in Q], out / f"{spec.stem}_incidence.png")
    panels(s, [s.Y[:, r] / s.rho for r in worst], [f"c{int(s.cols[r])} inner activation" for r in worst],
           out / f"{spec.stem}_reads.png", [s.CI[:, r] for r in worst])  # fmt: skip
    panels(s, [act_res[:, r] for r in worst], [f"c{int(s.cols[r])} residual (important entries)" for r in worst],
           out / f"{spec.stem}_residual.png")  # fmt: skip
    U, sv, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(3, len(sv))
    panels(s, [U[:, j] for j in range(k)],
           [f"residual singular function {j + 1} ({sv[j] ** 2 / max((sv**2).sum(), 1e-300):.3f})" for j in range(k)],
           out / f"{spec.stem}_residual_svd.png")  # fmt: skip
    rep["figures"] = [
        str(out / f"{spec.stem}_{x}.png")
        for x in ("incidence", "reads", "residual", "residual_svd")
    ]
    rep["_supports"] = {int(s.cols[r]): list(S) for r, (S, _) in sel.items()}
    (out / f"{spec.stem}_fit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "_supports"}, indent=1))
    return rep


def cmd_baseline(t: int, label: str, spec: Path) -> None:
    s = Site(t, label)
    Q = load_spec(spec)
    sc = Scorer5(s, Q)
    dj, _ = sc.dense()
    out = {"site": label, "t": t, "n_Q": len(Q), "B": B_BITS, "scored_readers": len(sc.readers),
           "unscored_readers": len(sc.unscored),
           "dense": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dj.items()}}  # fmt: skip
    allowed = list(range(len(Q)))
    for mode, flag in (("exhaustive", True), ("stepwise", False)):
        _G["exact"] = flag
        sel = sc.select(allowed)
        j = sc.J(allowed, sel)
        out[mode] = {"J": round(j["J"], 1), "r2": round(j["r2"], 4), "model_bits": round(j["model_bits"], 1),
                     "mean_support": round(float(np.mean([len(S) for S, _ in sel.values()])), 2)}  # fmt: skip
        out[mode + "_supports"] = {r: S for r, (S, _) in sel.items()}
    ex, sw = out.pop("exhaustive_supports"), out.pop("stepwise_supports")
    out["stepwise_gap_J"] = round(out["stepwise"]["J"] - out["exhaustive"]["J"], 1)
    out["readers_with_different_support"] = round(float(np.mean([ex[r] != sw[r] for r in ex])), 3)
    _G["exact"] = None
    allowed, sel, res = sc.run()
    out["final_after_outer_deletion"] = {"J": round(res["J"], 1), "r2": round(res["r2"], 4),
                                         "kept": [Q[q][0] for q in allowed],
                                         "mean_support": round(float(np.mean([len(S) for S, _ in sel.values()])), 2)}  # fmt: skip
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sites":
        qa.cmd_sites(int(sys.argv[2]))
        raise SystemExit
    cmd, t, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if cmd == "show":
        qa.cmd_show(t, label, Path(sys.argv[4]))
    elif cmd == "fit":
        cmd_fit(t, label, Path(sys.argv[4]), Path(sys.argv[5]))
    elif cmd == "baseline":  # not an agent tool: dense vs sparse, exhaustive vs stepwise
        cmd_baseline(t, label, Path(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
