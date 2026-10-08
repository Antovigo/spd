"""Joint fits of quantities to a site, scored in the reader metric, from cached Gram matrices.

Notation (AGENT.md): Z (D, k) the stream at the site in reader-span coordinates, W (k, n) the
readout, M = W W^T, so reader-metric squared norms are tr(E^T E M). Features of all candidates are
centred and stacked into P (D, C); a fit on a subset of columns is the ridge solution
G = (P_A^T P_A + lam I)^-1 P_A^T Z (lam = RIDGE x mean diagonal), computed from the Grams
P^T P, P^T Z, Z^T Z, so a fit costs O(C^2 k) instead of O(D C k).

Held-out fits: for each fold (a set of domain rows held out together), the fit uses the training
rows only, centred on the training means (intercept included), and predicts the fold; the fold's
Grams and sums are cached, so held-out scores cost the same as in-sample ones.

Held-out schemes (`folds`): which domain rows are held out together.
* "a-values": all rows with a in the held-out set of a values (t = 1, 2: the primary scheme);
* "pairs": all rows of held-out (a, b) pairs, both operations (t >= 3: the primary scheme);
* "b-values": all rows with b in the held-out set (t >= 3, diagnostic).
"""

import numpy as np
from qdata import Position, Site
from qfeat import Cand

RIDGE = 1e-3
IMPORTANT = 0.01


def folds(pos: Position, scheme: str, n_folds: int = 10, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if scheme == "a-values":
        key = pos.a
    elif scheme == "b-values":
        key = pos.b
    elif scheme == "pairs":
        key = (pos.a - 1) * 100 + (pos.b - 1)
    else:
        raise ValueError(scheme)
    vals = rng.permutation(np.unique(key))
    return [np.flatnonzero(np.isin(key, vals[f::n_folds])) for f in range(n_folds)]


def primary_scheme(t: int) -> str:
    return "a-values" if t <= 2 else "pairs"


def ridge(PP: np.ndarray, PZ: np.ndarray) -> np.ndarray:
    lam = max(
        RIDGE * np.trace(PP) / max(PP.shape[0], 1), 1e-12
    )  # > 0 also when every feature is constant
    return np.linalg.solve(PP + lam * np.eye(PP.shape[0]), PZ)


class Fitter:
    def __init__(self, site: Site, cands: list[Cand], fold_sets: list[np.ndarray]) -> None:
        self.site = site
        self.cands = {c.name: c for c in cands}
        self.Zc = site.Z - site.Z.mean(0)
        self.M = site.W @ site.W.T
        self.D = site.Z.shape[0]
        blocks, self.slc, s = [], {}, 0
        for c in cands:
            P = c.Phi - c.Phi.mean(0)
            blocks.append(P)
            self.slc[c.name] = np.arange(s, s + P.shape[1])
            s += P.shape[1]
        self.P = np.hstack(blocks)
        self.PP, self.PZ = self.P.T @ self.P, self.P.T @ self.Zc
        self.ZZ = self.Zc.T @ self.Zc
        self.tot = float(np.trace(self.ZZ @ self.M))
        self.folds = fold_sets
        self._fold_grams()

    def _fold_grams(self) -> None:
        """Per fold: test-row Grams and sums (P, Z globally centred, so training sums = -test sums)."""
        self.fg = []
        for te in self.folds:
            Pt, Zt = self.P[te], self.Zc[te]
            self.fg.append((Pt.T @ Pt, Pt.T @ Zt, Zt.T @ Zt, Pt.sum(0), Zt.sum(0), len(te)))

    def add(self, c: Cand) -> None:
        """Register an agent-made candidate after construction."""
        if c.name in self.cands:
            return
        P = c.Phi - c.Phi.mean(0)
        C0 = self.P.shape[1]
        self.slc[c.name] = np.arange(C0, C0 + P.shape[1])
        self.cands[c.name] = c
        cross = self.P.T @ P
        self.PP = np.block([[self.PP, cross], [cross.T, P.T @ P]])
        self.PZ = np.vstack([self.PZ, P.T @ self.Zc])
        self.P = np.hstack([self.P, P])
        self._fold_grams()

    def cols(self, names: list[str]) -> np.ndarray:
        return np.concatenate([self.slc[n] for n in names]) if names else np.zeros(0, int)

    def rss(self, names: list[str]) -> float:
        c = self.cols(names)
        if len(c) == 0:
            return self.tot
        G = ridge(self.PP[np.ix_(c, c)], self.PZ[c])
        return float(
            np.trace(self.ZZ @ self.M)
            - 2 * np.trace(G.T @ self.PZ[c] @ self.M)
            + np.trace(G.T @ self.PP[np.ix_(c, c)] @ G @ self.M)
        )

    def dims(self, names: list[str]) -> int:
        c = self.cols(names)
        if len(c) == 0:
            return 0
        ev = np.linalg.eigvalsh(self.PP[np.ix_(c, c)])
        return int((ev > 1e-8 * max(ev.max(), 1e-30)).sum())

    def increment(self, A: list[str], name: str) -> tuple[float, float]:
        """(increment, chance) of `name` added to the joint fit of A, as fractions of the total."""
        r0, r1 = self.rss(A), self.rss(A + [name])
        d_new = self.dims(A + [name]) - self.dims(A)
        df = max(self.D - 1 - self.dims(A), 1)
        return (r0 - r1) / self.tot, d_new / df * r0 / self.tot

    def drop_one(self, A: list[str]) -> list[tuple[str, float, float]]:
        return [(n, *self.increment([m for m in A if m != n], n)) for n in A]

    def _fold_fit(self, c: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(G, training mean of P[:, c], training mean of Z) for fold f."""
        PPt, PZt, _, sP, sZ, nte = self.fg[f]
        ntr = self.D - nte
        mP, mZ = -sP[c] / ntr, -sZ / ntr
        PP = self.PP[np.ix_(c, c)] - PPt[np.ix_(c, c)] - ntr * np.outer(mP, mP)
        PZ = self.PZ[c] - PZt[c] - ntr * np.outer(mP, mZ)
        return ridge(PP, PZ), mP, mZ

    def heldout_r2(self, names: list[str]) -> float:
        """1 - held-out error / held-out variance around the training mean, summed over folds."""
        c = self.cols(names)
        err = tot = 0.0
        for f, (PPt, PZt, ZZt, sP, sZ, nte) in enumerate(self.fg):
            mZ = -sZ / (self.D - nte)
            ZZc = (
                ZZt - np.outer(sZ, mZ) - np.outer(mZ, sZ) + nte * np.outer(mZ, mZ)
            )  # sum (Z - mZ)^T (Z - mZ)
            tot += float(np.trace(ZZc @ self.M))
            if len(c) == 0:
                err += float(np.trace(ZZc @ self.M))
                continue
            G, mP, _ = self._fold_fit(c, f)
            PZc = PZt[c] - np.outer(sP[c], mZ) - np.outer(mP, sZ) + nte * np.outer(mP, mZ)
            PPc = (
                PPt[np.ix_(c, c)]
                - np.outer(sP[c], mP)
                - np.outer(mP, sP[c])
                + nte * np.outer(mP, mP)
            )
            err += float(
                np.trace(ZZc @ self.M)
                - 2 * np.trace(G.T @ PZc @ self.M)
                + np.trace(G.T @ PPc @ G @ self.M)
            )
        return 1 - err / tot

    def reconstruct(self, names: list[str], heldout: bool = False) -> np.ndarray:
        """Zhat (D, k), in-sample or from held-out fits (each fold predicted by the others)."""
        c = self.cols(names)
        mu = self.site.Z.mean(0)
        if not heldout:
            if len(c) == 0:
                return np.tile(mu, (self.D, 1))
            return mu + self.P[:, c] @ ridge(self.PP[np.ix_(c, c)], self.PZ[c])
        out = np.zeros_like(self.Zc)
        for f, te in enumerate(self.folds):
            if len(c) == 0:
                out[te] = -self.fg[f][4] / (self.D - len(te))
                continue
            G, mP, mZ = self._fold_fit(c, f)
            out[te] = mZ + (self.P[np.ix_(te, c)] - mP) @ G
        return mu + out

    def important_share(self, names: list[str], heldout: bool = True) -> float:
        """Residual variance on important entries (CI > 0.01) / read variance there (deviations
        from the reader's domain mean: what a mean patch would remove)."""
        E = (self.site.Z - self.reconstruct(names, heldout)) @ self.site.W
        Y = self.site.Y
        Mk = self.site.CI > IMPORTANT
        return float(((E**2) * Mk).sum() / max((((Y - Y.mean(0)) ** 2) * Mk).sum(), 1e-30))

    def ci_weighted_residual(self, names: list[str]) -> np.ndarray:
        """Per reader: sum over the domain of CI x squared residual (ranks readers whose errors matter)."""
        E = (self.site.Z - self.reconstruct(names)) @ self.site.W
        return (self.site.CI * E**2).sum(0)


def fit_scalar(
    P: np.ndarray, y: np.ndarray, fold_sets: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """(in-sample, held-out) ridge predictions of a scalar y (D,) from features P (D, C), with
    training-mean centring per fold (used for the stream RMS rho)."""
    if P.shape[1] == 0:
        ho = np.empty_like(y)
        for te in fold_sets:
            ho[te] = np.delete(y, te).mean()
        return np.full_like(y, y.mean()), ho
    Pc, yc = P - P.mean(0), y - y.mean()
    ins = y.mean() + Pc @ ridge(Pc.T @ Pc, Pc.T @ yc)
    ho = np.empty_like(y)
    for te in fold_sets:
        tr = np.setdiff1d(np.arange(len(y)), te)
        mP, my = P[tr].mean(0), y[tr].mean()
        Ptr = P[tr] - mP
        ho[te] = my + (P[te] - mP) @ ridge(Ptr.T @ Ptr, Ptr.T @ (y[tr] - my))
    return ins, ho
