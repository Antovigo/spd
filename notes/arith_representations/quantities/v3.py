"""V3: the protocol (PROTOCOL.md) at token a: joint fits, drop-one losses, the important residual.

Notation as in PROTOCOL.md; at token position t = 1 the domain is a = 1..100. Every quantity is a
fixed feature matrix (100 x d) over a; all accepted quantities are refit jointly after each
change. CI: the filter's output CI on the original model at t = 1 (100 x n, a function of a).
"""

from dataclasses import dataclass

import numpy as np
from qsite import DATASET
from v2 import ROWS, A, reader_dirs, span_basis, stream

IMPORTANT = 0.01
RIDGE = 1e-3


@dataclass
class Cand:
    name: str
    Phi: np.ndarray  # (100, d), centred inside the fit

    @property
    def d(self) -> int:
        return int(np.linalg.matrix_rank(self.Phi - self.Phi.mean(0)))


@dataclass
class Site:
    label: str
    Z: np.ndarray  # (100, k)
    W: np.ndarray  # (k, n)
    names: list[str]
    cols: np.ndarray  # dataset columns of the readers
    CI: np.ndarray  # (100, n)

    @property
    def Y(self) -> np.ndarray:
        return self.Z @ self.W


def load(block: int, point: str) -> Site:
    from qsite import load_site

    s = load_site(block, point)
    Vt, names, _ = reader_dirs(block, point)
    Qb = span_basis(Vt)
    lpos = 2 * block + (point == "mlp")
    compact = DATASET / "original/ci_token_a.npy"  # (100, A): ci[ROWS, 1], for pods
    if compact.exists():
        CI = np.asarray(np.load(compact)[:, s.cols], np.float64)
    else:
        ci = np.load(DATASET / "original/ci.npy", mmap_mode="r")
        CI = np.asarray(ci[ROWS, 1][:, s.cols], np.float64)
    return Site(f"L{block}.{point}", stream(lpos) @ Qb, Qb.T @ Vt.T, names, s.cols, CI)


def design(H: list[Cand]) -> np.ndarray:
    if not H:
        return np.zeros((100, 0))
    P = np.hstack([c.Phi for c in H])
    return P - P.mean(0)


def fit(site: Site, H: list[Cand], rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Joint least squares of the centred Z on the quantities' features (fit on `rows`);
    returns the reconstruction of Z on all 100 values and the coefficients."""
    tr = np.arange(100) if rows is None else rows
    mu = site.Z[tr].mean(0)
    P = design(H)
    if P.shape[1] == 0:
        return np.tile(mu, (100, 1)), np.zeros((0, site.Z.shape[1]))
    Pm = P[tr].mean(0)
    Pc = P[tr] - Pm
    # ridge: overlapping class partitions are collinear on some training subsets; without a
    # penalty the held-out predictions explode. RIDGE is relative to the mean feature variance.
    G = np.linalg.solve(Pc.T @ Pc + RIDGE * np.trace(Pc.T @ Pc) / Pc.shape[1] * np.eye(Pc.shape[1]),
                        Pc.T @ (site.Z[tr] - mu))  # fmt: skip
    return mu + (P - Pm) @ G, G


def rss(site: Site, H: list[Cand]) -> float:
    Zh, _ = fit(site, H)
    return float((((site.Z - Zh) @ site.W) ** 2).sum())


def total(site: Site) -> float:
    Y = site.Y
    return float(((Y - Y.mean(0)) ** 2).sum())


def dims(H: list[Cand]) -> int:
    P = design(H)
    return int(np.linalg.matrix_rank(P)) if P.shape[1] else 0


def increment(site: Site, H: list[Cand], c: Cand) -> tuple[float, float]:
    """(increment, chance) of c added to the joint fit of H, as fractions of the total."""
    tot = total(site)
    r0 = rss(site, H)
    r1 = rss(site, H + [c])
    d_new = dims(H + [c]) - dims(H)
    df = max(99 - dims(H), 1)
    return (r0 - r1) / tot, d_new / df * r0 / tot


def drop_one(site: Site, H: list[Cand]) -> list[tuple[str, float, float]]:
    """(name, loss, chance) of removing each quantity from the joint fit of H."""
    out = []
    for i, c in enumerate(H):
        rest = H[:i] + H[i + 1 :]
        out.append((c.name, *increment(site, rest, c)))
    return out


def perm_p(site: Site, H: list[Cand], c: Cand, n: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    real = increment(site, H, c)[0]
    hits = 0
    for _ in range(n):
        cp = Cand(c.name, c.Phi[rng.permutation(100)])
        hits += increment(site, H, cp)[0] >= real
    return (hits + 1) / (n + 1)


def heldout_r2(site: Site, H: list[Cand], folds: int = 10) -> float:
    perm = np.random.default_rng(0).permutation(100)
    pred = np.zeros_like(site.Z)
    for f in range(folds):
        te = perm[f::folds]
        Zh, _ = fit(site, H, np.setdiff1d(np.arange(100), te))
        pred[te] = Zh[te]
    return 1 - float((((site.Z - pred) @ site.W) ** 2).sum()) / total(site)


def important_residual(site: Site, H: list[Cand]) -> dict:
    Zh, _ = fit(site, H)
    E = (site.Z - Zh) @ site.W
    Y = site.Y
    M = site.CI > IMPORTANT
    imp_tot = float((((Y - Y.mean(0)) ** 2) * M).sum())
    return {"share": float(((E**2) * M).sum() / max(imp_tot, 1e-30)),
            "frac_of_residual_on_important": float(((E**2) * M).sum() / max((E**2).sum(), 1e-30)),
            "important_entries": int(M.sum())}  # fmt: skip


def important_residual_cv(site: Site, H: list[Cand], folds: int = 10) -> dict:
    """The important residual of held-out predictions (directions fit on 90% of the values of a,
    predicted on the rest): an in-sample residual shrinks with every added dimension by chance."""
    perm = np.random.default_rng(0).permutation(100)
    pred = np.zeros_like(site.Z)
    for f in range(folds):
        te = perm[f::folds]
        Zh, _ = fit(site, H, np.setdiff1d(np.arange(100), te))
        pred[te] = Zh[te]
    E = (site.Z - pred) @ site.W
    Y = site.Y
    M = site.CI > IMPORTANT
    imp_tot = float((((Y - Y.mean(0)) ** 2) * M).sum())
    return {"share": float(((E**2) * M).sum() / max(imp_tot, 1e-30))}


def residual_svd(site: Site, H: list[Cand], important_only: bool = False, n_null: int = 50) -> dict:
    Zh, _ = fit(site, H)
    E = (site.Z - Zh) @ site.W
    if important_only:
        E = E * (site.CI > IMPORTANT)
    U, s, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    rng = np.random.default_rng(1)
    null = [np.linalg.svd(np.stack([rng.permutation(E[:, j]) for j in range(E.shape[1])], 1) - E.mean(0),
                          compute_uv=False)[0] for _ in range(n_null)]  # fmt: skip
    top = np.argsort(-np.abs(U[:, 0]))[:10]
    return {"sv1": float(s[0]), "null95": float(np.quantile(null, 0.95)), "u1": (U[:, 0] * s[0]).tolist(),
            "u1_top_values": [(int(A[i]), round(float(U[i, 0]), 3)) for i in top]}  # fmt: skip
