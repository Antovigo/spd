"""V2 of the quantity loop: quantities are directions in the span of the readers' read vectors.

Notation (a = 1..100 is the operand at the "a" token, token position t = 1):
* x(a) in R^4096: the raw residual stream of the alive-only model at the site; X (100, 4096).
* V~_r = gamma_l * V^_r: reader r's gain-folded unit read direction; S = span{V~_r} over the
  site's n active readers, k = dim S, Q (4096, k) an orthonormal basis of S.
* Z = X Q (100, k): the stream in reader-span coordinates; W = Q^T [V~_1 ... V~_n] (k, n), so the
  readers' raw reads are Y = Z W (exact).
* A quantity q: a feature map phi_q(a) in R^{d_q}, stacked over a into Phi_q (100, d_q), and its
  directions D_q (k, d_q): the model is Z = 1 mu^T + sum_q Phi_q D_q^T + E (E: residual).

Quantities are fit stagewise in a fixed order (each stage fits the residual of the previous
ones), so the variance each one explains is its increment given the earlier ones. Types:
* "fixed": given features (indicators, a circle's cos and sin), D = least squares.
* "curve": a smooth function of a with r directions, features unknown: reduced-rank regression
  of the residual on a cubic-spline basis B (rank r): f_1..f_r and D come from the top r singular
  pairs of the least-squares fit B G.
Generalisation: 10-fold cross-validation over values of a (fit on 90 values, predict the stream
at the 10 held-out ones from their features). A one-hot quantity predicts nothing at a held-out
value; a structured one does.
"""

from dataclasses import dataclass

import numpy as np
from qsite import DATASET, VW, load_site
from scipy.interpolate import BSpline

A = np.arange(1, 101)
ROWS = np.arange(100) * 100  # prompts op = +, b = 1, a = 1..100


def stream(l: int) -> np.ndarray:  # noqa: E741
    compact = VW / "alive_only/resid_token_a.npy"  # (65, 100, 4096): resid[:, ROWS, 1], for pods
    if compact.exists():
        return np.asarray(np.load(compact, mmap_mode="r")[l], np.float64)
    r = np.load(VW / "alive_only/resid.npy", mmap_mode="r")
    return np.asarray(r[l][ROWS, 1], np.float64)


def reader_dirs(block: int, point: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Gain-folded unit read directions (n, 4096) of the site's active readers, their names, and
    the readers' inner activations (100, n) in the ||V|| = 1 gauge."""
    site = load_site(block, point)
    ix = np.load(VW / "index.npz")
    row = {int(c): j for j, c in enumerate(ix["r_col"])}
    Vt = np.load(VW / "vectors.npz")["V_til"][[row[int(c)] for c in site.cols]]
    return Vt.astype(np.float64), site.names, site.X


def span_basis(Vt: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    U, s, _ = np.linalg.svd(Vt.T, full_matrices=False)
    return U[:, s > tol * s[0]]


def spline(n_basis: int = 6, degree: int = 3, log: bool = True) -> np.ndarray:
    x = np.log(A) if log else A.astype(float)
    inner = np.linspace(x[0], x[-1], n_basis - degree + 1)
    t = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
    return BSpline.design_matrix(x, t, degree).toarray()


def classes(labels: np.ndarray) -> np.ndarray:
    vals = np.unique(labels)
    return np.stack([(labels == v).astype(float) for v in vals], 1)


def fourier(period: int, harmonics: list[int]) -> np.ndarray:
    cols = []
    for j in harmonics:
        th = 2 * np.pi * j * A / period
        cols.append(np.cos(th))
        if 2 * j != period:
            cols.append(np.sin(th))
    return np.stack(cols, 1)


@dataclass
class Q:
    name: str
    kind: str  # "fixed" | "curve"
    Phi: np.ndarray  # fixed: features; curve: the smooth basis
    rank: int = 1


def _fit(q: Q, Ztr: np.ndarray, Ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Coefficients G (cols(Phi), k) and, for a curve, the projector onto its r directions."""
    Pc = Ptr - Ptr.mean(0)
    G, *_ = np.linalg.lstsq(Pc, Ztr, rcond=None)
    if q.kind == "fixed":
        return G, None
    _, _, Vh = np.linalg.svd(Pc @ G, full_matrices=False)
    P = Vh[: q.rank].T @ Vh[: q.rank]
    return G, P


def _predict(
    q: Q, G: np.ndarray, P: np.ndarray | None, Ptr_mean: np.ndarray, Pte: np.ndarray
) -> np.ndarray:
    out = (Pte - Ptr_mean) @ G
    return out if P is None else out @ P


def stagewise(Z: np.ndarray, H: list[Q], train: np.ndarray | None = None):  # noqa: ANN201
    """Fit H in order on the rows `train` (all if None); returns per-stage fitted parts on all
    rows (list of (100, k)) and the per-stage objects."""
    tr = np.arange(len(Z)) if train is None else train
    mu = Z[tr].mean(0)
    R = Z - mu
    parts, objs = [], []
    for q in H:
        G, P = _fit(q, R[tr], q.Phi[tr])
        part = _predict(q, G, P, q.Phi[tr].mean(0), q.Phi)
        parts.append(part)
        objs.append((G, P))
        R = R - part
    return mu, parts, objs


def variance_table(Z: np.ndarray, W: np.ndarray | None, H: list[Q], folds: int = 10) -> list[dict]:
    """Per stage: variance explained in the reader metric (Y = Z W; the stream metric Z when W
    is None), as an increment, against its chance level, and cumulatively; the cross-validated
    cumulative R^2 over held-out values of a (same metric); and how many readers the quantity
    explains at least 10% of the variance of.

    Chance level: d features fit to a residual with df remaining degrees of freedom (df = 99 minus
    the dimensions already used) capture d / df of it in expectation if they carry nothing.
    """
    M = W if W is not None else np.eye(Z.shape[1])
    mu, parts, _ = stagewise(Z, H)
    Yc = (Z - mu) @ M
    tot = (Yc**2).sum()
    perm = np.random.default_rng(0).permutation(len(Z))
    cv_pred = [np.zeros((len(Z), M.shape[1])) for _ in H]
    for f in range(folds):
        te = perm[f::folds]
        tr = np.setdiff1d(np.arange(len(Z)), te)
        mu_f, parts_f, _ = stagewise(Z, H, tr)
        cum = np.zeros_like(Z) + mu_f
        for s_, p in enumerate(parts_f):
            cum = cum + p
            cv_pred[s_][te] = (cum @ M)[te]
    rows, cum, used = [], np.zeros_like(Yc), 0
    Y = Z @ M
    for s_, (q, p) in enumerate(zip(H, parts, strict=True)):
        d = int(q.rank if q.kind == "curve" else np.linalg.matrix_rank(q.Phi - q.Phi.mean(0)))
        rest = ((Yc - cum) ** 2).sum()
        py = p @ M
        cum = cum + py
        df = max(99 - used, 1)
        incr = float((py**2).sum() / tot)
        chance = float(min(d, df) / df * rest / tot)
        per_reader = (py**2).sum(0) / np.maximum((Yc**2).sum(0), 1e-12)
        rows.append({
            "quantity": q.name, "dims": d, "incr": incr, "chance": chance, "excess": incr - chance,
            "cum": float(1 - ((Yc - cum) ** 2).sum() / tot),
            "cv_cum": float(1 - ((Y - cv_pred[s_]) ** 2).sum() / tot),
            "readers_10pct": int((per_reader > 0.1).sum()) if W is not None else None,
        })  # fmt: skip
        used += d
    return rows


def residual_svd(E: np.ndarray, n_null: int = 50) -> tuple[np.ndarray, np.ndarray, float]:
    U, s, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    rng = np.random.default_rng(1)
    null = [np.linalg.svd(np.stack([rng.permutation(E[:, j]) for j in range(E.shape[1])], 1)
                          - E.mean(0), compute_uv=False)[0] for _ in range(n_null)]  # fmt: skip
    return U, s, float(np.quantile(null, 0.95))


def circle_geometry(G: np.ndarray, rows: tuple[int, int]) -> dict:
    """Directions of a cos/sin pair (rows of G): norm ratio and angle between them."""
    dc, ds = G[rows[0]], G[rows[1]]
    nc, ns = np.linalg.norm(dc), np.linalg.norm(ds)
    return {"norm_ratio": float(min(nc, ns) / max(nc, ns)),
            "angle_deg": float(np.degrees(np.arccos(np.clip(dc @ ds / (nc * ns), -1, 1))))}  # fmt: skip


def tokens() -> np.ndarray:
    return np.load(DATASET / "index.npz")["tokens"][ROWS, 1]
