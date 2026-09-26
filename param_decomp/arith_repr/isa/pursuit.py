"""Reader-seeded sphere pursuit: find each quantity from a reader that reads it.

A reader of one quantity points into that quantity's subspace of the whitened reader space. From
the reader's direction, fit k-dimensional frames W (k = 1..kmax) that make the prompts' squared norm
inside the frame as constant as possible, scored against a Gaussian: `J_k = Var(|W^T y|^2) / (2k)`.
A Gaussian subspace has J = 1 for every k; a binary variable J = 0 at k = 1; a circle J = 0 at k = 2
(one of its axes alone scores 0.25); a simplex of K values J = 0 at k = K - 1; noise raises all of
them. The k with the lowest J is kept if J < `tol`; its frame is removed from the space, and the
next reader not yet explained seeds the next quantity. Readers are visited from the largest
variance down, so the structured readers lead.

"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Quantity:
    frame: np.ndarray  # (r, k) orthonormal frame in the whitened reader space
    J: float  # norm-constancy of the frame (0 = the prompts lie on a sphere)
    seed: int  # reader that seeded it


def parallel_rank(Hc: np.ndarray, sv: np.ndarray, reps: int = 5, seed: int = 0) -> int:
    """Horn's parallel analysis: keep the leading dimensions whose singular value beats the same
    rank after shuffling every reader independently over the prompts (which keeps each reader's
    distribution and destroys the shared structure)."""
    rng = np.random.default_rng(seed)
    null = np.zeros_like(sv)
    for _ in range(reps):
        P = np.stack([rng.permutation(col) for col in Hc.T], 1)
        null = np.maximum(null, np.linalg.svd(P, compute_uv=False))
    above = sv > null
    return int(np.argmin(above)) if not above.all() else len(sv)


def norm_var(Z: np.ndarray) -> float:
    k = Z.shape[1]
    return float(((Z**2).sum(1)).var() / (2 * k))


def fit_frame(
    U: np.ndarray, W0: np.ndarray, iters: int = 200, lr: float = 0.2
) -> tuple[np.ndarray, float]:
    """Projected gradient on orthonormal frames minimising Var(|U W|^2) / (2k), started at W0."""
    frame = W0.copy()
    k = frame.shape[1]
    for _ in range(iters):
        Z = U @ frame
        q = (Z**2).sum(1)
        G = 4 * U.T @ ((q - q.mean())[:, None] * Z) / (len(U) * 2 * k)
        Q, R = np.linalg.qr(frame - lr * G)
        frame = Q * np.sign(np.diag(R))
    return frame, norm_var(U @ frame)


def best_frame(U: np.ndarray, seed_dir: np.ndarray, kmax: int, rng: np.random.Generator,
               restarts: int = 4) -> tuple[np.ndarray, float]:  # fmt: skip
    """Lowest-J frame over k = 1..kmax, each started at the seed direction plus random others."""
    m = U.shape[1]
    best_W, best_J = seed_dir[:, None], norm_var(U @ seed_dir[:, None])
    for k in range(1, min(kmax, m) + 1):
        for _ in range(restarts if k > 1 else 1):
            W0 = np.linalg.qr(
                np.concatenate([seed_dir[:, None], rng.standard_normal((m, k - 1))], 1)
            )[0]
            W, J = fit_frame(U, W0)
            if best_J - 0.02 > J:  # a larger k must earn its keep
                best_W, best_J = W, J
    return best_W, best_J


def split(
    U: np.ndarray, W: np.ndarray, rng: np.random.Generator, tol: float, max_anti: float = 0.1
) -> list[np.ndarray]:
    """Split a frame into independent spheres, recursively.

    The union of two independent spheres is a sphere too (its norm is constant), so a frame may hold
    several quantities. For every size k1 < k, fit the most sphere-like k1-dim sub-frame A inside W
    and take B = its complement. Keep the split if both parts are sphere-like (J < tol) and their
    squared norms are not anti-correlated: the two halves of one circle have corr = -1, two
    independent spheres have corr ~ 0."""
    k = W.shape[1]
    if k < 2:
        return [W]
    V = U @ W
    best = None
    for k1 in range(1, k // 2 + 1):
        for _ in range(4):
            init = np.linalg.qr(rng.standard_normal((k, k1)))[0]
            part, JA = fit_frame(V, init)
            Bc = np.linalg.qr(np.concatenate([part, np.eye(k)], 1))[0][:, k1:k]
            JB = norm_var(V @ Bc)
            qa, qb = ((V @ part) ** 2).sum(1), ((V @ Bc) ** 2).sum(1)
            anti = -float(np.corrcoef(qa, qb)[0, 1]) if qa.std() > 1e-9 and qb.std() > 1e-9 else 0.0
            if tol > JA and tol > JB and anti < max_anti and (best is None or best[0] > JA + JB):
                best = (JA + JB, part, Bc)
    if best is None:
        return [W]
    _, part, Bc = best
    return split(U, W @ part, rng, tol, max_anti) + split(U, W @ Bc, rng, tol, max_anti)


def pursue(
    Y: np.ndarray,
    L: np.ndarray,
    order: np.ndarray,
    tol: float = 0.5,
    kmax: int = 4,
    explained: float = 0.5,
    seed: int = 0,
    Y_test: np.ndarray | None = None,
) -> list[Quantity]:
    """Y: (N, r) whitened scores; L: (n, r) reader loadings; order: readers to try as seeds.

    With `Y_test` (held-out prompts), frames are fitted on Y but accepted, and scored, on Y_test.
    A frame fitted to few distinct prompts can make their norms nearly equal by chance; only a real
    sphere keeps a low J on prompts it was not fitted to."""
    rng = np.random.default_rng(seed)
    r = Y.shape[1]
    basis = np.eye(r)  # basis of the space not yet explained
    found: list[Quantity] = []
    for c in order:
        if basis.shape[1] == 0:
            break
        lc = L[c]
        res = basis.T @ lc
        if (res**2).sum() < explained * (lc**2).sum():
            continue  # this reader is already mostly explained
        U = Y @ basis
        W, J_fit = best_frame(U, res / np.linalg.norm(res), kmax, rng)
        score = J_fit if Y_test is None else norm_var(Y_test @ basis @ W)
        if tol > score:
            for part in split(U, W, rng, tol):
                Jp = norm_var((Y if Y_test is None else Y_test) @ basis @ part)
                if Jp < tol:
                    found.append(Quantity(basis @ part, Jp, int(c)))
            Q = np.linalg.qr(np.concatenate([W, np.eye(basis.shape[1])], 1))[0]
            basis = basis @ Q[:, W.shape[1] : basis.shape[1]]
    return found


def assign(L: np.ndarray, found: list[Quantity]) -> np.ndarray:
    """(n, G + 1): share of each reader's loading energy in each quantity, last column = rest."""
    tot = np.maximum((L**2).sum(1), 1e-12)
    E = (
        np.stack([((L @ q.frame) ** 2).sum(1) for q in found], 1)
        if found
        else np.zeros((len(L), 0))
    )
    return np.concatenate([E, np.maximum(tot - E.sum(1), 0)[:, None]], 1) / tot[:, None]
