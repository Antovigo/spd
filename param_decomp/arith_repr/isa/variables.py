"""How quantities relate: is one quantity a function of another?

Quantities can carry the same information from orthogonal subspaces: a mod 10 is a function of
a mod 50. `cond_r2` measures, without labels, how much one quantity is a function of another (or of
a pair, for derived codes such as a + b). Mutual information was tried first and dropped: on
near-deterministic data a small shared impurity already gives a large value."""

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]


def cond_r2(Zi: np.ndarray, Zj: np.ndarray, k: int = 20, n: int = 4000, seed: int = 0) -> float:
    """How much of Zi is a function of Zj: 1 - E|Zi - E[Zi | Zj]|^2 / Var(Zi), with the conditional
    mean estimated by averaging Zi over the k nearest neighbours of each prompt in Zj's coordinates
    (the prompt itself excluded). 1 = Zi is determined by Zj; 0 = Zj says nothing about Zi. A small
    shared impurity gives a small value, unlike mutual information on near-deterministic data."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Zi), min(n, len(Zi)), replace=False)
    A, B = Zi[idx], Zj[idx] / Zj[idx].std(0)
    nb = cKDTree(B).query(B, k + 1)[1][:, 1:]
    pred = A[nb].mean(1)
    Ac = A - A.mean(0)
    return float(1 - ((A - pred) ** 2).sum() / max((Ac**2).sum(), 1e-12))


def r2_matrix(Zs: list[np.ndarray]) -> np.ndarray:
    """R[i, j] = cond_r2(Z_i, Z_j): how much quantity i is a function of quantity j."""
    G = len(Zs)
    R = np.eye(G)
    for i in range(G):
        for j in range(G):
            if i != j:
                R[i, j] = cond_r2(Zs[i], Zs[j])
    return R
