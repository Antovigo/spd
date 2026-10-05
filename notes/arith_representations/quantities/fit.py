"""MDL sparse fit of every reader of a site on the atom library.

Per reader x (100,), greedy orthogonal matching pursuit: start from the constant, repeatedly add
the atom that most reduces the residual sum of squares RSS, and keep it only if the total
description length drops:

    DL = (N / 2) log2(RSS / N)  +  sum over used atoms of (atom bits + (1/2) log2 N)

(N = 100 values of a; the first term is the Gaussian code length of the residual up to a constant,
the second names each atom and codes its weight). A family already used by another reader of the
site costs `REUSE` bits less per atom (a quantity is declared once per site and read many times).
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from library import Atom

N = 100
WEIGHT_BITS = 0.5 * np.log2(N)
REUSE = 4.0
RSS_FLOOR = 1e-6


@dataclass
class ReaderFit:
    atoms: list[int] = field(default_factory=list)
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fitted: np.ndarray = field(default_factory=lambda: np.zeros(N))
    r2: float = 0.0


def _lstsq(F: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, *_ = np.linalg.lstsq(F, x, rcond=None)
    return c, F @ c


def dl(rss: float, bits: float, n_atoms: int) -> float:
    return 0.5 * N * np.log2(max(rss, RSS_FLOOR) / N) + bits + n_atoms * WEIGHT_BITS


def fit_reader(x: np.ndarray, lib: list[Atom], Fall: np.ndarray, used_families: Counter,
               max_atoms: int = 12) -> ReaderFit:  # fmt: skip
    cols = [np.ones(N)]
    chosen: list[int] = []
    bits = 0.0
    _, fit = _lstsq(np.stack(cols, 1), x)
    best_dl = dl(float(((x - fit) ** 2).sum()), bits, 0)
    Fn = Fall / (np.linalg.norm(Fall, axis=0, keepdims=True) + 1e-12)
    for _ in range(max_atoms):
        res = x - fit
        # candidate atoms ranked by |correlation with the residual| (after projecting out the chosen ones)
        Q, _ = np.linalg.qr(np.stack(cols, 1))
        Fp = Fn - Q @ (Q.T @ Fn)
        nrm = np.linalg.norm(Fp, axis=0)
        score = np.abs(res @ Fp) / np.where(nrm > 1e-8, nrm, np.inf)
        best = None
        for j in np.argsort(-score)[:25]:
            c = lib[j]
            b = bits + c.bits - (REUSE if used_families[c.family] else 0.0)
            _, f2 = _lstsq(np.stack(cols + [Fall[:, j]], 1), x)
            d = dl(float(((x - f2) ** 2).sum()), b, len(chosen) + 1)
            if best is None or d < best[0]:
                best = (d, j, b, f2)
        if best is None or best[0] >= best_dl:
            break
        best_dl, j, bits, fit = best
        cols.append(Fall[:, j])
        chosen.append(j)
    coef, fit = _lstsq(np.stack(cols, 1), x)
    r2 = 1 - ((x - fit) ** 2).sum() / max(((x - x.mean()) ** 2).sum(), 1e-12)
    return ReaderFit(chosen, coef, fit, float(r2))


def fit_site(X: np.ndarray, lib: list[Atom], passes: int = 2) -> list[ReaderFit]:
    """Fit every reader; the second pass refits with the families the first pass used."""
    Fall = np.stack([a.f for a in lib], 1)
    used: Counter = Counter()
    fits: list[ReaderFit] = []
    for _ in range(passes):
        fits = [fit_reader(X[:, j], lib, Fall, used) for j in range(X.shape[1])]
        used = Counter(lib[i].family for f in fits for i in f.atoms)
    return fits
