"""Hypothesis sets, one function per iteration of the loop (each returns a list of Quantity).

Notation: a = 1..100 is the operand at the a token; theta_k(a) = 2 pi k a / 100 for an integer
frequency k; [P] is 1 if P holds, else 0.
"""

import numpy as np
from hyp_fit import Quantity

A = np.arange(1, 101)


def ind(mask: np.ndarray) -> np.ndarray:
    return mask.astype(float)[:, None]


def circle(k: int) -> np.ndarray:
    th = 2 * np.pi * k * A / 100
    return np.stack([np.cos(th), np.sin(th)], 1)


def onehot_classes(labels: np.ndarray) -> np.ndarray:
    """Indicators of each class but the first (the constant absorbs it)."""
    vals = np.unique(labels)
    return np.stack([(labels == v).astype(float) for v in vals[1:]], 1)


# ---------------------------------------------------------------- L0 attention input (site 1)
def s1_v1() -> list[Quantity]:
    """From the raw plots: a smooth magnitude, the digit count, round numbers."""
    return [
        Quantity("magnitude (log a, a)", np.stack([np.log(A), A / 100], 1), 6),
        Quantity("digit count ([a<=9], [a=100])", np.hstack([ind(A <= 9), ind(A == 100)]), 6),
        Quantity("round (a mod 10 = 0)", ind(A % 10 == 0), 6),
        Quantity("half-round (a mod 5 = 0)", ind(A % 5 == 0), 6),
    ]


def spline(x: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """Cubic B-spline basis on x (knots at quantiles), minus one column (the constant absorbs it)."""
    from scipy.interpolate import BSpline

    inner = np.quantile(x, np.linspace(0, 1, n_basis - degree + 1))
    t = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
    B = BSpline.design_matrix(x, t, degree).toarray()
    return B[:, 1:]


def s1_v2() -> list[Quantity]:
    """v1, with the magnitude as a smooth curve: a cubic spline in log a (6 dims)."""
    return [
        Quantity("magnitude (spline in log a)", spline(np.log(A), 7), 6),
        Quantity("digit count ([a<=9], [a=100])", np.hstack([ind(A <= 9), ind(A == 100)]), 6),
        Quantity("round (a mod 10 = 0)", ind(A % 10 == 0), 6),
        Quantity("half-round (a mod 5 = 0)", ind(A % 5 == 0), 6),
    ]


# ---------------------------------------------------------------- shared menu quantities
def token_identity() -> Quantity:
    """The embedding's binary quantities passed on: one indicator per value of a."""
    return Quantity("token identity (a = v)", np.eye(100), 4, True, tuple(f"a={v}" for v in A))


def residue_menu(m: int) -> Quantity:
    return Quantity(f"a mod {m}", np.stack([(A % m == r).astype(float) for r in range(m)], 1), 6, True,
                    tuple(f"a mod {m} = {r}" for r in range(m)))  # fmt: skip


def interval_menu(max_width: int = 40) -> Quantity:
    cols, labs = [], []
    for lo in range(1, 101):
        for hi in range(lo + 1, min(101, lo + max_width)):
            cols.append(((lo <= A) & (hi >= A)).astype(float))
            labs.append(f"[{lo},{hi}]")
    return Quantity("interval of a (place code)", np.stack(cols, 1), 6, True, tuple(labs))


def step_menu() -> Quantity:
    return Quantity("threshold a >= c", np.stack([(c <= A).astype(float) for c in range(2, 101)], 1), 6, True,
                    tuple(f"a>={c}" for c in range(2, 101)))  # fmt: skip


def s1_v3() -> list[Quantity]:
    """v2 + the token identity as a menu (replaces the ad-hoc outlier tokens)."""
    return s1_v2() + [token_identity()]


# ---------------------------------------------------------------- L0 MLP input (site 2)
def s2_v1() -> list[Quantity]:
    """From the raw plots: everything at site 1, plus units-digit indicators, intervals (bumps),
    thresholds, and the token identity."""
    return s1_v2() + [token_identity(), residue_menu(10), interval_menu(), step_menu()]


def residues_all(
    ms: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 25, 50),
) -> Quantity:
    """Residue indicators [a mod m = r] for several moduli m, one menu (the quantity: divisibility
    / units-digit-like classes)."""
    cols, labs = [], []
    for m in ms:
        for r in range(m):
            cols.append((A % m == r).astype(float))
            labs.append(f"a mod {m} = {r}")
    return Quantity("residue class (a mod m = r)", np.stack(cols, 1), 6, True, tuple(labs))


def residues_ramped(ms: tuple[int, ...] = (3, 5, 10)) -> Quantity:
    """[a mod m = r] * (a / 100): a residue class whose amplitude grows linearly with a."""
    cols, labs = [], []
    for m in ms:
        for r in range(m):
            cols.append((A % m == r) * A / 100.0)
            labs.append(f"[a mod {m} = {r}] * a/100")
    return Quantity("residue class x magnitude ramp", np.stack(cols, 1), 6, True, tuple(labs))


def bump_menu(widths: tuple[float, ...] = (1.0, 2.0, 3.5, 6.0, 10.0)) -> Quantity:
    """Soft bumps exp(-(a - c)^2 / (2 w^2)): a place code of the magnitude (centre c, width w)."""
    cols, labs = [], []
    for w in widths:
        for c in range(1, 101):
            cols.append(np.exp(-((A - c) ** 2) / (2 * w**2)))
            labs.append(f"bump c={c} w={w}")
    return Quantity("bump in a (place code)", np.stack(cols, 1), 6, True, tuple(labs))


def soft_step_menu(widths: tuple[float, ...] = (0.5, 2.0, 5.0)) -> Quantity:
    """Soft thresholds sigmoid((a - c) / w)."""
    cols, labs = [], []
    for w in widths:
        for c in range(2, 101):
            cols.append(1 / (1 + np.exp(-(A - c + 0.5) / w)))
            labs.append(f"step a>{c - 0.5} w={w}")
    return Quantity("soft threshold in a", np.stack(cols, 1), 6, True, tuple(labs))


def s2_v2() -> list[Quantity]:
    """v1 with: residues mod 3..12 (gate.c109 fires on multiples of 3), residue x ramp (amplitudes
    drift with a), soft bumps and soft steps (bumps had soft edges)."""
    return s1_v2()[:2] + [token_identity(), residues_all(), residues_ramped(), interval_menu(),
                          bump_menu(), soft_step_menu()]  # fmt: skip
