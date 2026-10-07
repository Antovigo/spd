"""Hypothesis sets for the layer sweep (each returns a list of v2.Q, the last one the one-hot
remainder)."""

import numpy as np
from v2 import A, Q, classes, fourier, spline

TENS = np.where((A >= 10) & (A <= 99), A // 10, 0)


def L0_set() -> list[Q]:
    """The set fit at L0 (V2 section of the README)."""
    return [
        Q("magnitude curve (1 dir)", "curve", spline(), 1),
        Q("smooth in log a, 2 more dirs", "curve", spline(), 2),
        Q("one digit [a <= 9]", "fixed", (A <= 9).astype(float)[:, None]),
        Q("a mod 10: circle (period 10)", "fixed", fourier(10, [1])),
        Q("a mod 10: harmonics 2-5", "fixed", fourier(10, [2, 3, 4, 5])),
        Q("a mod 3", "fixed", classes(A % 3)),
        Q("tens digit (10..99)", "fixed", classes(TENS)),
        Q("place code: smooth in a (25 dims)", "fixed", spline(25, log=False)),
        Q("one-hot remainder", "fixed", np.eye(100)),
    ]


def residues_set() -> list[Q]:
    """L0 set + residue classes a mod m for m = 4, 6, 7, 8, 9 (the residual spectra after the L0
    set peak at k = 25, 14/15/28, 12/37, 17, 22: the DFT signatures of these moduli), placed after
    the units digit and before the tens digit and the place code."""
    H = L0_set()
    extra = [Q(f"a mod {m}", "fixed", classes(A % m)) for m in (4, 6, 7, 8, 9)]
    return H[:6] + extra + H[6:]


def v2_adic(a: np.ndarray) -> np.ndarray:
    """2-adic valuation: the exponent of the largest power of 2 dividing a."""
    v = np.zeros_like(a)
    x = a.copy()
    while (m := (x % 2 == 0)).any():
        v[m] += 1
        x = np.where(m, x // 2, x)
    return v


def two_adic_set() -> list[Q]:
    """L0 set + the 2-adic valuation of a: the shared residual direction at L14+ is positive on
    every multiple of 8, negative on the even numbers = 2 mod 4, near 0 on odd numbers. Two forms,
    in order: one direction carrying min(v2(a), 4), then the classes odd / 2 mod 4 / 4 mod 8 /
    0 mod 8 (2 more dims)."""
    H = L0_set()
    v = np.minimum(v2_adic(A), 4).astype(float)
    cls = np.minimum(v2_adic(A), 3)
    extra = [Q("2-adic valuation min(v2(a), 4) (1 dir)", "fixed", v[:, None]),
             Q("2-adic classes (odd, 2|4|8: 2 more dims)", "fixed", classes(cls))]  # fmt: skip
    return H[:5] + extra + H[5:]


def final_set() -> list[Q]:
    """L0 set + the two features found in the residuals of the layer sweep, fit last before the
    remainder: the 2-adic valuation (one direction, min(v2(a), 4)) and repdigits ([a in {11, 22,
    ..., 99}], one direction)."""
    H = L0_set()
    v = np.minimum(v2_adic(A), 4).astype(float)
    rep = ((A % 11 == 0) & (A <= 99)).astype(float)
    extra = [Q("2-adic valuation (1 dir)", "fixed", v[:, None]),
             Q("repdigit [a in 11, 22, .., 99] (1 dir)", "fixed", rep[:, None])]  # fmt: skip
    return H[:-1] + extra + H[-1:]
