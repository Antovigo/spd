"""A library of candidate functions ("atoms") of the operand a = 1..100, grouped by quantity.

An atom is one function f: {1..100} -> R; a quantity is the variable an atom is a function of,
with an encoding (categorical value indicators, intervals, steps, a circle, a smooth ramp).
Every atom carries a description cost in bits, used by the MDL fit (`fit.py`).
"""

from dataclasses import dataclass

import numpy as np

A = np.arange(1, 101)
LOG2_100 = np.log2(100)


@dataclass(frozen=True)
class Atom:
    family: str  # the quantity + encoding, e.g. "units digit (a mod 10) indicator"
    name: str  # e.g. "a mod 10 = 3"
    f: np.ndarray  # (100,)
    bits: float  # cost of naming this atom within its family


def _ind(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.float64)


def build_library(max_window: int = 40) -> list[Atom]:
    L: list[Atom] = []
    for v in A:  # the token identity, passed on from the embedding
        L.append(Atom("token a = v", f"a = {v}", _ind(v == A), LOG2_100))
    for m in (2, 4, 5, 10, 20, 25, 50):  # residues
        for r in range(m):
            L.append(Atom(f"a mod {m}", f"a mod {m} = {r}", _ind(A % m == r), np.log2(m) + 3))
    for k in range(11):  # tens digit
        L.append(Atom("tens digit", f"floor(a/10) = {k}", _ind(k == A // 10), np.log2(11) + 3))
    for c in range(2, 101):  # thresholds: a >= c (magnitude, coarse)
        L.append(Atom("a >= c (step)", f"a >= {c}", _ind(c <= A), LOG2_100 + 1))
    for lo in range(1, 101):  # intervals (place code of the magnitude)
        for hi in range(lo + 1, min(101, lo + max_window)):
            L.append(Atom("a in [lo, hi] (interval)", f"a in [{lo}, {hi}]", _ind((lo <= A) & (hi >= A)),
                          2 * LOG2_100))  # fmt: skip
    for k in range(1, 51):  # circles: period 100 / k
        th = 2 * np.pi * k * A / 100
        L.append(Atom(f"circle k={k}", f"cos(2 pi {k} a / 100)", np.cos(th), np.log2(50) + 3))
        if k < 50:
            L.append(Atom(f"circle k={k}", f"sin(2 pi {k} a / 100)", np.sin(th), np.log2(50) + 3))
    for nm, f in (
        ("a", A / 100.0),
        ("log a", np.log(A) / np.log(100)),
        ("sqrt a", np.sqrt(A) / 10),
    ):
        L.append(Atom("smooth magnitude", nm, f, 4.0))
    return L
