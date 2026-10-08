"""Candidate quantities: named feature maps over a position's domain.

Variables: op (0 = +, 1 = -), operands a and b (1..100), and derived: s = a + b, d = a - b, the
result r = s if op = + else d, the units digits u_a = a mod 10, u_b = b mod 10, carry
[u_a + u_b >= 10] and borrow [u_a < u_b]. Only variables the position can depend on are used:
t = 1: a; t = 2: op, a; t >= 3: all.

Every candidate is a Cand (name, variable, Phi (D, d)). The agent adds its own candidates with
`Cand(name, var, Phi)` (e.g. a one-hot over listed values, a gated or restricted version of an
accepted quantity).
"""

import importlib
from dataclasses import dataclass

import numpy as np
from qdata import Position
from scipy.interpolate import BSpline


@dataclass
class Cand:
    name: str
    var: str  # the variable(s) it is a function of, e.g. "a", "op", "r", "op,a"
    Phi: np.ndarray  # (D, d)
    value: np.ndarray | None = (
        None  # (D,) value it encodes, for display (default Phi[:, 0] if d = 1)
    )
    requires: str | None = None  # a candidate that may enter a list only while this one is in it

    def shown(self) -> np.ndarray:
        """Natural value per domain row: the variable for curves and place codes, x mod P for a
        circle of period P, the class label for classes, the scalar itself otherwise."""
        return self.Phi[:, 0] if self.value is None else self.value

    @property
    def d(self) -> int:
        P = self.Phi - self.Phi.mean(0)
        return int(np.linalg.matrix_rank(P)) if P.size else 0


def spline(x: np.ndarray, lo: float, hi: float, n_basis: int, degree: int = 3) -> np.ndarray:
    inner = np.linspace(lo, hi, n_basis - degree + 1)
    t = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
    return BSpline.design_matrix(np.clip(x, lo, hi), t, degree).toarray()


def circle(x: np.ndarray, period: float) -> np.ndarray:
    th = 2 * np.pi * x / period
    return np.stack([np.cos(th), np.sin(th)], 1)


def classes(lab: np.ndarray) -> np.ndarray:
    return np.stack([(lab == v).astype(float) for v in np.unique(lab)], 1)


def ind(m: np.ndarray) -> np.ndarray:
    return m.astype(float)[:, None]


def two_adic(x: np.ndarray, cap: int = 4) -> np.ndarray:
    v = np.zeros_like(x)
    y = x.copy()
    for _ in range(cap):
        m = (y % 2 == 0) & (y != 0)
        v[m] += 1
        y = np.where(m, y // 2, y)
    return v.astype(float)


def bumps(x: np.ndarray, lo: int, hi: int, step: int, width: float) -> np.ndarray:
    cs = np.arange(lo + step // 2, hi + 1, step)
    return np.stack([np.exp(-((x - c) ** 2) / (2 * width**2)) for c in cs], 1)


def operand_family(x: np.ndarray, v: str) -> list[Cand]:
    """Quantities of an operand x in 1..100 (named after the variable v)."""
    tens = np.where((x >= 10) & (x <= 99), x // 10, 0)
    return [
        Cand(f"log {v}", v, np.log(x)[:, None]),
        Cand(
            f"magnitude of {v}: smooth curve in log {v} (5)",
            v,
            spline(np.log(x), 0, np.log(100), 6),
            x,
        ),
        Cand(f"one digit [{v} <= 9]", v, ind(x <= 9)),
        Cand(f"{v} = 100", v, ind(x == 100)),
        Cand(f"circle period 100 of {v}", v, circle(x, 100), x % 100),
        Cand(f"circle period 50 of {v}", v, circle(x, 50), x % 50),
        Cand(f"circle period 20 of {v}", v, circle(x, 20), x % 20),
        Cand(f"units digit circle of {v} (period 10)", v, circle(x, 10), x % 10),
        Cand(f"parity of {v}", v, ind(x % 2 == 1)),
        Cand(f"{v} mod 5 classes", v, classes(x % 5), x % 5),
        Cand(f"units digit classes of {v}", v, classes(x % 10), x % 10),
        Cand(f"tens digit classes of {v} (10..99)", v, classes(tens), tens),
        Cand(f"2-adic valuation of {v}", v, two_adic(x)[:, None]),
        Cand(f"repdigit {v} (11, 22, .., 99)", v, ind((x % 11 == 0) & (x <= 99))),
        Cand(f"{v} mod 3 classes", v, classes(x % 3), x % 3),
        Cand(f"place code of {v}: bumps every 5 (width 2.5)", v, bumps(x, 0, 100, 5, 2.5), x),
    ]


def result_family(r: np.ndarray, v: str, lo: int, hi: int) -> list[Cand]:
    """Quantities of an integer result r in lo..hi (s, d or the op-dependent result)."""
    return [
        Cand(f"magnitude of {v}: smooth curve (5)", v, spline(r.astype(float), lo, hi, 6), r),
        Cand(f"sign of {v} [{v} < 0]", v, ind(r < 0)),
        Cand(f"units digit classes of {v}", v, classes(np.mod(r, 10)), np.mod(r, 10)),
        Cand(f"units digit circle of {v} (period 10)", v, circle(r, 10), np.mod(r, 10)),
        Cand(f"circle period 100 of {v}", v, circle(r, 100), np.mod(r, 100)),
        Cand(f"tens digit classes of {v}", v, classes(np.abs(r) // 10), np.abs(r) // 10),
        Cand(
            f"place code of {v}: bumps every 10 (width 5)",
            v,
            bumps(r.astype(float), lo, hi, 10, 5.0),
            r,
        ),
    ]


def gated(c: Cand, op: np.ndarray, requires: str | None = None) -> Cand:
    """c on subtraction prompts only (an op-dependent encoding of c's variable)."""
    return Cand(f"[op = -] x {c.name}", f"op,{c.var}", c.Phi * (op == 1)[:, None],
                np.where(op == 1, c.shown(), np.nan), requires)  # fmt: skip


def pool(pos: Position) -> list[Cand]:
    t, op, a, b = pos.t, pos.op, pos.a, pos.b
    out = operand_family(a, "a")
    if t >= 2:
        out = [Cand("op [op = -]", "op", ind(op == 1))] + out
        for c in operand_family(a, "a"):
            if c.name.startswith(("magnitude", "units digit classes", "tens digit classes")):
                out.append(gated(c, op))
    if t >= 3:
        out += operand_family(b, "b")
        for c in operand_family(b, "b"):
            if c.name.startswith(("magnitude", "units digit classes", "tens digit classes")):
                out.append(gated(c, op))
        s, d = a + b, a - b
        r = np.where(op == 0, s, d)
        out += result_family(r, "r", -99, 200)
        out += [Cand("circle period 10 of s = a + b", "s", circle(s, 10), s % 10), Cand("circle period 100 of s = a + b", "s", circle(s, 100), s % 100),
                Cand("circle period 10 of d = a - b", "d", circle(d, 10), np.mod(d, 10)), Cand("circle period 100 of d = a - b", "d", circle(d, 100), np.mod(d, 100)),
                Cand("units digit classes of s = a + b", "s", classes(s % 10), s % 10), Cand("units digit classes of d = a - b", "d", classes(np.mod(d, 10)), np.mod(d, 10)),
                Cand("carry [u_a + u_b >= 10]", "u_a,u_b", ind(a % 10 + b % 10 >= 10)),
                Cand("borrow [u_a < u_b]", "u_a,u_b", ind(a % 10 < b % 10)),
                Cand("sign of a - b [a < b]", "a,b", ind(a < b))]  # fmt: skip
    return out


HINT_OPERATIONS = ("gate_by_op",)


def apply_hints(pos: Position, cands: list[Cand], hints: list[dict]) -> list[Cand]:
    """Candidates added by the researcher's hints (hints.json). Hints are operations on quantities,
    not quantities: "gate_by_op" adds, for every candidate that is not about op and not already
    op-gated, its subtraction-only version, which may enter a site's list only while the candidate
    itself is in it (t >= 2)."""
    out, names = [], {c.name for c in cands}
    for h in hints:
        assert h["operation"] in HINT_OPERATIONS, h["operation"]
        if h["operation"] == "gate_by_op" and pos.t >= 2:
            for c in cands:
                name = f"[op = -] x {c.name}"
                if c.var != "op" and not c.name.startswith("[op = -]") and name not in names:
                    out.append(gated(c, pos.op, requires=c.name))
                    names.add(name)
    return out


def candidates(
    pos: Position, extra: str | None = None, hints: list[dict] | None = None
) -> list[Cand]:
    """The standard pool, the agent's own candidates (`extra`: "module.function", a function
    Position -> list[Cand]) and the candidates the hints add."""
    cands = pool(pos)
    if extra:
        mod, fn = extra.rsplit(".", 1)
        cands += getattr(importlib.import_module(mod), fn)(pos)
    return cands + apply_hints(pos, cands, hints or [])
