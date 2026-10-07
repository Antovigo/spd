"""Decade parity vs period-20 circle at every site (token a), by fitting each first.

After a base (magnitude curve + 2 smooth dims + units digit, a mod 10 harmonics j = 1..5), in the
reader metric:
* sq: the decade-parity square wave s(a) = (-1)^floor(a/10) (1 dim; flips at a = 10, 20, ...);
* circ: the period-20 circle cos, sin(2 pi a / 20) (2 dims);
* P20: all functions with f(a + 10) = -f(a) on a = 1..100 (10 dims: [a mod 20 = r] - [a mod 20 =
  r + 10], r = 0..9), which contains both.
Order 1: sq, circ, P20; order 2: circ, sq, P20. Each increment is reported with its chance level
(d / df of the remaining residual). Also the increment of shifted square waves
s_c(a) = (-1)^floor((a - c) / 10), c = 0..9, each fit alone after the base (c = 0 is s).

    python v2_period20_test.py
"""

import json
from pathlib import Path

import numpy as np
from v2 import A, Q, fourier, reader_dirs, span_basis, spline, stagewise, stream
from v2_layers import site_list

HERE = Path(__file__).parent


def base() -> list[Q]:
    return [Q("magnitude", "curve", spline(), 1), Q("smooth", "curve", spline(), 2),
            Q("units", "fixed", fourier(10, [1, 2, 3, 4, 5]))]  # fmt: skip


def sq(c: int = 0) -> np.ndarray:
    return ((-1.0) ** np.floor((A - c) / 10))[:, None]


def P20() -> np.ndarray:
    return np.stack([(r == A % 20).astype(float) - (r + 10 == A % 20) for r in range(10)], 1)


def increments(Z: np.ndarray, W: np.ndarray, H: list[Q], used: int) -> list[tuple[float, float]]:
    """(increment, chance) of each quantity in H after the base, reader metric."""
    mu, parts, _ = stagewise(Z, base() + H)
    Yc = (Z - mu) @ W
    tot = (Yc**2).sum()
    rest = Yc - sum(p @ W for p in parts[:3])
    out = []
    for q, p in zip(H, parts[3:], strict=True):
        d = int(np.linalg.matrix_rank(q.Phi - q.Phi.mean(0)))
        r2 = (rest**2).sum()
        py = p @ W
        out.append((float((py**2).sum() / tot), float(min(d, 99 - used) / (99 - used) * r2 / tot)))
        rest = rest - py
        used += d
    return out


def main() -> None:
    res = []
    for b, pt, lpos in site_list():
        Vt, names, _ = reader_dirs(b, pt)
        if len(names) == 0:
            continue
        Qr = span_basis(Vt)
        Z, W = stream(lpos) @ Qr, Qr.T @ Vt.T
        used = 1 + 2 + 9
        o1 = increments(
            Z,
            W,
            [
                Q("sq", "fixed", sq()),
                Q("circ", "fixed", fourier(20, [1])),
                Q("P20", "fixed", P20()),
            ],
            used,
        )
        o2 = increments(
            Z,
            W,
            [
                Q("circ", "fixed", fourier(20, [1])),
                Q("sq", "fixed", sq()),
                Q("P20", "fixed", P20()),
            ],
            used,
        )
        shifts = [increments(Z, W, [Q("sq", "fixed", sq(c))], used)[0][0] for c in range(10)]
        res.append({"site": f"L{b}.{pt}", "sq_first": o1, "circ_first": o2, "shifted_sq": shifts})
        f = lambda t: f"{t[0]:.3f}/{t[1]:.3f}"  # noqa: E731
        print(f"L{b}.{pt}: sq {f(o1[0])} then circ {f(o1[1])} then rest-of-P20 {f(o1[2])} | "
              f"circ {f(o2[0])} then sq {f(o2[1])} then rest {f(o2[2])} | best shift c={int(np.argmax(shifts))}", flush=True)  # fmt: skip
    (HERE / "period20_test.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
