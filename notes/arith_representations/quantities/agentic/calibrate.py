"""Calibration of B_BITS (not an agent tool): at a few sites, greedy forward selection from a generic
candidate family by the score J, for several values of B; reports the held-out R^2 reached.

Generic family: for each variable x the position depends on (a; and b, a + b, a - b at t >= 3):
cubic splines with 4, 8, 16 functions, circles of period 10 and 100, classes of x mod 10 and of
x // 10; at t >= 2 also op and every candidate on subtraction prompts only.

    python calibrate.py
"""

import os
import sys

B_VALUES = (100, 300, 1000, 3000, 10000)
SITES = [
    (1, "L5.mlp"),
    (1, "L20.attn"),
    (2, "L10.mlp"),
    (2, "L25.attn"),
    (3, "L8.mlp"),
    (3, "L20.mlp"),
    (4, "L18.mlp"),
    (4, "L25.attn"),
]


def family(t: int) -> list[tuple[str, str]]:
    xs = (
        {"a": (1, 100)}
        if t <= 2
        else {"a": (1, 100), "b": (1, 100), "(a+b)": (2, 200), "(a-b)": (-99, 99)}
    )
    out = []
    for x, (lo, hi) in xs.items():
        out += [(f"spline {x} {n}", f"spline({x}, {lo}, {hi}, {n})") for n in (4, 8, 16)]
        out += [(f"circle {x} {p}", f"circle({x}, {p})") for p in (10, 100)]
        out += [(f"{x} mod 10", f"classes({x} % 10)"), (f"{x} // 10", f"classes({x} // 10)")]
    if t >= 2:
        out = [("op", "scalar(op)")] + out + [(f"sub {n}", f"gate(op == 1, {f})") for n, f in out]
    return out


def main() -> None:
    import qa

    for t, label in SITES:
        s = qa.Site(t, label)
        Q = family(t)
        sc = qa.Scorer(s, Q)
        res = []
        for B in B_VALUES:
            qa.B_BITS = B
            A: list[int] = []
            cur = sc.J(A)[0]
            while True:
                best = min(
                    ((sc.J(A + [i])[0], i) for i in range(len(Q)) if i not in A), default=None
                )
                if best is None or best[0] >= cur:
                    break
                cur, A = best[0], A + [best[1]]
            _, mb, r2 = sc.J(A)
            res.append((B, round(r2, 3), len(A), round(mb)))
        print(
            f"t={t} {label} N={s.N} k_eff={s.k_eff}: (B, held-out R2, quantities, model bits) {res}",
            flush=True,
        )


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
