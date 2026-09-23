"""Describe where a component is ON along one quantity, from its on-rate profile.

`describe(values, rate)`: the on-set `S = {v : rate(v) > 0.5}` summarised as the COARSEST period
`tau` (divisor of 100) whose residue classes explain the profile (group R^2 >= `R2_MIN`), the
residues `R` that are on, and whether `R` is an arc (one cyclic run) of Z/tau. `tau = 0` means
no period explains it (then `R` is the on-set itself and `arc` says whether it is one run on the
number line)."""

from typing import Any

import numpy as np

TAUS = (2, 4, 5, 10, 20, 25, 50, 100)
R2_MIN = 0.8


def cyclic_runs(mask: np.ndarray) -> int:
    if mask.all() or not mask.any():
        return int(mask.any())
    return int(np.sum(mask & ~np.roll(mask, 1)))


def linear_runs(mask: np.ndarray) -> int:
    return int(np.sum(mask & ~np.concatenate([[False], mask[:-1]])))


def describe(values: np.ndarray, rate: np.ndarray) -> dict[str, Any]:
    s = rate > 0.5
    out = {"n_on": int(s.sum()), "max_rate": float(rate.max())}
    var = float(((rate - rate.mean()) ** 2).sum())
    if var < 1e-12:
        return out | {"tau": -1, "R": [], "arc": False, "r2": 0.0}
    for tau in TAUS:
        cls = values % tau
        means = np.array([rate[cls == r].mean() if (cls == r).any() else 0.0 for r in range(tau)])
        r2 = 1.0 - float(((rate - means[cls]) ** 2).sum()) / var
        if r2 >= R2_MIN:
            R = np.flatnonzero(means > 0.5)
            present = np.zeros(tau, bool)
            present[R] = True
            return out | {"tau": tau, "R": R.tolist(), "arc": cyclic_runs(present) == 1, "r2": r2}
    return out | {"tau": 0, "R": values[s].tolist(), "arc": linear_runs(s) == 1, "r2": 1.0}


def fmt(d: dict[str, Any], name: str) -> str:
    if d.get("tau", -1) == -1:
        return f"{name}: flat"
    R = d["R"]
    rs = (
        f"{R[0]}..{R[-1]}"
        if d["arc"] and len(R) > 2 and list(range(R[0], R[-1] + 1)) == R
        else str(R[:12])
    )
    if d["tau"] == 0:
        return f"{name} in {rs}" + ("" if d["arc"] else " (irregular)")
    return f"{name} mod {d['tau']} in {rs}" + (" (arc)" if d["arc"] else "")


def ranges(xs: list[int]) -> str:
    """`[1, 2, 3, 7, 9, 10]` -> `1..3, 7, 9..10` (every element kept)."""
    if not xs:
        return "{}"
    xs = sorted(xs)
    parts, start, prev = [], xs[0], xs[0]
    for x in [*xs[1:], None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        parts.append(f"{start}" if start == prev else f"{start}..{prev}")
        if x is not None:
            start = prev = x
    return ", ".join(parts)


def describe_at(values: np.ndarray, rate: np.ndarray, name: str, tau: int) -> str:
    """The on-set along `name` at a GIVEN period (`tau = 0`: the values themselves): the classes
    whose mean on-rate is > 0.5, every one listed, then the coarsest period that already explains
    the profile (R^2 >= 0.8) when it is coarser than `tau`."""
    cls = values % tau if tau else values
    keys = np.unique(cls)
    means = np.array([rate[cls == k].mean() for k in keys])
    on = keys[means > 0.5].tolist()
    head = f"{name} mod {tau}" if tau else name
    txt = (
        f"{head} in {{{ranges(on)}}}"
        if on
        else f"{head}: no class above 0.5 (max {means.max():.2f})"
    )
    d = describe(values, rate)
    span = int(values.max() - values.min())
    redundant = d.get("tau", -1) == 100 and span < 100  # a, b: mod 100 is the value itself
    if d.get("tau", -1) > 0 and (tau == 0 or d["tau"] < tau) and not redundant:
        txt += f" [coarser: {name} mod {d['tau']} in {{{ranges(d['R'])}}}, R2 {d['r2']:.2f}]"
    return txt
