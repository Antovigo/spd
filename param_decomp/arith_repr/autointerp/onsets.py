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
