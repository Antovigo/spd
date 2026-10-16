"""Per-block, per-position summary of the variance results (report_variance.md).

All weights are a = Cov(Q, t_{d,w}) ||U_{d,w}||^2 (plan §8), summed over non-excluded writers.
Writes <src>/results/summary.json.

    python summarize.py [--src DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ids import component_ids, label
from variance import POSITIONS, SRC


def k_for(a: np.ndarray, frac: float) -> int:
    """Smallest number of entries whose |a| reach `frac` of sum |a|."""
    m = np.sort(np.abs(a))[::-1]
    return int(np.searchsorted(np.cumsum(m) / m.sum(), frac) + 1)


def summarize(src: Path, layer: int, tau: int) -> dict | None:
    d = src / "results" / f"L{layer}"
    if not (d / f"J_pos{tau}.npz").exists():
        return None
    z, j = np.load(d / f"variance_pos{tau}.npz"), np.load(d / f"J_pos{tau}.npz")
    ids = component_ids(src, layer)
    ok = ~z["excluded"]
    au, ag, rv = z["a_Mu"][ok], z["a_Mg"][ok], z["resid_var"][ok]
    a_all = np.concatenate([au, ag], 1)
    V = float(rv.sum())
    order = np.argsort(rv)[::-1]
    k90 = np.array([k_for(r, 0.9) for r in a_all])
    wmed = float(k90[np.argsort(k90)][np.searchsorted(np.cumsum(rv[np.argsort(k90)]) / V, 0.5)])
    okj = ~j["excluded"]
    aJ, aU, aU0 = j["a_J"][okj], j["a_Mu"][okj], j["a_Mu0"][okj]
    cols = np.flatnonzero(ok)
    top = []
    for w in order[:3]:
        r = a_all[w] / rv[w]
        best = np.argsort(np.abs(r))[::-1][:4]
        U = au.shape[1]
        top.append({"writer": label(layer, "down", cols[w], ids), "resid_var_frac": float(rv[w] / V),
                    "up_share": float(au[w].sum() / rv[w]),
                    "readers": [(label(layer, "up", b, ids) if b < U else
                                 label(layer, "gate", b - U, ids), round(float(r[b]), 3))
                                for b in best]})  # fmt: skip
    return {"layer": layer, "tau": tau, "D": int(z["excluded"].size), "excluded": int((~ok).sum()),
            "flagged": int(z["flagged"].sum()), "resid_var_total": V,
            "top5_frac": float(rv[order[:5]].sum() / V), "up_frac": float(au.sum() / V),
            "k90_weighted_median": wmed, "k90_median": float(np.median(k90)),
            "n_readers": int(a_all.shape[1]),
            "negative_frac": float(np.clip(-a_all, 0, None).sum() / np.abs(a_all).sum()),
            "routing_frac_of_up": float(aJ.sum() / aU.sum()), "mean_gate_frac_of_up": float(aU0.sum() / aU.sum()),
            "kl_note": None, "top_writers": top}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    args = ap.parse_args()
    rows = [r for layer in range(32) for tau in POSITIONS if (r := summarize(args.src, layer, tau))]
    meta = json.loads((args.src / "alive_only" / "meta.json").read_text())
    out = {"kl_last_mean": meta["kl_last_mean"], "n_prompts": meta["n_prompts"], "rows": rows}
    (args.src / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} (block, position) rows, KL mean {meta['kl_last_mean']:.4f}")


if __name__ == "__main__":
    main()
