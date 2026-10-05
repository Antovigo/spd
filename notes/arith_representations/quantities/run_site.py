"""Fit one site, write the per-reader table, fit/residual plots and a residual-structure report.

python run_site.py <block> <attn|mlp> <tag>
"""

import json
import sys
from collections import Counter, defaultdict

import numpy as np
from fit import fit_site
from library import build_library
from qsite import OUT, load_site, plot_curves


def residual_report(R: np.ndarray, X: np.ndarray) -> dict:
    """Shared structure left in the residuals: singular values of the (100, n) residual matrix vs
    a shuffled-in-a null (each reader's residual permuted over a independently)."""
    s = np.linalg.svd(R - R.mean(0), compute_uv=False)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(50):
        P = np.stack([rng.permutation(R[:, j]) for j in range(R.shape[1])], 1)
        null.append(np.linalg.svd(P - P.mean(0), compute_uv=False)[0])
    return {"resid_var_frac": float((R**2).sum() / ((X - X.mean(0)) ** 2).sum()),
            "top_sv": s[:5].round(3).tolist(), "null_top_sv_95": float(np.quantile(null, 0.95))}  # fmt: skip


def main() -> None:
    block, point, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    site = load_site(block, point)
    lib = build_library()
    fits = fit_site(site.X, lib)
    F = np.stack([f.fitted for f in fits], 1)
    R = site.X - F
    out = OUT / f"L{block}" / tag
    rows = []
    fam_readers: dict[str, list[str]] = defaultdict(list)
    for name, f in zip(site.names, fits, strict=True):
        atoms = [lib[i].name for i in f.atoms]
        for i in f.atoms:
            fam_readers[lib[i].family].append(name)
        rows.append({"reader": name, "r2": round(f.r2, 4), "atoms": atoms,
                     "coef": [round(float(c), 3) for c in f.coef]})  # fmt: skip
    rep = residual_report(R, site.X)
    summary = {
        "site": site.label, "n_readers": len(site.names), **rep,
        "families": {k: sorted(set(v)) for k, v in sorted(fam_readers.items(), key=lambda kv: -len(set(kv[1])))},
        "atom_counts": Counter(len(f.atoms) for f in fits),
        "readers": rows,
    }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "fit.json").write_text(json.dumps(summary, indent=1, default=str))
    plot_curves(
        {"A": site.X, "fit": F}, site.names, out / "fit.png", title=f"{site.label} fit ({tag})"
    )
    plot_curves(
        {"residual": R}, site.names, out / "residual.png", title=f"{site.label} residual ({tag})"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("site", "n_readers", "resid_var_frac", "top_sv", "null_top_sv_95")
            }
        )
    )
    for k, v in summary["families"].items():
        print(f"  {k}: {len(v)} readers")
    for r in rows:
        print(f"  {r['reader']:<16} R2={r['r2']:.3f}  {', '.join(r['atoms'])}")


if __name__ == "__main__":
    main()
