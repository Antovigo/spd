"""One iteration of the hypothesis loop on a site: fit, write plots and a JSON, print diagnostics.

    python iterate.py <block> <attn|mlp> <hypothesis set>

Target: the readers' RAW linear reads rho(a) * A_r(a) / mean(rho), i.e. the reader direction's
dot product with the raw residual stream, which is exactly linear in what was written there
(rho: the RMS of the stream at the site; reported separately, it is a shared per-a factor).
"""

import json
import sys

import hypotheses
import numpy as np
from hyp_fit import diagnostics, fit_site
from qsite import OUT, VW, load_site, plot_curves


def site_rho(block: int, point: str) -> np.ndarray:
    l = 2 * block + (0 if point == "attn" else 1)  # noqa: E741
    r = np.load(VW / "alive_only/resid.npy", mmap_mode="r")
    x = np.asarray(r[l][np.arange(100) * 100, 1], np.float64)
    return np.sqrt((x**2).mean(-1) + 1e-5)


def main() -> None:
    block, point, hname = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    site = load_site(block, point)
    rho = site_rho(block, point)
    X = site.X * (rho / rho.mean())[:, None]
    H = getattr(hypotheses, hname)()
    fits = fit_site(X, H)
    F = np.stack([f.fitted for f in fits], 1)
    R = X - F
    out = OUT / f"L{block}" / f"{point}_{hname}"
    out.mkdir(parents=True, exist_ok=True)
    tot = ((X - X.mean(0)) ** 2).sum()
    diag = diagnostics(R, site.names)
    per_q: dict[str, list[str]] = {}
    for nm, f in zip(site.names, fits, strict=True):
        for q in f.quantities:
            per_q.setdefault(q, []).append(nm)
    res = {
        "site": site.label, "hypotheses": hname, "n_readers": len(site.names),
        "explained_var": round(float(1 - (R**2).sum() / tot), 4),
        "median_reader_r2": round(float(np.median([f.r2 for f in fits])), 4),
        "quantities": {q: len(v) for q, v in sorted(per_q.items(), key=lambda kv: -len(kv[1]))},
        **diag,
        "readers": [{"reader": nm, "r2": round(f.r2, 3), "quantities": f.quantities, "unique": f.unique,
                     "items": f.items} for nm, f in zip(site.names, fits, strict=True)],
    }  # fmt: skip
    (out / "fit.json").write_text(json.dumps(res, indent=1))
    plot_curves(
        {"raw read": X, "fit": F}, site.names, out / "fit.png", title=f"{site.label} {hname}: fit"
    )
    plot_curves(
        {"residual": R}, site.names, out / "residual.png", title=f"{site.label} {hname}: residual"
    )
    U, sv, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    top = {f"sv{i + 1}={sv[i]:.2f}": U[:, i : i + 1] * sv[i] for i in range(3)}
    plot_curves({"u": np.hstack(list(top.values()))}, list(top), out / "residual_svd.png", ncol=3,
                title=f"{site.label} {hname}: top shared residual patterns (left singular vectors x sv)")  # fmt: skip
    load = {f"sv{i + 1}": [site.names[j] for j in np.argsort(-np.abs(Vt[i]))[:5]] for i in range(3)}
    res["residual_svd_top_readers"] = load
    (out / "fit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "readers"}, indent=0))
    for r in res["readers"]:
        print(f"{r['reader']:<15} R2={r['r2']:.3f} {r['quantities']} {r['items']}")


if __name__ == "__main__":
    main()
