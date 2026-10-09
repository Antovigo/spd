"""Final evaluation (not an agent tool): each site's final list (run1/t<t>/<site>/final.py), fit on all
training rows, scored on the test rows the agents never saw (rows with a held-out value of a, b,
a + b or a - b; see qa.py). Reports per site the test R^2 of the readers' raw reads (around the
training mean) and of the reads on important entries (CI > 0.01), and writes run1/evaluation.json.

    python evaluate.py <run dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import qa
from qdata import load


def main() -> None:
    run = Path(sys.argv[1])
    out = {}
    for t in (1, 2, 3, 4):
        pos = load(t)
        tr = qa.train_rows(t, pos.a, pos.b)
        te = np.setdiff1d(np.arange(pos.D), tr)
        for lab, s in pos.sites.items():
            spec = run / f"t{t}" / lab / "final.py"
            if not spec.exists():
                continue
            Q = qa.load_spec(spec)
            F = np.hstack([qa.evaluate(src, pos.op, pos.a, pos.b) for _, src in Q])
            mu, sd = F[tr].mean(0), F[tr].std(0)
            F = (F - mu) / np.where(sd > 1e-12, sd, 1.0)
            Y = s.Z @ s.W
            mY = Y[tr].mean(0)
            G = qa.ridge(F[tr], Y[tr] - mY)
            E = Y[te] - mY - F[te] @ G
            T = Y[te] - mY
            imp = s.CI[te] > qa.IMPORTANT
            r2 = 1 - float((E**2).sum() / (T**2).sum())
            r2i = 1 - float(((E**2) * imp).sum() / max(((T**2) * imp).sum(), 1e-30))
            out[f"{t}|{lab}"] = {"n_quantities": len(Q), "test_r2": round(r2, 4), "test_r2_important": round(r2i, 4),
                                 "n_train": int(len(tr)), "n_test": int(len(te))}  # fmt: skip
        vals = [v for k, v in out.items() if k.startswith(f"{t}|")]
        print(f"t={t}: {len(vals)} sites, median test R2 {np.median([v['test_r2'] for v in vals]):.3f}, "
              f"on important entries {np.median([v['test_r2_important'] for v in vals]):.3f}, "
              f"min {min(v['test_r2'] for v in vals):.3f}", flush=True)  # fmt: skip
    (run / "evaluation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
