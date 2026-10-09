"""Reconstructions of the agents' final lists for the closed patching test (qagent/qpatch.py --tag agentic).

Per position t and site, from run1/t<t>/<site>/final.py: the raw reads Y = Z W and the stream RMS rho
fit by ridge on the quantities' features (standardised), written in the format qpatch.py reads:
  Yhat, rhohat:                  fit on all domain rows;
  Yhat_heldout, rhohat_heldout:  fit on the training rows only (the test rows, with held-out values of
                                 a, b, a + b or a - b, are predicted out of sample).
Writes qagent/recon_t<t>_agentic.npz and qagent/qloop_t<t>_agentic.json (site list, quantity names).

    python recon.py <run dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import qa
from qdata import DATA, load


def fit_predict(F: np.ndarray, T: np.ndarray, rows: np.ndarray) -> np.ndarray:
    mu, sd = F[rows].mean(0), F[rows].std(0)
    Fs = (F - mu) / np.where(sd > 1e-12, sd, 1.0)
    mT = T[rows].mean(0)
    return mT + Fs @ qa.ridge(Fs[rows], T[rows] - mT)


def main() -> None:
    run = Path(sys.argv[1])
    for t in (1, 2, 3, 4):
        pos = load(t)
        tr = qa.train_rows(t, pos.a, pos.b)
        every = np.arange(pos.D)
        R, sites = {"rows": pos.rows}, []
        for lab, s in pos.sites.items():
            Y = s.Z @ s.W
            R[lab + "/Y"], R[lab + "/cols"], R[lab + "/rho"] = (
                Y.astype(np.float32),
                s.cols,
                s.rho.astype(np.float32),
            )
            spec = run / f"t{t}" / lab / "final.py"
            if not spec.exists():  # constant over the domain (L0.attn at the "=" token)
                for k in ("/Yhat", "/Yhat_heldout"):
                    R[lab + k] = Y.astype(np.float32)
                for k in ("/rhohat", "/rhohat_heldout"):
                    R[lab + k] = s.rho.astype(np.float32)
                sites.append({"site": lab, "constant": True, "accepted": []})
                continue
            Q = qa.load_spec(spec)
            F = np.hstack([qa.evaluate(src, pos.op, pos.a, pos.b) for _, src in Q])
            T = np.hstack([Y, s.rho[:, None]])
            full, held = fit_predict(F, T, every), fit_predict(F, T, tr)
            R[lab + "/Yhat"], R[lab + "/rhohat"] = (
                full[:, :-1].astype(np.float32),
                full[:, -1].astype(np.float32),
            )
            R[lab + "/Yhat_heldout"], R[lab + "/rhohat_heldout"] = (
                held[:, :-1].astype(np.float32),
                held[:, -1].astype(np.float32),
            )
            sites.append({"site": lab, "accepted": [{"name": n, "formula": f} for n, f in Q]})
        np.savez(DATA / f"recon_t{t}_agentic.npz", **R)
        (DATA / f"qloop_t{t}_agentic.json").write_text(
            json.dumps({"t": t, "nested": True, "source": str(run), "sites": sites}, indent=1)
        )
        print(f"t={t}: {len(sites)} sites", flush=True)


if __name__ == "__main__":
    main()
