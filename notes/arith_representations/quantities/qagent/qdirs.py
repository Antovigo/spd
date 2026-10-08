"""Stream directions of the accepted quantities at each site: the directions the readers read them along.

At a site (readers' read directions spanning Q, 4096 x k, orthonormal; reads Y = Z W with Z = X Q,
W k x n), the joint fit of the accepted list A (qfit's ridge) writes the stream, as the readers see
it, as Z(i) ~ mu + sum_q sum_j phi_qj(i) g_qj, with phi_qj quantity q's centred features (one for a
scalar, cos and sin for a circle, one per class) and g_qj in R^k. Mapped to the stream, d_qj = Q g_qj
(4096-vector): the stream moves by d_qj per unit of phi_qj. Only the part of the stream in span(Q)
changes a read, and d_qj lies in it. Reader r responds to q through d_qj . V~_r = (g_qj W)_r.

Saved per site and quantity (index j in A):
  <site>/<j>/D      d_q x 4096 float32: the directions d_qj, one row per feature;
  <site>/<j>/resp   d_q x n float32: each reader's response g_qj W (change of its raw read per unit
                    of phi_qj);
  <site>/<j>/basis  r x 4096 float32: an orthonormal basis of span{d_qj} (independent of how q's
                    features are written), from the SVD of the d_q x k matrix of g_qj;
  <site>/<j>/sv     its singular values.
JSON per site and quantity: dims; dependence: the R^2 of q's features on the other accepted
quantities' features over the domain (near 1: the readers cannot tell q from a combination of the
others, and how the directions are split between them is not determined by the data); readers:
the readers with >= 10% of their read variance from q's term phi_q g_q W.

    python qdirs.py <t> [--tag name]
"""

import argparse
import json

import numpy as np
from qdata import DATA, load
from qfeat import pool
from qfit import Fitter, folds, primary_scheme


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("t", type=int)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    run = json.loads((DATA / f"qloop_t{args.t}{tag}.json").read_text())
    accepted = {
        r["site"]: [a["name"] for a in r["accepted"]]
        for r in run["sites"]
        if not r.get("constant") and r["accepted"]
    }
    pos = load(args.t, list(accepted))
    cands = {c.name: c for c in pool(pos)}
    fs = folds(pos, primary_scheme(args.t))
    arrays, out = {}, []
    for lab, A in accepted.items():
        s = pos.sites[lab]
        F = Fitter(s, [cands[n] for n in A], fs)
        c = F.cols(A)
        G = F._solve(F.PP[np.ix_(c, c)], F.PZ[c])  # (features of A) x k
        Yc = s.Y - s.Y.mean(0)
        var_r = np.maximum((Yc**2).sum(0), 1e-30)
        rec = {"site": lab, "k": s.W.shape[0], "n": s.W.shape[1], "quantities": []}
        for j, q in enumerate(A):
            Gq = G[np.flatnonzero(np.isin(c, F.slc[q]))]  # d_q x k
            Pq = F.P[:, F.slc[q]]
            oth = F.cols([m for m in A if m != q])
            dep = 0.0
            if len(oth):
                fit = F.P[:, oth] @ F._solve(F.PP[np.ix_(oth, oth)], F.PP[np.ix_(oth, F.slc[q])])
                dep = 1 - float(((Pq - fit) ** 2).sum() / max((Pq**2).sum(), 1e-30))
            _, sv, Vt = np.linalg.svd(Gq, full_matrices=False)
            r = int((sv > 1e-6 * sv[0]).sum()) if sv[0] > 0 else 0
            term = Pq @ Gq @ s.W
            arrays[f"{lab}/{j}/D"] = (Gq @ s.Q.T).astype(np.float32)
            arrays[f"{lab}/{j}/resp"] = (Gq @ s.W).astype(np.float32)
            arrays[f"{lab}/{j}/basis"] = (Vt[:r] @ s.Q.T).astype(np.float32)
            arrays[f"{lab}/{j}/sv"] = sv[:r]
            rec["quantities"].append({
                "index": j, "name": q, "dims": r, "dependence": round(dep, 4),
                "readers": [f"c{int(s.cols[i])}" for i in np.flatnonzero((term**2).sum(0) / var_r >= 0.1)],
            })  # fmt: skip
        out.append(rec)
        print(lab, f"k={rec['k']}", [(e["name"][:26], e["dims"], e["dependence"], len(e["readers"])) for e in rec["quantities"]], flush=True)  # fmt: skip
    (DATA / f"dirs_t{args.t}{tag}.json").write_text(
        json.dumps({"t": args.t, "sites": out}, indent=1)
    )
    np.savez(DATA / f"dirs_t{args.t}{tag}.npz", **arrays)


if __name__ == "__main__":
    main()
