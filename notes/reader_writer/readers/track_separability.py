"""Are the residue clouds linearly separable at every boundary? Position 1, alive-only model, blocks 0..16.

Input: track_build.py's stream.npz. For each boundary (residual stream entering block b; pre_17 leaves block 16),
the 400 prompts' coordinates z(a) in two spaces: the frequency-1 plane P_1 (2 dims) and D^X (9 dims; notation of
track_analyze.py). Tests, each exact (linear-programming feasibility of y_i (w . z_i + c) >= 1, y_i = +-1):
  one_vs_rest  residue r against the other nine, r = 0..9
  pairwise     residue r against residue r', 45 pairs
with an offset c (affine) and with c = 0 (through the activation zero; the network has no biases). When a test
is infeasible, `errors` is the number of misclassified prompts of a linear SVM (C = 1e4) trained on the same split,
an upper bound on the minimum number of errors. Nothing is centred.

    python track_separability.py
"""

import json

import numpy as np
from mod10_L0 import A
from scipy.optimize import linprog
from sklearn.svm import LinearSVC
from track_analyze import basis
from track_build import LAST, OUT

R = A % 10


def separable(z: np.ndarray, y: np.ndarray, offset: bool) -> bool:
    Z = np.concatenate([z, np.ones((len(z), 1))], 1) if offset else z
    res = linprog(np.zeros(Z.shape[1]), A_ub=-(y[:, None] * Z), b_ub=-np.ones(len(z)),
                  bounds=[(None, None)] * Z.shape[1], method="highs")  # fmt: skip
    return res.status == 0


def svm_errors(z: np.ndarray, y: np.ndarray, offset: bool) -> int:
    s = z.std() or 1.0  # one global scale for the solver; the decision boundary is unchanged
    clf = LinearSVC(C=1e4, fit_intercept=offset, max_iter=200000).fit(z / s, y)
    return int((clf.predict(z / s) != y).sum())


def tests(z: np.ndarray) -> dict:
    out = {}
    for offset in (True, False):
        ovr, pairs = [], []
        for r in range(10):
            y = np.where(r == R, 1.0, -1.0)
            ok = separable(z, y, offset)
            ovr.append({"r": r, "separable": ok, "errors": 0 if ok else svm_errors(z, y, offset)})
        for r in range(10):
            for q in range(r + 1, 10):
                sel = (r == R) | (q == R)
                y = np.where(R[sel] == r, 1.0, -1.0)
                ok = separable(z[sel], y, offset)
                pairs.append(
                    {
                        "pair": [r, q],
                        "separable": ok,
                        "errors": 0 if ok else svm_errors(z[sel], y, offset),
                    }
                )
        out["affine" if offset else "through_zero"] = {
            "one_vs_rest_separable": sum(d["separable"] for d in ovr),
            "one_vs_rest": ovr,
            "pairs_separable": sum(d["separable"] for d in pairs),
            "pairs": pairs,
        }
    return out


def main() -> None:
    st = np.load(OUT / "stream.npz")
    res = {}
    for b in range(LAST + 2):
        X = st[f"pre_{b}"].astype(np.float64)
        P, D = basis(X)
        res[f"pre_{b}"] = {"P1": tests(X @ P[0]), "D": tests(X @ D)}
        line = []
        for sp in ("P1", "D"):
            for m in ("affine", "through_zero"):
                t = res[f"pre_{b}"][sp][m]
                err = sum(d["errors"] for d in t["one_vs_rest"])
                line.append(
                    f"{sp}/{m}: ovr {t['one_vs_rest_separable']}/10 (svm err {err}) pairs {t['pairs_separable']}/45"
                )
        print(f"pre_{b}", " | ".join(line), flush=True)
    (OUT / "separability.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
