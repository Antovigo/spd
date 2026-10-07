"""Refining the accepted quantities at the worst sites of the V3 patching test.

Candidate refinements, from inspecting their CI-weighted residuals (v3_inspect.py):
* one-hot over the values that single-value readers are important on: for each reader important
  (CI > 0.01) on at most 3 values of a, those values; one direction per value;
* a place code of the magnitude: Gaussian bumps exp(-(a - c)^2 / (2 w^2)), centres c every s
  values (s = 10, w = 4; s = 5, w = 2.5);
* residues: a mod 6, a mod 9, a mod 12 classes.
Each candidate is added to the site's accepted list; reported: its increment, chance and
permutation p (v3.py), the CI-weighted residual share (sum CI (Y - Yhat)^2 / sum CI (Y - mean)^2),
and the held-out R^2. Then the refined list (the accepted refinements, greedily, best first,
p < 0.01 and excess > 0) and its reconstruction (`v3_recon_refined.npz`, same keys as
v3_recon.npz) for patching.

    python v3_refine.py L0.mlp L16.attn ...
"""

import json
import sys
from pathlib import Path

import numpy as np
from v2 import A, classes
from v3 import Cand, fit, heldout_r2, increment, load, perm_p
from v3_run import pool

HERE = Path(__file__).parent


def ci_weighted_share(site, H) -> float:  # noqa: ANN001
    Zh, _ = fit(site, H)
    E = (site.Z - Zh) @ site.W
    Y = site.Y
    return float((site.CI * E**2).sum() / ((site.CI * (Y - Y.mean(0)) ** 2).sum() + 1e-30))


def refinements(site) -> list[Cand]:  # noqa: ANN001
    vals = sorted({int(a) + 1 for r in range(site.CI.shape[1]) if (site.CI[:, r] > 0.01).sum() <= 3
                   for a in np.flatnonzero(site.CI[:, r] > 0.01)})  # fmt: skip
    out = []
    if vals:
        out.append(
            Cand(
                f"one-hot over single-value readers' values {vals}",
                np.stack([(v == A).astype(float) for v in vals], 1),
            )
        )
    for s, w in ((10, 4.0), (5, 2.5)):
        cs = np.arange(s // 2, 101, s)
        out.append(
            Cand(
                f"place code: bumps every {s} (width {w})",
                np.stack([np.exp(-((A - c) ** 2) / (2 * w**2)) for c in cs], 1),
            )
        )
    for m in (6, 9, 12):
        out.append(Cand(f"a mod {m} classes", classes(A % m)))
    return out


def main() -> None:
    P = {c.name: c for c in pool()}
    runs = {s["site"]: s for s in json.loads((HERE / "v3_token_a.json").read_text())}
    R = dict(np.load(HERE / "v3_recon.npz"))
    report = {}
    for label in sys.argv[1:]:
        b, pt = int(label[1:].split(".")[0]), label.split(".")[1]
        site = load(b, pt)
        H = [P[a["name"]] for a in runs[label]["accepted"]]
        base = {"ci_weighted_share": ci_weighted_share(site, H), "heldout_r2": heldout_r2(site, H)}
        print(
            f"== {label}: accepted {len(H)} quantities; CI-weighted residual share {base['ci_weighted_share']:.3f}, held-out R2 {base['heldout_r2']:.3f}"
        )
        rows = []
        for c in refinements(site):
            inc, ch = increment(site, H, c)
            row = {"candidate": c.name, "dims": c.d, "increment": inc, "chance": ch, "p": perm_p(site, H, c),
                   "ci_weighted_share": ci_weighted_share(site, H + [c]), "heldout_r2": heldout_r2(site, H + [c])}  # fmt: skip
            rows.append((row, c))
            print(f"   {c.name[:70]:<70} d={c.d:2d} incr {inc:.4f} chance {ch:.4f} p {row['p']:.3f} "
                  f"CI-weighted share {row['ci_weighted_share']:.3f} held-out {row['heldout_r2']:.3f}")  # fmt: skip
        Href = list(H)
        for _row, c in sorted(rows, key=lambda t: t[0]["ci_weighted_share"]):
            inc, ch = increment(site, Href, c)
            if inc - ch > 0 and perm_p(site, Href, c) < 0.01:
                Href.append(c)
        ref = {"accepted_added": [c.name for c in Href[len(H) :]], "ci_weighted_share": ci_weighted_share(site, Href),
               "heldout_r2": heldout_r2(site, Href)}  # fmt: skip
        print(
            f"   refined: + {ref['accepted_added']}; CI-weighted share {ref['ci_weighted_share']:.3f}, held-out {ref['heldout_r2']:.3f}"
        )
        Zh, _ = fit(site, Href)
        R[label + "/Yhat"] = Zh @ site.W
        report[label] = {"base": base, "candidates": [r for r, _ in rows], "refined": ref,
                         "refined_list": [c.name for c in Href]}  # fmt: skip
    np.savez(HERE / "v3_recon_refined.npz", **R)
    (HERE / "v3_refine.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
