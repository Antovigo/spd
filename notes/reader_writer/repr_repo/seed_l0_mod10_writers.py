"""Add writer connections to the block-0 mod-10 readers of the representation repository (position 1).

Source: the exact variance split of the model with only the λ=5e-05 mask's components on
(`analysis/reader_writer/variance_kl5e-5/results/L0/{variance,J}_pos1.npz`; method in
`notes/reader_writer/plan_variance.md`, results in `report_variance.md`). Its block-0 readers are
exactly the repository's 153 block-0 readers.

Notation:
  c          a reader (block-0 gate or up component); w a writer (block-0 down component)
  phi_w,c    share of writer w's variance over the prompts credited to reader c (sums to 1 over c)
  a_w,c      phi_w,c Var(t_w) ||U_w||^2: the share times the variance writer w puts into the
             residual stream (t_w its activation, U_w its write vector)
  frac_w,c   |a_w,c| / sum over writers w' of |a_w',c|: how reader c's weight splits among writers
  J_w,c,c'   for an up reader c, the part of its term on writer w that gate reader c' passes or
             blocks by moving away from its mean; phi^J_w,c,c' its share of writer w's variance,
             measured on the split's 8,000-prompt subsample, as are phi^sub_w,c (the up reader's
             whole share there) and phi^0_w,c (its share through the mean gate);
             phi^sub = phi^0 + sum over c' of phi^J (plan eq. 5)
Kept per reader: the writers with the largest |a_w,c| until their frac_w,c add up to >= 0.9.
Kept per (up reader, writer): the gate readers with the largest |phi^J| until they hold >= 0.9 of
sum over c' of |phi^J|.
The mod-10 readers are the block-0 readers whose position-1 r2 is >= 0.5 (the page's threshold).

    python seed_l0_mod10_writers.py
"""

import datetime
import subprocess
from pathlib import Path

import numpy as np
from repo import Repo

SRC = Path(
    "/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/variance_kl5e-5/results/L0"
)
MOD10_R2 = 0.5
COVER = 0.9


def cover(x: np.ndarray) -> np.ndarray:
    """Indices of the largest |x| until they hold >= COVER of sum |x|."""
    order = np.argsort(-np.abs(x))
    cum = np.cumsum(np.abs(x[order])) / np.abs(x).sum()
    return order[: int(np.searchsorted(cum, COVER)) + 1]


def main() -> None:
    repo = Repo()
    var = np.load(SRC / "variance_pos1.npz")
    jz = np.load(SRC / "J_pos1.npz")
    for k in ("ids_gate", "ids_up", "ids_down"):
        assert (var[k] == jz[k]).all(), k
    ids = {"gate": list(var["ids_gate"]), "up": list(var["ids_up"])}
    block0 = {(rd["kind"], rd["component"]) for rd in repo.readers.values() if rd["block"] == 0}
    assert block0 == {(k, int(i)) for k in ids for i in ids[k]}, "reader sets differ"
    down = [int(i) for i in var["ids_down"]]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    today = datetime.date.today().isoformat()
    mod10 = [
        rd
        for rd in repo.readers.values()
        if rd["block"] == 0 and rd["positions"]["1"] and rd["positions"]["1"]["r2"] >= MOD10_R2
    ]
    assert len(mod10) == 11, [rd["id"] for rd in mod10]
    for rd in mod10:
        kind, j = rd["kind"], ids[rd["kind"]].index(rd["component"])
        key = "Mu" if kind == "up" else "Mg"
        a, phi = var[f"a_{key}"][:, j], var[f"phi_{key}"][:, j]
        frac = np.abs(a) / np.abs(a).sum()
        entries = []
        for w in cover(a):
            e = {
                "writer": f"L0.down.{down[w]}",
                "site": "layers.0.mlp.down_proj",
                "component": down[w],
                "phi": float(phi[w]),
                "reader_fraction": float(frac[w]),
            }
            if kind == "up":
                pj = jz["phi_J"][w, j]
                e["routing"] = {
                    "phi_sub": float(jz["phi_Mu"][w, j]),
                    "phi_mean_gate": float(jz["phi_Mu0"][w, j]),
                    "gates": [
                        {"reader": f"L0.gate.{int(ids['gate'][g])}", "phi_J": float(pj[g])}
                        for g in cover(pj)
                    ],
                }
            entries.append(e)
        rd["positions"]["1"]["writers"] = {
            "entries": entries,
            "covered_fraction": float(sum(e["reader_fraction"] for e in entries)),
            "source": str(SRC),
            "method": (
                "exact variance split (plan_variance.md) of the model with only the λ=5e-05 mask on; "
                "writers kept by largest |a| until reader_fraction sums to >= 0.9; phi = share of the "
                "writer's variance; reader_fraction = |a| / sum over writers |a|; routing (up readers): "
                "phi_J per gate reader on the 8,000-prompt subsample, kept until >= 0.9 of sum |phi_J|, "
                "phi_sub = phi_mean_gate + sum phi_J"
            ),
            "provenance": {
                "script": "notes/reader_writer/repr_repo/seed_l0_mod10_writers.py",
                "commit": commit,
            },  # fmt: skip
            "updated": today,
        }
        print(rd["id"], " ".join(f"{e['writer']}({e['reader_fraction']:.2f}, φ {e['phi']:+.2f})"
                                 for e in entries))  # fmt: skip
    repo.save()


if __name__ == "__main__":
    main()
