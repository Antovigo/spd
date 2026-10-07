"""The protocol (PROTOCOL.md) run at token a over all 63 read sites, in stream order.

Hypotheses are a fixed pool of specific quantities (functions of a), each one direction or a
small set; the loop at each site:
1. start from the previous site's accepted list A (passed on), refit, and remove any quantity
   whose drop-one loss is at most its chance level (backtrack);
2. forward step: fit every remaining candidate on top of A; take the one with the largest excess
   over chance per dimension; accept it if excess > 0 and its permutation p < 0.01, else stop;
3. after each acceptance, remove any accepted quantity whose drop-one loss falls to its chance
   level or below (backtrack: the new quantity made it redundant);
4. stop also when the important residual (CI > 0.01) is below 5% of the important variance.
Every decision is logged. Writes `v3_token_a.json` and per-site reconstructions `v3_recon.npz`.

    python v3_run.py
"""

import json
from pathlib import Path

import numpy as np
from v2 import A, classes, fourier, spline
from v2_layers import site_list
from v2_sets import v2_adic
from v3 import (
    Cand,
    drop_one,
    fit,
    heldout_r2,
    important_residual,
    important_residual_cv,
    increment,  # fmt: skip
    load,
    perm_p,
    residual_svd,
)

HERE = Path(__file__).parent
STOP_SHARE = 0.05


def pool() -> list[Cand]:
    tens = np.where((A >= 10) & (A <= 99), A // 10, 0)
    return [
        Cand("log a", np.log(A)[:, None]),
        Cand("magnitude: smooth curve in log a (5)", spline()),
        Cand("one digit [a <= 9]", (A <= 9).astype(float)[:, None]),
        Cand("a = 100", (A == 100).astype(float)[:, None]),
        Cand("circle period 100", fourier(100, [1])),
        Cand("circle period 50", fourier(100, [2])),
        Cand("parity (a mod 2)", (A % 2).astype(float)[:, None]),
        Cand("units digit circle (period 10)", fourier(10, [1])),
        Cand("a mod 5 classes", classes(A % 5)),
        Cand("units digit classes (a mod 10)", classes(A % 10)),
        Cand("decade parity (-1)^floor(a/10)", ((-1.0) ** (A // 10))[:, None]),
        Cand("circle period 20", fourier(20, [1])),
        Cand("tens digit classes (10..99)", classes(tens)),
        Cand("2-adic valuation min(v2(a), 4)", np.minimum(v2_adic(A), 4).astype(float)[:, None]),
        Cand("repdigit (11, 22, .., 99)", ((A % 11 == 0) & (A <= 99)).astype(float)[:, None]),
        Cand("a mod 3 classes", classes(A % 3)),
    ]


def run_site(site, inherited: list[str], log: list[str]) -> list[Cand]:  # noqa: ANN001
    P = {c.name: c for c in pool()}
    Acc = [P[nm] for nm in inherited]
    removed: set[str] = set()

    def prune(reason: str, test_p: bool = False) -> None:
        """Remove redundant quantities one at a time: drop-one loss <= chance (or, with test_p,
        drop-one permutation p >= 0.01); among several, the one with the most dims goes first
        (the protocol prefers the specific, low-dimensional encoding)."""
        while Acc:
            bad = []
            for (nm, loss, ch), c in zip(drop_one(site, Acc), Acc, strict=True):
                if loss - ch <= 0:
                    bad.append((c.d, loss - ch, nm, f"drop-one loss {loss:.4f} <= chance {ch:.4f}"))
                elif test_p:
                    p = perm_p(site, [a for a in Acc if a.name != nm], c)
                    if p >= 0.01:
                        bad.append((c.d, loss - ch, nm, f"drop-one p {p:.3f} >= 0.01"))
            if not bad:
                return
            d, _, nm, why = max(bad, key=lambda t: (t[0], -t[1]))
            Acc[:] = [c for c in Acc if c.name != nm]
            removed.add(nm)
            log.append(f"remove '{nm}' ({reason}): {why}")

    if Acc:
        log.append(f"inherit {[c.name for c in Acc]}")
        prune("inherited, not needed here", test_p=True)
    for _ in range(20):
        imp = important_residual_cv(site, Acc)
        if imp["share"] < STOP_SHARE:
            log.append(f"stop: held-out important residual share {imp['share']:.3f} < {STOP_SHARE}")
            break
        cands = [
            c for c in P.values() if c.name not in {a.name for a in Acc} and c.name not in removed
        ]
        scored = []
        for c in cands:
            inc, ch = increment(site, Acc, c)
            d = max(c.d, 1)
            scored.append(((inc - ch) / d, inc, ch, c))
        scored.sort(key=lambda t: -t[0])
        accepted = False
        for score, inc, ch, c in scored[:3]:
            if inc - ch <= 0:
                break
            p = perm_p(site, Acc, c)
            if p < 0.01:
                Acc.append(c)
                log.append(
                    f"accept '{c.name}': increment {inc:.4f}, chance {ch:.4f}, excess/dim {score:.4f}, p {p:.3f}"
                )
                prune(f"after adding '{c.name}'")
                accepted = True
                break
            log.append(f"reject '{c.name}': excess {inc - ch:.4f} but p {p:.3f}")
            removed.add(c.name)
        if not accepted:
            log.append("stop: no candidate with positive excess and p < 0.01")
            break
    return Acc


def main() -> None:
    out, recon = [], {}
    prev: list[str] = []
    for b, pt, _ in site_list():
        try:
            site = load(b, pt)
        except (IndexError, ValueError):
            continue
        if len(site.names) == 0:
            continue
        log: list[str] = []
        Acc = run_site(site, prev, log)
        prev = [c.name for c in Acc]
        tot_imp = important_residual(site, [])
        rec = {
            "site": site.label, "n": len(site.names), "k": int(site.Z.shape[1]),
            "accepted": [{"name": n, "drop_one_loss": round(loss, 4), "chance": round(c, 4)} for n, loss, c in drop_one(site, Acc)],
            "dims": int(sum(c.d for c in Acc)),
            "heldout_r2": round(heldout_r2(site, Acc), 4),
            "important": important_residual(site, Acc), "important_before": tot_imp,
            "important_cv": important_residual_cv(site, Acc),
            "resid_all": residual_svd(site, Acc), "resid_important": residual_svd(site, Acc, True),
            "log": log,
        }  # fmt: skip
        Zh, _ = fit(site, Acc)
        Yt = site.Y
        rec["r2_in"] = round(
            1
            - float((((site.Z - Zh) @ site.W) ** 2).sum()) / float(((Yt - Yt.mean(0)) ** 2).sum()),
            4,
        )
        recon[site.label + "/Yhat"] = Zh @ site.W
        recon[site.label + "/Y"] = Yt
        recon[site.label + "/cols"] = site.cols
        out.append(rec)
        print(f"{site.label}: n={rec['n']} A={[a['name'] for a in rec['accepted']]} dims={rec['dims']} "
              f"R2 {rec['r2_in']:.3f} heldout {rec['heldout_r2']:.3f} important share {rec['important']['share']:.3f} (held-out {rec['important_cv']['share']:.3f})", flush=True)  # fmt: skip
    (HERE / "v3_token_a.json").write_text(json.dumps(out, indent=1))
    np.savez(HERE / "v3_recon.npz", **recon)


if __name__ == "__main__":
    main()
