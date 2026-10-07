"""The automatic part of the loop (AGENT.md, steps 2-6) over the sites of one token position.

At each site, in stream order:
1. start from the previous site's accepted list, remove what is not needed here (drop-one loss at
   or below chance, or drop-one permutation p >= 0.01; the most dimensions first);
2. add, one at a time, the candidate with the largest excess over chance per dimension, if its
   increment is at least MIN_INC of the reads' variance and its permutation p < 0.01; after each
   addition remove what it made redundant (the most dimensions first);
3. stop when no candidate qualifies, or the held-out important residual is below STOP_SHARE.
The candidate pool is qfeat.pool plus the agent's extra candidates (`--extra module.function`,
a function (Position) -> list[Cand]). Writes OUT/qloop_t<t>[_tag].json (accepted lists, scores,
decision logs) and OUT/recon_t<t>[_tag].npz (reconstructed raw reads per site, in-sample and
held-out).

    python qloop.py <t> [--sites L0.attn,L0.mlp] [--extra mymod.extra] [--tag name]
"""

import argparse
import importlib
import json
import time

import numpy as np
from qdata import DATA, POS_NAMES, load
from qfeat import pool
from qfit import Fitter, folds, primary_scheme

MIN_INC = 0.002
STOP_SHARE = 0.05


def run_site(F: Fitter, inherited: list[str], log: list[str]) -> list[str]:
    A = [n for n in inherited if n in F.cands]
    removed: set[str] = set()

    def prune(reason: str, test_p: bool) -> None:
        while A:
            bad = []
            for n, loss, ch in F.drop_one(A):
                if loss - ch <= 0:
                    bad.append(
                        (F.dims([n]), loss - ch, n, f"drop-one loss {loss:.4f} <= chance {ch:.4f}")
                    )
                elif test_p:
                    p = F.perm_p([m for m in A if m != n], n)
                    if p >= 0.01:
                        bad.append((F.dims([n]), loss - ch, n, f"drop-one p {p:.3f}"))
            if not bad:
                return
            _, _, n, why = max(bad, key=lambda b: (b[0], -b[1]))
            A.remove(n)
            removed.add(n)
            log.append(f"remove '{n}' ({reason}): {why}")

    if A:
        log.append(f"inherit {A}")
        prune("inherited, not needed here", True)
    for _ in range(40):
        share = F.important_share(A)
        if share < STOP_SHARE:
            log.append(f"stop: held-out important residual share {share:.3f} < {STOP_SHARE}")
            break
        scored = []
        for n in F.cands:
            if n in A or n in removed:
                continue
            inc, ch = F.increment(A, n)
            scored.append(((inc - ch) / max(F.dims([n]), 1), inc, ch, n))
        scored.sort(key=lambda s: -s[0])
        added = False
        for score, inc, ch, n in scored[:3]:
            if inc - ch <= 0 or inc < MIN_INC:
                break
            p = F.perm_p(A, n)
            if p < 0.01:
                A.append(n)
                log.append(
                    f"accept '{n}': increment {inc:.4f}, chance {ch:.4f}, excess/dim {score:.4f}, p {p:.3f}"
                )
                prune(f"after adding '{n}'", False)
                added = True
                break
            log.append(f"reject '{n}': p {p:.3f}")
            removed.add(n)
        if not added:
            log.append(
                f"stop: no candidate with increment >= {MIN_INC}, positive excess and p < 0.01"
            )
            break
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("t", type=int)
    ap.add_argument("--sites", default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pos = load(args.t, args.sites.split(",") if args.sites else None)
    cands = pool(pos)
    if args.extra:
        mod, fn = args.extra.rsplit(".", 1)
        cands += getattr(importlib.import_module(mod), fn)(pos)
    fs = folds(pos, primary_scheme(pos.t))
    diag = {s: folds(pos, s) for s in (["b-values", "a-values"] if pos.t >= 3 else [])}
    out, recon, prev = [], {}, []
    t0 = time.time()
    for label, site in pos.sites.items():
        Y = site.Y
        if ((Y - Y.mean(0)) ** 2).sum() <= 1e-10 * (
            Y**2
        ).sum():  # constant over the domain (e.g. L0 at a fixed token)
            out.append({"site": label, "n": site.W.shape[1], "k": site.W.shape[0], "dims": 0, "accepted": [],
                        "constant": True, "log": ["constant over the domain: nothing to fit"]})  # fmt: skip
            for key in ("/Yhat", "/Yhat_heldout", "/Y"):
                recon[label + key] = Y.astype(np.float32)
            recon[label + "/cols"] = site.cols
            print(f"t={pos.t} {label}: constant over the domain", flush=True)
            continue
        F = Fitter(site, cands, fs)
        log: list[str] = []
        A = run_site(F, prev, log)
        prev = A
        Zin, Zho = F.reconstruct(A), F.reconstruct(A, heldout=True)
        rec = {
            "site": label, "n": site.W.shape[1], "k": site.W.shape[0], "dims": F.dims(A),
            "accepted": [{"name": n, "dims": F.dims([n]), "drop_one_loss": round(l_, 5), "chance": round(c_, 5)}
                         for n, l_, c_ in F.drop_one(A)],
            "r2_in": 1 - F.rss(A) / F.tot, "heldout_r2": F.heldout_r2(A),
            "important_share_in": F.important_share(A, heldout=False), "important_share_heldout": F.important_share(A),
            "log": log,
        }  # fmt: skip
        for s, fd in diag.items():
            rec[f"heldout_r2_{s}"] = Fitter(
                site, [F.cands[n] for n in A] or cands[:1], fd
            ).heldout_r2(A)
        recon[label + "/Yhat"] = (Zin @ site.W).astype(np.float32)
        recon[label + "/Yhat_heldout"] = (Zho @ site.W).astype(np.float32)
        recon[label + "/Y"] = Y.astype(np.float32)
        recon[label + "/cols"] = site.cols
        out.append(rec)
        print(f"t={pos.t} {label}: n={rec['n']} dims={rec['dims']} R2 {rec['r2_in']:.3f} held-out {rec['heldout_r2']:.3f} "
              f"important {rec['important_share_in']:.3f}/{rec['important_share_heldout']:.3f} [{time.time() - t0:.0f}s] "
              f"{[a['name'] for a in rec['accepted']]}", flush=True)  # fmt: skip
    tag = f"_{args.tag}" if args.tag else ""
    meta = {"t": pos.t, "position": POS_NAMES[pos.t], "D": pos.D, "scheme": primary_scheme(pos.t),
            "MIN_INC": MIN_INC, "STOP_SHARE": STOP_SHARE, "sites": out}  # fmt: skip
    (DATA / f"qloop_t{pos.t}{tag}.json").write_text(json.dumps(meta, indent=1))
    np.savez(DATA / f"recon_t{pos.t}{tag}.npz", rows=pos.rows, **recon)


if __name__ == "__main__":
    main()
