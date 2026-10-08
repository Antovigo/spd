"""The automatic part of the loop (AGENT.md, steps 2-6) over the sites of one token position.

A quantity q belongs in a site's accepted list A when all three hold:
  (i) its increment over the rest of A is at least MIN_INC of the reads' variance,
  (ii) the increment exceeds chance (d_q / df of the residual before it),
  (iii) the held-out R^2 of A is higher with q than without it.
At each site, in stream order:
1. start from the previous site's accepted list and remove what fails (i)-(iii) as a drop-one
   test (the most dimensions first, one at a time);
2. add, one at a time, the candidate with the largest excess over chance per dimension among
   those passing (i)-(iii) (the top 3 by excess per dimension are tried), then remove what it made
   redundant;
3. stop when no candidate qualifies or the held-out important residual is below STOP_SHARE.
The stream RMS rho (the norm's divisor) is fit as a scalar from the accepted quantities' features.

Selection is done on the rows it is scored on, so the held-out R^2 of the selected list is not
selection-independent. Nested selection: `--fold f` runs the whole loop on the training rows of
primary fold f only; `--combine` then predicts each fold from the list selected without it
(nested held-out reconstruction, used by the patching test's held-out variant) and records how
stable each accepted quantity is across the folds.

    python qloop.py <t> [--sites ...] [--extra mymod.extra] [--tag name]   # full run
    python qloop.py <t> --fold f [...]                                      # nested selection, f = 0..9
    python qloop.py <t> --combine [...]                                     # after the full run and all folds
Outputs: OUT/qloop_t<t>[_tag].json (accepted lists, scores, decision logs; after --combine also
nested scores and stability), OUT/recon_t<t>[_tag].npz (Y, Yhat, Yhat_heldout, rho, rhohat,
rhohat_heldout, cols per site), OUT/qloop_t<t>[_tag]_fold<f>.json.
"""

import argparse
import importlib
import json
import time

import numpy as np
from qdata import DATA, POS_NAMES, Position, load, restrict
from qfeat import Cand, pool
from qfit import Fitter, fit_scalar, folds, primary_scheme

MIN_INC = 0.002
STOP_SHARE = 0.05


def qualifies(F: Fitter, A: list[str], n: str) -> tuple[bool, str]:
    """Tests (i)-(iii) for n added to A (A without n)."""
    inc, ch = F.increment(A, n)
    if inc < MIN_INC:
        return False, f"increment {inc:.4f} < {MIN_INC}"
    if inc <= ch:
        return False, f"increment {inc:.4f} <= chance {ch:.4f}"
    h0, h1 = F.heldout_r2(A), F.heldout_r2(A + [n])
    if h1 <= h0:
        return False, f"held-out R2 {h1:.4f} <= {h0:.4f} without it"
    return True, f"increment {inc:.4f}, chance {ch:.4f}, held-out R2 {h0:.4f} -> {h1:.4f}"


def run_site(F: Fitter, inherited: list[str], log: list[str]) -> list[str]:
    A = [n for n in inherited if n in F.cands]
    removed: set[str] = set()

    def prune(reason: str) -> None:
        while A:
            bad = []
            for n in A:
                ok, why = qualifies(F, [m for m in A if m != n], n)
                if not ok:
                    bad.append((F.dims([n]), n, why))
            if not bad:
                return
            _, n, why = max(bad, key=lambda b: b[0])
            A.remove(n)
            removed.add(n)
            log.append(f"remove '{n}' ({reason}): {why}")

    if A:
        log.append(f"inherit {A}")
        prune("inherited, not needed here")
    for _ in range(40):
        share = F.important_share(A)
        if share < STOP_SHARE:
            log.append(f"stop: held-out important residual share {share:.3f} < {STOP_SHARE}")
            break
        scored = []
        for n in F.cands:
            if (
                n in A or n in removed or F.dims([n]) == 0
            ):  # constant on these rows (e.g. a = 100 held out)
                continue
            inc, ch = F.increment(A, n)
            scored.append(((inc - ch) / max(F.dims([n]), 1), n))
        scored.sort(key=lambda s: -s[0])
        added = False
        for _, n in scored[:3]:
            ok, why = qualifies(F, A, n)
            if ok:
                A.append(n)
                log.append(f"accept '{n}': {why}")
                prune(f"after adding '{n}'")
                added = True
                break
            log.append(f"reject '{n}': {why}")
            removed.add(n)
        if not added:
            log.append("stop: none of the top candidates passes")
            break
    return A


def candidates(pos: Position, extra: str | None) -> list[Cand]:
    cands = pool(pos)
    if extra:
        mod, fn = extra.rsplit(".", 1)
        cands += getattr(importlib.import_module(mod), fn)(pos)
    return cands


def constant(Y: np.ndarray) -> bool:
    return bool(((Y - Y.mean(0)) ** 2).sum() <= 1e-10 * (Y**2).sum())  # e.g. L0 at the "=" token


def loop(pos: Position, cands: list[Cand]) -> dict[str, tuple[list[str], list[str]]]:
    """Accepted list and decision log per non-constant site, in stream order."""
    fs = folds(pos, primary_scheme(pos.t))
    out, prev, t0 = {}, [], time.time()
    for label, site in pos.sites.items():
        if constant(site.Y):
            continue
        F = Fitter(site, cands, fs)
        log: list[str] = []
        A = run_site(F, prev, log)
        prev = A
        out[label] = (A, log)
        print(f"t={pos.t} {label}: {len(A)} quantities [{time.time() - t0:.0f}s]", flush=True)
    return out


def combine(pos: Position, cands: list[Cand], tag: str) -> None:
    """Nested held-out reconstruction and stability from the full run and the per-fold runs."""
    fs = folds(pos, primary_scheme(pos.t))
    byname = {c.name: c for c in cands}
    meta = json.loads((DATA / f"qloop_t{pos.t}{tag}.json").read_text())
    nested = [
        json.loads((DATA / f"qloop_t{pos.t}{tag}_fold{f}.json").read_text())["sites"]
        for f in range(len(fs))
    ]
    R = dict(np.load(DATA / f"recon_t{pos.t}{tag}.npz"))
    for rec in meta["sites"]:
        label = rec["site"]
        if rec.get("constant"):
            continue
        site = pos.sites[label]
        F = Fitter(site, cands, fs)
        Zho, rho_ho = np.zeros_like(site.Z), np.zeros_like(site.rho)
        err = tot = 0.0
        for f, te in enumerate(fs):
            Af = nested[f].get(label, [])
            Zho[te] = F.reconstruct(Af, heldout=True)[te]  # fold f predicted from its training rows
            Pf = np.hstack([byname[n].Phi for n in Af]) if Af else np.zeros((pos.D, 0))
            rho_ho[te] = fit_scalar(Pf, site.rho, [te])[1][te]
            E = (site.Z[te] - Zho[te]) @ site.W
            Y0 = (site.Z[te] - np.delete(site.Z, te, 0).mean(0)) @ site.W
            err += float((E**2).sum())
            tot += float((Y0**2).sum())
        rec["heldout_r2_nested"] = 1 - err / tot
        rec["rho_r2_nested"] = 1 - float(
            ((site.rho - rho_ho) ** 2).sum() / ((site.rho - site.rho.mean()) ** 2).sum()
        )
        counts: dict[str, int] = {}
        for f in range(len(fs)):
            for n in nested[f].get(label, []):
                counts[n] = counts.get(n, 0) + 1
        for a in rec["accepted"]:
            a["stability"] = counts.get(a["name"], 0) / len(fs)
        accepted = {a["name"] for a in rec["accepted"]}
        rec["fold_only"] = {n: c / len(fs) for n, c in counts.items() if n not in accepted}
        R[label + "/Yhat_heldout"] = (Zho @ site.W).astype(np.float32)
        R[label + "/rhohat_heldout"] = rho_ho.astype(np.float32)
        print(f"t={pos.t} {label}: nested held-out R2 {rec['heldout_r2_nested']:.3f} (refit only {rec['heldout_r2']:.3f}), "
              f"rho {rec['rho_r2_nested']:.3f}", flush=True)  # fmt: skip
    meta["nested"] = True
    (DATA / f"qloop_t{pos.t}{tag}.json").write_text(json.dumps(meta, indent=1))
    np.savez(DATA / f"recon_t{pos.t}{tag}.npz", **R)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("t", type=int)
    ap.add_argument("--sites", default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    pos = load(args.t, args.sites.split(",") if args.sites else None)
    fs = folds(pos, primary_scheme(pos.t))

    if args.fold is not None:  # nested selection on the training rows of one primary fold
        sub = restrict(pos, np.setdiff1d(np.arange(pos.D), fs[args.fold]))
        res = loop(sub, candidates(sub, args.extra))
        (DATA / f"qloop_t{pos.t}{tag}_fold{args.fold}.json").write_text(
            json.dumps({"fold": args.fold, "sites": {k: A for k, (A, _) in res.items()}}, indent=1)
        )
        return
    cands = candidates(pos, args.extra)
    if args.combine:
        combine(pos, cands, tag)
        return

    byname = {c.name: c for c in cands}
    diag = {s: folds(pos, s) for s in (["b-values", "a-values"] if pos.t >= 3 else [])}
    out, recon = [], {}
    res = loop(pos, cands)
    for label, site in pos.sites.items():
        Y = site.Y
        recon[label + "/Y"] = Y.astype(np.float32)
        recon[label + "/cols"] = site.cols
        recon[label + "/rho"] = site.rho.astype(np.float32)
        if label not in res:
            out.append({"site": label, "n": site.W.shape[1], "k": site.W.shape[0], "dims": 0, "accepted": [],
                        "constant": True, "log": ["constant over the domain: nothing to fit"]})  # fmt: skip
            for key in ("/Yhat", "/Yhat_heldout"):
                recon[label + key] = Y.astype(np.float32)
            for key in ("/rhohat", "/rhohat_heldout"):
                recon[label + key] = site.rho.astype(np.float32)
            continue
        A, log = res[label]
        F = Fitter(site, cands, fs)
        P = np.hstack([byname[n].Phi for n in A]) if A else np.zeros((pos.D, 0))
        rho_in, rho_ho = fit_scalar(P, site.rho, fs)
        rec = {
            "site": label, "n": site.W.shape[1], "k": site.W.shape[0], "dims": F.dims(A),
            "accepted": [{"name": n, "dims": F.dims([n]), "drop_one_loss": round(l_, 5), "chance": round(c_, 5)}
                         for n, l_, c_ in F.drop_one(A)],
            "r2_in": 1 - F.rss(A) / F.tot, "heldout_r2": F.heldout_r2(A),
            "important_share_in": F.important_share(A, heldout=False), "important_share_heldout": F.important_share(A),
            "rho_r2_in": 1 - float(((site.rho - rho_in) ** 2).sum() / ((site.rho - site.rho.mean()) ** 2).sum()),
            "log": log,
        }  # fmt: skip
        for s, fd in diag.items():
            rec[f"heldout_r2_{s}"] = Fitter(
                site, [byname[n] for n in A] or cands[:1], fd
            ).heldout_r2(A)
        recon[label + "/Yhat"] = (F.reconstruct(A) @ site.W).astype(np.float32)
        recon[label + "/Yhat_heldout"] = (F.reconstruct(A, heldout=True) @ site.W).astype(
            np.float32
        )  # refit only; --combine replaces it
        recon[label + "/rhohat"] = rho_in.astype(np.float32)
        recon[label + "/rhohat_heldout"] = rho_ho.astype(np.float32)
        out.append(rec)
        print(f"t={pos.t} {label}: n={rec['n']} dims={rec['dims']} R2 {rec['r2_in']:.3f} held-out {rec['heldout_r2']:.3f} "
              f"important {rec['important_share_in']:.3f}/{rec['important_share_heldout']:.3f} rho R2 {rec['rho_r2_in']:.3f} "
              f"{[a['name'] for a in rec['accepted']]}", flush=True)  # fmt: skip
    meta = {"t": pos.t, "position": POS_NAMES[pos.t], "D": pos.D, "scheme": primary_scheme(pos.t),
            "MIN_INC": MIN_INC, "STOP_SHARE": STOP_SHARE, "nested": False, "sites": out}  # fmt: skip
    (DATA / f"qloop_t{pos.t}{tag}.json").write_text(json.dumps(meta, indent=1))
    np.savez(DATA / f"recon_t{pos.t}{tag}.npz", rows=pos.rows, **recon)


if __name__ == "__main__":
    main()
