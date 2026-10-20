"""Create the representation repository (repo.py) and seed it with the block-0 a mod 10 Fourier planes.

Representations: for k in 1..5, span{Re Psi_k, Im Psi_k} of the block-0 MLP input at position 1, with
Psi_k = sum_r mu_r exp(-2 pi i k r / 10), mu_r the residue means of x_V(a) - trend (notation of
../readers/mod10_L0.py; i is the imaginary unit). k = 1, 3 carry a mod 10, k = 2, 4 carry a mod 5,
k = 5 (a line) carries a mod 2.

Readers: every kept alive gate / up component of blocks 0..31, positions unmeasured (null), except
block 0 at position 1. There, for each representation R, r2_R = 1 - sum_a (f~ - g_R)^2 / sum_a f~^2 over
a = 1..400, with f~ = f_c - mean f_c and g_R = V_c . Pi_R (x_V(a) - mean_a x_V) / ||V_c||. responds_to
holds the R with r2_R >= the threshold, and r2 is the same quantity for the span of responds_to.

    python seed_l0_mod10.py [--threshold 0.05] [--commit SHA]
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mod10_L0 import FILTER, SRC, VALID, A, N, class_means, kept_readers, moving_trend  # noqa: E402
from repo import ROOT, Repo  # noqa: E402

ENCODES = {1: "a mod 10", 2: "a mod 5", 3: "a mod 10", 4: "a mod 5", 5: "a mod 2"}
SPACE = (
    "block-0 MLP input (output of the post-attention RMSNorm, R^4096) at position 1 (token a), "
    "alive-only model of the ceiling filter addsub-08-last-pos, projected onto the span of the kept "
    "block-0 readers' read vectors"
)


def orth(X: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def all_kept_readers() -> list[dict]:
    text = (FILTER / "ab_grids/pruned.js").read_text().strip()
    doc = json.loads(text[len("window.PRUNED = ") :].rstrip(";"))
    (mask,) = [m for m in doc["masks"] if m["name"] == "kl_sampled/lam_5e-05"]
    alive = json.loads((FILTER / "alive/alive.json").read_text())["components"]
    out = []
    for block in range(32):
        for kind in ("gate", "up"):
            site = f"layers.{block}.mlp.{kind}_proj"
            for cid in sorted(set(alive[site]) - set(mask["pruned"].get(site, []))):
                out.append(
                    {
                        "id": f"L{block}.{kind}.{cid}",
                        "block": block,
                        "kind": kind,
                        "component": int(cid),
                        "site": site,
                        "positions": {t: None for t in ("1", "2", "3", "4")},
                    }
                )
    return out


def r2(
    Fc: np.ndarray, xc: np.ndarray, B: np.ndarray, V: np.ndarray, vnorm: np.ndarray
) -> np.ndarray:
    g = (xc @ B) @ (B.T @ V) / vnorm
    return 1 - ((Fc - g) ** 2).sum(0) / (Fc**2).sum(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args()
    today = datetime.date.today().isoformat()
    readers = all_kept_readers()
    repo = Repo.create(
        ROOT,
        {
            "run": "p-ba8b0c08",
            "step": 40000,
            "filter": "addsub-08-last-pos",
            "mask": "kl_sampled/lam_5e-05 (kl_int, sampled mask, λ=5e-05)",
            "activations": str(SRC.parent),
            "model": "alive-only: every ceiling-filter component on, everything else and the weight delta off",
            "pool": "a, b in 1..400, ops + then -",
            "positions": {"1": "a", "2": "op", "3": "b", "4": "="},
            "response_threshold": args.threshold,
            "response_threshold_meaning": "a reader responds to a representation when projecting its input onto "
            "that representation alone explains >= this share of its variance",
            "created": today,
        },
        readers,
    )
    print(f"{len(readers)} kept readers")

    cols, names, V = kept_readers()
    vnorm = np.linalg.norm(V, axis=0)
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    T = np.asarray(inner[(A - 1) * N, 0, :][:, cols], np.float64)
    F = T / vnorm
    W = V @ np.linalg.pinv(V.T @ V)
    xv = (W @ T.T).T
    xc = xv - xv.mean(axis=0)
    mu = W @ class_means(T - moving_trend(T, 10), 10, VALID).T
    muc = mu - mu.mean(axis=1, keepdims=True)
    rr = np.arange(10)
    prov = {
        "script": "notes/reader_writer/repr_repo/seed_l0_mod10.py",
        "commit": args.commit,
        "date": today,
        "notebook": "Block-0 readers of a mod 10; Block-0 a mod 10: the Fourier planes",
    }
    rep_ids = []
    for k in range(1, 6):
        psi = mu @ np.exp(-2j * np.pi * k * rr / 10)
        raw = np.stack([psi.real, psi.imag], 1) if k < 5 else psi.real[:, None]
        B = orth(raw)
        rid = f"L0.p1.{ENCODES[k].replace(' ', '_')}.k{k}"
        rep = {
            "id": rid,
            "block": 0,
            "position": 1,
            "space": SPACE,
            "dim": int(B.shape[1]),
            "encodes": ENCODES[k],
            "kind": "fourier_plane" if k < 5 else "fourier_line",
            "details": {
                "frequency_k": k,
                "period": 10 / k,
                "residue_angle": f"residue r = a mod 10 sits at angle 2*pi*{k}*r/10",
                "npz_keys": {
                    "basis": "(4096, d) orthonormal basis",
                    "re_im": "(4096, 2 or 1) Re Psi_k, Im Psi_k before orthonormalising",
                    "residue_coords": "(10, d) coordinates of mu_r - mu_bar in basis, row r",
                },
            },
            "provenance": prov,
        }
        repo.add_representation(rep, {"basis": B, "re_im": raw, "residue_coords": muc.T @ B})
        rep_ids.append(rid)

    Fc = F - F.mean(axis=0)
    per = {rid: r2(Fc, xc, repo.basis(rid), V, vnorm) for rid in rep_ids}
    counts = dict.fromkeys(rep_ids, 0)
    for j, name in enumerate(names):
        r2_by = {rid: float(per[rid][j]) for rid in rep_ids}
        resp = [rid for rid in rep_ids if r2_by[rid] >= args.threshold]
        for rid in resp:
            counts[rid] += 1
        total = 0.0
        if resp:
            B = orth(np.concatenate([repo.basis(rid) for rid in resp], 1))
            total = float(r2(Fc[:, [j]], xc, B, V[:, [j]], vnorm[[j]])[0])
        kind = {"g": "gate", "u": "up"}[name[0]]
        repo.set_position(
            f"L0.{kind}.{name[1:]}",
            1,
            {
                "r2": total,
                "responds_to": resp,
                "r2_by_rep": r2_by,
                "method": "input x_V(a) replaced by its projection onto span(responds_to), centred; "
                "a = 1..400 (position 1 depends on a only)",
                "updated": today,
            },
        )
    repo.save()
    l0 = [rd["positions"]["1"] for rd in repo.readers.values() if rd["block"] == 0]
    print("readers responding per representation:", counts)
    print(
        "L0 p1 r2 quantiles (0, .5, .9, 1):",
        np.quantile([p["r2"] for p in l0], [0, 0.5, 0.9, 1]).round(3).tolist(),
    )
    print(
        "L0 readers responding to >= 1 representation:",
        sum(bool(p["responds_to"]) for p in l0),
        "of",
        len(l0),
    )
    best = sorted(
        (
            (rd["id"], rd["positions"]["1"]["r2"])
            for rd in repo.readers.values()
            if rd["block"] == 0
        ),
        key=lambda t: -t[1],
    )[:20]
    print("top r2:", best)


if __name__ == "__main__":
    main()
