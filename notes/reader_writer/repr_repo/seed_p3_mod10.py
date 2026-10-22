"""Add the b mod 10 representations at position 3 (the token b): block-0 Fourier planes, blocks 1..11 residue-mean
spans, every kept reader's position-3 response for blocks 0..11, and the block-0 mod-10 readers' writers.

Same definitions as seed_l0_mod10.py (block 0) and seed_blocks_mod10.py (blocks 1..11), with r = b mod 10 and the
residue means taken over all 320,000 prompts (every a, b and op; 32,000 per residue), from the stored inner
activations of the alive-only model at position 3 (`variance/alive_only/L<b>/inner.npy`[:, 2]). Each block's input
is used in the coordinates y = Q^T x_V of an orthonormal basis Q of the kept readers' read vectors (x_V = Q y), so
the 4096-dimensional x_V is never formed; stored bases are lifted back to R^4096. Writers: as in
seed_l0_mod10_writers.py, from variance_kl5e-5/results/L0/{variance,J}_pos3.npz. Nothing is centred.

    python seed_p3_mod10.py
"""

import datetime
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mod10_L0 import RUN  # noqa: E402
from repo import ROOT, Repo  # noqa: E402
from seed_blocks_mod10 import circulant  # noqa: E402
from seed_l0_mod10_writers import cover  # noqa: E402

SRC = RUN / "analysis/reader_writer/variance/alive_only"
VAR = RUN / "analysis/reader_writer/variance_kl5e-5/results/L0"
N = 400
ROWS = np.arange(2 * N * N)
LAB = (ROWS % N + 1) % 10  # b mod 10
ENC = {1: "b mod 10", 2: "b mod 5", 3: "b mod 10", 4: "b mod 5", 5: "b mod 2"}
MOD10_R2 = 0.5


def orth(X: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def block_inputs(repo: Repo, b: int):
    rds = [rd for rd in repo.readers.values() if rd["block"] == b]
    z = np.load(SRC / f"L{b}" / "comps.npz")
    col = {(str(k), int(i)): j for j, (k, i) in enumerate(zip(z["kind"], z["index"], strict=True))}
    cols = np.asarray([col[(rd["kind"], rd["component"])] for rd in rds])
    V = np.stack([z[f"V_{rd['kind']}"][:, list(z["index"][z["kind"] == rd["kind"]]).index(rd["component"])]
                  for rd in rds], 1).astype(np.float64)  # fmt: skip
    T = np.asarray(
        np.load(SRC / f"L{b}" / "inner.npy", mmap_mode="r")[:, 2, :][:, cols], np.float64
    )
    Q = np.linalg.qr(V)[0]
    y = T @ (Q.T @ V @ np.linalg.pinv(V.T @ V)).T  # (320000, K) coordinates of x_V
    W = (Q.T @ V) / np.linalg.norm(V, axis=0)  # g_c = y . W[:, c]
    return rds, Q, y, W, T / np.linalg.norm(V, axis=0)


def r2(f: np.ndarray, y: np.ndarray, B: np.ndarray, W: np.ndarray) -> np.ndarray:
    g = (y @ B) @ (B.T @ W)
    return 1 - ((f - g) ** 2).sum(0) / (f**2).sum(0)


def main() -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copytree(ROOT, ROOT.parent / f"representation_repository.bak-{stamp}")
    repo = Repo()
    thr = repo.meta["response_threshold"]
    today = datetime.date.today().isoformat()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()  # fmt: skip
    prov = {"script": "notes/reader_writer/repr_repo/seed_p3_mod10.py", "commit": commit, "date": today,
            "notebook": "Mod 10 representations at b (position 3)"}  # fmt: skip
    method = ("input x_V replaced by its projection onto the representation, uncentred: r2 = 1 - sum (f - g)^2 / "
              "sum f^2 over all 320,000 prompts (every a, b, op) at position 3")  # fmt: skip
    for b in range(12):
        rds, Q, y, W, f = block_inputs(repo, b)
        mu = np.stack([y[r == LAB].mean(0) for r in range(10)])
        psi = np.exp(-2j * np.pi * np.outer(np.arange(1, 6), np.arange(10)) / 10) @ mu
        space = (f"block-{b} MLP input (output of the post-attention RMSNorm, R^4096) at position 3 (token b), "
                 "alive-only model of the ceiling filter addsub-08-last-pos, projected onto the span of the kept "
                 f"block-{b} readers' read vectors")  # fmt: skip
        if b == 0:
            reps, bases = [], {}
            for k in range(1, 6):
                raw = (
                    np.stack([psi[k - 1].real, psi[k - 1].imag], 1)
                    if k < 5
                    else psi[k - 1].real[:, None]
                )
                Bc = orth(raw)
                rid = f"L0.p3.{ENC[k].replace(' ', '_')}.k{k}"
                repo.add_representation({
                    "id": rid, "block": 0, "position": 3, "space": space, "dim": int(Bc.shape[1]), "encodes": ENC[k],
                    "kind": "fourier_plane" if k < 5 else "fourier_line",
                    "details": {"frequency_k": k, "period": 10 / k,
                                "residue_angle": f"residue r = b mod 10 sits at angle 2*pi*{k}*r/10",
                                "circulant": circulant(mu),
                                "npz_keys": {"basis": "(4096, d) orthonormal basis",
                                             "residue_coords": "(10, d) coordinates of mu_r (raw, uncentred) in basis"}},
                    "provenance": prov,
                }, {"basis": Q @ Bc, "residue_coords": mu @ Bc})  # fmt: skip
                reps.append(rid)
                bases[rid] = Bc
            r2s = {rid: r2(f, y, B, W) for rid, B in bases.items()}
            for i, rd in enumerate(rds):
                by = {rid: float(v[i]) for rid, v in r2s.items()}
                resp = [rid for rid, v in by.items() if v >= thr]
                Bs = orth(np.concatenate([bases[r] for r in resp], 1)) if resp else None
                repo.set_position(rd["id"], 3, {
                    "r2": float(r2(f[:, i : i + 1], y, Bs, W[:, i : i + 1])[0]) if resp else 0.0,
                    "responds_to": resp, "r2_by_rep": by, "method": method, "updated": today,
                })  # fmt: skip
        else:
            B = orth(mu.T)
            D = orth(np.concatenate([np.stack([p.real, p.imag], 1) for p in psi], 1))
            rid = f"L{b}.p3.b_mod_10.means"
            repo.add_representation({
                "id": rid, "block": b, "position": 3, "space": space, "dim": int(B.shape[1]), "encodes": "b mod 10",
                "kind": "residue_mean_span",
                "details": {"circulant": circulant(mu),
                            "meaning": "span of the ten raw residue means over b mod 10 (all a, op); see L<b>.p1.a_mod_10.means",
                            "npz_keys": {"basis": "(4096, 10) orthonormal basis", "mu": "(10, 4096) raw residue means",
                                         "period10_basis": "(4096, 9) frequency 1..5 part"}},
                "provenance": prov,
            }, {"basis": Q @ B, "mu": mu @ Q.T, "period10_basis": Q @ D})  # fmt: skip
            r2f, r2p = r2(f, y, B, W), r2(f, y, D, W)
            for i, rd in enumerate(rds):
                repo.set_position(rd["id"], 3, {
                    "r2": float(r2f[i]) if r2f[i] >= thr else 0.0, "responds_to": [rid] if r2f[i] >= thr else [],
                    "r2_by_rep": {rid: float(r2f[i])}, "r2_period10": float(r2p[i]),
                    "method": method + "; r2_period10 uses the frequency 1..5 part only", "updated": today,
                })  # fmt: skip
        print(f"block {b}: {len(rds)} readers", flush=True)
    # writers of the block-0 position-3 mod-10 readers
    var, jz = np.load(VAR / "variance_pos3.npz"), np.load(VAR / "J_pos3.npz")
    ids = {"gate": list(var["ids_gate"]), "up": list(var["ids_up"])}
    down = [int(i) for i in var["ids_down"]]
    for rd in repo.readers.values():
        p = rd["positions"]["3"]
        if rd["block"] != 0 or not p or p["r2"] < MOD10_R2:
            continue
        j = ids[rd["kind"]].index(rd["component"])
        key = "Mu" if rd["kind"] == "up" else "Mg"
        a, phi = var[f"a_{key}"][:, j], var[f"phi_{key}"][:, j]
        frac = np.abs(a) / np.abs(a).sum()
        entries = []
        for w in cover(a):
            e = {"writer": f"L0.down.{down[w]}", "site": "layers.0.mlp.down_proj", "component": down[w],
                 "phi": float(phi[w]), "reader_fraction": float(frac[w])}  # fmt: skip
            if rd["kind"] == "up":
                pj = jz["phi_J"][w, j]
                e["routing"] = {"phi_sub": float(jz["phi_Mu"][w, j]), "phi_mean_gate": float(jz["phi_Mu0"][w, j]),
                                "gates": [{"reader": f"L0.gate.{int(ids['gate'][g])}", "phi_J": float(pj[g])}
                                          for g in cover(pj)]}  # fmt: skip
            entries.append(e)
        p["writers"] = {"entries": entries, "covered_fraction": float(sum(e["reader_fraction"] for e in entries)),
                        "source": str(VAR), "method": "as at position 1 (seed_l0_mod10_writers.py), position 3",
                        "provenance": prov, "updated": today}  # fmt: skip
        print(
            rd["id"],
            [
                (e["writer"], round(e["reader_fraction"], 2), round(e["phi"], 2))
                for e in entries[:3]
            ],
        )
    repo.save()


if __name__ == "__main__":
    main()
