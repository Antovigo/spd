"""Add blocks 1..11's a mod 10 representations, their readers' responses, and the transitions of blocks 0..11.

Representations (kind residue_mean_span), one per block b = 1..11, id L<b>.p1.a_mod_10.means: the span of the
ten raw residue means mu_r of the block-b MLP input at position 1, in the space of block 0's planes (MLP input
projected onto the span of the kept block-b readers' read vectors; inner activations from the stored
alive-only model, variance/alive_only/L<b>/inner.npy). The span includes the direction shared by all residues.
Stored statistics:
  circulant     G = [mu_r . mu_r'] (10 x 10); F the unitary 10-point DFT (F[k, r] = exp(-2 pi i k r/10)/sqrt(10));
                offdiag = share of |F G F^H|^2 off the diagonal within frequencies 1..5 (0 for residues equally
                spaced on circles); the same for 200 random relabellings of the residues (seed 0)
  fourier_amp   ||Psi_k||, k = 1..5;  roundness: minor/major axis of span{Re Psi_k, Im Psi_k}, k = 1..4
  shared_share  share of trace(G) in frequency 0 (the direction shared by all residues)
Readers of blocks 1..11 at position 1: r2 of their inner activation f_c = t_c / ||V_c|| when the input is
projected onto the span (as for block 0), and r2_period10 for its frequency 1..5 part alone. Nothing is centred.
Transitions L<b>.p1.a_mod_10, b = 0..11: from track_mod10/{analysis,residues}.json and the notebook's
per-block descriptions (../readers/track_page.py NOTES). Migrates the repository to schema 2.

    python seed_blocks_mod10.py
"""

import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mod10_L0 import RUN, A, N  # noqa: E402
from repo import ROOT, SCHEMA, Repo  # noqa: E402
from track_page import NOTES  # noqa: E402

SRC = RUN / "analysis/reader_writer/variance/alive_only"
TRACK = RUN / "analysis/reader_writer/track_mod10"
BLOCKS = range(1, 12)
R = A % 10
F = np.exp(-2j * np.pi * np.outer(np.arange(10), np.arange(10)) / 10) / np.sqrt(10)


def orth(X: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def offdiag(mu: np.ndarray) -> float:
    H = (F @ (mu @ mu.T) @ F.conj().T)[1:, 1:]
    return float((np.abs(H - np.diag(np.diag(H))) ** 2).sum() / (np.abs(H) ** 2).sum())


def circulant(mu: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    sh = np.asarray([offdiag(mu[rng.permutation(10)]) for _ in range(200)])
    H = F @ (mu @ mu.T) @ F.conj().T
    psi = np.exp(-2j * np.pi * np.outer(np.arange(1, 6), np.arange(10)) / 10) @ mu
    rnd = [float((lambda s: s[-1] / s[0])(np.linalg.svd(np.stack([p.real, p.imag], 1), compute_uv=False)))
           for p in psi[:4]]  # fmt: skip
    return {
        "offdiag_share_k1to5": offdiag(mu),
        "shuffled_median": float(np.median(sh)),
        "shuffled_min": float(sh.min()),
        "shuffled_n": 200,
        "shuffled_fraction_below": float((sh <= offdiag(mu)).mean()),
        "fourier_amp": [float(np.linalg.norm(p)) for p in psi],
        "roundness": rnd,
        "shared_share": float(H[0, 0].real / np.trace(H).real),
    }


def main() -> None:
    meta_p = ROOT / "meta.json"
    meta = json.loads(meta_p.read_text())
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copytree(
        ROOT,
        ROOT.parent / f"representation_repository.bak-{stamp}",
        ignore=shutil.ignore_patterns("*.bak*"),
    )
    if meta["schema"] != SCHEMA:
        meta["schema"] = SCHEMA
        meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    repo = Repo()
    today = datetime.date.today().isoformat()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()  # fmt: skip
    prov = {"script": "notes/reader_writer/repr_repo/seed_blocks_mod10.py", "commit": commit, "date": today,
            "notebook": "Mod 10 representations, section 8"}  # fmt: skip
    thr = repo.meta["response_threshold"]
    rep_of = {0: [r for r in repo.representations if r.startswith("L0.p1.")]}
    for b in BLOCKS:
        z = np.load(SRC / f"L{b}" / "comps.npz")
        col = {
            (str(k), int(i)): j for j, (k, i) in enumerate(zip(z["kind"], z["index"], strict=True))
        }
        rds = [rd for rd in repo.readers.values() if rd["block"] == b]
        cols = np.asarray([col[(rd["kind"], rd["component"])] for rd in rds])
        V = np.concatenate([z[f"V_{rd['kind']}"][:, [list(z["index"][z["kind"] == rd["kind"]]).index(rd["component"])]]
                            for rd in rds], 1).astype(np.float64)  # fmt: skip
        T = np.asarray(
            np.load(SRC / f"L{b}" / "inner.npy", mmap_mode="r")[(A - 1) * N, 0, :][:, cols],
            np.float64,
        )
        vnorm = np.linalg.norm(V, axis=0)
        Fc = T / vnorm
        xv = (V @ np.linalg.pinv(V.T @ V) @ T.T).T
        mu = np.stack([xv[r == R].mean(0) for r in range(10)])  # (10, 4096) raw residue means
        B = orth(mu.T)
        psi = np.exp(-2j * np.pi * np.outer(np.arange(1, 6), np.arange(10)) / 10) @ mu
        D = orth(np.concatenate([np.stack([p.real, p.imag], 1) for p in psi], 1))
        rid = f"L{b}.p1.a_mod_10.means"
        repo.add_representation(
            {
                "id": rid,
                "block": b,
                "position": 1,
                "space": f"block-{b} MLP input (output of the post-attention RMSNorm, R^4096) at position 1 (token a), "
                "alive-only model of the ceiling filter addsub-08-last-pos, projected onto the span of the kept "
                f"block-{b} readers' read vectors",
                "dim": int(B.shape[1]),
                "encodes": "a mod 10",
                "kind": "residue_mean_span",
                "details": {
                    "circulant": circulant(mu),
                    "meaning": "span of the ten raw residue means; circulant.offdiag_share_k1to5 is 0 when the residues "
                    "sit equally spaced on circles (Fourier planes meaningful) and is compared with random relabellings",
                    "npz_keys": {
                        "basis": "(4096, 10) orthonormal basis of span{mu_r}",
                        "mu": "(10, 4096) raw residue means",
                        "period10_basis": "(4096, 9) orthonormal basis of the frequency 1..5 Fourier components",
                    },
                },
                "provenance": prov,
            },  # fmt: skip
            {"basis": B, "mu": mu, "period10_basis": D},
        )
        rep_of[b] = [rid]
        g_full = (xv @ B) @ (B.T @ V) / vnorm
        g_p10 = (xv @ D) @ (D.T @ V) / vnorm
        r2f = 1 - ((Fc - g_full) ** 2).sum(0) / (Fc**2).sum(0)
        r2p = 1 - ((Fc - g_p10) ** 2).sum(0) / (Fc**2).sum(0)
        for i, rd in enumerate(rds):
            repo.set_position(rd["id"], 1, {
                "r2": float(r2f[i]) if r2f[i] >= thr else 0.0,
                "responds_to": [rid] if r2f[i] >= thr else [],
                "r2_by_rep": {rid: float(r2f[i])},
                "r2_period10": float(r2p[i]),
                "method": "input x_V(a) replaced by its projection onto the span of the residue means, uncentred: "
                "r2 = 1 - sum (f - g)^2 / sum f^2; a = 1..400; r2_period10 uses the frequency 1..5 part only",
                "updated": today,
            })  # fmt: skip
        n_resp = int((r2f >= thr).sum())
        print(f"block {b}: {len(rds)} readers, rank {B.shape[1]}, offdiag {circulant(mu)['offdiag_share_k1to5']:.3f}, "
              f"respond {n_resp}, r2_period10 >= 0.5: {int((r2p >= 0.5).sum())}", flush=True)  # fmt: skip
    an = json.loads((TRACK / "analysis.json").read_text())
    rq = json.loads((TRACK / "residues.json").read_text())
    defs = {
        "space": "residual stream at position 1 (pre-norm), alive-only model rebuilt in float64 (track_build.py)",
        "D": "D^X: span of the frequency 1..5 Fourier components of the residue means of the stream X entering the sublayer",
        "Delta": "S(X) - S(X - Pi_D X): what sublayer S writes because its input contains D^X",
        "out_gain": "||Psi_k(S(X))|| / ||Psi_k(X)||, k = 1..5",
        "out_share_from_D": "Re<Psi_k(S(X)), Psi_k(Delta)> / ||Psi_k(S(X))||^2",
        "r2_lin": "uncentred r2 of Delta ~= L z, z = coordinates of X in D^X, one 4096 x 9 matrix L, no intercept",
        "secant": "uncentred r2 of Delta ~= sum_c kappa_c dt_c over MLP readers c (dt_c: change of t_c when D is removed)",
        "additivity": "uncentred r2 of Delta ~= sum_c Delta_c (one reader changed at a time)",
        "write_into_output_rel": "per residue: ||Pi_D' w_r|| / ||Pi_D mu_r||, w_r the residue mean of the MLP output, D' of the stream after",
        "sep": "||c_r - c_r'|| / sqrt((s_r^2 + s_r'^2)/2), c_r the residue's mean in D coordinates, s_r its RMS spread",
        "acc_loo": "leave-one-out nearest-mean accuracy of r from the D coordinates (chance 0.1)",
    }
    for b in range(0, 12):
        blk = an["blocks"][str(b)]
        tr = {
            "id": f"L{b}.p1.a_mod_10", "block": b, "position": 1,
            "input_reps": rep_of[b], "output_reps": rep_of.get(b + 1, []),
            "attention": {k: blk["attn"][k] for k in ("out_gain", "out_share_from_D", "r2_lin", "delta_over_input_D")}
            | {"r2_value_path": blk["attn"]["mechanism"]["r2_value_path"],
               "weight_on_self_mean": blk["attn"]["mechanism"]["p_self_mean"]},
            "mlp": {k: blk["mlp"][k] for k in ("out_gain", "out_share_from_D", "r2_lin", "delta_over_input_D", "beta_re",
                                               "in_input_plane")}
            | {"secant": blk["mlp"]["mechanism"]["r2_secant_sum"], "additivity": blk["mlp"]["mechanism"]["r2_sum_single"],
               "top_readers": blk["mlp"]["mechanism"]["top90"],
               "plane_cos_across": [float(np.mean(c)) for c in an["boundaries"][f"pre_{b + 1}"]["cos_prev"]]},
            "residues": {"write_into_output_rel": rq["sublayers"][f"pre_{b + 1}"]["write_into_output_rel"],
                         "radius_ratio": rq["sublayers"][f"pre_{b + 1}"]["radius_ratio"],
                         "before": rq["boundaries"][f"mid_{b}"], "after": rq["boundaries"][f"pre_{b + 1}"]},
            "description": NOTES[b],
            "definitions": defs,
            "provenance": prov | {"data": str(TRACK)},
        }  # fmt: skip
        repo.transitions[tr["id"]] = tr
    repo.save()
    print(len(repo.representations), "representations;", len(repo.transitions), "transitions")


if __name__ == "__main__":
    main()
