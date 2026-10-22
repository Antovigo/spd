"""Block 0 at position 3 (the token b) versus position 1 (the token a): planes, readers and writers.

Alive-only model, stored inner activations (`variance/alive_only/L0/inner.npy`, all 320,000 prompts; position index
0 is position 1, index 2 is position 3). Nothing is centred. Notation as on the page (mod10_page.py):
  x_V          block-0 MLP input projected onto span{V_c} of the 153 kept readers; coordinates y = Q^T x_V with Q an
               orthonormal basis of that span (x_V = Q y), computed from t_c = x . V_c without forming x_V
  position 1   residue r = a mod 10, 400 prompts (a = 1..400; b, op do not matter there)
  position 3   residue r = b mod 10, 320,000 prompts (every a, b, op): means mu_r over 32,000 prompts each
  Psi_k, P_k, D as on the page; r2_D(c) uncentred, f_c = t_c / ||V_c||
Also: the a mod 10 structure at position 3 (Fourier amplitudes of the means over a mod 10), the op structure, and
the writers of the position-3 mod-10 readers from the variance split (variance_kl5e-5/results/L0/variance_pos3.npz).

    python mod10_p3_L0.py        # writes <run>/analysis/reader_writer/track_mod10_p3/L0.json
"""

import json

import numpy as np
from mod10_L0 import RUN, SRC, kept_readers

OUT = RUN / "analysis/reader_writer/track_mod10_p3"
VAR = RUN / "analysis/reader_writer/variance_kl5e-5/results/L0"
N = 400


def fourier(mu: np.ndarray) -> np.ndarray:
    return np.stack([np.exp(-2j * np.pi * k * np.arange(10) / 10) @ mu for k in range(1, 6)])


def planes(ps: np.ndarray) -> list[np.ndarray]:
    out = []
    for p in ps:
        u, s, _ = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)
        out.append(u[:, s > 1e-9 * s.max()])
    return out


def offdiag(mu: np.ndarray) -> float:
    F = np.exp(-2j * np.pi * np.outer(np.arange(10), np.arange(10)) / 10) / np.sqrt(10)
    H = (F @ (mu @ mu.T) @ F.conj().T)[1:, 1:]
    return float((np.abs(H - np.diag(np.diag(H))) ** 2).sum() / (np.abs(H) ** 2).sum())


def main() -> None:
    cols, names, V = kept_readers()
    vnorm = np.linalg.norm(V, axis=0)
    Q = np.linalg.qr(V)[0]
    M = Q.T @ V @ np.linalg.pinv(V.T @ V)  # y = M t
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    rows = np.arange(2 * N * N)
    a, b, op = rows % (N * N) // N + 1, rows % N + 1, rows // (N * N)
    T3 = np.asarray(inner[:, 2, :][:, cols], np.float64)
    T1 = np.asarray(inner[(np.arange(1, N + 1) - 1) * N, 0, :][:, cols], np.float64)
    y3, y1 = T3 @ M.T, T1 @ M.T
    lab1, lab3 = np.arange(1, N + 1) % 10, b % 10
    mu1 = np.stack([y1[lab1 == r].mean(0) for r in range(10)])
    mu3 = np.stack([y3[lab3 == r].mean(0) for r in range(10)])
    mu3a = np.stack([y3[a % 10 == r].mean(0) for r in range(10)])
    F1, F3, F3a = fourier(mu1), fourier(mu3), fourier(mu3a)
    P1, P3 = planes(F1), planes(F3)
    D1 = np.linalg.qr(np.concatenate(P1, 1))[0]
    D3 = np.linalg.qr(np.concatenate(P3, 1))[0]
    f3, f1 = T3 / vnorm, T1 / vnorm
    W = (Q.T @ V) / vnorm  # (153, K): g_c = y . W[:, c]

    def r2(f: np.ndarray, y: np.ndarray, B: np.ndarray) -> np.ndarray:
        g = (y @ B) @ (B.T @ W)
        return 1 - ((f - g) ** 2).sum(0) / (f**2).sum(0)

    r2_3, r2_1 = r2(f3, y3, D3), r2(f1, y1, D1)
    r2_3_with1 = r2(f3, y3, D1)  # position-3 activations explained by the position-1 subspace
    mod3 = [names[i] for i in np.argsort(-r2_3) if r2_3[i] >= 0.5]
    mod1 = [names[i] for i in np.argsort(-r2_1) if r2_1[i] >= 0.5]
    # residue profiles and one-hotness at position 3 (raw class means of f over b mod 10)
    prof = {
        names[i]: [float(f3[lab3 == r, i].mean()) for r in range(10)]
        for i in range(len(names))
        if r2_3[i] >= 0.5
    }
    # within-class: how much of each mod-10 reader's activation varies with the context (a, op) for fixed b
    ctx_share = {}
    for i in [names.index(n) for n in mod3]:
        f = f3[:, i].reshape(2, N, N)  # (op, a, b)
        ctx_mean = f.mean((0, 1), keepdims=True)  # the part that depends on b only
        ctx_share[names[i]] = float(((f - ctx_mean) ** 2).sum() / (f**2).sum())
    # writers at position 3 (variance split) for the position-3 mod-10 readers
    vz = np.load(VAR / "variance_pos3.npz")
    writers = {}
    for n in mod3:
        k = "Mu" if n[0] == "u" else "Mg"
        ids = list(vz["ids_up" if n[0] == "u" else "ids_gate"])
        j = ids.index(int(n[1:]))
        aa = vz[f"a_{k}"][:, j]
        o = np.argsort(-np.abs(aa))[:3]
        writers[n] = [[f"d{int(vz['ids_down'][w])}", float(np.abs(aa[w]) / np.abs(aa).sum()), float(vz[f"phi_{k}"][w, j])]
                      for w in o]  # fmt: skip
    res = {
        "amp_p1": [float(np.linalg.norm(p)) for p in F1],
        "amp_p3": [float(np.linalg.norm(p)) for p in F3],
        "amp_p3_a": [float(np.linalg.norm(p)) for p in F3a],
        "cos_planes_p1_p3": [np.linalg.svd(P1[k].T @ P3[k], compute_uv=False).tolist() for k in range(5)],
        "offdiag_p1": offdiag(mu1), "offdiag_p3": offdiag(mu3),
        "norm_mu_p1": float(np.linalg.norm(mu1, axis=1).mean()), "norm_mu_p3": float(np.linalg.norm(mu3, axis=1).mean()),
        "op_diff_norm_p3": float(np.linalg.norm(y3[op == 0].mean(0) - y3[op == 1].mean(0))),
        "mod10_readers_p1": mod1, "mod10_readers_p3": mod3,
        "r2_D_p3": {names[i]: float(r2_3[i]) for i in np.argsort(-r2_3)[:20]},
        "r2_D_p1_of_same": {n: float(r2_1[names.index(n)]) for n in set(mod1) | set(mod3)},
        "r2_p3_explained_by_p1_subspace": {n: float(r2_3_with1[names.index(n)]) for n in mod3},
        "profiles_p3": prof,
        "context_share_p3": ctx_share,
        "writers_p3": writers,
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "L0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("profiles_p3",)}, indent=1))


if __name__ == "__main__":
    main()
