"""Which components write each Fourier code of a quantity at a read point (direct contributions).

    python -m param_decomp.arith_repr.autointerp.code_attrib --dataset <filter>/dataset
        --resid <original-resid dir> --uv uv_alive.npz --wiring wiring.npz --out code_attrib.npz

For a read point r (`attn_in.l` / `mlp_in.l`), position p, operation o and quantity q in
{a, b, sum (a+b), diff (a-b)}, the code at frequency k is the complex vector
`z(k) = mean_i exp(-2 pi i k q_i / 100) x_i` of the post-norm read input `x`. A residual writer c
upstream of r adds `w_c(i) U_c` to the raw stream at p, i.e. about `w_c(i) (ln_r * U_c) / rms_r,p`
to x; its share of the code is `Re <z(k), z_c(k)> / |z(k)|^2` (shares of all writers, the token
embedding and the unexplained rest sum to 1). `share` (n_cases, n_writer + 1, 51): last row =
the embedding (position 1/3 only). `cases` lists (r, p, o, q)."""

import argparse
import json
from pathlib import Path

import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.resid import load_read_input

CASES = [
    *[(f"mlp_in.{li}", 1, 0, "a") for li in (0, 1, 2, 3, 5, 8, 12)],
    *[(f"attn_in.{li}", 1, 0, "a") for li in (16, 18, 20)],
    *[(f"attn_in.{li}", 3, 0, "b") for li in (2, 15)],
    *[(f"mlp_in.{li}", 3, 0, "b") for li in (0, 1, 2, 5)],
    *[(f"mlp_in.{li}", 4, o, q) for li in (15, 16, 17) for o in (0, 1) for q in ("a", "b")],
    *[(f"mlp_in.{li}", 4, 0, "sum") for li in (19, 20, 21, 23, 26, 29, 31)],
    *[(f"mlp_in.{li}", 4, 1, "diff") for li in (19, 20, 21, 23, 26, 29, 31)],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--wiring", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ix = np.load(args.dataset / "index.npz")
    a, b, op, tokens = ix["a"], ix["b"], ix["op"], ix["tokens"]
    site, cidx, kind, layer = ix["comp_site"], ix["comp_index"], ix["comp_kind"], ix["comp_layer"]
    wz = np.load(args.wiring)
    writers, rms = wz["writers"], wz["rms"]
    uv = np.load(args.uv)
    U = np.zeros((writers.size, 4096), np.float32)
    for s in np.unique(site[writers]):
        ids = {int(i): j for j, i in enumerate(uv[s + ".ids"])}
        Us = uv[s + ".U"]
        for n, c in enumerate(writers):
            if site[c] == s:
                U[n] = Us[ids[int(cidx[c])]]
    norms = np.load(args.resid / "norms.npz")
    w = Weights()
    ci_mm = np.load(args.dataset / "original" / "ci.npy", mmap_mode="r")
    inner_mm = np.load(args.dataset / "original" / "inner.npy", mmap_mode="r")
    W_by_pos: dict[int, np.ndarray] = {}
    embed_rows: dict[int, np.ndarray] = {}
    ks = np.arange(51)
    shares, zpow = [], []
    for r, p, o, q in CASES:
        if p not in W_by_pos:
            W_by_pos[p] = (np.asarray(inner_mm[:, p, :])[:, writers]) * (
                np.asarray(ci_mm[:, p, :], np.float32)[:, writers] > 0.01
            )
            tok = np.unique(tokens[:, p])
            E = w.get_rows("model.embed_tokens.weight", tok)
            emb = np.zeros((tokens[:, p].max() + 1, 4096), np.float32)
            emb[tok] = E
            embed_rows[p] = emb[tokens[:, p]]
        stream, li = r.split(".")
        li = int(li)
        m = op == o
        qv = {"a": a, "b": b, "sum": a + b, "diff": a - b}[q][m]
        phi = np.exp(-2j * np.pi * np.outer(ks, qv) / 100) / m.sum()  # (K, n)
        x = load_read_input(args.resid, r, p)[m]
        z = phi @ x  # (K, d)
        ln = (norms["ln1"] if stream == "attn_in" else norms["ln2"])[li]
        rp = rms[2 * li + (stream == "mlp_in"), p]
        # writers strictly upstream of this read point
        up = (layer[writers] < li) | (
            (layer[writers] == li) & (kind[writers] == "self_attn.o_proj") & (stream == "mlp_in")
        )
        u = (U * ln[None]) / rp * up[:, None]  # (n_w, d)
        s = phi @ W_by_pos[p][m]  # (K, n_w)
        M = np.conj(z) @ u.T  # (K, n_w)
        norm2 = np.maximum((np.abs(z) ** 2).sum(1), 1e-30)
        sh = np.real(s * M) / norm2[:, None]
        e = phi @ embed_rows[p][m]  # (K, d)
        e_share = np.real((np.conj(z) * (e * ln[None] / rp)).sum(1)) / norm2
        shares.append(np.concatenate([sh.T, e_share[None]], 0).astype(np.float32))
        tot = (x - x.mean(0)) ** 2
        zpow.append(norm2 / tot.sum(1).mean())
        print(
            r,
            p,
            o,
            q,
            "writers sum k=1,10,20:",
            sh.sum(1)[[1, 10, 20]].round(2),
            "embed",
            e_share[[1, 10, 20]].round(2),
            flush=True,
        )
    np.savez(
        args.out, share=np.stack(shares), zpow=np.stack(zpow), writers=writers,
        cases=np.array([json.dumps(c) for c in CASES]),
    )  # fmt: skip


if __name__ == "__main__":
    main()
