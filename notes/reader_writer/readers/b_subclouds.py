"""The two sub-clouds of the copy of b mod 10 at "=" in the first Fourier plane (block 15, set "b", 2,000 prompts;
alive-only and original models).

O_3 is block 15's attention write at "=" (position 4) from b's position (3); P_1 is the k = 1 Fourier plane (4096 x 2,
orthonormal) of O_3's residue means, residue r = b mod 10. For a prompt n of residue r, rho = (P_1^T O_3(n)) . u_r is its
radial coordinate (u_r the unit vector of the residue-r mean) and rho_n = rho / mean_r rho its radius relative to its
residue. The two sub-clouds are the two groups of a 2-means split of rho_n. Head 13's part is rho_13 = p_13 * c, with
p_13 its attention weight on position 3 and c = rho of its value write at weight 1 (depends on the stream at
position 3, which sees <BOS>, a, op and b).

    python b_subclouds.py
"""

import glob
import json
from pathlib import Path

import numpy as np
from copy_analyze import attention_parts, copied
from copy_build import OUT
from copy_figs import SUB, planes
from mlp0_mod10 import rope
from mod10_L0 import HUB, bf16_tensor
from track_build import load_blocks, rmsnorm

L, S, H = 15, 3, 13


def orig_parts(snap, wmap, cfg, X):
    """Original block L: pattern at "=" (n, 32, 5), the write from S (n, 4096) and head H's part of it."""

    def w(name):
        return bf16_tensor(snap + wmap[name], name).astype(np.float64)

    rep = np.arange(32) // 4
    pre = f"model.layers.{L}."
    n, T, _ = X.shape
    x = rmsnorm(X, w(pre + "input_layernorm.weight"))
    q = rope((x[:, 4] @ w(pre + "self_attn.q_proj.weight").T).reshape(n, 32, 128), 4, cfg)
    Wk, Wv = w(pre + "self_attn.k_proj.weight"), w(pre + "self_attn.v_proj.weight")
    k = np.stack([rope((x[:, t] @ Wk.T).reshape(n, 8, 128), t, cfg) for t in range(T)], 1)[
        :, :, rep
    ]
    v = (x[:, S] @ Wv.T).reshape(n, 8, 128)[:, rep]
    s = np.einsum("nhd,nthd->nht", q, k) / np.sqrt(128)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    Wo = w(pre + "self_attn.o_proj.weight").T
    z = p[:, :, S, None] * v
    Os = z.reshape(n, 32 * 128) @ Wo
    Oh = z[:, H] @ Wo[H * 128 : (H + 1) * 128]
    return p, Os, Oh


def kmeans2(y):
    c = np.percentile(y, [10, 90]).astype(float)
    for _ in range(100):
        g = np.abs(y[:, None] - c[None]).argmin(1)
        c = np.array([y[g == j].mean() for j in range(2)])
    return g, c


def rate(g, lab):
    return {str(int(u)): float((g[lab == u] == 0).mean()) for u in np.unique(lab)}


def main():
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    st = np.load(OUT / "stream_b.npz")
    sub = SUB["b"]
    a, b, op = st["a"][sub], st["b"][sub], st["op"][sub]
    lab = b % 10
    blk = load_blocks()[0][L]
    out = {}
    for name in ("alive-only", "original"):
        if name == "alive-only":
            X = st[f"X{L}"][sub].astype(np.float64)
            parts = attention_parts(blk, X)
            p = parts["p"]
            Os = copied(blk, parts, S)
            Oh = copied(blk, parts, S, heads=np.array([H]))
        else:
            X = np.load(OUT / "orig_cache_b.npz")["Xor"].astype(np.float64)
            p, Os, Oh = orig_parts(snap, wmap, cfg, X)
        P1 = planes(Os, lab)[0][0]
        mu = np.stack([(Os[lab == q] @ P1).mean(0) for q in range(10)])
        u = (mu / np.linalg.norm(mu, axis=1, keepdims=True))[lab]
        rho = ((Os @ P1) * u).sum(1)
        rho_h = ((Oh @ P1) * u).sum(1)
        mr = np.array([rho[lab == q].mean() for q in range(10)])[lab]
        rn, rhn = rho / mr, rho_h / mr
        g, cen = kmeans2(rn)
        if cen[0] > cen[1]:
            g, cen = 1 - g, cen[::-1]
        ph = p[:, H, S]
        c = rho_h / ph
        r = {
            "centres_rho_n_small_big": cen.tolist(),
            "fraction_small": float((g == 0).mean()),
            "small_rate_by_op": rate(g, op),
            "small_rate_by_a": rate(g, a),
            "small_rate_by_b_range": {
                f"{lo}-{hi}": float((g[(b >= lo) & (b <= hi)] == 0).mean())
                for lo, hi in ((1, 9), (10, 99), (100, 199), (200, 299), (300, 400))
            },  # fmt: skip
            "small_rate_by_residue": rate(g, lab),
            "by_cloud": {},
        }
        for j, nm in ((0, "small"), (1, "big")):
            m = g == j
            r["by_cloud"][nm] = {
                "n": int(m.sum()),
                "rho_n_mean": float(rn[m].mean()),
                "head13_rho_n_mean": float(rhn[m].mean()),
                "other_heads_rho_n_mean": float((rn - rhn)[m].mean()),
                "p13_mean": float(ph[m].mean()),
                "c_over_residue_mean_rho": float((c / mr)[m].mean()),
                "pattern_head13_mean": p[m, H].mean(0).tolist(),
            }
        out[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
    (OUT / "b_subclouds.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
