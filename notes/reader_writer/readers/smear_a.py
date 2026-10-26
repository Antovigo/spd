"""Within-a spread of the copy of a at "=" after block 16's attention write (set "a", 2,000 prompts), alive-only and
original. Groups: prompts with the same integer a (1..400, 4-6 prompts each, varying b and op). In the period-10
subspace D of E' (stream at "=" after the write), splits the within-group spread into the part along the group mean
(radial) and the rest, for E' and for each source position's write O_j = sum_h p_h(j) v_h(j) W_o (j = 0..4).

    python smear_a.py
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

L, S = 16, 1


def orig_parts(snap, wmap, cfg, X):
    """Original block L: pattern at "=" (n, 32, 5) and the per-source writes O_j (n, 5, 4096)."""

    def w(name):
        return bf16_tensor(snap + wmap[name], name).astype(np.float64)

    H, KV, dh = 32, 8, 128
    rep = np.arange(H) // (H // KV)
    pre = f"model.layers.{L}."
    n, T, _ = X.shape
    x = rmsnorm(X, w(pre + "input_layernorm.weight"))
    q = rope((x[:, 4] @ w(pre + "self_attn.q_proj.weight").T).reshape(n, H, dh), 4, cfg)
    Wk, Wv = w(pre + "self_attn.k_proj.weight"), w(pre + "self_attn.v_proj.weight")
    k = np.stack([rope((x[:, t] @ Wk.T).reshape(n, KV, dh), t, cfg) for t in range(T)], 1)[
        :, :, rep
    ]
    v = np.stack([(x[:, t] @ Wv.T).reshape(n, KV, dh) for t in range(T)], 1)[:, :, rep]
    s = np.einsum("nhd,nthd->nht", q, k) / np.sqrt(dh)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    Wo = w(pre + "self_attn.o_proj.weight").T
    Osrc = np.stack([(p[:, :, t, None] * v[:, t]).reshape(n, H * dh) @ Wo for t in range(T)], 1)
    Oh = np.einsum(
        "nh,nhd->nhd", p[:, :, S], v[:, S]
    )  # per-head source-1 values, for the head split
    O21 = Oh[:, 21] @ Wo[21 * dh : 22 * dh]
    return p, Osrc, O21


def split(Z, g):
    """Within-group spread of Z (n, d): total, radial (along each group's mean), and per-prompt scale s = z.m/|m|^2."""
    tot = rad = 0.0
    s = np.zeros(len(Z))
    for u in np.unique(g):
        z = Z[g == u]
        m = z.mean(0)
        mh = m / np.linalg.norm(m)
        dev = z - m
        tot += (dev**2).sum()
        rad += ((dev @ mh) ** 2).sum()
        s[g == u] = z @ m / (m @ m)
    return tot, rad, s


def main():
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    st = np.load(OUT / "stream_a.npz")
    sub = SUB["a"]
    a, b, op = st["a"][sub], st["b"][sub], st["op"][sub]
    lab = a % 10
    blocks, _, _ = load_blocks()
    blk = blocks[L]
    out = {}
    for name in ("alive-only", "original"):
        if name == "alive-only":
            X = st[f"X{L}"][sub].astype(np.float64)
            parts = attention_parts(blk, X)
            p = parts["p"]
            Osrc = np.stack([copied(blk, parts, j) for j in range(5)], 1)
            O21 = copied(blk, parts, S, heads=np.array([21]))
        else:
            X = np.load(OUT / "orig_cache_a.npz")["Xor"].astype(np.float64)
            p, Osrc, O21 = orig_parts(snap, wmap, cfg, X)
        E = X[:, 4]
        Ep = E + Osrc.sum(1)
        _, D = planes(Ep, lab)
        r = {}
        tE, rE, sE = split(Ep @ D, a)
        r["Eprime_within_a_spread"] = tE
        r["Eprime_radial_share"] = rE / tE
        # share of E' within-a spread carried by each source (covariance with E' deviation, sums to 1 with E)
        devs = {}
        for key, Z in [("E", E @ D)] + [(f"O{j}", Osrc[:, j] @ D) for j in range(5)]:
            dz = Z.copy()
            for u in np.unique(a):
                dz[a == u] -= Z[a == u].mean(0)
            devs[key] = dz
        dE = sum(devs.values())
        for key, dz in devs.items():
            r[f"{key}_cov_share"] = float((dz * dE).sum() / (dE * dE).sum())
        t1, r1, s1 = split(Osrc[:, S] @ D, a)
        r["O1_radial_share"] = r1 / t1
        t21, r21, s21 = split(O21 @ D, a)
        r["O1_head21_radial_share"] = r21 / t21
        r["O1_head21_share_of_O1_norm2_in_D"] = float(
            ((O21 @ D) ** 2).sum() / ((Osrc[:, S] @ D) ** 2).sum()
        )
        r["corr_sEprime_p21"] = float(np.corrcoef(sE, p[:, 21, S])[0, 1])
        r["corr_sO1_p21"] = float(np.corrcoef(s1, p[:, 21, S])[0, 1])
        r["sEprime_range_5_95"] = np.percentile(sE, [5, 95]).tolist()
        r["p21_range_5_95"] = np.percentile(p[:, 21, S], [5, 95]).tolist()
        r["p21_mean_by_b"] = {int(v): float(p[b == v, 21, S].mean()) for v in np.unique(b)}
        r["p21_mean_by_op"] = [float(p[op == o, 21, S].mean()) for o in (0, 1)]
        r["sEprime_mean_by_b"] = {int(v): float(sE[b == v].mean()) for v in np.unique(b)}
        r["sEprime_mean_by_op"] = [float(sE[op == o].mean()) for o in (0, 1)]
        r["corr_sEprime_rms_eq_after"] = float(np.corrcoef(sE, np.sqrt((Ep**2).mean(1)))[0, 1])
        out[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
    (OUT / "smear_a.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
