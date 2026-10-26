"""Why head 21's weight on a (position 1) at "=" varies in block 16 (set "a", 2,000 prompts), alive-only and original.
s_j: score q.k_j / sqrt(128) of head 21 at "=" (position 4) on position j (0..4); p_j = softmax_j(s_j).
g = s_1 - log sum_{j != 1} exp(s_j): the logit gap, p_1 = sigmoid(g). Splits var(g) over prompts by additive least-squares
fits on the integer a (1..400) and on the (b, op) cell, and reports which position takes the weight when p_1 is low.

    python p21_vary.py
"""

import glob
import json
from pathlib import Path

import numpy as np
from copy_analyze import attention_parts
from copy_build import OUT
from copy_figs import SUB
from mlp0_mod10 import rope
from mod10_L0 import HUB, bf16_tensor
from track_build import load_blocks, rmsnorm

L, H = 16, 21


def scores_orig(snap, wmap, cfg, X):
    def w(name):
        return bf16_tensor(snap + wmap[name], name).astype(np.float64)

    pre = f"model.layers.{L}."
    x = rmsnorm(X, w(pre + "input_layernorm.weight"))
    Wq = w(pre + "self_attn.q_proj.weight")[H * 128 : (H + 1) * 128]
    kv = H // 4
    Wk = w(pre + "self_attn.k_proj.weight")[kv * 128 : (kv + 1) * 128]
    n = len(X)
    q = rope((x[:, 4] @ Wq.T).reshape(n, 1, 128), 4, cfg)[:, 0]
    k = np.stack([rope((x[:, t] @ Wk.T).reshape(n, 1, 128), t, cfg)[:, 0] for t in range(5)], 1)
    return np.einsum("nd,ntd->nt", q, k) / np.sqrt(128), q, k


def r2_fit(y, cols):
    """R^2 of a least-squares fit of y on one-hot columns of the given label arrays (plus the constant)."""
    M = np.concatenate(
        [np.ones((len(y), 1))]
        + [(c[:, None] == np.unique(c)[None]).astype(float)[:, 1:] for c in cols],
        1,
    )
    beta = np.linalg.lstsq(M, y, rcond=None)[0]
    return float(1 - ((y - M @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    st = np.load(OUT / "stream_a.npz")
    sub = SUB["a"]
    a, b, op = st["a"][sub], st["b"][sub], st["op"][sub]
    cell = b * 2 + op
    out = {}
    for name in ("alive-only", "original"):
        if name == "alive-only":
            X = st[f"X{L}"][sub].astype(np.float64)
            blk = load_blocks()[0][L]
            parts = attention_parts(blk, X)
            p = parts["p"][:, H]
            s = np.log(p)  # scores up to a per-prompt constant (enough for gaps)
        else:
            X = np.load(OUT / "orig_cache_a.npz")["Xor"].astype(np.float64)
            s, q, k = scores_orig(snap, wmap, cfg, X)
            p = np.exp(s - s.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
        others = [0, 2, 3, 4]
        g = s[:, 1] - np.log(np.exp(s[:, others]).sum(1))
        lo = p[:, 1] < np.percentile(p[:, 1], 10)
        r = {
            "p1_mean_sd": [float(p[:, 1].mean()), float(p[:, 1].std())],
            "g_mean_sd": [float(g.mean()), float(g.std())],
            "r2_g_on_a": r2_fit(g, [a]),
            "r2_g_on_b_op_cell": r2_fit(g, [cell]),
            "r2_g_on_op": r2_fit(g, [op]),
            "r2_g_on_b": r2_fit(g, [b]),
            "r2_g_on_a_plus_cell": r2_fit(g, [a, cell]),
            "corr_g_a": float(np.corrcoef(g, a)[0, 1]),
            "pattern_lowest_decile_of_p1": p[lo].mean(0).tolist(),
            "pattern_rest": p[~lo].mean(0).tolist(),
            "p1_mean_by_a_quintile": [
                float(p[(a > 80 * i) & (a <= 80 * (i + 1)), 1].mean()) for i in range(5)
            ],
            "p1_mean_by_op_b": {
                f"op{o}_b{v}": float(p[(op == o) & (b == v), 1].mean())
                for o in (0, 1)
                for v in np.unique(b)
            },
        }
        # score differences s_1 - s_j: share of var(g) from each competitor's gap
        for j in others:
            r[f"corr_g_s1_minus_s{j}"] = float(np.corrcoef(g, s[:, 1] - s[:, j])[0, 1])
        if name == "original":
            r["corr_g_s1"] = float(np.corrcoef(g, s[:, 1])[0, 1])
            r["corr_g_qnorm"] = float(np.corrcoef(g, np.linalg.norm(q, axis=1))[0, 1])
            r["corr_g_k1norm"] = float(np.corrcoef(g, np.linalg.norm(k[:, 1], axis=1))[0, 1])
            for t in range(5):
                r[f"s{t}_mean_sd"] = [float(s[:, t].mean()), float(s[:, t].std())]
            # query vs key: a fixed-key / fixed-query swap. g with q replaced by its mean (key-driven variation) and with
            # k_1 replaced by its per-a value averaged... k_1 depends on a only; q depends on (a, b, op).
            qm = q.mean(0)
            sq = np.einsum("d,ntd->nt", qm, k) / np.sqrt(128)
            gq = sq[:, 1] - np.log(np.exp(sq[:, others]).sum(1))
            r["var_g_with_mean_query_over_var_g"] = float(gq.var() / g.var())
        out[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
    (OUT / "p21_vary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
