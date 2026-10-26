"""What makes block 16 head 21's query at "=" vary across prompts (set "a", 2,000 prompts, alive-only model).
s_t: head 21's score at "=" (position 4) on position t (0..4); D13 = s_1 - s_3 (a minus b), the gap that tracks the
weight on a. The stream at "=" entering block 16 is E = X15 + A15 + M15 (stream entering block 15, block 15's attention
write, block 15's MLP write, all at "="). The query is linear in x = g1 * E / rms(E), so s_t splits exactly into the
three sources (sharing the same rms) and, inside the query, into query components c: s_t = sum_c t_c w_c(t), with
t_c = x . V_q[:, c] the component's inner activation. Variance shares are covariances with the total over var(total).

    python q21_vary.py
"""

import json

import numpy as np
from copy_build import OUT
from copy_figs import SUB
from mlp0_mod10 import rope
from track_build import load_blocks, rmsnorm
from track_build_p3 import attn_seq

H, KV = 21, 5


def r2(y, lab):
    m = np.zeros_like(y)
    for u in np.unique(lab):
        m[lab == u] = y[lab == u].mean()
    return float(1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def share(parts, tot):
    d = tot - tot.mean()
    return {k: float(((v - v.mean()) * d).sum() / (d * d).sum()) for k, v in parts.items()}


def main():
    st = np.load(OUT / "stream_a.npz")
    sub = SUB["a"]
    a, b, op = st["a"][sub], st["b"][sub], st["op"][sub]
    X15, X16 = st["X15"][sub].astype(np.float64), st["X16"][sub].astype(np.float64)
    blocks = load_blocks()[0]
    b15, b16 = blocks[15], blocks[16]
    A15 = attn_seq(b15, X15)[:, 4]
    M15 = X16[:, 4] - X15[:, 4] - A15
    M15_direct = b15.mlp((X15 + attn_seq(b15, X15))[:, 4:5])[:, 0]
    print(
        "MLP15 residual check",
        float(np.abs(M15 - M15_direct).max() / np.abs(M15).max()),
        flush=True,
    )
    E = X16[:, 4]
    rms = np.sqrt((E**2).mean(1, keepdims=True) + 1e-5)
    cfg = b16.cfg
    n = len(a)
    xall = rmsnorm(X16, b16.g1)
    k = np.stack(
        [rope(b16.proj(xall[:, t], "k").reshape(n, 8, 128), t, cfg)[:, KV] for t in range(5)], 1
    )

    def scores(xq):
        q = rope(b16.proj(xq, "q").reshape(n, 32, 128), 4, cfg)[:, H]
        return np.einsum("nd,ntd->nt", q, k) / np.sqrt(128)

    s = scores(xall[:, 4])
    D13 = s[:, 1] - s[:, 3]
    src = {"X15": X15[:, 4], "A15": A15, "M15": M15}
    s_src = {name: scores(b16.g1 * v / rms) for name, v in src.items()}
    res = {
        "D13_mean_sd": [float(D13.mean()), float(D13.std())],
        "s1_sd": float(s[:, 1].std()),
        "s3_sd": float(s[:, 3].std()),
        "share_of_var_D13_by_source": share(
            {kk: v[:, 1] - v[:, 3] for kk, v in s_src.items()}, D13
        ),
        "share_of_var_s1_by_source": share({kk: v[:, 1] for kk, v in s_src.items()}, s[:, 1]),
        "share_of_var_s3_by_source": share({kk: v[:, 3] for kk, v in s_src.items()}, s[:, 3]),
        "r2_D13": {
            "a": r2(D13, a),
            "op": r2(D13, op),
            "b": r2(D13, b),
            "op_b": r2(D13, b * 2 + op),
        },
        "corr_D13_log_rms": float(np.corrcoef(D13, np.log(rms[:, 0]))[0, 1]),
        "r2_log_rms": {"a": r2(np.log(rms[:, 0]), a), "op_b": r2(np.log(rms[:, 0]), b * 2 + op)},
    }
    # key side: fix the query at its mean
    q = rope(b16.proj(xall[:, 4], "q").reshape(n, 32, 128), 4, cfg)[:, H]
    sk = np.einsum("d,ntd->nt", q.mean(0), k) / np.sqrt(128)
    res["var_D13_with_mean_query_over_var_D13"] = float((sk[:, 1] - sk[:, 3]).var() / D13.var())
    # query components: contribution of each to D13
    Vq, Uq = b16.a["V_q"], b16.a["U_q"]
    t = xall[:, 4] @ Vq  # (n, Cq)
    Uh = rope(Uq.reshape(-1, 32, 128), 4, cfg)[:, H]  # (Cq, 128)
    w = np.einsum("cd,ntd->nct", Uh, k) / np.sqrt(128)  # (n, Cq, 5)
    contrib = t * (w[:, :, 1] - w[:, :, 3])  # (n, Cq)
    d = D13 - D13.mean()
    sh = ((contrib - contrib.mean(0)) * d[:, None]).sum(0) / (d * d).sum()
    order = np.argsort(-sh)
    top = []
    for c in order[:8]:
        y = t[:, c]
        top.append({"index": int(c), "share_of_var_D13": float(sh[c]),
                    "inner_act_r2": {"a": r2(y, a), "op": r2(y, op), "op_b": r2(y, b * 2 + op)},
                    "inner_act_mean_by_op": [float(y[op == o].mean()) for o in (0, 1)],
                    "corr_inner_act_a": float(np.corrcoef(y, a)[0, 1])})  # fmt: skip
    res["top_query_components"] = top
    res["n_query_components_for_80pct"] = int(np.searchsorted(np.cumsum(sh[order]), 0.8) + 1)
    # inner acts of the top components: which source writes them
    for e in top:
        c = e["index"]
        parts = {nm: (b16.g1 * v / rms) @ Vq[:, c] for nm, v in src.items()}
        e["inner_act_share_by_source"] = share(parts, t[:, c])
    print(json.dumps(res, indent=1), flush=True)
    (OUT / "q21_vary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
