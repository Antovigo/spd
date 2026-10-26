"""Which writers' activation variance becomes variance in block 16 head 21's attention logit s_1 (query at "=",
position 4; key at a, position 1), set "a" (2,000 prompts), alive-only model.

The stream entering block 16 at position t is X_t = sum_w y_w(t) U_w: the token embedding (y = 1) plus every writer w
of blocks 0..15 (attention output components, MLP down components) with inner activation y_w(t) (the component's read
of its input) and write vector U_w (4096). With r_t = rms(X_t), x_t = g1 * X_t / r_t:
  q = sum_w (y_w(4) / r_4) Q_w,   Q_w = head 21's query of g1 * U_w at position 4 (128)
  k = sum_w (y_w(1) / r_1) K_w,   K_w = KV head 5's key of g1 * U_w at position 1 (128)
  s_1 = q . k / sqrt(128)
Two exact splits of s_1 per prompt: query side c^Q_w = (y_w(4) / r_4) Q_w . k / sqrt(128) (the actual key), and key
side c^K_w = q . (y_w(1) / r_1) K_w / sqrt(128) (the actual query); each sums to s_1 over w. A writer's share of
var(s_1) is cov(c_w, s_1) / var(s_1). Shares can be negative; they sum to 1 on each side.

    python logit_writers.py
"""

import json

import numpy as np
from copy_build import OUT
from copy_figs import SUB
from mlp0_mod10 import rope
from track_build import load_blocks, rmsnorm

L, H, KV, DH = 16, 21, 5, 128
ALIVE = OUT.parent.parent.parent / "addsub-08-last-pos-40000/alive/alive.json"


def attn_heads(blk, X):
    """Concatenated head outputs at every position (n, T, 4096), before o_proj."""
    n, T, _ = X.shape
    x = rmsnorm(X, blk.g1)
    rep = np.arange(32) // 4
    q = np.stack([rope(blk.proj(x[:, t], "q").reshape(n, 32, DH), t, blk.cfg) for t in range(T)], 1)
    k = np.stack([rope(blk.proj(x[:, t], "k").reshape(n, 8, DH), t, blk.cfg) for t in range(T)], 1)[
        :, :, rep
    ]
    v = np.stack([blk.proj(x[:, t], "v").reshape(n, 8, DH) for t in range(T)], 1)[:, :, rep]
    s = np.einsum("nthd,nshd->nhts", q, k) / np.sqrt(DH)
    s = np.where(np.tril(np.ones((T, T), bool))[None, None], s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("nhts,nshd->nthd", p, v).reshape(n, T, 32 * DH)


def r2(y, lab):
    m = np.zeros_like(y)
    for u in np.unique(lab):
        m[lab == u] = y[lab == u].mean()
    v = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - m) ** 2).sum() / v) if v > 0 else 0.0


def main():
    st = np.load(OUT / "stream_a.npz")
    sub = SUB["a"]
    a, b, op = st["a"][sub], st["b"][sub], st["op"][sub]
    X = st["X16"][sub].astype(np.float64)
    alive = json.loads(ALIVE.read_text())["components"]
    blocks = load_blocks()[0]
    b16 = blocks[L]
    n = len(a)
    # writers: embedding + per block (attention o comps, MLP down comps); keep inner acts at positions 1 and 4
    import glob
    from pathlib import Path

    from mod10_L0 import HUB, bf16_tensor
    from transformers import AutoTokenizer

    (snap,) = glob.glob(HUB)
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    toks = np.asarray(
        [
            tok.encode(f"{x}{'+-'[o]}{y}=", add_special_tokens=True)
            for x, y, o in zip(a, b, op, strict=True)
        ]
    )
    uniq, inv = np.unique(toks.ravel(), return_inverse=True)
    Z = bf16_tensor(snap + wmap["model.embed_tokens.weight"], "model.embed_tokens.weight", uniq)[
        inv
    ]
    Z = Z.reshape(*toks.shape, -1).astype(np.float64)
    names, Y1, Y4, U = ["embed"], [np.ones((n, 1))], [np.ones((n, 1))], []
    emb1, emb4 = Z[:, 1].copy(), Z[:, 4].copy()
    for blk in blocks[:L]:
        hd = attn_heads(blk, Z)
        yo = hd @ blk.a["V_o"]
        Z = Z + yo @ blk.a["U_o"]
        ids_o = alive[f"layers.{blk.b}.self_attn.o_proj"]
        flat = Z.reshape(n * 5, -1)
        tg, tu = blk.readers(flat)
        m, td = blk.mlp_from_t(tg, tu)
        Z = Z + m.reshape(n, 5, -1)
        td = td.reshape(n, 5, -1)
        ids_d = blk.index[blk.kind == "down"]
        Ud = blk.m["U_down"]
        assert Ud.shape[0] == td.shape[2] == len(ids_d)
        for kind, y, Uw, ids in (("attn", yo, blk.a["U_o"], ids_o), ("mlp", td, Ud, ids_d)):
            names += [f"L{blk.b}.{kind}.{c}" for c in ids]
            Y1.append(y[:, 1])
            Y4.append(y[:, 4])
            U.append(Uw)
        print(blk.b, flush=True)
    print("rebuild rel err", float(np.abs(Z - X).max() / np.abs(X).max()), flush=True)
    Y1, Y4 = np.concatenate(Y1, 1), np.concatenate(Y4, 1)
    U = np.concatenate(U, 0)
    r1 = np.sqrt((Z[:, 1] ** 2).mean(1) + 1e-5)
    r4 = np.sqrt((Z[:, 4] ** 2).mean(1) + 1e-5)
    g1 = b16.g1
    # per-writer query / key vectors; the embedding is per prompt at position 1, constant at position 4
    Qw = rope(((g1 * U) @ b16.a["V_q"] @ b16.a["U_q"]).reshape(-1, 32, DH), 4, b16.cfg)[:, H]
    Kw = rope(((g1 * U) @ b16.a["V_k"] @ b16.a["U_k"]).reshape(-1, 8, DH), 1, b16.cfg)[:, KV]
    qe = rope(((g1 * emb4) @ b16.a["V_q"] @ b16.a["U_q"]).reshape(n, 32, DH), 4, b16.cfg)[:, H]
    ke = rope(((g1 * emb1) @ b16.a["V_k"] @ b16.a["U_k"]).reshape(n, 8, DH), 1, b16.cfg)[:, KV]
    q = (qe + Y4[:, 1:] @ Qw) / r4[:, None]
    k = (ke + Y1[:, 1:] @ Kw) / r1[:, None]
    s1 = (q * k).sum(1) / np.sqrt(DH)
    xs = rmsnorm(X, g1)
    q_ref = rope(b16.proj(xs[:, 4], "q").reshape(n, 32, DH), 4, b16.cfg)[:, H]
    k_ref = rope(b16.proj(xs[:, 1], "k").reshape(n, 8, DH), 1, b16.cfg)[:, KV]
    s1_ref = (q_ref * k_ref).sum(1) / np.sqrt(DH)
    print("s1 check", float(np.abs(s1 - s1_ref).max()), float(s1.std()), flush=True)
    cQ = np.concatenate(
        [((qe * k).sum(1) / r4)[:, None], (Y4[:, 1:] / r4[:, None]) * (k @ Qw.T)], 1
    ) / np.sqrt(DH)
    cK = np.concatenate(
        [((ke * q).sum(1) / r1)[:, None], (Y1[:, 1:] / r1[:, None]) * (q @ Kw.T)], 1
    ) / np.sqrt(DH)
    d = s1 - s1.mean()

    def shares(c):
        return ((c - c.mean(0)) * d[:, None]).sum(0) / (d * d).sum()

    sQ, sK = shares(cQ), shares(cK)
    # first-order split of var(s1): query variation with mean key, key variation with mean query, rest
    qb, kb = q.mean(0), k.mean(0)
    vq = ((q - qb) @ kb) / np.sqrt(DH)
    vk = ((k - kb) @ qb) / np.sqrt(DH)
    res = {
        "s1_mean_sd": [float(s1.mean()), float(s1.std())],
        "var_share_query_variation_at_mean_key": float(np.cov(vq, s1)[0, 1] / s1.var()),
        "var_share_key_variation_at_mean_query": float(np.cov(vk, s1)[0, 1] / s1.var()),
        "r2_s1": {"a": r2(s1, a), "op": r2(s1, op), "op_b": r2(s1, b * 2 + op), "a_mod10": r2(s1, a % 10)},
        "corr_s1_r1": float(np.corrcoef(s1, r1)[0, 1]),
        "corr_s1_r4": float(np.corrcoef(s1, r4)[0, 1]),
        "var_share_sd_of_1_over_r1_r4": [float(np.std(1 / r1) / np.mean(1 / r1)), float(np.std(1 / r4) / np.mean(1 / r4))],
    }  # fmt: skip
    for side, sh, Yp in (("query", sQ, Y4), ("key", sK, Y1)):
        o = np.argsort(-np.abs(sh))
        top = []
        for i in o[:15]:
            y = Yp[:, i] if i > 0 else Yp[:, 0]
            top.append({"writer": names[i], "share": float(sh[i]),
                        "y_mean_sd": [float(y.mean()), float(y.std())],
                        "y_r2": {"a": r2(y, a), "op": r2(y, op), "op_b": r2(y, b * 2 + op)}})  # fmt: skip
        res[f"{side}_top"] = top
        byblk = {}
        for i, nm in enumerate(names):
            key = nm.rsplit(".", 1)[0] if nm != "embed" else "embed"
            byblk[key] = byblk.get(key, 0.0) + float(sh[i])
        res[f"{side}_by_block"] = dict(sorted(byblk.items(), key=lambda kv: -abs(kv[1]))[:12])
        res[f"{side}_n_writers_for_80pct_positive"] = int(
            np.searchsorted(np.cumsum(np.sort(sh)[::-1]), 0.8) + 1
        )
        res[f"{side}_sum_shares"] = float(sh.sum())
    print(json.dumps(res, indent=1), flush=True)
    (OUT / "logit_writers.json").write_text(json.dumps(res, indent=1))
    np.savez(OUT / "logit_writers.npz", names=np.asarray(names), sQ=sQ, sK=sK)


if __name__ == "__main__":
    main()
