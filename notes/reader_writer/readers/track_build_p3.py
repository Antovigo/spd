"""Rebuild the alive-only model at positions 0..3 (<BOS>, a, op, b) through block LAST, in float64, for position 3.

Prompts: `<BOS> a op b =` tokenized as the pool does (f"{a}{op}{b}=", no spaces), for the operands a in A_SUB
(ten values, one per residue a mod 10, spread over 1..400), every b = 1..400 and op in (+, -): 8,000 prompts,
ordered (op, a, b). Every block's alive attention and MLP components are on, every other component and the weight
delta are off (the model of `variance/alive_only`), whose stored MLP inner activations at position 3
(`L<l>/inner.npy`[:, 2], rows op * 160000 + (a - 1) * 400 + (b - 1)) check the rebuild block by block. Writes

    <run>/analysis/reader_writer/track_mod10_p3/stream.npz
        pre_<l>, mid_<l>        (8000, 4096) residual stream at position 3 entering block l / after its attention
        ctx_pre_<l>, ctx_mid_<l> (20, 3, 4096) positions 0..2 for the 20 (op, a) contexts (row op * 10 + i_a)
        ctx                      (8000,) context row of each prompt; a, b, op (8000,) the operands and op index
    <run>/analysis/reader_writer/track_mod10_p3/build.json

    python track_build_p3.py
"""

import glob
import json
from pathlib import Path

import numpy as np
from mlp0_mod10 import BOS, rope
from mod10_L0 import HUB
from track_build import LAST, SRC, Block, attn_all, bf16_tensor, rmsnorm

OUT = Path(str(SRC).replace("variance/alive_only", "track_mod10_p3"))
A_SUB = np.asarray([7 + 41 * i for i in range(10)])  # residues 7, 8, 9, 0, 1, ..., 6
OPS = ("+", "-")
B_ALL = np.arange(1, 401)


def attn_seq(blk: Block, X: np.ndarray) -> np.ndarray:
    """Causal attention output at every position of X (n, T, 4096)."""
    H, KV, dh = blk.cfg["num_attention_heads"], blk.cfg["num_key_value_heads"], 128
    n, T, _ = X.shape
    x = rmsnorm(X, blk.g1)
    rep = np.arange(H) // (H // KV)
    q = np.stack([rope(blk.proj(x[:, t], "q").reshape(n, H, dh), t, blk.cfg) for t in range(T)], 1)
    k = np.stack(
        [rope(blk.proj(x[:, t], "k").reshape(n, KV, dh), t, blk.cfg) for t in range(T)], 1
    )[:, :, rep]
    v = np.stack([blk.proj(x[:, t], "v").reshape(n, KV, dh) for t in range(T)], 1)[:, :, rep]
    s = np.einsum("nthd,nshd->nhts", q, k) / np.sqrt(dh)
    s = np.where(np.tril(np.ones((T, T), bool))[None, None], s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    heads = np.einsum("nhts,nshd->nthd", p, v)
    return blk.proj(heads.reshape(n, T, H * dh), "o")


def attn_last(blk: Block, ctx: np.ndarray, X: np.ndarray, value_from: np.ndarray | None = None):
    """Attention output at the last position, keys/values 0..T-2 from ctx (n, T-1, 4096) and the last from X (n, 4096).
    With value_from, the last position's value is computed from value_from instead of X (pattern from X).
    Returns the output (n, 4096) and the pattern (n, H, T)."""
    H, KV, dh = blk.cfg["num_attention_heads"], blk.cfg["num_key_value_heads"], 128
    n, Tm1, _ = ctx.shape
    T = Tm1 + 1
    xc, xl = rmsnorm(ctx, blk.g1), rmsnorm(X, blk.g1)
    rep = np.arange(H) // (H // KV)
    q = rope(blk.proj(xl, "q").reshape(n, H, dh), T - 1, blk.cfg)
    ks = [rope(blk.proj(xc[:, t], "k").reshape(n, KV, dh), t, blk.cfg) for t in range(Tm1)]
    ks.append(rope(blk.proj(xl, "k").reshape(n, KV, dh), T - 1, blk.cfg))
    vl = blk.proj(xl if value_from is None else rmsnorm(value_from, blk.g1), "v").reshape(n, KV, dh)
    vs = [blk.proj(xc[:, t], "v").reshape(n, KV, dh) for t in range(Tm1)] + [vl]
    k = np.stack(ks, 1)[:, :, rep]
    v = np.stack(vs, 1)[:, :, rep]
    s = np.einsum("nhd,nshd->nhs", q, k) / np.sqrt(dh)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return blk.proj(np.einsum("nhs,nshd->nhd", p, v).reshape(n, H * dh), "o"), p


def load() -> tuple[list[Block], np.ndarray, dict[str, np.ndarray]]:
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]

    def wt(name: str, rows: np.ndarray | None = None) -> np.ndarray:
        return bf16_tensor(snap + wmap[name], name, rows)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    rows, meta = [], {"a": [], "b": [], "op": [], "ctx": []}
    for o, sym in enumerate(OPS):
        for i, a in enumerate(A_SUB):
            for b in B_ALL:
                t = tok.encode(f"{a}{sym}{b}=", add_special_tokens=True)
                assert len(t) == 5 and t[0] == BOS, (a, sym, b, t)
                rows.append(t[:4])
                meta["a"].append(a)
                meta["b"].append(b)
                meta["op"].append(o)
                meta["ctx"].append(o * len(A_SUB) + i)
    tokens = np.asarray(rows)
    uniq, inv = np.unique(tokens.ravel(), return_inverse=True)
    emb = wt("model.embed_tokens.weight", uniq)[inv].reshape(*tokens.shape, -1)
    at = attn_all()
    return (
        [Block(b, at, wt, cfg) for b in range(LAST + 1)],
        emb,
        {k: np.asarray(v) for k, v in meta.items()},
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, X, meta = load()
    first = np.unique(meta["ctx"], return_index=True)[1]  # one prompt per context
    rows = meta["op"] * 160000 + (meta["a"] - 1) * 400 + (meta["b"] - 1)
    save: dict[str, np.ndarray] = {k: v for k, v in meta.items()}
    checks = {}
    for blk in blocks:
        b = blk.b
        save[f"pre_{b}"], save[f"ctx_pre_{b}"] = (
            X[:, 3].astype(np.float32),
            X[first, :3].astype(np.float32),
        )
        X = X + np.concatenate([attn_seq(blk, X[i : i + 1000]) for i in range(0, len(X), 1000)])
        save[f"mid_{b}"], save[f"ctx_mid_{b}"] = (
            X[:, 3].astype(np.float32),
            X[first, :3].astype(np.float32),
        )
        tg, tu = blk.readers(X[:, 3])
        m3, td = blk.mlp_from_t(tg, tu)
        ref = np.asarray(
            np.load(SRC / f"L{b}" / "inner.npy", mmap_mode="r")[np.sort(rows), 2, :], np.float64
        )
        ref = ref[np.argsort(np.argsort(rows))]
        ck = {}
        for name, mine in (("gate", tg), ("up", tu), ("down", td)):
            r = ref[:, blk.kind == name]
            ck[f"{name}_rel_max"] = float(np.abs(mine - r).max() / np.abs(r).max())
            ck[f"{name}_r2"] = float(1 - ((mine - r) ** 2).sum() / (r**2).sum())
        checks[b] = ck
        print(b, {k: round(v, 4) for k, v in ck.items()}, flush=True)
        n, T, d = X.shape
        flat = X.reshape(n * T, d)
        X = X + np.concatenate(
            [blk.mlp(flat[i : i + 2000]) for i in range(0, n * T, 2000)]
        ).reshape(n, T, d)
    save[f"pre_{LAST + 1}"], save[f"ctx_pre_{LAST + 1}"] = (
        X[:, 3].astype(np.float32),
        X[first, :3].astype(np.float32),
    )
    np.savez(
        OUT / "stream.npz",
        **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in save.items()},
    )
    (OUT / "build.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
