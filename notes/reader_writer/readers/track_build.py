"""Rebuild the alive-only model at positions 0 (<BOS>) and 1 (the token a) through block LAST, in float64.

Every block's alive attention (q/k/v/o) and MLP (gate/up/down) components are on, every other component and
the weight delta are off: the model of `variance/alive_only`. Its stored MLP inner activations at position 1
(`L<l>/inner.npy`, bf16 forward) check the rebuild block by block. Writes, for a = 1..400:

    <run>/analysis/reader_writer/track_mod10/stream.npz
        pre_<l>   (400, 4096) residual stream entering block l at position 1 (pre_0 = embedding)
        mid_<l>   (400, 4096) after block l's attention add
        bos_pre_<l>, bos_mid_<l>  (4096,) the same at position 0
    <run>/analysis/reader_writer/track_mod10/build.json   per-block checks

    python track_build.py
"""

import glob
import json
from pathlib import Path

import numpy as np
from mlp0_mod10 import ALIVE, BOS, CKPT, rope
from mod10_L0 import HUB, RUN, A, N, bf16_tensor

LAST = 16
SRC = RUN / "analysis/reader_writer/variance/alive_only"
OUT = RUN / "analysis/reader_writer/track_mod10"
EPS = 1e-5


def attn_all() -> dict[str, np.ndarray]:
    path = OUT / f"attn_alive_L0-{LAST}.npz"
    if path.exists():
        return dict(np.load(path))
    import jax
    import orbax.checkpoint as ocp

    jax.config.update("jax_platforms", "cpu")
    sh = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    ck = ocp.PyTreeCheckpointer()
    meta = ck.metadata(str(CKPT))
    tree = getattr(meta, "item_metadata", meta)
    leaves, td = jax.tree.flatten(tree, is_leaf=lambda x: hasattr(x, "shape"))
    ra = jax.tree.unflatten(
        td, [ocp.ArrayRestoreArgs(sharding=sh, dtype=np.float32) for _ in leaves]
    )
    t = ck.restore(str(CKPT), args=ocp.args.PyTreeRestore(restore_args=ra))
    alive = json.loads(ALIVE.read_text())["components"]
    out = {}
    for k in "qkvo":
        V, U = (np.asarray(x) for x in t["components"]["stacks"][k])
        for b in range(LAST + 1):
            ids = np.asarray(alive[f"layers.{b}.self_attn.{k}_proj"])
            out[f"L{b}_V_{k}"], out[f"L{b}_U_{k}"] = V[b][:, ids], U[b][ids]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def rmsnorm(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    return g * x / np.sqrt((x**2).mean(-1, keepdims=True) + EPS)


def silu(h: np.ndarray) -> np.ndarray:
    return h / (1 + np.exp(-h))


class Block:
    """Block b of the alive-only model; position-1 attention over keys {<BOS>, a}."""

    def __init__(self, b: int, at: dict, wt, cfg: dict) -> None:
        self.b, self.cfg = b, cfg
        self.g1 = wt(f"model.layers.{b}.input_layernorm.weight")
        self.g2 = wt(f"model.layers.{b}.post_attention_layernorm.weight")
        self.a = {k: at[f"L{b}_{k}"].astype(np.float64) for k in
                  ("V_q", "U_q", "V_k", "U_k", "V_v", "U_v", "V_o", "U_o")}  # fmt: skip
        z = np.load(SRC / f"L{b}" / "comps.npz")
        self.m = {k: z[k].astype(np.float64) for k in z.files if k[0] in "UV"}
        self.kind, self.index = z["kind"], z["index"]

    def proj(self, x: np.ndarray, k: str) -> np.ndarray:
        return (x @ self.a[f"V_{k}"]) @ self.a[f"U_{k}"]

    def attn_parts(self, bos: np.ndarray, X: np.ndarray):
        """Per-head pattern p0 (weight on <BOS>), head values at both keys, and the output at position 1."""
        H, KV, dh = self.cfg["num_attention_heads"], self.cfg["num_key_value_heads"], 128
        n = X.shape[0]
        xb, x1 = rmsnorm(bos, self.g1)[None], rmsnorm(X, self.g1)
        q1 = rope(self.proj(x1, "q").reshape(n, H, dh), 1, self.cfg)
        k0 = rope(self.proj(xb, "k").reshape(1, KV, dh), 0, self.cfg)
        k1 = rope(self.proj(x1, "k").reshape(n, KV, dh), 1, self.cfg)
        rep = np.arange(H) // (H // KV)
        v0 = self.proj(xb, "v").reshape(1, KV, dh)[:, rep]
        v1 = self.proj(x1, "v").reshape(n, KV, dh)[:, rep]
        s0 = (q1 * k0[:, rep]).sum(-1) / np.sqrt(dh)
        s1 = (q1 * k1[:, rep]).sum(-1) / np.sqrt(dh)
        p0 = 1 / (1 + np.exp(s1 - s0))
        return p0, v0, v1

    def attn_out(self, p0: np.ndarray, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
        heads = p0[..., None] * v0 + (1 - p0)[..., None] * v1
        return self.proj(heads.reshape(heads.shape[0], -1), "o")

    def attn(self, bos: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.attn_out(*self.attn_parts(bos, X))

    def attn_bos(self, bos: np.ndarray) -> np.ndarray:
        H, KV, dh = self.cfg["num_attention_heads"], self.cfg["num_key_value_heads"], 128
        rep = np.arange(H) // (H // KV)
        v0 = self.proj(rmsnorm(bos, self.g1)[None], "v").reshape(1, KV, dh)[:, rep]
        return self.proj(v0.reshape(1, -1), "o")[0]

    def readers(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = rmsnorm(X, self.g2)
        return x @ self.m["V_gate"], x @ self.m["V_up"]

    def mlp_from_t(self, tg: np.ndarray, tu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hg, hu = tg @ self.m["U_gate"], tu @ self.m["U_up"]
        td = (silu(hg) * hu) @ self.m["V_down"]
        return td @ self.m["U_down"], td

    def mlp(self, X: np.ndarray) -> np.ndarray:
        return self.mlp_from_t(*self.readers(X))[0]


def load_blocks() -> tuple[list[Block], np.ndarray, np.ndarray]:
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]

    def wt(name: str, rows: np.ndarray | None = None) -> np.ndarray:
        return bf16_tensor(snap + wmap[name], name, rows)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    ids = [tok.encode(str(a), add_special_tokens=False)[0] for a in A]
    e = wt("model.embed_tokens.weight", np.asarray(ids))
    e_bos = wt("model.embed_tokens.weight", np.asarray([BOS]))[0]
    at = attn_all()
    return [Block(b, at, wt, cfg) for b in range(LAST + 1)], e, e_bos


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, X, bos = load_blocks()
    save, checks = {}, {}
    for blk in blocks:
        b = blk.b
        save[f"pre_{b}"], save[f"bos_pre_{b}"] = X, bos
        X = X + blk.attn(bos, X)
        bos = bos + blk.attn_bos(bos)
        save[f"mid_{b}"], save[f"bos_mid_{b}"] = X, bos
        tg, tu = blk.readers(X)
        m, td = blk.mlp_from_t(tg, tu)
        ref = np.asarray(
            np.load(SRC / f"L{b}" / "inner.npy", mmap_mode="r")[(A - 1) * N, 0, :], np.float64
        )
        ck = {}
        for name, mine in (("gate", tg), ("up", tu), ("down", td)):
            r = ref[:, blk.kind == name]
            ck[f"{name}_rel_max"] = float(np.abs(mine - r).max() / np.abs(r).max())
            ck[f"{name}_r2"] = float(1 - ((mine - r) ** 2).sum() / (r**2).sum())
        checks[b] = ck
        print(b, {k: round(v, 4) for k, v in ck.items()}, flush=True)
        X = X + m
        bos = bos + blk.mlp(bos[None])[0]
    save[f"pre_{LAST + 1}"], save[f"bos_pre_{LAST + 1}"] = X, bos
    np.savez(OUT / "stream.npz", **{k: v.astype(np.float32) for k, v in save.items()})
    (OUT / "build.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
