"""Rebuild the ORIGINAL Llama-3.1-8B (no decomposition) at positions 0 (<BOS>) and 1 (the token a) through block
LAST, in float64 from the bf16 weights: the neuron-basis twin of track_build.py's alive-only rebuild.

Position 1 attends only to <BOS> and to itself, so 400 two-token forwards give every distinct value. Writes, for
a = 1..400:

    <run>/analysis/reader_writer/track_orig/stream.npz
        pre_<l>   (400, 4096) residual stream entering block l at position 1 (pre_0 = embedding)
        mid_<l>   (400, 4096) after block l's attention add
        bos_pre_<l>, bos_mid_<l>  (4096,) the same at position 0
    <run>/analysis/reader_writer/track_orig/build.json   per block, the cosine of the stream with the
        alive-only rebuild's (track_mod10/stream.npz), averaged over a

    python track_build_orig.py
"""

import glob
import json
from pathlib import Path

import numpy as np
from mlp0_mod10 import BOS, rope
from mod10_L0 import HUB, RUN, A, bf16_tensor
from track_build import LAST, rmsnorm, silu
from track_build import OUT as ALIVE_OUT

OUT = RUN / "analysis/reader_writer/track_orig"


class OrigBlock:
    """Block b of the original model; position-1 attention over keys {<BOS>, a}. Same interface as
    track_build.Block, with the MLP readers being neurons: readers(X) returns the gate and up pre-activations
    (400, 14336) and mlp_from_t maps them to the MLP output."""

    def __init__(self, b: int, wt, cfg: dict) -> None:
        self.b, self.cfg = b, cfg
        p = f"model.layers.{b}."
        self.g1 = wt(p + "input_layernorm.weight")
        self.g2 = wt(p + "post_attention_layernorm.weight")
        self.W = {k: wt(p + f"self_attn.{k}_proj.weight") for k in "qkvo"}
        self.Wg = wt(p + "mlp.gate_proj.weight")  # (14336, 4096)
        self.Wu = wt(p + "mlp.up_proj.weight")
        self.Wd = wt(p + "mlp.down_proj.weight")  # (4096, 14336)

    def proj(self, x: np.ndarray, k: str) -> np.ndarray:
        return x @ self.W[k].T

    def attn_parts(self, bos: np.ndarray, X: np.ndarray):
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
        return x @ self.Wg.T, x @ self.Wu.T

    def mlp_from_t(self, tg: np.ndarray, tu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        act = silu(tg) * tu
        return act @ self.Wd.T, act

    def mlp(self, X: np.ndarray) -> np.ndarray:
        return self.mlp_from_t(*self.readers(X))[0]


def loader():
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]

    def wt(name: str, rows: np.ndarray | None = None) -> np.ndarray:
        return bf16_tensor(snap + wmap[name], name, rows)

    return snap, cfg, wt


def load_block(b: int) -> OrigBlock:
    _, cfg, wt = loader()
    return OrigBlock(b, wt, cfg)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap, cfg, wt = loader()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    ids = [tok.encode(str(a), add_special_tokens=False)[0] for a in A]
    X = wt("model.embed_tokens.weight", np.asarray(ids))
    bos = wt("model.embed_tokens.weight", np.asarray([BOS]))[0]
    ref = np.load(ALIVE_OUT / "stream.npz")
    save, checks = {}, {}

    def cos(name: str, Y: np.ndarray) -> float:
        Z = ref[name].astype(np.float64)
        return float(
            ((Y * Z).sum(1) / np.linalg.norm(Y, axis=1) / np.linalg.norm(Z, axis=1)).mean()
        )

    for b in range(LAST + 1):
        blk = OrigBlock(b, wt, cfg)
        save[f"pre_{b}"], save[f"bos_pre_{b}"] = X, bos
        X = X + blk.attn(bos, X)
        bos = bos + blk.attn_bos(bos)
        save[f"mid_{b}"], save[f"bos_mid_{b}"] = X, bos
        checks[b] = {
            "cos_pre_alive": cos(f"pre_{b}", save[f"pre_{b}"]),
            "cos_mid_alive": cos(f"mid_{b}", X),
        }
        print(b, checks[b], flush=True)
        X = X + blk.mlp(X)
        bos = bos + blk.mlp(bos[None])[0]
        del blk
    save[f"pre_{LAST + 1}"], save[f"bos_pre_{LAST + 1}"] = X, bos
    checks[LAST + 1] = {"cos_pre_alive": cos(f"pre_{LAST + 1}", X)}
    np.savez(OUT / "stream.npz", **{k: v.astype(np.float32) for k, v in save.items()})
    (OUT / "build.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
