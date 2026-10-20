"""Block-0 MLP input of the ORIGINAL Llama-3.1-8B at position 1 of `<BOS> a ...`, for a = 1..400 (numpy, CPU).

Position 1 is the token a; causal attention makes everything there a function of a alone, so 400 two-token
forwards (<BOS>, a) give every distinct value. The forward is embedding -> input RMSNorm -> block-0
attention (GQA, llama3 RoPE) -> residual add -> post-attention RMSNorm, in float32 from the bf16 weights.

    OUT/l0_input.npz   x (400, 4096): post-attention-RMSNorm output at position 1; h (400, 4096): the
                       residual stream before that norm; tokens (400,)

    python neurons_l0_input.py --out <dir>
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from mod10_L0 import HUB, A, bf16_tensor

D_MODEL, N_HEADS, N_KV, HEAD = 4096, 32, 8, 128
EPS = 1e-5


def rms_norm(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w * (h / np.sqrt((h**2).mean(axis=-1, keepdims=True) + EPS))


def llama3_inv_freq(cfg: dict) -> np.ndarray:
    inv = 1.0 / (cfg["rope_theta"] ** (np.arange(0, HEAD, 2, dtype=np.float64) / HEAD))
    rs = cfg["rope_scaling"]
    factor, lo, hi, old = (
        rs["factor"],
        rs["low_freq_factor"],
        rs["high_freq_factor"],
        rs["original_max_position_embeddings"],
    )
    wavelen = 2 * np.pi / inv
    out = np.where(wavelen > old / lo, inv / factor, inv)
    smooth = (old / wavelen - lo) / (hi - lo)
    mid = (wavelen <= old / lo) & (wavelen >= old / hi)
    return np.where(mid, (1 - smooth) * inv / factor + smooth * inv, out)


def rope(x: np.ndarray, pos: int, inv: np.ndarray) -> np.ndarray:
    """x (..., HEAD) rotated to position pos (HF rotate_half convention)."""
    ang = np.concatenate([pos * inv, pos * inv])
    x1, x2 = x[..., : HEAD // 2], x[..., HEAD // 2 :]
    return x * np.cos(ang) + np.concatenate([-x2, x1], axis=-1) * np.sin(ang)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    ids = []
    for a in A:
        enc = tok.encode(str(a), add_special_tokens=False)
        assert len(enc) == 1, (a, enc)
        ids.append(enc[0])
    idx = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]

    def w(name: str, rows: np.ndarray | None = None) -> np.ndarray:
        full = f"model.{name}"
        return bf16_tensor(snap + idx[full], full, rows).astype(np.float32)

    emb = w("embed_tokens.weight", np.asarray([cfg["bos_token_id"], *ids]))
    e_bos, e = emb[0], emb[1:]  # (4096,), (400, 4096)
    g_in, g_post = (
        w("layers.0.input_layernorm.weight"),
        w("layers.0.post_attention_layernorm.weight"),
    )
    Wq, Wk = w("layers.0.self_attn.q_proj.weight"), w("layers.0.self_attn.k_proj.weight")
    Wv, Wo = w("layers.0.self_attn.v_proj.weight"), w("layers.0.self_attn.o_proj.weight")
    inv = llama3_inv_freq(cfg)
    n_bos, n_a = rms_norm(e_bos, g_in), rms_norm(e, g_in)
    k0 = rope((Wk @ n_bos).reshape(N_KV, HEAD), 0, inv)  # (8, 128)
    v0 = (Wv @ n_bos).reshape(N_KV, HEAD)
    q1 = rope((n_a @ Wq.T).reshape(-1, N_HEADS, HEAD), 1, inv)  # (400, 32, 128)
    k1 = rope((n_a @ Wk.T).reshape(-1, N_KV, HEAD), 1, inv)
    v1 = (n_a @ Wv.T).reshape(-1, N_KV, HEAD)
    grp = np.arange(N_HEADS) // (N_HEADS // N_KV)
    s0 = (q1 * k0[grp]).sum(-1) / np.sqrt(HEAD)  # (400, 32)
    s1 = (q1 * k1[:, grp]).sum(-1) / np.sqrt(HEAD)
    mx = np.maximum(s0, s1)
    p0, p1 = np.exp(s0 - mx), np.exp(s1 - mx)
    p0, p1 = p0 / (p0 + p1), p1 / (p0 + p1)
    o = p0[..., None] * v0[grp] + p1[..., None] * v1[:, grp]  # (400, 32, 128)
    h = e + o.reshape(-1, D_MODEL) @ Wo.T
    x = rms_norm(h, g_post)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "l0_input.npz", x=x, h=h, tokens=np.asarray(ids), attn_to_bos=p0)
    print("x", x.shape, "mean attention to BOS", float(p0.mean()))


if __name__ == "__main__":
    main()
