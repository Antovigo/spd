"""What the block-0 MLP does to the a mod 10 representation, at position 1 (the token a), alive-only model.

Nothing is centred: zero is the activation-space zero (the network has no biases).

Notation (a = integer operand 1..400; r = a mod 10; k = integer frequency 0..5; i = imaginary unit):
  e(a)        token embedding of a (R^4096); position 0 is <BOS>
  r_mid(a)    residual stream entering the block-0 MLP = e(a) + block-0 attention output (alive-only)
  x(a)        MLP input = gamma_2 * r_mid / rms(r_mid)  (post-attention RMSNorm)
  m(a)        MLP output = sum_w t_w(a) U_w over the block's alive writers w
  r_out(a)    = r_mid(a) + m(a), the residual stream after block 0
  mu_r^X      mean of a field X over the 40 values of a with a mod 10 = r
  Psi_k^X     = sum_r mu_r^X exp(-2 pi i k r / 10) in C^4096;  R = Re Psi, I = Im Psi
  P_k^X       orthonormal basis of span{R, I}: a plane for k = 1..4, a line for k = 5
  D_x         span of P_1^x .. P_5^x (the input representation, in the full 4096-dim MLP input)
The alive-only block 0 is rebuilt in float64 from the checkpoint's alive components (weight delta off)
and checked against the stored inner activations (`variance/alive_only/L0/inner.npy`, bf16 forward).

    python mlp0_mod10.py            # writes <run>/analysis/reader_writer/mlp0_mod10/
"""

import glob
import json
from pathlib import Path

import numpy as np
from mod10_L0 import HUB, RUN, SRC, A, N, bf16_tensor

OUT = RUN / "analysis/reader_writer/mlp0_mod10"
ALIVE = RUN / "addsub-08-last-pos-40000/alive/alive.json"
CKPT = RUN / "ckpts/40000/decomposition"
MOD10 = (
    "up.794",
    "gate.43",
    "gate.69",
    "up.60",
    "gate.62",
    "up.52",
    "up.54",
    "gate.58",
    "gate.59",
    "gate.47",
)  # fmt: skip  (u31 left out)
EPS = 1e-5
BOS = 128000


def attn_components() -> dict[str, np.ndarray]:
    """V (4096, C_alive) and U (C_alive, d_out) of block 0's alive q/k/v/o components, cached."""
    path = OUT / "attn0_alive.npz"
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
        ids = np.asarray(alive[f"layers.0.self_attn.{k}_proj"])
        V, U = (np.asarray(x[0]) for x in t["components"]["stacks"][k])
        out[f"V_{k}"], out[f"U_{k}"] = V[:, ids], U[ids]
        out[f"ids_{k}"] = ids
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x**2).mean(-1, keepdims=True) + EPS)


def rope(x: np.ndarray, pos: int, cfg: dict) -> np.ndarray:
    """HF Llama rotary embedding (rotate_half layout) with llama3 frequency scaling; x (..., 128)."""
    d = x.shape[-1]
    inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, d, 2) / d)
    rs = cfg["rope_scaling"]
    lo_w = rs["original_max_position_embeddings"] / rs["low_freq_factor"]
    hi_w = rs["original_max_position_embeddings"] / rs["high_freq_factor"]
    wl = 2 * np.pi / inv
    smooth = (rs["original_max_position_embeddings"] / wl - rs["low_freq_factor"]) / (
        rs["high_freq_factor"] - rs["low_freq_factor"]
    )
    scaled = np.where(wl > lo_w, inv / rs["factor"], inv)
    mid = (wl <= lo_w) & (wl >= hi_w)
    scaled = np.where(mid, (1 - smooth) * inv / rs["factor"] + smooth * inv, scaled)
    ang = pos * np.concatenate([scaled, scaled])
    half = np.concatenate([-x[..., d // 2 :], x[..., : d // 2]], -1)
    return x * np.cos(ang) + half * np.sin(ang)


def fourier(X: np.ndarray) -> np.ndarray:
    """Psi_k^X for k = 0..5, (6, 4096) complex, from a field X (400, 4096) indexed by a - 1."""
    mu = np.stack([X[r == A % 10].mean(0) for r in range(10)])
    return np.stack([np.exp(-2j * np.pi * k * np.arange(10) / 10) @ mu for k in range(6)])


def plane(psi: np.ndarray) -> np.ndarray:
    M = np.stack([psi.real, psi.imag], 1)
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def mlp(x: np.ndarray, comps: dict, drop: np.ndarray | None = None, cols: np.ndarray | None = None):
    """MLP output m (400, 4096) and writer activations; with `drop`, the readers in `cols` see
    x - Pi_drop x instead of x (drop: orthonormal basis (4096, d))."""
    tg = x @ comps["V_gate"]
    tu = x @ comps["V_up"]
    if drop is not None:
        xd = x - (x @ drop) @ drop.T
        G = comps["V_gate"].shape[1]
        sel_g, sel_u = cols[cols < G], cols[cols >= G] - G
        tg[:, sel_g] = xd @ comps["V_gate"][:, sel_g]
        tu[:, sel_u] = xd @ comps["V_up"][:, sel_u]
    hg, hu = tg @ comps["U_gate"], tu @ comps["U_up"]
    o = hg / (1 + np.exp(-hg)) * hu
    td = o @ comps["V_down"]
    return td @ comps["U_down"], tg, tu, td


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]

    def w(name: str, rows: np.ndarray | None = None) -> np.ndarray:
        return bf16_tensor(snap + wmap[name], name, rows)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    ids = [tok.encode(str(a), add_special_tokens=False) for a in A]
    assert all(len(t) == 1 for t in ids)
    e = w("model.embed_tokens.weight", np.asarray([t[0] for t in ids]))  # (400, 4096)
    e_bos = w("model.embed_tokens.weight", np.asarray([BOS]))[0]
    g1 = w("model.layers.0.input_layernorm.weight")
    g2 = w("model.layers.0.post_attention_layernorm.weight")

    at = {k: v.astype(np.float64) for k, v in attn_components().items()}
    H, KV, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], 128

    def proj(x: np.ndarray, k: str) -> np.ndarray:
        return (x @ at[f"V_{k}"]) @ at[f"U_{k}"]

    xa0, xa1 = g1 * e_bos / rms(e_bos), g1 * e / rms(e)
    q1 = rope(proj(xa1, "q").reshape(N, H, dh), 1, cfg)
    k0 = rope(proj(xa0[None], "k").reshape(1, KV, dh), 0, cfg)
    k1 = rope(proj(xa1, "k").reshape(N, KV, dh), 1, cfg)
    v0, v1 = proj(xa0[None], "v").reshape(1, KV, dh), proj(xa1, "v").reshape(N, KV, dh)
    rep = np.arange(H) // (H // KV)
    s0 = (q1 * k0[:, rep]).sum(-1) / np.sqrt(dh)
    s1 = (q1 * k1[:, rep]).sum(-1) / np.sqrt(dh)
    p0 = 1 / (1 + np.exp(s1 - s0))  # softmax over keys {0, 1}
    heads = p0[..., None] * v0[:, rep] + (1 - p0)[..., None] * v1[:, rep]
    attn = proj(heads.reshape(N, H * dh), "o")
    r_mid = e + attn
    x = g2 * r_mid / rms(r_mid)

    comps = {k: v.astype(np.float64) for k, v in np.load(SRC / "comps.npz").items() if k[0] in "UV"}
    kind, cidx = np.load(SRC / "comps.npz")["kind"], np.load(SRC / "comps.npz")["index"]
    inner = np.asarray(np.load(SRC / "inner.npy", mmap_mode="r")[(A - 1) * N, 0, :], np.float64)
    m, tg, tu, td = mlp(x, comps)
    res: dict = {"attn_alive": {k: int(at[f"ids_{k}"].size) for k in "qkvo"}}
    for name, mine, kk in (("gate", tg, "gate"), ("up", tu, "up"), ("down", td, "down")):
        ref = inner[:, kind == kk]
        res[f"check_{name}_rel"] = float(np.abs(mine - ref).max() / np.abs(ref).max())
        res[f"check_{name}_corr_min"] = float(
            min(
                np.corrcoef(mine[:, j], ref[:, j])[0, 1]
                for j in range(ref.shape[1])
                if ref[:, j].std() > 0
            )
        )
    print(json.dumps({k: v for k, v in res.items() if k.startswith("check")}, indent=1), flush=True)
    # the MLP output from the STORED writer activations (the model's own numbers)
    m_st = inner[:, kind == "down"] @ comps["U_down"]
    r_out = r_mid + m

    F = {n: fourier(X) for n, X in (("e", e), ("attn", attn), ("r_mid", r_mid), ("x", x), ("m", m),
                                     ("m_stored", m_st), ("r_out", r_out))}  # fmt: skip
    norm_mid = float(np.linalg.norm(r_mid, axis=1).mean())
    res["mean_norm"] = {n: float(np.linalg.norm(X, axis=1).mean()) for n, X in
                        (("r_mid", r_mid), ("m", m), ("r_out", r_out), ("e", e), ("attn", attn))}  # fmt: skip
    res["per_k"] = {}
    for k in range(1, 6):
        pm, pmm, po = F["r_mid"][k], F["m"][k], F["r_out"][k]
        Pmid, Pm = plane(pm), plane(pmm)
        inplane = lambda v, P: (
            np.linalg.norm(P.T @ v.real) ** 2 + np.linalg.norm(P.T @ v.imag) ** 2
        )  # noqa: E731
        beta = np.vdot(pm, pmm) / np.vdot(pm, pm)
        ell = lambda v: (lambda s: float(s[-1] / s[0]))(
            np.linalg.svd(np.stack([v.real, v.imag], 1), compute_uv=False)
        )  # noqa: E731
        res["per_k"][k] = {
            "amp": {n: float(np.linalg.norm(F[n][k])) for n in F},
            "amp_over_mean_norm_r_mid": float(np.linalg.norm(pm) / norm_mid),
            "beta_m_on_mid": [float(beta.real), float(beta.imag)],
            "m_share_in_mid_plane": float(inplane(pmm, Pmid) / np.linalg.norm(pmm) ** 2),
            "cos_planes_m_mid": np.linalg.svd(Pm.T @ Pmid, compute_uv=False).tolist(),
            "out_in_mid_plane_over_mid": float(np.sqrt(inplane(po, Pmid)) / np.linalg.norm(pm)),
            "out_amp_over_mid": float(np.linalg.norm(po) / np.linalg.norm(pm)),
            "ellipse_minor_over_major": {n: ell(F[n][k]) for n in ("r_mid", "x", "m", "r_out")}
            if k < 5
            else None,
            "cos_planes_x_mid": np.linalg.svd(plane(F["x"][k]).T @ Pmid, compute_uv=False).tolist(),
        }
    # mediation: remove D_x from the input of chosen readers
    Dx = np.linalg.qr(np.concatenate([plane(F["x"][k]) for k in range(1, 6)], 1))[0]
    names = [f"{kd}.{int(i)}" for kd, i in zip(kind, cidx, strict=True) if kd != "down"]
    ten = np.asarray([names.index(n) for n in MOD10])
    allr = np.arange(len(names))
    others = np.setdiff1d(allr, ten)
    variants = {"D_removed_all": allr, "D_removed_except_10": others, "D_removed_from_10": ten}
    res["dim_Dx"] = int(Dx.shape[1])
    res["mediation"] = {}
    for vn, cols in variants.items():
        mv = mlp(x, comps, Dx, cols)[0]
        Fv = fourier(mv)
        res["mediation"][vn] = {
            k: {
                "amp_ratio": float(np.linalg.norm(Fv[k]) / np.linalg.norm(F["m"][k])),
                "overlap": float((np.vdot(F["m"][k], Fv[k]) / np.vdot(F["m"][k], F["m"][k])).real),
            }  # fmt: skip
            for k in range(1, 6)
        }
        res["mediation"][vn]["m_change_rel"] = float(np.linalg.norm(mv - m) / np.linalg.norm(m))
    np.savez(OUT / "fields.npz", r_mid=r_mid, x=x, m=m, m_stored=m_st, attn=attn, e=e, Dx=Dx)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
