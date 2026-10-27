"""Data for the rotatable 3D views of the four quarter clumps at the source positions just before the copy to "=":
a at position 1 entering block 16, b at position 3 entering block 15, in the alive-only model (components) and the
original model (neurons). Nothing is centred: the origin of every view is the activation zero.

Notation (o = the copied operand, a or b, integer 1..400; s = o mod 20; r = o mod 10; Q = floor(s / 5), integer 0..3):
  x(p)      attention input of the copying block (its input RMSNorm output) at the source position, for prompt p
  nu_Q      mean of x over the prompts of quarter Q;  G = [nu_1 - nu_0, nu_2 - nu_0, nu_3 - nu_0] (4096 x 3)
  P         the frequency-1 plane of o mod 20 of x;  D10 its 9-dimensional o mod 10 subspace (both from the residue means)
  basis     e1, e2: orthonormal basis of the projection of span(G) onto P (the part of the clump layout inside the mod-20
            plane); e3: the unit vector along what is left of span(G) after removing e1 and e2. A point's coordinates are
            (e1 . x, e2 . x, e3 . x)
  shares    share of ||G||_F^2 in P, in D10, and in span(e1, e2, e3)
Prompts: a, the 400 values of a (position 1 depends on a alone; components model from track_build.py, original model from
track_build_orig.py, stream entering block 16); b, copy_build.py's set "b" (ten a, every b, both ops: 8,000 prompts) and
copy_figs.py's 2,000-prompt original-model cache.
With --after, the same views after the copy, at "=" (position 4), for the same prompt sets: "copy" is the part of the
copying block's attention output at "=" that comes from the operand's position (all heads; copy_analyze.copied, and its
original-model twin from mod20_copy.orig_parts); "eq" is the stream at "=" after the whole attention write (the stream
entering the block plus the attention output from all positions). For a, set "a" of copy_build.py (every a, ten b,
both ops; 8,000 prompts; original model: copy_figs.py's 2,000-prompt cache); for b, set "b" as above.
Writes <out>/mod20_clumps3d.json (before the copy) or <out>/mod20_clumps3d_after.json (--after).

    python mod20_clumps3d.py --out notebook_mod20 [--after]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from copy_build import OUT as COPY
from copy_figs import SUB
from mod10_L0 import HUB, bf16_tensor
from track_build import OUT as AOUT
from track_build import load_blocks, rmsnorm
from track_build_orig import OUT as OOUT


def means(Z: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    return np.stack([Z[lab == q].mean(0) for q in range(n)])


def view(x: np.ndarray, oper: np.ndarray) -> dict:
    s = oper % 20
    Q = s // 5
    mu = means(x, s, 20)
    p = np.exp(-2j * np.pi * np.arange(20) / 20) @ mu
    P = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)[0]
    m10 = means(x, oper % 10, 10)
    F = np.fft.fft(m10, axis=0)
    D10 = np.linalg.svd(np.concatenate([np.stack([F[k].real, F[k].imag], 1) for k in range(1, 5)] + [F[5].real[:, None]], 1),
                        full_matrices=False)[0]  # fmt: skip
    nu = means(x, Q, 4)
    G = (nu[1:] - nu[0]).T
    E12 = np.linalg.svd(P @ (P.T @ G), full_matrices=False)[0][:, :2]
    rest = G - E12 @ (E12.T @ G)
    e3 = np.linalg.svd(rest, full_matrices=False)[0][:, :1]
    B = np.concatenate([E12, e3], 1)
    # orient: residue 0's mean on +e1 side, e3 so that quarter 1 (r = 5..9) is positive
    if (mu[0] - mu[10]) @ B[:, 0] < 0:
        B[:, 0] *= -1
    if ((nu[1] - nu[0]) @ B[:, 2]) < 0:
        B[:, 2] *= -1
    tot = (G**2).sum()
    return {
        "points": np.round(x @ B, 4).tolist(),
        "s": s.tolist(),
        "operand": oper.tolist(),
        "residue_means": np.round(mu @ B, 4).tolist(),
        "quarter_means": np.round(nu @ B, 4).tolist(),
        "shares": {
            "G_in_P": float(((P.T @ G) ** 2).sum() / tot),
            "G_in_D10": float(((D10.T @ G) ** 2).sum() / tot),
            "G_in_view": float(((B.T @ G) ** 2).sum() / tot),
        },  # fmt: skip
        "G_sv": np.linalg.svd(G, compute_uv=False).tolist(),
        "e3_in_D10": float(((D10.T @ B[:, 2]) ** 2).sum()),
        "e3_in_P": float(((P.T @ B[:, 2]) ** 2).sum()),
    }


def after(out_dir: Path) -> None:
    import gc

    from copy_analyze import attention_parts, copied
    from mod20_copy import orig_parts

    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    blocks = load_blocks()[0]
    out = {}
    for which, (L, u) in {"a": (16, 1), "b": (15, 3)}.items():
        st = np.load(COPY / f"stream_{which}.npz")
        X = st[f"X{L}"].astype(np.float64)
        oper = st[which].astype(int)
        parts = attention_parts(blocks[L], X)
        cp = copied(blocks[L], parts, u)
        att = sum(copied(blocks[L], parts, t) for t in range(5))
        out[f"alive_{which}_copy"] = view(cp, oper)
        out[f"alive_{which}_eq"] = view(X[:, 4] + att, oper)
        del parts, X, att, cp
        gc.collect()
        Xo = np.load(COPY / f"orig_cache_{which}.npz")["Xor"].astype(np.float64)
        opo = st[which][SUB[which]].astype(int)
        _, p, v, Wo = orig_parts(snap, wmap, cfg, L, Xo)
        cpo = np.einsum("nhd,ohd->no", p[:, :, u, None] * v[:, u], Wo)
        atto = np.einsum("nht,nthd->nhd", p, v).reshape(len(Xo), -1) @ Wo.reshape(4096, -1).T
        out[f"orig_{which}_copy"] = view(cpo, opo)
        out[f"orig_{which}_eq"] = view(Xo[:, 4] + atto, opo)
        del p, v, Xo
        gc.collect()
    for k, v_ in out.items():
        print(k, len(v_["points"]), {kk: round(vv, 3) for kk, vv in v_["shares"].items()}, np.round(v_["G_sv"], 3),
              f"e3 in D10 {v_['e3_in_D10']:.3f}", flush=True)  # fmt: skip
    (out_dir / "mod20_clumps3d_after.json").write_text(json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--after", action="store_true", help="views at the last position after the copy"
    )
    args = ap.parse_args()
    if args.after:
        after(args.out)
        return
    (snap,) = glob.glob(HUB)
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]

    def g1(L: int) -> np.ndarray:
        name = f"model.layers.{L}.input_layernorm.weight"
        return bf16_tensor(snap + wmap[name], name)

    blocks = load_blocks()[0]
    A = np.arange(1, 401)
    out = {}
    xa = rmsnorm(np.load(AOUT / "stream.npz")["pre_16"].astype(np.float64), blocks[16].g1)
    out["alive_a"] = view(xa, A)
    xo = rmsnorm(np.load(OOUT / "stream.npz")["pre_16"].astype(np.float64), g1(16))
    out["orig_a"] = view(xo, A)
    st = np.load(COPY / "stream_b.npz")
    xb = rmsnorm(st["X15"][:, 3].astype(np.float64), blocks[15].g1)
    out["alive_b"] = view(xb, st["b"].astype(int))
    Xo = np.load(COPY / "orig_cache_b.npz")["Xor"][:, 3].astype(np.float64)
    out["orig_b"] = view(rmsnorm(Xo, g1(15)), st["b"][SUB["b"]].astype(int))
    for k, v in out.items():
        print(k, len(v["points"]), {kk: round(vv, 3) for kk, vv in v["shares"].items()}, np.round(v["G_sv"], 3),
              f"e3 in D10 {v['e3_in_D10']:.3f} in P {v['e3_in_P']:.3f}", flush=True)  # fmt: skip
    (args.out / "mod20_clumps3d.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
