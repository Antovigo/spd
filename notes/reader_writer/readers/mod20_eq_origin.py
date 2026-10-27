"""What the stream at "=" already holds, along the 3D clump axes of section 5.3, before the copying attention writes, and
which earlier sublayers wrote it (alive-only model; the original model at the copying block only).

Nothing is centred except in the factor split below, which says so. Notation (copy of o = a by block L = 16 from position
u = 1, or o = b by block L = 15 from position u = 3; copy_build.py's prompt sets, 8,000 prompts: every o, ten values of the
other operand o', both operations op):
  E          the stream at "=" entering block L;  C the part of block L's attention output at "=" from position u (the copy);
             R the rest of block L's attention output at "=" (positions 0..4 except u);  E' = E + C + R
  B          the 4096 x 3 basis of section 5.3, computed from E' (axes 1-2: the clump layout's part in the mod-20 plane;
             axis 3: the rest, in the mod-10 subspace)
  y(Z)       = Z B, a term's coordinates on the three axes, for each prompt
  factor split of y over the full factorial (o, context), context = (o', op), 20 contexts:
             offset (the mean over prompts, a constant), own (mean over contexts per o, minus the offset), context (mean
             over o per context, minus the offset), interaction (the rest). Shares are of sum_prompts ||y||^2 and add to 1;
             own_s is the part of "own" explained by o mod 20 alone
  sublayer terms of E: the embedding at "=", and every attention and MLP output at "=" of blocks 0..L-1 (a float64 rebuild of
             positions 0..4 as in copy_build.py), each projected on B
Writes <run>/analysis/reader_writer/track_mod20/eq_origin.json.

    python mod20_eq_origin.py
"""

import glob
import json
from pathlib import Path

import numpy as np
from copy_analyze import attention_parts, copied
from copy_build import OUT as COPY
from copy_build import prompts
from copy_figs import SUB
from mod10_L0 import HUB, bf16_tensor
from mod20_copy import orig_parts
from mod20_track import OUT
from track_build import load_blocks
from track_build_p3 import attn_seq


def means(Z: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    return np.stack([Z[lab == q].mean(0) for q in range(n)])


def basis(x: np.ndarray, oper: np.ndarray) -> np.ndarray:
    """Section 5.3's axes (mod20_clumps3d.view), from the field x (n, 4096) and the operand."""
    s = oper % 20
    mu = means(x, s, 20)
    p = np.exp(-2j * np.pi * np.arange(20) / 20) @ mu
    P = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)[0]
    nu = means(x, s // 5, 4)
    G = (nu[1:] - nu[0]).T
    E12 = np.linalg.svd(P @ (P.T @ G), full_matrices=False)[0][:, :2]
    e3 = np.linalg.svd(G - E12 @ (E12.T @ G), full_matrices=False)[0][:, :1]
    B = np.concatenate([E12, e3], 1)
    if (mu[0] - mu[10]) @ B[:, 0] < 0:
        B[:, 0] *= -1
    if (nu[1] - nu[0]) @ B[:, 2] < 0:
        B[:, 2] *= -1
    return B


def split(y: np.ndarray, oper: np.ndarray, ctx: np.ndarray) -> dict:
    """Factor split of y (n, 3) over operand values x contexts (least squares on the two sets of indicators, which is the
    plain two-way split on a full factorial)."""
    tot = float((y**2).sum())
    off = y.mean(0)
    uo, io = np.unique(oper, return_inverse=True)
    uc, ic = np.unique(ctx, return_inverse=True)
    own = means(y, io, len(uo)) - off
    con = means(y, ic, len(uc)) - off
    fit_own, fit_con = own[io], con[ic]
    inter = y - off - fit_own - fit_con
    own_s = means(fit_own, oper % 20, 20)[oper % 20]
    return {"total": tot, "offset": float(len(y) * (off**2).sum()) / tot, "own": float((fit_own**2).sum()) / tot,
            "own_s": float((own_s**2).sum()) / tot, "context": float((fit_con**2).sum()) / tot,
            "interaction": float((inter**2).sum()) / tot, "offset_vec": off.tolist(),
            "context_means": con.tolist(), "context_ids": uc.tolist()}  # fmt: skip


def main() -> None:
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    blocks = load_blocks()[0]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    res: dict = {}
    for which, (L, u) in {"a": (16, 1), "b": (15, 3)}.items():
        other = "b" if which == "a" else "a"
        st = np.load(COPY / f"stream_{which}.npz")
        X = st[f"X{L}"].astype(np.float64)
        oper, oth, op = (st[k].astype(int) for k in (which, other, "op"))
        ctx = op * 1000 + oth
        parts = attention_parts(blocks[L], X)
        C = copied(blocks[L], parts, u)
        Rr = sum(copied(blocks[L], parts, t) for t in range(5) if t != u)
        E = X[:, 4]
        B = basis(E + C + Rr, oper)
        r = {"terms": {}}
        for name, Z in (("E (stream before the attention)", E), ("C (copy from the operand)", C),
                        ("R (attention from the other positions)", Rr), ("E' (after the attention)", E + C + Rr)):  # fmt: skip
            r["terms"][name] = split(Z @ B, oper, ctx)
        del parts, C, Rr
        # sublayer terms of E: rebuild positions 0..4 through blocks 0..L-1, project each write at "=" on B
        meta = prompts(which)
        assert (meta[which] == st[which]).all() and (meta[other] == st[other]).all()
        toks = np.asarray([tok.encode(f"{a}{'+-'[o]}{b}=", add_special_tokens=True)
                           for a, b, o in zip(meta["a"], meta["b"], meta["op"], strict=True)])  # fmt: skip
        uniq, inv = np.unique(toks.ravel(), return_inverse=True)
        Y = bf16_tensor(
            snap + wmap["model.embed_tokens.weight"], "model.embed_tokens.weight", uniq
        )[inv]
        Y = Y.reshape(*toks.shape, -1)
        sub = {"embedding": split(Y[:, 4] @ B, oper, ctx)}
        proj = Y[:, 4] @ B
        for blk in blocks[:L]:
            att = np.concatenate([attn_seq(blk, Y[i : i + 1000]) for i in range(0, len(Y), 1000)])
            Y = Y + att
            sub[f"L{blk.b} attention"] = split(att[:, 4] @ B, oper, ctx)
            proj = proj + att[:, 4] @ B
            n, T, d = Y.shape
            flat = Y.reshape(n * T, d)
            m = np.concatenate(
                [blk.mlp(flat[i : i + 2000]) for i in range(0, n * T, 2000)]
            ).reshape(n, T, d)
            Y = Y + m
            sub[f"L{blk.b} MLP"] = split(m[:, 4] @ B, oper, ctx)
            proj = proj + m[:, 4] @ B
            print(which, blk.b, flush=True)
        r["rebuild_check"] = float(np.abs(proj - E @ B).max() / np.abs(E @ B).max())
        # contribution of each term to E's coordinates: <y_term, y_E> / ||y_E||^2 (sums to 1), per factor
        yE = E @ B
        sE = split(yE, oper, ctx)
        r["sublayers"] = sub
        r["E_split"] = sE
        res[f"alive_{which}"] = r
        # original model: the same terms at block L only
        Xo = np.load(COPY / f"orig_cache_{which}.npz")["Xor"].astype(np.float64)
        sl = SUB[which]
        opo, otho, opop = (st[k][sl].astype(int) for k in (which, other, "op"))
        _, p, v, Wo = orig_parts(snap, wmap, cfg, L, Xo)
        Co = np.einsum("nhd,ohd->no", p[:, :, u, None] * v[:, u], Wo)
        Ro = (
            np.einsum("nht,nthd->nhd", np.delete(p, u, 2), np.delete(v, u, 1)).reshape(len(Xo), -1)
            @ Wo.reshape(4096, -1).T
        )
        Eo = Xo[:, 4]
        Bo = basis(Eo + Co + Ro, opo)
        ro = {"terms": {}}
        for name, Z in (("E (stream before the attention)", Eo), ("C (copy from the operand)", Co),
                        ("R (attention from the other positions)", Ro), ("E' (after the attention)", Eo + Co + Ro)):  # fmt: skip
            ro["terms"][name] = split(Z @ Bo, opo, opop * 1000 + otho)
        res[f"orig_{which}"] = ro
        del p, v, Xo
        for k in (f"alive_{which}", f"orig_{which}"):
            for name, t in res[k]["terms"].items():
                print(k, name, {kk: round(t[kk], 3) for kk in ("offset", "own", "own_s", "context", "interaction")},
                      "rms", round((t["total"] / (8000 if k.startswith("alive") else 2000)) ** 0.5, 3), flush=True)  # fmt: skip
        print("rebuild check", r["rebuild_check"], flush=True)
    (OUT / "eq_origin.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
