"""Where the copying head's attention weight on the source varies from, split exactly between its query and its keys
(sets "a" and "b", 2,000 prompts each; alive-only and original models).

Copy of a: block 16, head 21, source position 1. Copy of b: block 15, head 13, source position 3. At "=" (position 4)
the head's weight on the source is p = softmax_t(q . k_t / sqrt(128))[source], with q the head's query at "=" and k_t
its key at position t (0..4). Three factors: Q (the query), K_src (the key at the source), K_rest (the keys at the other
positions). For a subset of factors, p is recomputed with the other factors replaced by their mean over prompts; the
Shapley value of each factor, per prompt, sums with the others to p - p(all at mean). A factor's share is the covariance
of its Shapley value with p over the variance of p, both within a residue r of the copied operand mod 10, so the shares
cover the spread of p inside each residue's cloud and sum to 1. The prompt variables each factor follows are reported as
within-residue R^2 on the exact copied operand, the operator, and (other operand, operator).

    python weight_qk_split.py
"""

import glob
import itertools
import json
from math import factorial
from pathlib import Path

import numpy as np
from copy_build import OUT
from copy_figs import SUB
from mlp0_mod10 import rope
from mod10_L0 import HUB, bf16_tensor
from track_build import load_blocks, rmsnorm

CASES = {"a": (16, 21, 1), "b": (15, 13, 3)}
FACT = ("Q", "K_src", "K_rest")


def qk_alive(blk, X, H):
    n = len(X)
    x = rmsnorm(X, blk.g1)
    q = rope(blk.proj(x[:, 4], "q").reshape(n, 32, 128), 4, blk.cfg)[:, H]
    k = np.stack(
        [rope(blk.proj(x[:, t], "k").reshape(n, 8, 128), t, blk.cfg)[:, H // 4] for t in range(5)],
        1,
    )
    return q, k


def qk_orig(snap, wmap, cfg, L, X, H):
    def w(name):
        return bf16_tensor(snap + wmap[name], name).astype(np.float64)

    pre = f"model.layers.{L}."
    n = len(X)
    x = rmsnorm(X, w(pre + "input_layernorm.weight"))
    Wq = w(pre + "self_attn.q_proj.weight")[H * 128 : (H + 1) * 128]
    Wk = w(pre + "self_attn.k_proj.weight")[(H // 4) * 128 : (H // 4 + 1) * 128]
    q = rope((x[:, 4] @ Wq.T).reshape(n, 1, 128), 4, cfg)[:, 0]
    k = np.stack([rope((x[:, t] @ Wk.T).reshape(n, 1, 128), t, cfg)[:, 0] for t in range(5)], 1)
    return q, k


def weight(q, k, s):
    z = np.einsum("nd,ntd->nt", q, k) / np.sqrt(128)
    z = np.exp(z - z.max(1, keepdims=True))
    return z[:, s] / z.sum(1)


def within(y, lab):
    d = y.astype(np.float64).copy()
    for r in range(10):
        d[lab == r] -= y[lab == r].mean()
    return d


def r2_within(y, lab, g):
    d = within(y, lab)
    m = np.zeros_like(d)
    for u in np.unique(g):
        m[g == u] = d[g == u].mean()
    return float(1 - ((d - m) ** 2).sum() / (d**2).sum())


def main():
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    blocks = load_blocks()[0]
    out = {}
    for w_, (L, H, s) in CASES.items():
        st = np.load(OUT / f"stream_{w_}.npz")
        sub = SUB[w_]
        a, b, op = st["a"][sub], st["b"][sub], st["op"][sub]
        cop, oth = (a, b) if w_ == "a" else (b, a)
        lab = cop % 10
        rest = [t for t in range(5) if t != s]
        for name in ("alive-only", "original"):
            if name == "alive-only":
                q, k = qk_alive(blocks[L], st[f"X{L}"][sub].astype(np.float64), H)
            else:
                q, k = qk_orig(
                    snap,
                    wmap,
                    cfg,
                    L,
                    np.load(OUT / f"orig_cache_{w_}.npz")["Xor"].astype(np.float64),
                    H,
                )
            qm, km = np.broadcast_to(q.mean(0), q.shape), np.broadcast_to(k.mean(0), k.shape)

            def p_of(S, q=q, k=k, qm=qm, km=km, s=s, rest=rest):
                kk = km.copy()
                if "K_src" in S:
                    kk[:, s] = k[:, s]
                if "K_rest" in S:
                    kk[:, rest] = k[:, rest]
                return weight(q if "Q" in S else qm, kk, s)

            vals = {S: p_of(S) for r_ in range(4) for S in itertools.combinations(FACT, r_)}
            full = vals[FACT]
            phi = {}
            for f in FACT:
                others = [g for g in FACT if g != f]
                tot = np.zeros(len(full))
                for r_ in range(3):
                    for S in itertools.combinations(others, r_):
                        wgt = factorial(len(S)) * factorial(2 - len(S)) / factorial(3)
                        Sf = tuple(g for g in FACT if g in S or g == f)
                        tot += wgt * (vals[Sf] - vals[S])
                phi[f] = tot
            d = within(full, lab)
            res = {
                "p_mean_within_sd": [float(full.mean()), float(d.std())],
                "share": {f: float((within(phi[f], lab) * d).sum() / (d * d).sum()) for f in FACT},
                "factor_r2_within": {
                    f: {
                        "copied operand": r2_within(phi[f], lab, cop),
                        "op": r2_within(phi[f], lab, op),
                        "other operand and op": r2_within(phi[f], lab, oth * 2 + op),
                    }
                    for f in FACT
                },  # fmt: skip
                "p_r2_within": {
                    "copied operand": r2_within(full, lab, cop),
                    "op": r2_within(full, lab, op),
                    "other operand and op": r2_within(full, lab, oth * 2 + op),
                },  # fmt: skip
            }
            out[f"{w_}/{name}"] = res
            print(w_, name, json.dumps(res, indent=1), flush=True)
    (OUT / "weight_qk_split.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
