"""Which upstream writers put the b mod 50 pattern into the input of the block-15 gate readers g19 and g29 (position 3).

Model: the alive-only model of variance/alive_only, rebuilt at position 3 by readers/track_build_p3.py on its
8,000 prompts (op in {+, -}, a in A_SUB = ten values 7 + 41 i, i = 0..9, every b = 1..400); its stream.npz holds
the residual stream at position 3 entering block l (pre_l) and after its attention (mid_l), and positions 0..2 of
the 20 (op, a) contexts (ctx_pre_l). Nothing is centred: zero is the zero of the activation space.

Notation (position 3 throughout; x a prompt; b its second operand, integer 1..400):
  c            a reader: block-15 gate component 19 or 29; V_c in R^4096 its read vector, gamma in R^4096 the gain
               of block 15's post-attention RMSNorm
  X(x)         mid_15(x) in R^4096, the stream the reader reads; rho(x) = sqrt(mean_j X_j(x)^2 + 1e-5)
  f_c(x)       = (gamma * V_c) . X(x) / (rho(x) ||V_c||): the reader's inner activation divided by ||V_c||
  writers w    the embedding of the token b (one term, "emb"); every alive attention o component of blocks 0..15 and
               every alive MLP down component of blocks 0..14; t_w(x) its activation at position 3 (for o: the
               concatenated head outputs . V_o,w; for down: the MLP hidden activation . V_down,w), U_w its write
               vector. X(x) = emb(x) + sum_w t_w(x) U_w exactly
  e_w(x)       = t_w(x) (gamma * V_c) . U_w / (rho(x) ||V_c||): writer w's term in f_c; sum_w e_w = f_c exactly
               (rho is the reader's own per-prompt norm, shared by every term)
  F(b)         mean of a field over the 20 prompts with that b (10 a values, 2 ops)
  m(r)         mean of F over the 4 values b with b mod 100 = r, r in 0..99
  M(k)         sum_r m(r) exp(-2 pi i k r / 100), k in 0..99 (i the imaginary unit); tau(k) = 100 / gcd(k, 100)
  P_S F        the part of m(b mod 100) made of the frequencies k in the set S, a function of b (400 values)
  S50          {k : tau(k) = 50}, i.e. gcd(k, 100) = 2: the exact-period-50 part
  S|50         {k != 0 : tau(k) divides 50}: every non-constant function of b mod 50 (k even, k != 0)
  share_w(S)   <P_S F_{e_w}, P_S F_{f_c}> / ||P_S F_{f_c}||^2 over b = 1..400: writer w's share of the reader's S-part
               (sums to 1 over the writers, signed: a negative share opposes the reader's pattern)
  own_w(S)     ||P_S F_{e_w}||^2 / ||P_S F_{f_c}||^2: the size of writer w's S-part next to the reader's
  energy shares of f_c by exact period tau as in seed_periodicity.py, but over these 8,000 prompts

Writes <run>/analysis/reader_writer/mod50_p3/l15_writers.json.

    python mod50_l15_writers.py
"""

import json
import sys
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mlp0_mod10 import ALIVE, rope  # noqa: E402
from track_build import Block, rmsnorm  # noqa: E402
from track_build_p3 import OUT as STREAM  # noqa: E402
from track_build_p3 import load  # noqa: E402

RUN = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
OUT = RUN / "analysis/reader_writer/mod50_p3"
READ_BLOCK = 15
READERS = (19, 29)
K = np.arange(100)
TAU = np.asarray([100 // gcd(int(k), 100) for k in K])
SETS = {"50": TAU == 50, "div50": (50 % TAU == 0) & (K != 0)}
TOP = 25


def o_acts(blk: Block, ctx: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Activations of the alive o components at position 3: keys/values 0..2 from ctx (n, 3, 4096), 3 from X."""
    H, KV, dh = blk.cfg["num_attention_heads"], blk.cfg["num_key_value_heads"], 128
    n = X.shape[0]
    xc, xl = rmsnorm(ctx, blk.g1), rmsnorm(X, blk.g1)
    rep = np.arange(H) // (H // KV)
    q = rope(blk.proj(xl, "q").reshape(n, H, dh), 3, blk.cfg)
    ks = [rope(blk.proj(xc[:, t], "k").reshape(n, KV, dh), t, blk.cfg) for t in range(3)]
    ks.append(rope(blk.proj(xl, "k").reshape(n, KV, dh), 3, blk.cfg))
    vs = [blk.proj(xc[:, t], "v").reshape(n, KV, dh) for t in range(3)]
    vs.append(blk.proj(xl, "v").reshape(n, KV, dh))
    k, v = np.stack(ks, 1)[:, :, rep], np.stack(vs, 1)[:, :, rep]
    s = np.einsum("nhd,nshd->nhs", q, k) / np.sqrt(dh)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    heads = np.einsum("nhs,nshd->nhd", p, v).reshape(n, H * dh)
    return heads @ blk.a["V_o"]


def by_b(F: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, ...) -> (400, ...) mean over the prompts with each b."""
    out = np.zeros((400, *F.shape[1:]))
    np.add.at(out, b - 1, F)
    return out / np.bincount(b - 1, minlength=400).reshape(-1, *[1] * (F.ndim - 1))


def parts(Fb: np.ndarray) -> dict[str, np.ndarray]:
    """(400, m) -> {set: (400, m)} the S-parts as functions of b = 1..400."""
    bb = np.arange(1, 401)
    mr = np.zeros((100, Fb.shape[1]))
    for r in range(100):
        mr[r] = Fb[(bb % 100) == r].mean(0)
    M = np.exp(-2j * np.pi * np.outer(K, np.arange(100)) / 100) @ mr  # (100, m)
    out = {}
    for name, sel in SETS.items():
        rec = (np.exp(2j * np.pi * np.outer(np.arange(100), K[sel]) / 100) @ M[sel]).real / 100
        out[name] = rec[bb % 100]
    return out


def shares_by_period(fb: np.ndarray) -> dict[str, float]:
    bb = np.arange(1, 401)
    mr = np.stack([fb[(bb % 100) == r].mean() for r in range(100)])
    M = np.exp(-2j * np.pi * np.outer(K, np.arange(100)) / 100) @ mr
    E = (fb**2).sum()
    return {
        str(t): float(4 * (np.abs(M[t == TAU]) ** 2).sum() / (100 * E)) for t in sorted(set(TAU))
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, _, _ = load()
    alive = json.loads(ALIVE.read_text())["components"]
    z = np.load(STREAM / "stream.npz")
    b, ctx = z["b"], z["ctx"]
    rd = blocks[READ_BLOCK]
    gi = {int(c): j for j, c in enumerate(rd.index[rd.kind == "gate"])}
    Vc = rd.m["V_gate"][:, [gi[c] for c in READERS]]  # (4096, 2)
    read = rd.g2[:, None] * Vc / np.linalg.norm(Vc, axis=0)  # (4096, 2): gamma * V_c / ||V_c||
    X = z[f"mid_{READ_BLOCK}"].astype(np.float64)
    rho = np.sqrt((X**2).mean(-1) + 1e-5)
    f = (X @ read) / rho[:, None]  # (n, 2)
    ref = rd.readers(X)[0][:, [gi[c] for c in READERS]] / np.linalg.norm(Vc, axis=0)
    assert np.allclose(f, ref, rtol=1e-6, atol=1e-6), np.abs(f - ref).max()

    names: list[str] = ["emb"]
    terms: list[np.ndarray] = [(z["pre_0"].astype(np.float64) @ read)[:, None, :]]  # (n, 1, 2)
    for blk in blocks[: READ_BLOCK + 1]:
        blk_i = blk.b
        pre = z[f"pre_{blk_i}"].astype(np.float64)
        cpre = z[f"ctx_pre_{blk_i}"].astype(np.float64)[ctx]
        to = np.concatenate([o_acts(blk, cpre[i : i + 1000], pre[i : i + 1000])
                             for i in range(0, len(pre), 1000)])  # fmt: skip
        mid = z[f"mid_{blk_i}"].astype(np.float64)
        err = np.abs(to @ blk.a["U_o"] - (mid - pre)).max() / np.abs(mid - pre).max()
        assert err < 1e-3, (blk_i, "attn", err)
        terms.append(to[:, :, None] * (blk.a["U_o"] @ read)[None])
        names += [f"L{blk_i}.o.{c}" for c in alive[f"layers.{blk_i}.self_attn.o_proj"]]
        if blk_i < READ_BLOCK:
            _, td = blk.mlp_from_t(*blk.readers(mid))
            nxt = z[f"pre_{blk_i + 1}"].astype(np.float64)
            err = np.abs(td @ blk.m["U_down"] - (nxt - mid)).max() / np.abs(nxt - mid).max()
            assert err < 1e-3, (blk_i, "mlp", err)
            terms.append(td[:, :, None] * (blk.m["U_down"] @ read)[None])
            names += [f"L{blk_i}.down.{c}" for c in blk.index[blk.kind == "down"]]
        print(blk_i, flush=True)
    E = np.concatenate(terms, 1) / rho[:, None, None]  # (n, W, 2)
    gap = np.abs(E.sum(1) - f).max() / np.abs(f).max()
    assert gap < 1e-3, gap

    res: dict[str, Any] = {"definitions": __doc__, "readers": {}}
    for j, c in enumerate(READERS):
        cid = f"L{READ_BLOCK}.gate.{c}"
        Fb = by_b(E[:, :, j], b)  # (400, W)
        fb = by_b(f[:, j], b)
        Pw, Pf = parts(Fb), parts(fb[:, None])
        entry: dict[str, Any] = {
            "period_shares": shares_by_period(fb),
            "reconstruction_rel_max": float(gap),
        }
        for s in SETS:
            nf = (Pf[s][:, 0] ** 2).sum()
            sh = Pw[s].T @ Pf[s][:, 0] / nf
            own = (Pw[s] ** 2).sum(0) / nf
            order = np.argsort(-np.abs(sh))
            sub = {}
            for nm, v in zip(names, sh, strict=True):
                key = (
                    "emb"
                    if nm == "emb"
                    else nm.split(".")[0] + "." + ("attn" if ".o." in nm else "mlp")
                )
                sub[key] = sub.get(key, 0.0) + float(v)
            entry[s] = {
                "sum_shares": float(sh.sum()),
                "top": [
                    {"writer": names[w], "share": float(sh[w]), "own": float(own[w])}
                    for w in order[:TOP]
                ],  # fmt: skip
                "by_sublayer": dict(sorted(sub.items(), key=lambda kv: -abs(kv[1]))),
            }
        res["readers"][cid] = entry
        print(cid, json.dumps(entry["period_shares"]))
        for s in SETS:
            print(f"  {s}: sum {entry[s]['sum_shares']:.3f}")
            print(
                "   sublayers:",
                {k: round(v, 3) for k, v in list(entry[s]["by_sublayer"].items())[:10]},
            )
            print(
                "   top:",
                [
                    (t["writer"], round(t["share"], 3), round(t["own"], 3))
                    for t in entry[s]["top"][:15]
                ],
            )
    (OUT / "l15_writers.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
