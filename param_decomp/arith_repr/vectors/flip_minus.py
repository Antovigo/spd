"""How much of the op-constant's effect goes through neurons that push the "-" token, versus the other
op-dependent neurons of layers 16-31 (alive-only model, `=`).

    python -m param_decomp.arith_repr.vectors.flip_minus select      # CPU -> OUT/flip/opflow/minus_sets.json
    python -m param_decomp.arith_repr.vectors.flip_minus run <cond>  # CPU -> OUT/flip/opflow/minus_<cond>.json

select: per layer 16-31, the N_TOP neurons with the largest op-odd write energy (`flip_opflow.anova` of
act, times |w_n|^2, w_n = sum_c V_down[n, c] U_down[c]). A neuron is a "minus" neuron if its write's
direct logit on "-" (w_n scaled by the final norm gain, through the unembedding) differs from the mean
of its logits on the number tokens 0..200 by >= MINUS_Z of their standard deviation; the others are
"rest".
run: runs A and B of `flip_readers` (A: the flip swapped at `=` entering block 16; B: the flip and the
op constant swapped there); in A, the chosen neurons' activations at `=` are set to their values in B
(through the down inner: h_down += (act_B - act_A)[chosen] @ V_down[chosen]). Scored against the
unedited model's top-1 on 2500 random (a, b) pairs, both ops: swap / same per op, and on a > b and
a < b separately."""

import json
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.vectors.flip_opflow import (
    EQ,
    ODD,
    OUT,
    alive_model,
    anova,
    grid,
    stream_at,
)
from param_decomp.arith_repr.vectors.flip_terms import odd_terms

N_TOP = 100
MINUS_Z = 2.0
LAYERS = tuple(range(16, 32))
CONDS = ("all", "minus", "rest")


def select() -> None:
    cm = alive_model()
    g = grid(cm)
    H = dict(np.load(OUT / "h_eq.npz"))
    tok = np.r_[np.asarray(cm.num_ids), int(cm.minus)]
    Eg = (
        np.asarray(cm.unembed[tok], np.float64) * np.asarray(cm.final, np.float64)[None]
    )  # (202, d)
    sets: dict[str, dict[str, list[int]]] = {"minus": {}, "rest": {}}
    for li in LAYERS:
        pre = {
            k: H[f"{li}.{k}"].astype(np.float64) @ np.asarray(cm.sites[li][k].U, np.float64)
            for k in ("gate", "up")
        }
        act = pre["gate"] / (1 + np.exp(-pre["gate"])) * pre["up"]
        del pre
        W = np.asarray(cm.sites[li]["down"].V, np.float64) @ np.asarray(
            cm.sites[li]["down"].U, np.float64
        )
        ra = anova(act, g)
        oe = sum(ra[k] for k in ODD) * (W**2).sum(1)
        top = np.argsort(-oe)[:N_TOP]
        lg = W[top] @ Eg.T  # (N_TOP, 202)
        z = (lg[:, 201] - lg[:, :201].mean(1)) / lg[:, :201].std(1)
        mi = top[np.abs(z) >= MINUS_Z]
        sets["minus"][str(li)] = [int(n) for n in mi]
        sets["rest"][str(li)] = [int(n) for n in top if n not in set(mi.tolist())]
        print(f"L{li}: top-{N_TOP} op-odd neurons, {len(mi)} minus neurons "
              f"(op-odd energy share of minus in top: {oe[mi].sum() / oe[top].sum():.2f})", flush=True)  # fmt: skip
    (OUT / "minus_sets.json").write_text(json.dumps(sets, indent=1))


def run(cond: str) -> None:
    S = json.loads((OUT / "minus_sets.json").read_text())
    chosen = {
        li: sorted(S["minus"][str(li)] + S["rest"][str(li)]) if cond == "all" else S[cond][str(li)]
        for li in LAYERS
    }
    cm = alive_model()
    g = grid(cm)
    H = dict(np.load(OUT / "h_eq.npz"))
    t = 32
    T = odd_terms(stream_at(cm, H, t), g)
    del H
    dA = (-2 * T["op*b"]).astype(np.float32)
    dB = (-2 * (T["op*b"] + T["op"])).astype(np.float32)
    del T
    rng = np.random.default_rng(0)
    pick = rng.choice(10000, 2500, replace=False)
    rows = np.concatenate([g[0].reshape(-1)[pick], g[1].reshape(-1)[pick]])
    part_pos = np.r_[np.arange(2500, 5000), np.arange(2500)]
    op, gt, lt = cm.op[rows], cm.a[rows] > cm.b[rows], cm.a[rows] < cm.b[rows]
    Ug = {li: jnp.asarray(np.asarray(cm.sites[li]["gate"].U)[:, chosen[li]]) for li in LAYERS}
    Uu = {li: jnp.asarray(np.asarray(cm.sites[li]["up"].U)[:, chosen[li]]) for li in LAYERS}
    Vd = {li: jnp.asarray(np.asarray(cm.sites[li]["down"].V)[chosen[li]]) for li in LAYERS}
    actB: dict[int, np.ndarray] = {
        li: np.zeros((len(rows), len(chosen[li])), np.float32) for li in LAYERS
    }
    live: dict[str, Any] = {}

    def run_one(delta: np.ndarray | None, mode: str) -> np.ndarray:
        def resid(tt: int, xx: Any, idx: np.ndarray) -> Any:
            if delta is not None and tt == t:
                return xx.at[:, EQ].add(jnp.asarray(delta[rows[idx]]))
            return xx

        def inner(li: int, kind: str, h: Any, _rms: Any, idx: np.ndarray) -> Any:
            if li not in chosen or not chosen[li] or mode == "plain":
                return h
            if kind in ("gate", "up"):
                live[kind] = h[:, EQ] @ (Ug[li] if kind == "gate" else Uu[li])
                return h
            if kind == "down":
                a = live["gate"] / (1 + jnp.exp(-live["gate"])) * live["up"]
                if mode == "capture":
                    actB[li][idx] = np.asarray(a)
                    return h
                return h.at[:, EQ].add((jnp.asarray(actB[li][idx]) - a) @ Vd[li])
            return h

        lp, _ = cm.forward(rows, resid=resid, inner=inner)
        return lp.argmax(-1)

    top0 = run_one(None, "plain")
    run_one(dB, "capture")
    top = run_one(dA, "patch")
    r: dict[str, Any] = {"cond": cond, "n neurons": sum(len(v) for v in chosen.values())}
    for o, nm in ((0, "add"), (1, "sub")):
        for tag, sel in (("", op == o), ("_gt", (op == o) & gt), ("_lt", (op == o) & lt)):
            r[f"swap_{nm}{tag}"] = float((top[sel] == top0[part_pos][sel]).mean())
            r[f"same_{nm}{tag}"] = float((top[sel] == top0[sel]).mean())
    print(
        cond,
        " ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
            for k, v in r.items()
            if k != "cond"
        ),
        flush=True,
    )
    (OUT / f"minus_{cond}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select()
    else:
        run(CONDS[int(sys.argv[2])])
