"""Which op-dependent part of the stream at `=` decides the answer, layer by layer (alive-only model).

    python -m param_decomp.arith_repr.vectors.flip_terms <layer>   # CPU -> OUT/flip/opflow/terms_L<layer>.json
    (needs OUT/flip/opflow/h_eq.npz from `flip_opflow capture`)

The stream x at `=` entering block <layer> (before its attention) is split over the full prompt grid
op x a x b into the grand mean and the ANOVA terms of `flip_opflow`. The four op-odd terms (op, op x a,
op x b, op x a x b) change sign between a prompt and its partner (same a, b, other op), so subtracting
twice a set of them hands each prompt its partner's value of exactly those terms; subtracting all four
gives the partner's stream. Conditions: each op-odd term alone, some pairs, all four. Scored on 2500
random (a, b) pairs (both ops), against the UNEDITED model's top-1: `swap` = top-1 equals the
unedited top-1 on the partner prompt, `same` = equals its own unedited top-1, per op, also on a > b."""

import json
import sys
from typing import Any

import jax
import numpy as np

from param_decomp.arith_repr.vectors.flip_opflow import EQ, OUT, alive_model, grid, stream_at

TERMS_ODD = ("op", "op*a", "op*b", "op*a*b")
CONDS = {
    "op": ("op",),
    "op*a": ("op*a",),
    "op*b": ("op*b",),
    "op*a*b": ("op*a*b",),
    "op+op*b": ("op", "op*b"),
    "op*b+op*a*b": ("op*b", "op*a*b"),
    "op+op*a+op*b": ("op", "op*a", "op*b"),
    "all": TERMS_ODD,
}


def odd_terms(x: np.ndarray, g: np.ndarray) -> dict[str, np.ndarray]:
    """x (N, d) per prompt -> each op-odd ANOVA term as a per-prompt (N, d) array."""
    Y = x[g]  # (2, 100, 100, d)
    m = Y.mean((0, 1, 2))
    mo = Y.mean((1, 2)) - m
    ma = Y.mean((0, 2)) - m
    mb = Y.mean((0, 1)) - m
    moa = Y.mean(2) - m - mo[:, None] - ma[None]
    mob = Y.mean(1) - m - mo[:, None] - mb[None]
    mab = Y.mean(0) - m - ma[:, None] - mb[None]
    rest = Y - m - mo[:, None, None] - ma[None, :, None] - mb[None, None] - moa[:, :, None]
    rest = rest - mob[:, None] - mab[None]
    assert np.abs(rest[0] + rest[1]).max() < 1e-6 * np.abs(rest).max() + 1e-9
    full = {"op": np.broadcast_to(mo[:, None, None], Y.shape), "op*a": np.broadcast_to(moa[:, :, None], Y.shape),
            "op*b": np.broadcast_to(mob[:, None], Y.shape), "op*a*b": rest}  # fmt: skip
    out = {}
    for k, v in full.items():
        t = np.empty_like(x)
        t[g.reshape(-1)] = v.reshape(-1, x.shape[1])
        out[k] = t
    return out


def main(layer: int) -> None:
    cm = alive_model()
    g = grid(cm)
    H = dict(np.load(OUT / "h_eq.npz"))
    t = 2 * layer
    x = stream_at(cm, H, t)
    del H
    T = odd_terms(x, g)
    partner = np.empty(len(x), int)
    partner[g[0].ravel()], partner[g[1].ravel()] = g[1].ravel(), g[0].ravel()
    err = np.abs(x - 2 * np.sum(list(T.values()), axis=0) - x[partner]).max() / np.abs(x).max()
    assert err < 1e-6, f"all-terms swap != partner stream ({err:.1e})"
    rng = np.random.default_rng(0)
    pick = rng.choice(10000, 2500, replace=False)
    rows = np.concatenate([g[0].reshape(-1)[pick], g[1].reshape(-1)[pick]])
    part_pos = np.r_[
        np.arange(2500, 5000), np.arange(2500)
    ]  # position of each row's partner in rows
    op, gt = cm.op[rows], cm.a[rows] > cm.b[rows]

    def run(delta: np.ndarray | None) -> np.ndarray:
        def resid(tt: int, xx: Any, idx: np.ndarray) -> Any:
            if delta is None or tt != t:
                return xx
            return xx.at[:, EQ].add(jax.numpy.asarray(delta[rows[idx]], xx.dtype))

        lp, _ = cm.forward(rows, resid=resid)
        return lp.argmax(-1)

    top0 = run(None)
    res: dict[str, Any] = {"layer": layer, "t": t}
    for name, keys in CONDS.items():
        delta = (-2 * np.sum([T[k] for k in keys], axis=0)).astype(np.float32)
        top = run(delta)
        r = {}
        for o, nm in ((0, "add"), (1, "sub")):
            for tag, sel in (("", op == o), ("_gt", (op == o) & gt)):
                r[f"swap_{nm}{tag}"] = float((top[sel] == top0[part_pos][sel]).mean())
                r[f"same_{nm}{tag}"] = float((top[sel] == top0[sel]).mean())
        r["energy"] = float(sum((T[k][rows] ** 2).sum() for k in keys))
        res[name] = r
        print(
            f"L{layer} {name:14s} "
            + " ".join(f"{k} {v:.3f}" for k, v in r.items() if k != "energy"),
            flush=True,
        )
    (OUT / f"terms_L{layer}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
