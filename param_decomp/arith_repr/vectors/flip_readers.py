"""Which components read the op constant that, together with the b flip, decides the operation (alive-only).

    python -m param_decomp.arith_repr.vectors.flip_readers <cond index>
    # CPU -> OUT/flip/opflow/readers_<cond>.json (needs OUT/flip/opflow/h_eq.npz from `flip_opflow capture`)

`flip_terms` found that at the input of block 16, swapping the op term (the op constant) and the op x b
term (the flip) of the stream at `=` swaps most answers, and neither does alone. Two edited runs:
* A: the flip swapped (x -> x - 2 (op x b term), at `=` entering block 16);
* B: the flip and the op constant swapped there (x -> x - 2 (op term + op x b term)).
Each condition runs A, but the chosen components at `=` get their inner activations from B (activation
patching from B into A): they read the partner's op constant, with B's whole upstream state, while
everything else sees A. `B` itself and `A` itself are conditions too. Scored on 2500 random (a, b) pairs
(both ops), against the UNEDITED model's top-1: `swap` = top-1 equals the unedited top-1 on the
partner prompt (same a, b, other op), `same` = equals its own unedited top-1, per op, also on a > b."""

import json
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.vectors.flip_opflow import EQ, OUT, alive_model, grid, stream_at
from param_decomp.arith_repr.vectors.flip_terms import odd_terms

READ = ("q", "k", "v", "gate", "up")
GU = ("gate", "up")


def _sites(layers: range | tuple[int, ...], kinds: tuple[str, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((li, k) for li in layers for k in kinds)


CONDS: dict[str, tuple[tuple[int, str], ...] | str] = {
    "A": (),
    "B": "B",
    "gateup16-18": _sites(range(16, 19), GU),
    "gateup19-24": _sites(range(19, 25), GU),
    "gateup25-31": _sites(range(25, 32), GU),
    "gateup16-31": _sites(range(16, 32), GU),
    "qkv16-31": _sites(range(16, 32), ("q", "k", "v")),
    "readers16-31": _sites(range(16, 32), READ),
    "readers16-24": _sites(range(16, 25), READ),
    "readers25-31": _sites(range(25, 32), READ),
    "gateup16": _sites((16,), GU),
    "gateup16-17": _sites(range(16, 18), GU),
    "switches16-31": "switches",  # gate / up components of layers 16-31 whose op term is >= 0.5 of h's variance
    "nonswitch16-31": "nonswitch",  # the other gate / up components of layers 16-31
}
SWITCH_SHARE = 0.5


def switch_cols(cm: Any, H: dict[str, np.ndarray]) -> dict[tuple[int, str], np.ndarray]:
    """(layer, kind) -> boolean mask of the switch components (op term share of h's variance >= 0.5)."""
    out = {}
    for li in range(16, 32):
        for k in GU:
            h = H[f"{li}.{k}"].astype(np.float64)
            mo = np.stack([h[cm.op == o].mean(0) for o in (0, 1)]) - h.mean(0)
            out[(li, k)] = (mo**2).mean(0) / np.maximum(h.var(0), 1e-30) >= SWITCH_SHARE
    return out


FLIP_AT = 16


def main(cond: str) -> None:
    sites = CONDS[cond]
    cm = alive_model()
    g = grid(cm)
    H = dict(np.load(OUT / "h_eq.npz"))
    cols: dict[tuple[int, str], np.ndarray] = {}
    if sites in ("switches", "nonswitch"):
        sw = switch_cols(cm, H)
        cols = {key: (m if sites == "switches" else ~m) for key, m in sw.items()}
        print(
            "switch components per layer:",
            {f"{li}.{k}": int(m.sum()) for (li, k), m in sw.items()},
            flush=True,
        )
        sites = tuple(cols)
    t = 2 * FLIP_AT
    T = odd_terms(stream_at(cm, H, t), g)
    del H
    dA = (-2 * T["op*b"]).astype(np.float32)
    dB = (-2 * (T["op*b"] + T["op"])).astype(np.float32)
    del T
    rng = np.random.default_rng(0)
    pick = rng.choice(10000, 2500, replace=False)
    rows = np.concatenate([g[0].reshape(-1)[pick], g[1].reshape(-1)[pick]])
    part_pos = np.r_[np.arange(2500, 5000), np.arange(2500)]
    op, gt = cm.op[rows], cm.a[rows] > cm.b[rows]
    want = set() if isinstance(sites, str) else set(sites)
    cap: dict[tuple[int, str], np.ndarray] = {}

    def run(delta: np.ndarray | None, mode: str) -> np.ndarray:
        """mode: 'plain', 'capture' (store the inners of `want`), 'patch' (overwrite them from `cap`)."""

        def resid(tt: int, xx: Any, idx: np.ndarray) -> Any:
            if delta is not None and tt == t:
                return xx.at[:, EQ].add(jnp.asarray(delta[rows[idx]]))
            return xx

        def inner(li: int, kind: str, h: Any, _rms: Any, idx: np.ndarray) -> Any:
            if (li, kind) not in want:
                return h
            if mode == "capture":
                buf = cap.setdefault((li, kind), np.zeros((len(rows), h.shape[-1]), np.float32))
                buf[idx] = np.asarray(h[:, EQ])
                return h
            if mode == "patch":
                new = jnp.asarray(cap[(li, kind)][idx])
                if (li, kind) in cols:  # patch only the chosen columns of this site
                    new = jnp.where(jnp.asarray(cols[(li, kind)])[None], new, h[:, EQ])
                return h.at[:, EQ].set(new)
            return h

        lp, _ = cm.forward(rows, resid=resid, inner=inner)
        return lp.argmax(-1)

    top0 = run(None, "plain")
    if sites == "B":
        top = run(dB, "plain")
    elif not sites:
        top = run(dA, "plain")
    else:
        run(dB, "capture")
        top = run(dA, "patch")
    r: dict[str, Any] = {"cond": cond}
    for o, nm in ((0, "add"), (1, "sub")):
        for tag, sel in (("", op == o), ("_gt", (op == o) & gt)):
            r[f"swap_{nm}{tag}"] = float((top[sel] == top0[part_pos][sel]).mean())
            r[f"same_{nm}{tag}"] = float((top[sel] == top0[sel]).mean())
    print(cond, " ".join(f"{k} {v:.3f}" for k, v in r.items() if k != "cond"), flush=True)
    (OUT / f"readers_{cond}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(list(CONDS)[int(sys.argv[1])])
