"""Per-prompt inner activations at `=` of every alive component, alive-only model (every alive component on
everywhere, no delta), for the account of where the operation goes after layer 15.

    python -m param_decomp.arith_repr.vectors.flip_opflow capture   # -> OUT/flip/opflow/h_eq.npz
    python -m param_decomp.arith_repr.vectors.flip_opflow account   # -> stdout + OUT/flip/opflow/account.json

At `=` every sublayer's write is h @ U of that position's inners, so the stream at `=` before any
sublayer is the `=` embedding plus the writes of the sublayers before it; `capture` checks this
against the captured stream entering layer 15's and layer 24's MLPs.

`account`: the prompts form the full grid op x a x b (2 x 100 x 100). Any per-prompt quantity X (a
scalar or a vector) splits orthogonally into the grand mean and seven effects (two-way and three-way
ANOVA terms): op, a, b, op x a, op x b, a x b, op x a x b; the energy of a term is the sum over the
20000 prompts of its squared norm. The op-odd part D(a, b) = (X(add, a, b) - X(sub, a, b)) / 2 carries
the four terms with op in them. Reported:
* every sublayer's write at `=` (attention o, MLP down), layers 0-31, and the stream entering every
  MLP: the energy of each term;
* the op x b and op x a terms' flip shares: with f(b) the b effect per op (mean over a, centred) and
  f(b)_odd = (f(b) - f(100 - b)) / 2 (b -> -b mod 100), share = |odd(op x b)|^2 / (|odd(op x b)|^2 +
  |odd(b)|^2), summed over b (1 = b reversed on sub, 0 = same on both); likewise for a;
* the share of D's variance (D minus its mean over (a, b)) explained by D's class means over a + b,
  over a - b, over a, over b;
* MLP neurons, layers 15-31: act = silu(gate) * up per prompt (gate, up rebuilt from the components'
  inners at `=`); each neuron's write is act * w_n, w_n = sum_c V_down[n, c] U_down[c]; its energy per
  term is the act's energy per term times |w_n|^2. Top neurons by op-odd write energy, with the share
  of the gate's and up's variance carried by the op term, the flip shares, the D groupings, and the
  cosine of w_n with the op flag (op term of the stream after the MLP). At layer 15 also neurons 9816,
  10519 and the eight flip neurons, and the write directions of the top ones against the quantity
  codes of the stream after the L15 MLP (`flip_dirs.profile`);
* components, layers 15-31: per site, the components with the largest op-odd energy of h_c |U_c|."""

import json
import sys
from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.isa.atlas import ATLAS
from param_decomp.arith_repr.isa.components_model import ComponentsModel, T
from param_decomp.arith_repr.vectors.flip_components import DIR

EQ = 4  # position of `=`
KINDS = ("q", "k", "v", "o", "gate", "up", "down")
OUT = DIR / "opflow"


def alive_model() -> ComponentsModel:
    cm = ComponentsModel()
    ncols = {
        li: int(np.load(ATLAS / "masks" / f"L{li}.npy", mmap_mode="r").shape[2]) for li in range(32)
    }
    cm.masks = lambda layer, rows: np.ones((len(rows), T, ncols[layer]), np.float32)  # type: ignore[method-assign]
    return cm


def stream_at(cm: ComponentsModel, H: dict[str, np.ndarray], t: int) -> np.ndarray:
    """Raw stream at `=` at point t (2l before block l's attention, 2l + 1 before its MLP), (N, d)."""
    x = np.asarray(cm.embed[int(cm.tokens[0, EQ])], np.float64)[None]
    for li in range(32):
        for off, kind in ((0, "o"), (1, "down")):
            if 2 * li + off >= t:
                return x
            x = x + H[f"{li}.{kind}"].astype(np.float64) @ np.asarray(
                cm.sites[li][kind].U, np.float64
            )
    return x


def capture() -> None:
    cm = alive_model()
    assert len(set(cm.tokens[:, EQ].tolist())) == 1, "`=` token differs across prompts"
    rows = np.arange(len(cm.op))
    H = {
        f"{li}.{k}": np.zeros((len(rows), cm.sites[li][k].V.shape[1]), np.float32)
        for li in range(32)
        for k in KINDS
    }

    def inner(li: int, kind: str, h: Any, _rms: Any, idx: np.ndarray) -> Any:
        H[f"{li}.{kind}"][idx] = np.asarray(h[:, EQ])
        return h

    _, caps = cm.forward(rows, inner=inner, capture={31, 49})
    for t, c in caps.items():
        rec = stream_at(cm, H, t)
        ref = c[:, EQ].astype(np.float64)
        err = np.linalg.norm(rec - ref, axis=1) / np.linalg.norm(ref, axis=1)
        print(
            f"stream at t={t}: relative reconstruction error median {np.median(err):.2e} max {err.max():.2e}",
            flush=True,
        )
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "h_eq.npz", **cast(dict[str, Any], H))
    print("saved", OUT / "h_eq.npz", flush=True)


TERMS = ("op", "a", "b", "op*a", "op*b", "a*b", "op*a*b")
ODD = ("op", "op*a", "op*b", "op*a*b")
PERM = (98 - np.arange(100)) % 100  # index of 100 - v for the index of v (v = index + 1)


def grid(cm: ComponentsModel) -> np.ndarray:
    """(2, 100, 100) row index of prompt (op, a, b)."""
    g = np.full((2, 100, 100), -1)
    g[cm.op, cm.a - 1, cm.b - 1] = np.arange(len(cm.op))
    assert (g >= 0).all() and len(np.unique(g)) == 20000, "prompts are not the full grid"
    return g


def _labels() -> dict[str, np.ndarray]:
    a, b = np.meshgrid(np.arange(1, 101), np.arange(1, 101), indexing="ij")
    return {"a+b": (a + b).ravel(), "a-b": (a - b).ravel(), "a": a.ravel(), "b": b.ravel()}


LABELS = _labels()


def _onehot(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inv = np.unique(lab, return_inverse=True)
    M = np.eye(inv.max() + 1)[inv]  # (10000, groups)
    return M, M.sum(0)


ONEHOT = {q: _onehot(lab) for q, lab in LABELS.items()}


def anova(X: np.ndarray, g: np.ndarray, chunk: int = 512) -> dict[str, np.ndarray]:
    """X (20000, n) per prompt -> per column: energies of TERMS, flip shares of b and a, and the shares
    of D's variance explained by its class means over a + b, a - b, a, b."""
    n = X.shape[1]
    out = {
        k: np.zeros(n) for k in (*TERMS, "total", "flip b", "flip a", *(f"D|{q}" for q in LABELS))
    }
    for c0 in range(0, n, chunk):
        Y = X[:, c0 : c0 + chunk][g].astype(np.float64)  # (2, 100, 100, m)
        m = Y.mean((0, 1, 2))
        mo = Y.mean((1, 2)) - m
        ma = Y.mean((0, 2)) - m
        mb = Y.mean((0, 1)) - m
        moa = Y.mean(2) - m - mo[:, None] - ma[None]
        mob = Y.mean(1) - m - mo[:, None] - mb[None]
        mab = Y.mean(0) - m - ma[:, None] - mb[None]
        tot = ((Y - m) ** 2).sum((0, 1, 2))
        e = {"op": 10000 * (mo**2).sum(0), "a": 200 * (ma**2).sum(0), "b": 200 * (mb**2).sum(0),
             "op*a": 100 * (moa**2).sum((0, 1)), "op*b": 100 * (mob**2).sum((0, 1)),
             "a*b": 2 * (mab**2).sum((0, 1))}  # fmt: skip
        e["op*a*b"] = tot - sum(e.values())
        e["total"] = tot
        for q, mq, mi in (("b", mb, mob[0]), ("a", ma, moa[0])):
            fo, so = (mi - mi[PERM]) / 2, (mq - mq[PERM]) / 2
            fo2, so2 = (fo**2).sum(0), (so**2).sum(0)
            e[f"flip {q}"] = fo2 / np.maximum(fo2 + so2, 1e-30)
        D0 = ((Y[0] - Y[1]) / 2).reshape(10000, -1)
        D = D0 - D0.mean(0)
        dt = np.maximum((D**2).sum(0), 1e-30)
        for q, (M, cnt) in ONEHOT.items():
            S = M.T @ D  # (groups, m) sums of D per class
            e[f"D|{q}"] = ((S**2) / cnt[:, None]).sum(0) / dt  # sum over classes of count * mean^2
        for k, v in e.items():
            out[k][c0 : c0 + chunk] = v
    return out


def summed(r: dict[str, np.ndarray]) -> dict[str, float]:
    """Energies of a vector quantity (sum over its columns); flip shares and D shares re-weighted."""
    o = {k: float(r[k].sum()) for k in (*TERMS, "total")}
    return o


def odd(r: dict[str, Any]) -> Any:
    return sum(r[k] for k in ODD)


def group_sums(x: np.ndarray, cm: ComponentsModel) -> np.ndarray:
    """(N, d) per prompt -> (800, d) sums by (op, value) of b, a, (a + b) mod 100, (a - b) mod 100, in
    the row layout of `flip_dirs.codes`."""
    out = np.zeros((800, x.shape[1]))
    for gi, v in enumerate((cm.b - 1, cm.a - 1, (cm.a + cm.b) % 100, (cm.a - cm.b) % 100)):
        np.add.at(out, gi * 200 + cm.op * 100 + v, x)
    return out


def account() -> None:
    from param_decomp.arith_repr.vectors.flip_dirs import codes, fmt, profile, span_basis

    cm = ComponentsModel()
    g = grid(cm)
    # self-test on synthetic quantities of the same grid: b reversed on sub, an op constant, an a effect
    sgn = np.where(cm.op == 0, 1.0, -1.0)
    sb = np.sin(2 * np.pi * 3 * cm.b / 100)
    t = anova(np.stack([sgn * sb, sgn, np.cos(2 * np.pi * cm.a / 100), sb], 1), g)
    assert abs(t["op*b"][0] / t["total"][0] - 1) < 1e-9 and abs(t["flip b"][0] - 1) < 1e-9, t
    assert abs(t["D|b"][0] - 1) < 1e-9 and abs(t["op"][1] / t["total"][1] - 1) < 1e-9, t
    assert abs(t["a"][2] / t["total"][2] - 1) < 1e-9 and t["flip b"][3] < 1e-9, t
    print("anova self-test passed", flush=True)
    H = dict(np.load(OUT / "h_eq.npz"))
    res: dict[str, Any] = {"writes": {}, "stream": {}, "neurons": {}, "components": {}}

    def fmt_e(e: dict[str, float]) -> str:
        return (
            " ".join(f"{k} {e[k]:9.1f}" for k in TERMS)
            + f" | op-odd {sum(e[k] for k in ODD):9.1f} of {e['total']:9.1f}"
        )

    print("== energies per term at `=` (sum over prompts of squared norms)", flush=True)
    x = np.asarray(cm.embed[int(cm.tokens[0, EQ])], np.float64)[None].repeat(20000, 0)
    for li in range(32):
        for off, kind in ((0, "o"), (1, "down")):
            if off == 1:
                rs = summed(anova(x, g))
                res["stream"][li] = rs
                print(f"stream into L{li:2d} MLP  {fmt_e(rs)}", flush=True)
            w = H[f"{li}.{kind}"].astype(np.float64) @ np.asarray(cm.sites[li][kind].U, np.float64)
            r = anova(w, g)
            rw = summed(r)
            res["writes"][f"{li}.{kind}"] = rw
            print(f"write L{li:2d} {'attn' if off == 0 else 'mlp '}     {fmt_e(rw)}", flush=True)
            x = x + w
    x_after15 = None

    for li in range(15, 32):
        Ug = np.asarray(cm.sites[li]["gate"].U, np.float64)
        Uu = np.asarray(cm.sites[li]["up"].U, np.float64)
        gp = H[f"{li}.gate"].astype(np.float64) @ Ug
        up = H[f"{li}.up"].astype(np.float64) @ Uu
        act = gp / (1 + np.exp(-gp)) * up
        Vd = np.asarray(cm.sites[li]["down"].V, np.float64)
        Ud = np.asarray(cm.sites[li]["down"].U, np.float64)
        err = np.abs(act @ Vd - H[f"{li}.down"]).max() / np.abs(H[f"{li}.down"]).max()
        assert err < 1e-3, f"L{li}: act @ V_down != captured down inner ({err:.1e})"
        wn2 = ((Vd @ Ud) ** 2).sum(1)  # |w_n|^2
        ra, rg, ru = anova(act, g), anova(gp, g), anova(up, g)
        # op flag of the stream after this MLP
        xs = stream_at(cm, H, 2 * li + 2)
        flag = xs[cm.op == 0].mean(0) - xs[cm.op == 1].mean(0)
        if li == 15:
            x_after15 = xs
        W = Vd @ Ud
        cosf = W @ flag / np.maximum(np.linalg.norm(W, axis=1), 1e-30) / np.linalg.norm(flag)
        ow = odd(ra) * wn2
        top = list(np.argsort(-ow)[:12])
        extra = [9816, 10519, 12769, 6456, 9205, 9057, 13193, 7446, 11305, 130] if li == 15 else []
        rows = []
        print(f"\n== L{li} MLP neurons by op-odd write energy (layer MLP op-odd write {odd(res['writes'][f'{li}.down']):.1f};"
              f" sum over all neurons {ow.sum():.1f}, top 12 {ow[top].sum():.1f})", flush=True)  # fmt: skip
        for n in top + [e for e in extra if e not in top]:
            ta = {k: float(ra[k][n] * wn2[n]) for k in (*TERMS, "total")}
            row: dict[str, Any] = {"neuron": int(n), "write": ta, "op share gate": float(rg["op"][n] / max(rg["total"][n], 1e-30)),
                   "op share up": float(ru["op"][n] / max(ru["total"][n], 1e-30)),
                   "mean act add/sub": [float(act[cm.op == o, n].mean()) for o in (0, 1)],
                   "mean gate add/sub": [float(gp[cm.op == o, n].mean()) for o in (0, 1)],
                   "mean up add/sub": [float(up[cm.op == o, n].mean()) for o in (0, 1)],
                   "flip b": float(ra["flip b"][n]), "flip a": float(ra["flip a"][n]),
                   "D": {q: float(ra[f"D|{q}"][n]) for q in LABELS}, "cos flag": float(cosf[n])}  # fmt: skip
            rows.append(row)
            tt = ta["total"]
            print(f"  n{n:5d} op-odd {sum(ta[k] for k in ODD):7.1f} ({sum(ta[k] for k in ODD) / tt:.2f} of its {tt:7.1f}) | "
                  + " ".join(f"{k} {ta[k] / tt:.2f}" for k in TERMS)
                  + f" | op share gate {row['op share gate']:.2f} up {row['op share up']:.2f}"
                  + f" | act {row['mean act add/sub'][0]:+.2f}/{row['mean act add/sub'][1]:+.2f}"
                  + f" gate {row['mean gate add/sub'][0]:+.2f}/{row['mean gate add/sub'][1]:+.2f}"
                  + f" up {row['mean up add/sub'][0]:+.2f}/{row['mean up add/sub'][1]:+.2f}"
                  + f" | flip b {row['flip b']:.2f} a {row['flip a']:.2f} | D|a+b {row['D']['a+b']:.2f} a-b {row['D']['a-b']:.2f}"
                  + f" a {row['D']['a']:.2f} b {row['D']['b']:.2f} | cos(w, op flag) {row['cos flag']:+.2f}", flush=True)  # fmt: skip
        res["neurons"][li] = rows
        if li == 15 and x_after15 is not None:
            d_out, F_out = codes(group_sums(x_after15, cm))
            Q_out = span_basis(d_out, F_out)
            print(
                "  write directions w_n vs the codes of the stream after the L15 MLP:", flush=True
            )
            for n in [9816, 10519, *top[:6]]:
                print(f"   n{n}: {fmt(profile(W[n], d_out, F_out, Q_out))}", flush=True)
        del act, gp, up
        comps_out = {}
        for kind in ("q", "k", "v", "o", "gate", "up", "down"):
            hc = H[f"{li}.{kind}"].astype(np.float64)
            rc = anova(hc, g)
            u2 = (np.asarray(cm.sites[li][kind].U, np.float64) ** 2).sum(1)
            oc = odd(rc) * u2
            names = cm.sites[li][kind].names
            tops = np.argsort(-oc)[:5]
            comps_out[kind] = [{"name": names[j], "op-odd": float(oc[j]), "op share": float(rc["op"][j] / max(rc["total"][j], 1e-30)),
                                "terms": {k: float(rc[k][j] / max(rc["total"][j], 1e-30)) for k in TERMS},
                                "mean add/sub": [float(hc[cm.op == o, j].mean()) for o in (0, 1)]} for j in tops]  # fmt: skip
            print(f"  {kind:4s} comps by op-odd energy of h_c U_c: " + "; ".join(
                f"{c['name'].split('.')[-1]} {c['op-odd']:.1f} (op {c['terms']['op']:.2f} op*a {c['terms']['op*a']:.2f} op*b {c['terms']['op*b']:.2f}"
                f" op*a*b {c['terms']['op*a*b']:.2f}; h {c['mean add/sub'][0]:+.1f}/{c['mean add/sub'][1]:+.1f})" for c in comps_out[kind]), flush=True)  # fmt: skip
        res["components"][li] = comps_out
    (OUT / "account.json").write_text(json.dumps(res, indent=1, default=float))
    print("saved", OUT / "account.json", flush=True)


if __name__ == "__main__":
    cast(Any, {"capture": capture, "account": account})[sys.argv[1]]()
