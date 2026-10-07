"""What the core flip components do besides the flip, in the alive-only model.

    python -m param_decomp.arith_repr.vectors.flip_core run   # GPU -> OUT/flip/core.json + stdout

Core: the L15 op switches gate c0, gate c72, up c34, up c117 and the flip writers down c16, down c4, at
`=`. F8 = the seven b-flip neurons and neuron 130 (the neuron down c4 reads).

Conditions (each against the base alive-only model):
* `core off`: the six removed at `=`; `core off + flip back`: the same, with the flip part of the change
  of the L15 MLP write (the b-odd, op-odd part of its (op, b) group means) added back as a fixed
  per-(op, b) vector. What this still breaks is the core's non-flip function.
* switch c `via F8` / `via rest`: c's write into the gate / up pre-activations removed only on the
  F8 neurons / only on the other 14,328, at `=`; also all four switches at once.
* writer c: one piece of its inner h_c (x . V_c) removed at a time, as the fixed per-(op, b) vector
  piece(op, b) U_c, from the group means G(op, b) = m(op) + C(op, b):
    `op const` +-(m_add - m_sub)/2;  `bias` (m_add + m_sub)/2;  `b same` (C_add + C_sub)/2 (b's code
    on both operations);  `flip` the b-odd part of (C_add - C_sub)/2;  `op-odd b-even` the rest of
    (C_add - C_sub)/2;  `within (op, b)` h - G(op, b) (what depends on a), removed by replacing h_c by
    G(op, b).

Scores: KL to the base, accuracy on add with a + b < 100 / >= 100 and on sub with a >= b; the flip
Phi relative to base at the L16, L18, L20, L24, L31 readers. For `core off` and the `via rest`
conditions, the change of the L15 MLP write at `=` in the L16 readers' view: energies of its op-even
constant, op constant, the codes of b, a, a + b, a - b and the flip, each relative to the same energy
of the base L15 MLP write."""

import json
from typing import Any

import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.components_model import kl
from param_decomp.arith_repr.vectors.flip_components import DIR
from param_decomp.arith_repr.vectors.flip_model import FLIP
from param_decomp.arith_repr.vectors.flip_set import PERM, READ, Net, flip_part, measure

L, EQ = 15, 4
SWITCHES = (("gate", "c0"), ("gate", "c72"), ("up", "c34"), ("up", "c117"))
WRITERS = (("down", "c16"), ("down", "c4"))
F8 = np.r_[FLIP, 130]
SHOW = (16, 18, 20, 24, 31)
GROUPS = ("b", "a", "a+b", "a-b")


def main() -> None:
    net = Net(dense=False, alive=True)
    cm = net.cm
    s0 = net.scales_one()
    rows = np.arange(20000)
    pm = jnp.asarray(np.eye(5, dtype=np.float32)[EQ])
    G16 = np.asarray(net.G[16])
    base_full = net.full(s0)
    base = base_full.pop("lp")
    phi0 = measure({f"r{li}": base_full[f"x{li}"][:200] @ np.asarray(net.G[li]) for li in READ})
    res: dict[str, Any] = {}

    def idx(kind: str, c: str) -> tuple[int, int]:
        names = cm.sites[L][kind].names
        j = names.index(f"L{L}.{kind}.{c}")
        return j, int(cm.sites[L][kind].cols[j])

    def scores(lp: np.ndarray, st: dict[int, tuple[float, float, float]]) -> dict[str, Any]:
        top = lp.argmax(-1)
        d = np.concatenate(
            [kl(base[r0 : r0 + 1000], lp[r0 : r0 + 1000]) for r0 in range(0, len(lp), 1000)]
        )
        add, sub = cm.op == 0, cm.op == 1
        lo = add & (cm.a + cm.b < 100)
        hi = add & (cm.a + cm.b >= 100)
        ok = sub & (cm.a >= cm.b)
        return {
            "kl_add": float(d[add].mean()), "kl_sub": float(d[sub].mean()),
            "acc_add_lo": float((top[lo] == cm.answer[lo]).mean()),
            "acc_add_hi": float((top[hi] == cm.answer[hi]).mean()),
            "acc_sub": float((top[ok] == cm.answer[ok]).mean()),
            "phi": {li: st[li][0] / phi0[li][0] for li in SHOW},
        }  # fmt: skip

    def describe(w: np.ndarray) -> dict[str, float]:
        """w (800, d): group sums of an L15 MLP write (or of its change) at `=` -> energies in the L16
        readers' view, each a mean over the 200 (op, value) groups of a squared norm: `bias` (the
        op-even constant), `op const` (+-(m_add - m_sub)/2), the centred code of each grouping, and
        the flip (b-odd, op-odd part of the b grouping)."""
        r = (w / 100.0) @ G16  # (800, n) group means in reader view
        y = r.reshape(4, 2, 100, -1)
        m = y[0].mean(1)  # (2, n) op means (the same for every grouping)
        out = {
            "bias": float((((m[0] + m[1]) / 2) ** 2).sum()),
            "op const": float((((m[0] - m[1]) / 2) ** 2).sum()),
        }
        for g, nm in enumerate(GROUPS):
            out[nm] = float(((y[g] - m[:, None]) ** 2).sum(-1).mean())
        out["flip"] = float((flip_part(w[:200] @ G16) ** 2).sum(-1).mean())
        return out

    def run(name: str, s: list[Any] | None = None, dU: dict[str, Any] | None = None,
            wclamp: np.ndarray | None = None, full: bool = False) -> dict[str, Any]:  # fmt: skip
        net.P["dU"] = [{} for _ in range(32)]
        if dU:
            net.P["dU"][L] = dU
        net.P["wclamp"] = {L: jnp.asarray(wclamp, jnp.float32)} if wclamp is not None else {}
        ss = s if s is not None else s0
        o = None
        if full:
            o = net.full(ss)
            lp = o.pop("lp")
            st = measure({f"r{li}": o[f"x{li}"][:200] @ np.asarray(net.G[li]) for li in READ})
        else:
            lp = net.logprobs(ss, rows)
            st = measure(net.light(ss))
        sc = scores(lp, st)
        if o is not None:
            dw = describe(o["w31"] - base_full["w31"])
            sc["dw31"] = {
                k: v / max(w31_base[k], 1e-12) for k, v in dw.items()
            }  # relative to the base write
        res[name] = sc
        print(f"{name}: KL {sc['kl_add']:.3f}/{sc['kl_sub']:.3f} acc add<100 {sc['acc_add_lo']:.3f} add>=100 "
              f"{sc['acc_add_hi']:.3f} sub {sc['acc_sub']:.3f} | Phi rel " + " ".join(f"L{li} {v:.2f}" for li, v in sc["phi"].items())
              + ("" if "dw31" not in sc else " | dw31 / base w31 " + " ".join(f"{k} {v:.2f}" for k, v in sc["dw31"].items())), flush=True)  # fmt: skip
        (DIR / "core.json").write_text(json.dumps(res, indent=1))
        net.P["dU"] = [{} for _ in range(32)]
        net.P["wclamp"] = {}
        return o or {}

    w31_base = describe(base_full["w31"])
    res["w31 base energies"] = w31_base
    print("base L15 MLP write at `=`, energies in the L16 readers' view:", {k: f"{v:.3g}" for k, v in w31_base.items()}, flush=True)  # fmt: skip
    run("base", full=False)
    # core off, then with the flip put back
    s_core = net.scales_one()
    for kind, c in SWITCHES + WRITERS:
        _, col = idx(kind, c)
        s_core[L] = s_core[L].at[col, EQ].set(0.0)
    o = run("core off", s_core, full=True)
    fp = flip_part(o["w31"][:200] - base_full["w31"][:200])  # (100, d): flip part of the change
    run("core off + flip back", s_core, wclamp=np.concatenate([fp, -fp]))
    # the switches through F8 / through the other neurons
    inF = np.zeros(14336, bool)
    inF[F8] = True
    for part, msk in (("via F8", inF), ("via rest", ~inF)):
        allsw: dict[str, Any] = {}
        for kind, c in SWITCHES:
            j, _ = idx(kind, c)
            Urow = np.asarray(cm.sites[L][kind].U)[j]
            dU = {kind: (jnp.asarray([j]), jnp.asarray(-(Urow * msk)[None], jnp.float32), pm)}
            run(f"{kind} {c} {part}", dU=dU, full=part == "via rest")
            if kind in allsw:
                jj, du, _ = allsw[kind]
                allsw[kind] = (
                    jnp.concatenate([jj, jnp.asarray([j])]),
                    jnp.concatenate([du, dU[kind][1]]),
                    pm,
                )
            else:
                allsw[kind] = dU[kind]
        run(f"all switches {part}", dU=allsw, full=part == "via rest")
    # the writers' pieces
    for kind, c in WRITERS:
        j, col = idx(kind, c)
        Uw = np.asarray(cm.sites[L][kind].U)[j].astype(np.float64)  # (4096,)
        Gm = base_full[f"h{L}.{kind}"][EQ, :200, j].reshape(2, 100) / 100.0  # (op, b) group means
        m = Gm.mean(1)
        C = Gm - m[:, None]
        odd = (C - C[:, PERM]) / 2
        opodd = (C[0] - C[1]) / 2
        flip = (odd[0] - odd[1]) / 2
        pieces = {
            "op const": np.repeat([(m[0] - m[1]) / 2, -(m[0] - m[1]) / 2], 100),
            "bias": np.full(200, (m[0] + m[1]) / 2),
            "b same": np.tile((C[0] + C[1]) / 2, 2),
            "flip": np.r_[flip, -flip],
            "op-odd b-even": np.r_[opodd - flip, -(opodd - flip)],
        }
        for nm, tab in pieces.items():
            run(f"{c} {nm}", wclamp=np.outer(tab, Uw))
        s_w = net.scales_one()
        s_w[L] = s_w[L].at[col, EQ].set(0.0)
        run(f"{c} within (op, b)", s_w, wclamp=-np.outer(Gm.reshape(-1), Uw))
        run(f"{c} whole", s_w)


if __name__ == "__main__":
    main()
