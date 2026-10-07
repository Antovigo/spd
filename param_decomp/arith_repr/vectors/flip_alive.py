"""The b flip in the alive-only model: every alive component on for every prompt and position (no
masks), no delta, no dead components.

    python -m param_decomp.arith_repr.vectors.flip_alive baseline   # GPU -> OUT/flip/alive_baseline.json
    python -m param_decomp.arith_repr.vectors.flip_set search alive # GPU -> OUT/flip/set_alive.json
    python -m param_decomp.arith_repr.vectors.flip_alive evaluate   # GPU -> OUT/flip/eval_alive*.{json,npz}
    python -m param_decomp.arith_repr.vectors.flip_alive describe   # CPU, prints per-component profiles

Flip measures (Phi, Sig, Ev per reader layer) as in `flip_set`.

baseline: the alive-only model against the original (dense) model and the masked components model:
last-position KL(original || model), top-1 agreement with the original, accuracy (sub: a >= b), and
the flip measures of each model.

evaluate (the set found by `flip_set search alive`):
* `base`, `set` (every pair of the set removed);
* `clamp16`: the base model's flip part F_16 (the b-odd, op-odd part of the stream's (op, b) group
  means at `=`, before L16's MLP) is subtracted from every prompt's stream there: +F_16(b) on add,
  -F_16(b) on sub; `clamp_all`: the same at L16 and, at every later MLP input l, the flip part newly
  present in the base model, F_l - F_(l-1). The oracle removes the flip and nothing else, so its
  cost is what losing the flip costs; the set's cost beyond it is what the set does besides;
* every pair of the set removed alone: change of the flip, KL, accuracy.

describe: each pair's inner activation in the base alive-only model, at its position: shares of its
variance in the group means by a, b, (a + b) mod 100, (a - b) mod 100 (per op, averaged), the flip
share (b-odd, op-odd), the op share ((mean_add - mean_sub)^2 / 4 over the variance plus that), and
the strongest harmonic of each grouping (period, value where it peaks)."""

import json
import sys
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.components_model import kl
from param_decomp.arith_repr.vectors.flip_components import DIR
from param_decomp.arith_repr.vectors.flip_set import READ, Net, flip_part, measure

POS = ("BOS", "a", "op", "b", "=")
GROUPS = ("b", "a", "a+b", "a-b")  # order of the 800 group columns of the full captures


def metrics(base: np.ndarray, lp: np.ndarray, net: Net) -> dict[str, float]:
    cm = net.cm
    top = lp.argmax(-1)
    d = kl(base, lp)
    out = {}
    for o, nm in ((0, "add"), (1, "sub")):
        sel = (cm.op == o) & ((o == 0) | (cm.a >= cm.b))
        out[f"kl_{nm}"] = float(d[cm.op == o].mean())
        out[f"acc_{nm}"] = float((top[sel] == cm.answer[sel]).mean())
    return out


def baseline() -> None:
    nets = {
        "alive": Net(dense=False, alive=True),
        "comp": Net(dense=False),
        "dense": Net(dense=True),
    }
    s = {k: v.scales_one() for k, v in nets.items()}
    cm = nets["dense"].cm
    res: dict[str, Any] = {}
    agg = {k: {"kl": [], "agree": [], "top": []} for k in ("alive", "comp")}
    tops_d = []
    for r0 in range(0, 20000, 2000):
        rows = np.arange(r0, r0 + 2000)
        lpd = nets["dense"].logprobs(s["dense"], rows)
        tops_d.append(lpd.argmax(-1))
        for k in ("alive", "comp"):
            lp = nets[k].logprobs(s[k], rows)
            agg[k]["kl"].append(kl(lpd, lp))
            agg[k]["agree"].append(lp.argmax(-1) == lpd.argmax(-1))
            agg[k]["top"].append(lp.argmax(-1))
    td = np.concatenate(tops_d)
    for k in ("alive", "comp", "dense"):
        top = td if k == "dense" else np.concatenate(agg[k]["top"])
        r: dict[str, Any] = {}
        for o, nm in ((0, "add"), (1, "sub")):
            sel = (cm.op == o) & ((o == 0) | (cm.a >= cm.b))
            r[f"acc_{nm}"] = float((top[sel] == cm.answer[sel]).mean())
            if k != "dense":
                r[f"kl_{nm}"] = float(np.concatenate(agg[k]["kl"])[cm.op == o].mean())
                r[f"agree_{nm}"] = float(np.concatenate(agg[k]["agree"])[cm.op == o].mean())
        st = measure(nets[k].light(s[k]))
        r["flip"] = {str(li): v for li, v in st.items()}
        res[k] = r
        print(k, {a: round(b, 4) for a, b in r.items() if a != "flip"}, flush=True)
        print("   Phi/Sig/Ev, flip share Phi/(Phi+Sig):", " ".join(f"L{li} {v[0]:.3g}/{v[1]:.3g}/{v[2]:.3g} ({v[0] / (v[0] + v[1]):.2f})" for li, v in st.items()), flush=True)  # fmt: skip
    (DIR / "alive_baseline.json").write_text(json.dumps(res, indent=1))


def evaluate() -> None:
    S = json.loads((DIR / "set_alive.json").read_text())["removed"]
    net = Net(dense=False, alive=True)
    s0 = net.scales_one()
    res: dict[str, Any] = {}
    out = net.full(s0)
    base = out.pop("lp")
    np.savez(DIR / "eval_alive_base.npz", **cast(dict[str, Any], out))
    st0 = measure({f"r{li}": out[f"x{li}"][:200] @ np.asarray(net.G[li]) for li in READ})
    res["base"] = {**metrics(base, base, net), "flip": {str(k): v for k, v in st0.items()}}

    def record(name: str, lp: np.ndarray, st: dict[int, tuple[float, float, float]]) -> None:
        rel = {li: st[li][0] / st0[li][0] for li in READ}
        res[name] = {**metrics(base, lp, net), "flip": {str(k): v for k, v in st.items()}}
        print(name, {k: round(v, 4) for k, v in res[name].items() if k != "flip"},
              "Phi rel:", " ".join(f"L{li} {rel[li]:.2f}" for li in READ), flush=True)  # fmt: skip
        (DIR / "eval_alive.json").write_text(json.dumps(res, indent=1))

    # the whole set
    s = net.scales_one()
    for p in S:
        s[p["layer"]] = s[p["layer"]].at[p["col"], p["pos"]].set(0.0)
    o_set = net.full(s)
    lp = o_set.pop("lp")
    np.savez(DIR / "eval_alive_set.npz", **cast(dict[str, Any], o_set))
    record(
        "set", lp, measure({f"r{li}": o_set[f"x{li}"][:200] @ np.asarray(net.G[li]) for li in READ})
    )

    # oracle flip removal
    F = {li: flip_part(out[f"x{li}"][:200].astype(np.float64)) for li in READ}  # (100, d)
    table = {li: np.concatenate([F[li], -F[li]]).astype(np.float32) for li in READ}
    for name, clamp in (
        ("clamp16", {16: table[16]}),
        ("clamp_all", {16: table[16]} | {li: table[li] - table[li - 1] for li in READ if li > 16}),
    ):
        net.P["clamp"] = {li: jnp.asarray(v) for li, v in clamp.items()}
        oc = net.full(s0)
        lp = oc.pop("lp")
        record(
            name, lp, measure({f"r{li}": oc[f"x{li}"][:200] @ np.asarray(net.G[li]) for li in READ})
        )
    net.P["clamp"] = {}

    # each pair alone
    rows = np.arange(20000)
    for p in S:
        s = net.scales_one()
        s[p["layer"]] = s[p["layer"]].at[p["col"], p["pos"]].set(0.0)
        lp = net.logprobs(s, rows)
        record(f"{p['name']}@{POS[p['pos']]}", lp, measure(net.light(s)))


def describe() -> None:
    S = json.loads((DIR / "set_alive.json").read_text())["removed"]
    z = np.load(DIR / "eval_alive_base.npz")
    ev = (
        json.loads((DIR / "eval_alive.json").read_text())
        if (DIR / "eval_alive.json").exists()
        else {}
    )
    net_names: dict[tuple[int, str], list[str]] = {}
    from param_decomp.arith_repr.isa.components_model import ComponentsModel

    cm = ComponentsModel()
    for li in range(32):
        for kind, site in cm.sites[li].items():
            net_names[(li, kind)] = site.names
    base_phi = {li: ev["base"]["flip"][str(li)][0] for li in READ} if ev else {}
    print("per pair (alive-only model, base): variance shares of its inner at its position — a / b / a+b / a-b"
          " group means (avg over ops), flip (b-odd op-odd), op; strongest harmonic per grouping;"
          " removal alone: mean Phi change L16-18, KL add/sub")  # fmt: skip
    for p in sorted(S, key=lambda q: (q["layer"], q["pos"], q["name"])):
        li, kind = p["layer"], p["name"].split(".")[1]
        j = net_names[(li, kind)].index(p["name"])
        t = p["pos"]
        H = z[f"h{li}.{kind}"][t, :, j] / 100.0  # (800,) group means
        Q = z[f"q{li}.{kind}"][:, t, j] / 10000.0  # (2,) mean square per op
        mean_op = np.array([H[o * 100 : (o + 1) * 100].mean() for o in (0, 1)])
        var_op = Q - mean_op**2
        var = float(var_op.mean())
        shares, harm = {}, {}
        for g, nm in enumerate(GROUPS):
            y = H[g * 200 : (g + 1) * 200].reshape(2, 100)
            yc = y - y.mean(1, keepdims=True)
            shares[nm] = float((yc**2).mean(1).mean() / max(var, 1e-12))
            o_big = int(np.argmax((yc**2).sum(1)))  # the op where this grouping varies most
            f = np.fft.fft(yc[o_big])[1:51] / 100
            k = int(np.argmax(np.abs(f))) + 1
            if nm in ("b", "a"):  # row i holds value i + 1: refer the phase to the value
                f = f * np.exp(-2j * np.pi * np.arange(1, 51) / 100)
            per = 100 // np.gcd(k, 100)
            peak = float(np.mod(-np.angle(f[k - 1]) * 100 / (2 * np.pi * k), 100 / k))
            harm[nm] = f"T{per}@{peak:.1f}"
        flip = float(
            (flip_part(100.0 * H[:200].astype(np.float64)) ** 2).mean() / max(var, 1e-12)
        )  # group sums in
        dm = (mean_op[0] - mean_op[1]) / 2
        op_share = float(dm**2 / (var + dm**2))
        key = f"{p['name']}@{POS[t]}"
        rm = ""
        if key in ev:
            d = np.mean([ev[key]["flip"][str(li2)][0] / base_phi[li2] - 1 for li2 in (16, 17, 18)])
            rm = f" | alone: dPhi {d:+.2f}, KL {ev[key]['kl_add']:.3f}/{ev[key]['kl_sub']:.3f}"
        print(f"  {key:22s} " + " ".join(f"{nm} {shares[nm]:.2f}" for nm in GROUPS)
              + f" flip {flip:.2f} op {op_share:.2f} | " + " ".join(f"{nm}:{harm[nm]}" for nm in GROUPS) + rm)  # fmt: skip


def flipswap() -> None:
    """Is the flip the only difference between the operations from L16 on? At the stream entering
    L16's MLP at `=`, the base model's flip part F_16 (b-odd, op-odd part of the (op, b) group means)
    is removed (`clamp16`: -F on add, +F on sub) or reversed (`swap16`: -2F on add, +2F on sub, so each
    operation gets the other's flip). Scores per operation, against the UNEDITED model's top-1:
    `swap` = top-1 on (o, a, b) equals the unedited top-1 on (1 - o, a, b) (row i <-> i +- 10000);
    `same` = equals the unedited top-1 on the prompt itself; both also on the pairs with a > b."""
    S = np.load(DIR / "eval_alive_base.npz")
    F = flip_part(S["x16"][:200].astype(np.float64))  # (100, d), the add-side sign
    net = Net(dense=False, alive=True)
    cm = net.cm
    s0 = net.scales_one()
    rows = np.arange(20000)
    partner = np.r_[np.arange(10000, 20000), np.arange(10000)]
    gt = cm.a > cm.b
    res: dict[str, Any] = {}
    top0 = None
    for name, scale in (("base", 0.0), ("clamp16", 1.0), ("swap16", 2.0)):
        tab = scale * np.concatenate([F, -F]).astype(np.float32)
        net.P["clamp"] = {16: jnp.asarray(tab)} if scale else {}
        top = net.logprobs(s0, rows).argmax(-1)
        if top0 is None:
            top0 = top
        r = {}
        for o, nm in ((0, "add"), (1, "sub")):
            for tag, sel in (("", cm.op == o), ("_gt", (cm.op == o) & gt)):
                r[f"swap_{nm}{tag}"] = float((top[sel] == top0[partner][sel]).mean())
                r[f"same_{nm}{tag}"] = float((top[sel] == top0[sel]).mean())
        res[name] = r
        print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
    net.P["clamp"] = {}
    (DIR / "flipswap.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    cast(
        Any,
        {"baseline": baseline, "evaluate": evaluate, "describe": describe, "flipswap": flipswap},
    )[sys.argv[1]]()
