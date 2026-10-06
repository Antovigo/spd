"""Which L15 MLP components carry the seven b-flip neurons, and is that all they do?

    python -m param_decomp.arith_repr.vectors.flip_components static   # CPU -> OUT/flip/static.npz + stdout
    python -m param_decomp.arith_repr.vectors.flip_components rest     # CPU -> OUT/flip/rest.json + stdout
    python -m param_decomp.arith_repr.vectors.flip_components ablate   # GPU -> OUT/flip/ablate.json + stdout

Model: the components-only model of the dataset (`decomposed/`): every L15 site is `((x V) * m) U`
with the masks m of `ATLAS/masks`, no weight delta. F is the set of the seven neurons that write the
op-odd (sign-flipped) part of b's code at `=` (report_auto_interp_vectors.md, section 5).

static
1. Check: rebuilding the decomposed gate/up pre-activations from inner x mask x U.
2. The flip neurons in the decomposed model: op-odd b-line coefficient of act = silu(g) u at k = 2,
   10, 20, against the original model; rank among all 14,336 neurons by |odd coef| x |W_down col|.
3. Feeders: for each flip neuron n, the share of g_n and of u_n carried by each gate / up component,
   `sum_i C_c(i) y_n(i) / sum_i y_n(i)^2` with `C_c = h_c m_c U_c[n]` (additive, sums to 1),
   at `=`, plus the share of the add-sub difference of the means.
   Readers: the share of each down component in the neuron's write `act_n sum_d m_d V_d[n] U_d`
   (energy without cross terms).
4. For every component found: weight on F (share of |U_c|^2 for gate / up, of |V_c|^2 for down),
   and the effect of removing it, split into the part through F and the part through the other
   neurons, per position and op:
     write   = mean |h m|^2 |U_c[S]|^2                 what it puts into the neurons S,
     act     = mean |delta act_S|^2                    after SiLU gating (up: silu(g) h m U;
                                                       gate: (silu(g) - silu(g - h m U)) u),
     out     = mean |delta out_S|^2                    after the alive down components,
   with delta out_S = sum_d m_d (delta act_S . V_d) U_d. Down components: out_S = mean (m h_S)^2 |U_c|^2,
   h_S = act_S . V_c[S].

ablate
Last-position KL(base || ablated) and accuracy on all 20,000 prompts, base = the components model,
for each component: off everywhere, off at `=` only, its path through F removed, its path through the
other neurons removed; the seven neurons zeroed (everywhere / at `=`); and the whole set."""

import json
import sys
from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.isa.atlas import ATLAS
from param_decomp.arith_repr.vectors.common import AUTOINTERP, DATASET, OUT, comp_table
from param_decomp.arith_repr.vectors.flip_model import FLIP, FlipModel, L
from param_decomp.arith_repr.vectors.mirror_neurons import b_line

FLIP_K = {12769: 2, 6456: 2, 9205: 10, 9057: 10, 13193: 10, 7446: 20, 11305: 20}
KINDS = ("gate", "up", "down")
POS = ("BOS", "a", "op", "b", "=")
DIR = OUT / "flip"


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def layer_sites() -> dict[str, dict[str, Any]]:
    """Per kind: V, U, component names, and the columns in the dataset / in the layer's mask array."""
    comps = comp_table()
    uv = np.load(AUTOINTERP / "uv_alive.npz")
    lcols = np.flatnonzero(comps["layer"] == L)
    out = {}
    for kind in KINDS:
        key = f"layers.{L}.mlp.{kind}_proj"
        sel = np.flatnonzero(comps["site"][lcols] == key)
        ids = {int(i): j for j, i in enumerate(uv[key + ".ids"])}
        order = [ids[int(comps["cidx"][lcols[s]])] for s in sel]
        out[kind] = {
            "V": uv[key + ".V"][:, order].astype(np.float32),
            "U": uv[key + ".U"][order].astype(np.float32),
            "names": [f"L{L}.{kind}.c{int(comps['cidx'][lcols[s]])}" for s in sel],
            "mcols": sel,
            "dcols": lcols[sel],
        }
    return out


def load_layer() -> dict[str, dict[str, Any]]:
    """`layer_sites` plus each site's decomposed-model inner activations h and masks m, (N, T, c)."""
    S = layer_sites()
    masks = np.load(ATLAS / "masks" / f"L{L}.npy", mmap_mode="r")
    inner_mm = np.load(DATASET / "decomposed/inner.npy", mmap_mode="r")
    allcols = np.concatenate([S[k]["dcols"] for k in KINDS])
    inner_all = np.asarray(inner_mm[:, :, allcols], np.float32)  # (N, T, n)
    off = np.cumsum([0] + [len(S[k]["dcols"]) for k in KINDS])
    for j, k in enumerate(KINDS):
        S[k]["h"] = inner_all[:, :, off[j] : off[j + 1]]
        S[k]["m"] = np.asarray(masks[:, :, S[k]["mcols"]], np.float32)
    return S


def static() -> None:
    DIR.mkdir(exist_ok=True)
    S = load_layer()
    print({k: len(S[k]["names"]) for k in KINDS}, flush=True)

    # per position: g, u (decomposed, from the dataset) for all neurons
    g_mm = np.load(DATASET / "decomposed/mlp_gate.npy", mmap_mode="r")
    u_mm = np.load(DATASET / "decomposed/mlp_up.npy", mmap_mode="r")
    go_mm = np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")
    uo_mm = np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")

    # 1. reconstruction check at '='
    t = 4
    g = np.asarray(g_mm[L, :, t], np.float32)
    u = np.asarray(u_mm[L, :, t], np.float32)
    for nm, y in (("gate", g), ("up", u)):
        rec = (S[nm]["h"][:, t] * S[nm]["m"][:, t]) @ S[nm]["U"]
        err = np.linalg.norm(rec - y) / np.linalg.norm(y)
        print(
            f"check: {nm} pre-activation at '=' rebuilt from inner*mask*U, relative error {err:.4f}"
        )

    # 2. flip neurons in the decomposed model vs original
    from param_decomp.arith_repr.autointerp.llama_weights import Weights

    Wd = Weights().get(f"model.layers.{L}.mlp.down_proj.weight").astype(np.float32)  # (4096, 14336)
    wnorm = np.linalg.norm(Wd, axis=0)
    act_d = silu(g) * u
    go = np.asarray(go_mm[L, :, t], np.float32)
    uo = np.asarray(uo_mm[L, :, t], np.float32)
    act_o = silu(go) * uo
    odd = {}
    for nm, act in (("decomposed", act_d), ("original", act_o)):
        cb = [b_line(act[o * 10000 : (o + 1) * 10000]) for o in (0, 1)]  # (50, 14336) each
        odd[nm] = (cb[0] - cb[1]) / 2
    print("\n2. op-odd b-line coefficient of act at '=' (|O_n(k)|), decomposed vs original")
    for k in (2, 10, 20):
        score = np.abs(odd["decomposed"][k - 1]) * wnorm
        rank = np.argsort(-score)
        top = rank[:5]
        print(f"  k{k}: top-5 neurons by |O| x |W_down col| in decomposed: {top.tolist()}")
        for n in FLIP:
            if FLIP_K[int(n)] != k:
                continue
            od, oo = odd["decomposed"][k - 1, n], odd["original"][k - 1, n]
            print(f"    n{n}: |O| decomposed {abs(od):.3f} original {abs(oo):.3f}, phase diff "
                  f"{np.degrees(np.angle(od * np.conj(oo))):+.0f} deg, rank {int(np.flatnonzero(rank == n)[0])}")  # fmt: skip
    del act_o, go, uo

    # 3. feeders and readers of each flip neuron at '='
    print("\n3. components carrying each flip neuron at '=' (shares of the neuron's pre-activation"
          " second moment / of its add-sub mean difference)")  # fmt: skip
    opv = np.repeat([0, 1], 10000)
    feed: dict[str, dict[str, float]] = {}
    for n in FLIP:
        print(f"  neuron {n} (k{FLIP_K[int(n)]}):")
        for nm, y in (("gate", g), ("up", u)):
            C = S[nm]["h"][:, t] * S[nm]["m"][:, t] * S[nm]["U"][:, n][None]  # (N, c)
            yn = y[:, n]
            share = (C * yn[:, None]).sum(0) / (yn**2).sum()
            dmean = yn[opv == 1].mean() - yn[opv == 0].mean()
            dshare = (C[opv == 1].mean(0) - C[opv == 0].mean(0)) / dmean
            order = np.argsort(-np.abs(share))[:4]
            txt = ", ".join(f"{S[nm]['names'][c].split('.')[-1]} {share[c]:+.2f}/{dshare[c]:+.2f}"
                            for c in order if abs(share[c]) > 0.03)  # fmt: skip
            print(
                f"    {nm}: mean add {yn[opv == 0].mean():+.2f} sub {yn[opv == 1].mean():+.2f}; {txt}"
            )
            for c in range(len(share)):
                key = S[nm]["names"][c]
                feed.setdefault(key, {})[f"n{n}"] = float(share[c])
        # readers
        md = S["down"]["m"][:, t]
        Vn = S["down"]["V"][n]  # (d,)
        Un = np.linalg.norm(S["down"]["U"], axis=1)
        e = ((md * act_d[:, n : n + 1]) ** 2).mean(0) * Vn**2 * Un**2
        es = e / e.sum()
        order = np.argsort(-es)[:4]
        print("    down: " + ", ".join(f"{S['down']['names'][c].split('.')[-1]} {es[c]:.2f}"
                                       for c in order if es[c] > 0.03))  # fmt: skip
        for c in range(len(es)):
            feed.setdefault(S["down"]["names"][c], {})[f"n{n}"] = float(es[c])

    chosen = sorted(
        (k for k, v in feed.items() if max(abs(x) for x in v.values()) >= 0.1),
        key=lambda s: (KINDS.index(s.split(".")[1]), int(s.split(".c")[1])),
    )
    print("\nselected (>= 0.1 of some flip neuron):", chosen)
    (DIR / "chosen.json").write_text(json.dumps({"chosen": chosen, "feed": feed}, indent=1))

    # 4. localisation: weight and effect, through F and through the rest
    inF = np.zeros(14336, bool)
    inF[FLIP] = True
    print("\n4. localisation. weight on F = share of |U| (gate/up) or |V| (down) on the 7 neurons;"
          " n90 = neurons holding 90% of it.")  # fmt: skip
    print("   effect rows: per position and op, mean squared norm of the removal effect at each stage,"
          " F / rest (rest share of the total in brackets); on = fraction of prompts with the mask on")  # fmt: skip
    rows: dict[str, Any] = {}
    U_down, V_down = S["down"]["U"], S["down"]["V"]
    gs: dict[int, np.ndarray] = {4: g}
    us: dict[int, np.ndarray] = {4: u}
    for name in chosen:
        kind = name.split(".")[1]
        c = S[kind]["names"].index(name)
        w = (S[kind]["U"][c] if kind != "down" else S[kind]["V"][:, c]) ** 2
        ws = np.sort(w)[::-1]
        n90 = int(np.searchsorted(np.cumsum(ws) / ws.sum(), 0.9)) + 1
        topn = np.argsort(-w)[:6]
        print(f"\n  {name}: weight on F {w[FLIP].sum() / w.sum():.2f} (n90 {n90}); top neurons "
              + ", ".join(f"{int(i)}{'*' if inF[i] else ''} {w[i] / w.sum():.2f}" for i in topn))  # fmt: skip
        rows[name] = {"wF": float(w[FLIP].sum() / w.sum()), "n90": n90, "pos": {}}
        for t in (1, 2, 3, 4):
            if t not in gs:
                gs[t] = np.asarray(g_mm[L, :, t], np.float32)
                us[t] = np.asarray(u_mm[L, :, t], np.float32)
            gt, ut = gs[t], us[t]
            md = S["down"]["m"][:, t]
            for o in (0, 1):
                sl = slice(o * 10000, (o + 1) * 10000)
                hm = S[kind]["h"][sl, t, c] * S[kind]["m"][sl, t, c]
                on = float(S[kind]["m"][sl, t, c].mean())
                if on == 0:
                    rows[name]["pos"][f"{POS[t]}_{o}"] = {"on": 0.0}
                    continue
                res: dict[str, float] = {"on": on}
                if kind == "down":
                    act = silu(gt[sl]) * ut[sl]
                    for part, msk in (("F", inF), ("rest", ~inF)):
                        hS = act[:, msk] @ S[kind]["V"][msk, c]
                        res[f"act_{part}"] = float((hS**2).mean())
                        res[f"out_{part}"] = float(((S[kind]["m"][sl, t, c] * hS) ** 2).mean()
                                                   * (S[kind]["U"][c] ** 2).sum())  # fmt: skip
                else:
                    Uc = S[kind]["U"][c]
                    for part, msk in (("F", inF), ("rest", ~inF)):
                        res[f"write_{part}"] = float((hm**2).mean() * (Uc[msk] ** 2).sum())
                    dW = hm[:, None] * Uc[None]
                    if kind == "up":
                        dact = silu(gt[sl]) * dW
                    else:
                        dact = (silu(gt[sl]) - silu(gt[sl] - dW)) * ut[sl]
                    for part, msk in (("F", inF), ("rest", ~inF)):
                        da = np.where(msk[None], dact, 0.0)
                        res[f"act_{part}"] = float((da**2).sum(1).mean())
                        dout = ((da @ V_down) * md[sl]) @ U_down
                        res[f"out_{part}"] = float((dout**2).sum(1).mean())
                    dout = ((dact @ V_down) * md[sl]) @ U_down
                    res["out_all"] = float((dout**2).sum(1).mean())
                rows[name]["pos"][f"{POS[t]}_{o}"] = res
                stages = [s for s in ("write", "act", "out") if f"{s}_F" in res]
                txt = "  ".join(f"{s} {res[f'{s}_F']:.3g} / {res[f'{s}_rest']:.3g} "
                                f"({res[f'{s}_rest'] / max(res[f'{s}_F'] + res[f'{s}_rest'], 1e-30):.2f})"
                                for s in stages)  # fmt: skip
                print(f"    {POS[t]:>2} {'add' if o == 0 else 'sub'} on {on:.2f}: {txt}")
        # where the rest goes: top non-F neurons by weight, their SiLU and down readership at '='
        if kind != "down":
            rest_top = [int(i) for i in np.argsort(-np.where(inF, 0, w))[:8]]
            gt, ut = gs[4], us[4]
            mon = S[kind]["m"][:, 4, c] > 0
            readw = (V_down**2 * (S["down"]["m"][:, 4].mean(0) * (U_down**2).sum(1))[None]).sum(1)
            txt = []
            for i in rest_top:
                sv = silu(gt[mon, i]) if mon.any() else np.zeros(1)
                txt.append(f"{i} w{w[i] / w.sum():.2f} |silu| {np.abs(sv).mean():.2f} "
                           f"u {np.abs(ut[mon, i]).mean() if mon.any() else 0:.2f} read {readw[i]:.2g}")  # fmt: skip
            print(
                "    top other neurons at '=' (mean over prompts with the mask on): "
                + "; ".join(txt)
            )
            rows[name]["rest_top"] = rest_top
    print("\n   reference: down readership of the flip neurons "
          + ", ".join(f"{n} {(V_down[n] ** 2 * (S['down']['m'][:, 4].mean(0) * (U_down ** 2).sum(1))).sum():.2g}"
                      for n in FLIP))  # fmt: skip
    out_ref = {}
    for t in (1, 2, 3, 4):
        act = silu(gs[t]) * us[t]
        mlp_out = ((act @ V_down) * S["down"]["m"][:, t]) @ U_down
        out_ref[POS[t]] = float((mlp_out**2).sum(1).mean())
    print(
        "   reference: mean |L15 MLP output|^2 per position:",
        {k: f"{v:.3g}" for k, v in out_ref.items()},
    )
    (DIR / "static.json").write_text(json.dumps({"rows": rows, "out_ref": out_ref}, indent=1))


def grid_split(x: np.ndarray) -> dict[str, float]:
    """x (10000, d) over one op's (a, b) grid -> shares of mean |x|^2: constant (grid mean), a-only,
    b-only, and the rest (a-b interaction)."""
    tot = float((x**2).sum(1).mean())
    g = x.reshape(100, 100, -1)
    mu = g.mean((0, 1))
    ma, mb = g.mean(1) - mu, g.mean(0) - mu
    out = {"total": tot, "const": float((mu**2).sum()) / tot}
    out["a"] = float((ma**2).sum(1).mean()) / tot
    out["b"] = float((mb**2).sum(1).mean()) / tot
    out["ab"] = 1 - out["const"] - out["a"] - out["b"]
    return out


def rest() -> None:
    """What each selected component writes through the neurons outside F: the removal effect on the
    L15 MLP output split into grid-constant / a-only / b-only / interaction parts, per position and
    op, and the neurons carrying it."""
    S = load_layer()
    chosen = json.loads((DIR / "chosen.json").read_text())["chosen"]
    g_mm = np.load(DATASET / "decomposed/mlp_gate.npy", mmap_mode="r")
    u_mm = np.load(DATASET / "decomposed/mlp_up.npy", mmap_mode="r")
    inF = np.zeros(14336, bool)
    inF[FLIP] = True
    U_down, V_down = S["down"]["U"], S["down"]["V"]
    gu = {
        t: (np.asarray(g_mm[L, :, t], np.float32), np.asarray(u_mm[L, :, t], np.float32))
        for t in (3, 4)
    }
    out: dict[str, Any] = {}
    print("removal effect on the L15 MLP output, through F / through the rest: mean |.|^2 and its shares"
          " const / a / b / ab (grid mean, a-only, b-only, interaction)")  # fmt: skip
    for name in chosen:
        kind = name.split(".")[1]
        c = S[kind]["names"].index(name)
        print(f"\n  {name}")
        for t in (3, 4):
            gt, ut = gu[t]
            md = S["down"]["m"][:, t]
            Weff = md.mean(0)[:, None] * U_down  # (d, 4096): mean-mask down map per component
            for o in (0, 1):
                sl = slice(o * 10000, (o + 1) * 10000)
                if S[kind]["m"][sl, t, c].mean() < 0.05:
                    continue
                hm = S[kind]["h"][sl, t, c] * S[kind]["m"][sl, t, c]
                parts = {}
                if kind == "down":
                    act = silu(gt[sl]) * ut[sl]
                    for part, msk in (("F", inF), ("rest", ~inF)):
                        hS = (act[:, msk] @ S[kind]["V"][msk, c]) * S[kind]["m"][sl, t, c]
                        parts[part] = hS[:, None] * S[kind]["U"][c][None]
                    e = (act * S[kind]["V"][:, c][None]) ** 2
                else:
                    dW = hm[:, None] * S[kind]["U"][c][None]
                    if kind == "up":
                        dact = silu(gt[sl]) * dW
                    else:
                        dact = (silu(gt[sl]) - silu(gt[sl] - dW)) * ut[sl]
                    for part, msk in (("F", inF), ("rest", ~inF)):
                        da = np.where(msk[None], dact, 0.0)
                        parts[part] = ((da @ V_down) * md[sl]) @ U_down
                    e = (dact**2) * ((V_down @ Weff) ** 2).sum(1)[None]
                en = e.mean(0)
                en[inF] = 0
                top = np.argsort(-en)[:5]
                res = {p: grid_split(x) for p, x in parts.items()}
                out[f"{name}|{POS[t]}|{o}"] = {
                    **res,
                    "top": top.tolist(),
                    "top_share": (en[top] / en.sum()).tolist(),
                }
                txt = "  ".join(f"{p} {r['total']:.3g} [{r['const']:.2f} {r['a']:.2f} {r['b']:.2f} {r['ab']:.2f}]"
                                for p, r in res.items())  # fmt: skip
                print(f"    {POS[t]:>2} {'add' if o == 0 else 'sub'}: {txt}; rest carried by "
                      + ", ".join(f"{int(n)} {en[n] / en.sum():.2f}" for n in top))  # fmt: skip
    (DIR / "rest.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- ablations (components model, GPU)


def ablate() -> None:
    import jax

    from param_decomp.arith_repr.isa.components_model import kl

    print(jax.devices(), flush=True)
    chosen = json.loads((DIR / "chosen.json").read_text())["chosen"]

    M = FlipModel()
    rows = np.arange(20000)
    base, _ = M.forward(rows)
    ans = M.answer
    a, b, op = M.a, M.b, M.op
    valid = (op == 0) | (a >= b)
    flipped = np.where(
        op == 0, M.num_ids[np.clip(a - b, 0, 200)], M.num_ids[np.clip(a + b, 0, 200)]
    )
    flip_ok = ((op == 0) & (a > b)) | (op == 1)  # the other op's answer is a number token

    def metrics(lp: np.ndarray) -> dict[str, float]:
        top = lp.argmax(-1)
        d = kl(base, lp)
        r = {}
        for o, nm in ((0, "add"), (1, "sub")):
            sel = (op == o) & valid
            r[f"kl_{nm}"] = float(d[op == o].mean())
            r[f"acc_{nm}"] = float((top[sel] == ans[sel]).mean())
            fs = (op == o) & flip_ok
            r[f"other_op_{nm}"] = float((top[fs] == flipped[fs]).mean())
        return r

    out: dict[str, Any] = {"base": metrics(base)}
    print("base", out["base"], flush=True)
    allT = np.ones(5, bool)
    eqT = np.array([0, 0, 0, 0, 1], bool)
    inF = np.zeros(14336, bool)
    inF[FLIP] = True
    allN = np.ones(14336, bool)

    def run(label: str, comps: list[Any], neurons: Any = None) -> None:
        M.spec = {"comps": comps, "neurons": neurons}
        lp, _ = M.forward(rows)
        out[label] = metrics(lp)
        print(label, {k: round(v, 4) for k, v in out[label].items()}, flush=True)

    run("neurons F off", [], (inF, allT))
    run("neurons F off at =", [], (inF, eqT))
    S = {k: M.sites[L][k].names for k in KINDS}
    spec_of = {nm: (nm.split(".")[1], S[nm.split(".")[1]].index(nm)) for nm in chosen}
    for nm in chosen:
        kind, c = spec_of[nm]
        run(f"{nm} off", [(kind, c, allN, allT)])
        run(f"{nm} off at =", [(kind, c, allN, eqT)])
        run(f"{nm} via F off", [(kind, c, inF, allT)])
        run(f"{nm} via rest off", [(kind, c, ~inF, allT)])
    run("all chosen off", [(*spec_of[nm], allN, allT) for nm in chosen])
    run("all chosen via F off", [(*spec_of[nm], inF, allT) for nm in chosen])
    run("all chosen via rest off", [(*spec_of[nm], ~inF, allT) for nm in chosen])
    (DIR / "ablate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    cast(Any, {"static": static, "rest": rest, "ablate": ablate})[sys.argv[1]]()
