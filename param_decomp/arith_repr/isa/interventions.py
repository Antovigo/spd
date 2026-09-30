"""Causal tests of the atlas on the components-only model: interchange interventions on the quantities,
and ablations of the part of the readers the quantities do not explain.

    python -m param_decomp.arith_repr.isa.interventions maps <layer>   # per read point, CPU
    python -m param_decomp.arith_repr.isa.interventions swap           # -> INTERV/swap.json
    python -m param_decomp.arith_repr.isa.interventions ablate <kind>  # layers | windows -> INTERV/ablate_<kind>.json
    python -m param_decomp.arith_repr.isa.interventions local          # -> INTERV/local.json, CPU
    python -m param_decomp.arith_repr.isa.interventions labels <layer> # -> INTERV/labels/L<layer>.json, CPU

Read point (layer, site, position) as in `atlas.py`; its readers' activations times the stream RMS are
`H = x G` with `G = g * V` (ln gain times the readers' V), centred `Hc = H - mu`.

Maps (per read point). The quantities' coordinates Z (N, K) are a linear read of the stream,
`Z = (x - xbar) F` with `F = G W`, `W = lstsq(Hc, Z)`; their patterns `P = E[(x - xbar) Z]` are where
they are written. Each reader splits into three uncorrelated parts over the prompts:
    E = Z B, B = lstsq(Z, Hc)            the part the quantities explain,
    U = Hc Q Q^T - E                     the rest of the top-r principal subspace (r: parallel analysis),
    T = Hc - Hc Q Q^T                    the tail below the parallel-analysis rank.

Swap (interchange). Base prompts are additions the model gets right; each source prompt differs from
its base in one input (a, b or the operation) or, for result codes, is another addition. The
quantity's value is set to the source's by moving the stream along its patterns only,
`x' = x + ((x_src - x) F) (P^T F)^-1 P^T`: `x' F = x_src F`, and quantities uncorrelated with it
(`F2^T P = Cov(z2, z) = 0`) keep their base values. A quantity with period T passes the test when the
answer's residue mod T follows the source. Controls: a random filter of the same dimension (its
pattern from the stream covariance) and the whole stream at the position.

Local swap. The answer is a poor readout for a quantity: until about L24 later heads re-read the
operands at their own positions, and late result codes sit beside answer directions no reader reads.
So each MLP at `=` is tested on its own output, with its masks held at the base prompt's:
    IIA = 1 - sum |w - w_stream|^2 / sum |w_base - w_stream|^2,
w the MLP's write after the swap, w_stream its write when its whole input is the source's. The swaps:
single quantities, all quantities of one variable, all quantities, a random subspace, and, in reader
space, the source's explained part E or its unexplained part U + T, with the base's or the source's
stream RMS (the readers see `x G / rms`, so the RMS is a channel of its own).

Ablate. Each part of the readers at the chosen read points is kept, replaced by its mean (0), or
resampled from another prompt with the same operation, or with the same operation and result."""

import json
import re
import sys
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.atlas import ATLAS, POSITIONS, SITES, read_point_index, results
from param_decomp.arith_repr.isa.components_model import ComponentsModel, kl
from param_decomp.arith_repr.vectors.common import DATASET, RESID, comp_table, load_uv

MODEL = "decomposed"
INTERV = results(MODEL) / "interventions"
N_BASE = 500
POS_OF = {v: k for k, v in POSITIONS.items()}


def maps(layer: int) -> None:
    comps = comp_table()
    gains = np.load(RESID / "norms.npz")
    cols = np.flatnonzero(comps["layer"] == layer)
    inner = np.load(ATLAS / MODEL / "inner" / f"L{layer}.npy", mmap_mode="r")
    resid = np.load(DATASET / MODEL / "resid.npy", mmap_mode="r")
    (INTERV / "maps").mkdir(parents=True, exist_ok=True)
    for site, kinds in SITES.items():
        sel = np.flatnonzero(np.isin(comps["kind"][cols], kinds))
        names = np.array(
            [f"L{layer}.{comps['kind'][cols[i]]}.c{comps['cidx'][cols[i]]}" for i in sel]
        )
        V, _ = load_uv(comps, cols[sel])
        G_all = np.stack(V, 1) * gains["ln1" if site == "attn" else "ln2"][layer][:, None]
        for pos, pname in POSITIONS.items():
            key = f"L{layer}_{site}_{pname}".replace("=", "eq")
            d = json.loads((results(MODEL) / "points" / f"{key}.json").read_text())
            arr = np.load(results(MODEL) / "points" / f"{key}.npz")
            x = np.asarray(resid[read_point_index(layer, site), :, pos], np.float64)
            N = len(x)
            rms = np.sqrt((x**2).mean(1) + 1e-5)
            H = np.asarray(inner[:, pos][:, sel], np.float64) * rms[:, None]
            var = H.var(0)
            keep = var > 1e-8 * max(float(var.max()), 1e-30)
            if keep.sum() == 0:
                continue
            mu = H[:, keep].mean(0)
            Hc, G = H[:, keep] - mu, G_all[:, keep]
            xbar = x.mean(0)
            xc = x - xbar
            dims = [q["k"] for q in d["quantities"]]
            Z = (
                np.concatenate([arr[f"z{i}"] for i in range(len(dims))], 1).astype(np.float64)
                if dims
                else np.zeros((N, 0))
            )
            r = int(d["r"])
            Q = (
                np.linalg.svd(Hc / Hc.std(), full_matrices=False)[2][:r].T
                if r
                else np.zeros((Hc.shape[1], 0))
            )
            W = np.linalg.lstsq(Hc, Z, rcond=None)[0] if dims else np.zeros((Hc.shape[1], 0))
            B = np.linalg.lstsq(Z, Hc, rcond=None)[0] if dims else np.zeros((0, Hc.shape[1]))
            F = G @ W
            P = xc.T @ Z / N
            E = Z @ B
            top = Hc @ Q @ Q.T
            parts = {"E": E, "U": top - E, "T": Hc - top}
            share = {p: (v.var(0) / np.maximum(var[keep], 1e-30)) for p, v in parts.items()}
            Fr = np.random.default_rng(layer * 100 + pos).standard_normal(
                (x.shape[1], max(sum(dims), 1))
            )
            Pr = xc.T @ (xc @ Fr) / N
            fit = 1 - float(((xc @ F - Z) ** 2).sum() / max((Z**2).sum(), 1e-12))
            np.savez(INTERV / "maps" / f"{key}.npz", names=names[keep], mu=mu, xbar=xbar, dims=np.array(dims, int),
                     F=F, P=P, B=B, Q=Q, FQ=G @ Q, Fr=Fr, Pr=Pr, var=var[keep],
                     **{f"share_{p}": v for p, v in share.items()})  # fmt: skip
            tot = var[keep].sum()
            print(f"{key}: readers {keep.sum()} K {sum(dims)} r {r} fit {fit:.4f} "
                  + " ".join(f"{p} {(v * var[keep]).sum() / tot:.3f}" for p, v in share.items()), flush=True)  # fmt: skip


@jax.jit
def _swap(x: jax.Array, xs: jax.Array, F: jax.Array, M: jax.Array) -> jax.Array:
    return x + ((xs - x) @ F) @ M


@jax.jit
def _parts(x: jax.Array, G: jax.Array, mu: jax.Array, xbar: jax.Array, F: jax.Array, B: jax.Array,
           FQ: jax.Array, Qt: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:  # fmt: skip
    Hc = x @ G - mu
    xc = x - xbar
    E = (xc @ F) @ B
    top = (xc @ FQ) @ Qt
    return E, top - E, Hc - top


def load_map(key: str) -> dict[str, np.ndarray]:
    return dict(np.load(INTERV / "maps" / f"{key}.npz"))


def key_parts(key: str) -> tuple[int, str, int]:
    lay, site, pos = key.split("_")
    return int(lay[1:]), site, POS_OF["=" if pos == "eq" else pos]


def residue_metrics(M: ComponentsModel, lp: np.ndarray, lp_base: np.ndarray, lp_src: np.ndarray,
                    period: int | None) -> dict[str, float]:  # fmt: skip
    pi, pb, ps = lp.argmax(-1), lp_base.argmax(-1), lp_src.argmax(-1)
    out = {"exact_src": float((pi == ps).mean()), "exact_base": float((pi == pb).mean()),
           "kl": float(kl(lp_base, lp).mean())}  # fmt: skip
    if period is not None:
        vi, vb, vs = M.value(pi), M.value(pb), M.value(ps)
        diff = (vb >= 0) & (vs >= 0) & (vs % period != vb % period)
        if not diff.any():
            return out
        out |= {"n": int(diff.sum()),
                "follows_src": float(((vi % period == vs % period) & (vi >= 0))[diff].mean()),
                "keeps_base": float(((vi % period == vb % period) & (vi >= 0))[diff].mean())}  # fmt: skip
    return out


def swap() -> None:
    M = ComponentsModel()
    rng = np.random.default_rng(0)
    INTERV.mkdir(parents=True, exist_ok=True)
    clean_pred = np.concatenate(
        [M.forward(np.arange(s, s + 2000))[0].argmax(-1) for s in range(0, 20000, 2000)]
    )
    correct = clean_pred == M.answer
    np.save(INTERV / "clean_pred.npy", clean_pred)
    print("clean accuracy", correct.mean(), flush=True)
    base = rng.choice(np.flatnonzero(correct & (M.op == 0)), N_BASE, replace=False)
    a, b = M.a[base], M.b[base]
    other = lambda v: (v - 1 + rng.integers(1, 100, len(v))) % 100 + 1  # noqa: E731
    src = {"a": (other(a) - 1) * 100 + b - 1, "b": (a - 1) * 100 + other(b) - 1, "op": 10000 + base,
           "result": rng.choice(np.flatnonzero(M.op == 0), N_BASE)}  # fmt: skip
    atlas = json.loads((results(MODEL) / "atlas.json").read_text())
    keys = sorted(
        {q["point"] for q in atlas["quantities"] if q["pos"] == "="},
        key=lambda k: (key_parts(k)[0], k),
    )
    ts = sorted({read_point_index(key_parts(k)[0], key_parts(k)[1]) for k in keys})
    lp_base = M.forward(base)[0]
    lp_src, caps = {}, {}
    for v, rows_src in src.items():
        lp_src[v], caps[v] = M.forward(rows_src, capture=set(ts))

    def run(t: int, pos: int, var: str, F: np.ndarray | None, P: np.ndarray | None) -> np.ndarray:
        """Base prompts with the quantities (F, P) at stream point t, position pos, set to the source's
        (the whole stream at the position when F is None)."""
        xs_all = jnp.asarray(caps[var][t], jnp.float32)
        FM = (
            None
            if F is None or P is None
            else (
                jnp.asarray(F, jnp.float32),
                jnp.asarray(np.linalg.solve(P.T @ F, P.T), jnp.float32),
            )
        )

        def hook(tt: int, x: jax.Array, idx: np.ndarray) -> jax.Array:
            if tt != t:
                return x
            xs = xs_all[idx, pos]
            return x.at[:, pos].set(xs if FM is None else _swap(x[:, pos], xs, *FM))

        return M.forward(base, resid=hook)[0]

    def variable(name: str) -> tuple[str, int | None]:
        head = name.split(",")[0].split(" ")[0]
        m = re.search(r"mod (\d+)", name)
        var = head if head in ("op", "a", "b") else "result"
        return var, (None if var == "op" else int(m.group(1)) if m else 100)

    rows: list[dict[str, Any]] = []
    for key in keys:
        layer, site, pos = key_parts(key)
        t = read_point_index(layer, site)
        mp = load_map(key)
        offs = np.concatenate([[0], np.cumsum(mp["dims"])]).astype(int)
        qs = [q for q in atlas["quantities"] if q["point"] == key]
        groups: dict[str, list[int]] = {}
        for q in qs:
            var, period = variable(q["name"])
            groups.setdefault(var, []).append(q["qi"])
            sl = slice(offs[q["qi"]], offs[q["qi"] + 1])
            k = int(sl.stop - sl.start)
            for kind, Fq, Pq in (
                ("quantity", mp["F"][:, sl], mp["P"][:, sl]),
                ("random", mp["Fr"][:, :k], mp["Pr"][:, :k]),
            ):
                lp = run(t, pos, var, Fq, Pq)
                rows.append({"point": key, "layer": layer, "site": site, "qi": q["qi"], "name": q["name"], "code": q["code"],
                             "var": var, "period": period, "k": k, "kind": kind} | residue_metrics(M, lp, lp_base, lp_src[var], period))  # fmt: skip
        for var, qis in list(groups.items()) + [("all", [q["qi"] for q in qs])]:
            cols = np.concatenate([np.arange(int(offs[i]), int(offs[i + 1])) for i in qis])
            v = "result" if var == "all" else var
            for kind, Fq, Pq in (("group", mp["F"][:, cols], mp["P"][:, cols]), ("random", mp["Fr"][:, : cols.size], mp["Pr"][:, : cols.size]),
                               ("stream", None, None)):  # fmt: skip
                lp = run(t, pos, v, Fq, Pq)
                rows.append({"point": key, "layer": layer, "site": site, "qi": -1, "name": f"all {var} quantities", "var": var,
                             "period": None, "k": int(cols.size), "kind": kind} | residue_metrics(M, lp, lp_base, lp_src[v], None))  # fmt: skip
        print(key, "done", flush=True)
        (INTERV / "swap.json").write_text(json.dumps(rows))


def ablate(kind: str) -> None:
    """`kind` "layers": every layer, position and site on its own; "windows": layer windows at `=`,
    and whole readers (components) at `=` chosen by how much of them the quantities explain."""
    M = ComponentsModel()
    rng = np.random.default_rng(1)
    INTERV.mkdir(parents=True, exist_ok=True)
    base = np.concatenate(
        [rng.choice(np.flatnonzero(M.op == o), N_BASE, replace=False) for o in (0, 1)]
    )
    same_op = np.array([rng.choice(np.flatnonzero(M.op == M.op[i])) for i in base])
    pool = {}
    for i in np.arange(20000):
        pool.setdefault((M.op[i], M.result[i]), []).append(i)
    same_res = np.array(
        [rng.choice([j for j in pool[(M.op[i], M.result[i])] if j != i] or [i]) for i in base]
    )
    keys = sorted(
        (p.stem for p in (INTERV / "maps").glob("*.npz")), key=lambda k: (key_parts(k)[0], k)
    )
    mps = {k: load_map(k) for k in keys}
    ts = set(range(2 * M.n_layer))
    lp_base, _ = M.forward(base)
    caps = {
        "src_op": M.forward(same_op, capture=ts)[1],
        "src_res": M.forward(same_res, capture=ts)[1],
    }
    ans = M.answer[base]
    ok = lp_base.argmax(-1) == ans
    print("clean accuracy add", ok[:N_BASE].mean(), "sub", ok[N_BASE:].mean(), flush=True)

    # per (layer, kind): the read points' device arrays
    dev: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for key, mp in mps.items():
        layer, site, pos = key_parts(key)
        for kd in SITES[site]:
            j = np.flatnonzero([n.split(".")[1] == kd for n in mp["names"]])
            if j.size == 0:
                continue
            col = np.array([M.column(layer, str(n))[1] for n in mp["names"][j]])
            s = M.sites[layer][kd]
            g = jnp.asarray(M.ln1[layer] if site == "attn" else M.ln2[layer])
            dev.setdefault((layer, kd), []).append({
                "key": key, "pos": pos, "t": read_point_index(layer, site), "col": jnp.asarray(col), "j": j,
                "G": s.V[:, col] * g[:, None], "mu": jnp.asarray(mp["mu"][j], jnp.float32),
                "xbar": jnp.asarray(mp["xbar"], jnp.float32), "F": jnp.asarray(mp["F"], jnp.float32),
                "B": jnp.asarray(mp["B"][:, j], jnp.float32), "FQ": jnp.asarray(mp["FQ"], jnp.float32),
                "Qt": jnp.asarray(mp["Q"][j].T, jnp.float32)})  # fmt: skip

    def run(
        scope: set[str],
        mode: dict[str, str],
        drop: dict[tuple[int, str], tuple[jax.Array, jax.Array | None]] | None = None,
    ) -> np.ndarray:
        """`drop[(layer, kind)] = (columns, mean)`: those readers at `=` set to their mean (times the
        RMS), or to zero (the component switched off) when mean is None."""
        stash: dict[int, jax.Array] = {}

        def rhook(t: int, x: jax.Array, _idx: np.ndarray) -> jax.Array:
            stash[t] = x
            return x

        def ihook(
            layer: int, kind: str, h: jax.Array, rms: jax.Array, idx: np.ndarray
        ) -> jax.Array:
            for d in dev.get((layer, kind), []):
                if d["key"] not in scope:
                    continue
                p = d["pos"]
                args = (d["G"], d["mu"], d["xbar"], d["F"], d["B"], d["FQ"], d["Qt"])
                parts = dict(zip("EUT", _parts(stash[d["t"]][:, p], *args), strict=True))
                srcs = {s: dict(zip("EUT", _parts(jnp.asarray(caps[s][d["t"]][idx, p]), *args), strict=True))
                        for s in set(mode.values()) - {"mean"}}  # fmt: skip
                new = sum(
                    (parts[q] if q not in mode else 0.0 if mode[q] == "mean" else srcs[mode[q]][q])
                    for q in "EUT"
                )
                h = h.at[:, p, d["col"]].set((d["mu"] + new) / rms[:, p, None])
            if drop is not None and (layer, kind) in drop:
                c, mu = drop[(layer, kind)]
                h = h.at[:, 4, c].set(0.0 if mu is None else mu / rms[:, 4, None])
            return h

        return M.forward(base, resid=rhook, inner=ihook)[0]

    answer_ids = np.concatenate([M.num_ids, [M.minus]])

    def metrics(lp: np.ndarray) -> dict[str, float]:
        """Accuracy (top token is the answer's first token), accuracy among the answer tokens only
        ("0".."200" and "-"), and the probability on those tokens."""
        pred = lp.argmax(-1)
        pred_num = answer_ids[lp[:, answer_ids].argmax(-1)]
        n = np.arange(len(ans))
        return {"acc_add": float((pred == ans)[:N_BASE].mean()), "acc_sub": float((pred == ans)[N_BASE:].mean()),
                "num_acc_add": float((pred_num == ans)[:N_BASE].mean()), "num_acc_sub": float((pred_num == ans)[N_BASE:].mean()),
                "p_number": float(np.exp(lp[:, answer_ids]).sum(-1).mean()),
                "agree": float((pred == lp_base.argmax(-1)).mean()), "kl": float(kl(lp_base, lp).mean()),
                "dlogp": float((lp[n, ans] - lp_base[n, ans])[ok].mean())}  # fmt: skip

    modes = {"drop tail": {"T": "mean"}, "drop unexplained in rank": {"U": "mean"},
             "keep quantities only": {"U": "mean", "T": "mean"}, "drop quantities": {"E": "mean"},
             "mean readers": {"E": "mean", "U": "mean", "T": "mean"},
             "resample unexplained, same op": {"U": "src_op", "T": "src_op"},
             "resample unexplained, same op and result": {"U": "src_res", "T": "src_res"},
             "resample quantities, same op": {"E": "src_op"},
             "resample quantities, same op and result": {"E": "src_res"}}  # fmt: skip
    rows: list[dict[str, Any]] = []
    preds: dict[str, np.ndarray] = {}

    def record(entry: dict[str, Any], lp: np.ndarray) -> None:
        rows.append(entry | metrics(lp))
        preds[" | ".join(str(v) for v in entry.values())] = lp.argmax(-1)
        print(rows[-1], flush=True)
        (INTERV / f"ablate_{kind}.json").write_text(json.dumps(rows))

    np.save(INTERV / f"ablate_{kind}_base.npy", base)
    if kind == "layers":
        scopes = {"all": set(keys)} | {
            f"L{li}": {k for k in keys if key_parts(k)[0] == li} for li in range(M.n_layer)
        }
        scopes |= {
            f"pos {p}": {k for k in keys if k.endswith("_" + p.replace("=", "eq"))}
            for p in POSITIONS.values()
        }
        scopes |= {f"site {s}": {k for k in keys if f"_{s}_" in k} for s in SITES}
        for sname, scope in scopes.items():
            for mname, mode in modes.items():
                record({"scope": sname, "mode": mname}, run(scope, mode))
    else:
        eq = [k for k in keys if k.endswith("_eq")]
        windows = [
            (0, 13),
            (14, 17),
            (18, 21),
            (22, 25),
            (26, 31),
            (14, 31),
            (18, 31),
            (22, 31),
            (26, 31),
        ]
        for lo, hi in windows:
            scope = {k for k in eq if lo <= key_parts(k)[0] <= hi}
            for mname, mode in modes.items():
                record({"scope": f"= L{lo}-{hi}", "mode": mname}, run(scope, mode))
        # whole readers at `=` by the share of their variance the quantities explain
        for lo, hi in ((14, 31), (18, 31), (22, 31)):
            pts = [k for k in eq if lo <= key_parts(k)[0] <= hi]
            readers = []
            for k in pts:
                mp, layer = mps[k], key_parts(k)[0]
                big = mp["var"] >= 1e-3 * mp["var"].max()
                for j in np.flatnonzero(big):
                    readers.append(
                        (
                            layer,
                            str(mp["names"][j]),
                            float(mp["share_E"][j]),
                            float(mp["mu"][j]),
                            float(mp["var"][j]),
                        )
                    )
            tot = sum(r[4] for r in readers)
            sets = {"unexplained (E < 0.2)": [r for r in readers if r[2] < 0.2],
                    "explained (E > 0.8)": [r for r in readers if r[2] > 0.8]}  # fmt: skip
            n_u = len(sets["unexplained (E < 0.2)"])
            sets["random, as many as unexplained"] = [
                readers[i]
                for i in np.random.default_rng(3).choice(len(readers), n_u, replace=False)
            ]
            for sname, sel in sets.items():
                for how in ("zero", "mean"):
                    drop: dict[tuple[int, str], tuple[list[int], list[float]]] = {}
                    for layer, name, _, mu, _ in sel:
                        kd, c = M.column(layer, name)
                        drop.setdefault((layer, kd), ([], []))[0].append(c)
                        drop[(layer, kd)][1].append(mu)
                    dj = {
                        lk: (jnp.asarray(c), None if how == "zero" else jnp.asarray(m, jnp.float32))
                        for lk, (c, m) in drop.items()
                    }
                    record({"scope": f"= L{lo}-{hi} readers", "mode": f"{how} {sname}", "n": len(sel), "of": len(readers),
                            "var_share": sum(r[4] for r in sel) / tot}, run(set(), {}, dj))  # fmt: skip
    np.savez(
        INTERV / f"ablate_{kind}_preds.npz",
        keys=np.array(list(preds)),
        preds=np.stack(list(preds.values())),
    )


@dataclass
class LocalMLP:
    """One MLP at `=` run on its own, with the base prompts' masks `mb`: its write for a raw input
    stream, and the parts E, U, T of its kept gate and up readers."""

    mp: dict[str, np.ndarray]
    G: np.ndarray  # (d, gate + up readers): ln2 gain times V
    kept: np.ndarray  # the map's readers among G's columns
    gi: np.ndarray
    ui: np.ndarray
    di: np.ndarray
    Ug: np.ndarray
    Uu: np.ndarray
    Vd: np.ndarray
    Ud: np.ndarray
    mb: np.ndarray
    eps: float

    def parts(self, x: np.ndarray) -> dict[str, np.ndarray]:
        mp = self.mp
        xc = x - mp["xbar"]
        E = (xc @ mp["F"]) @ mp["B"]
        top = (xc @ mp["FQ"]) @ mp["Q"].T
        return {"E": E, "U": top - E, "T": x @ self.G[:, self.kept] - mp["mu"] - top}

    def write(self, x: np.ndarray, override: dict[str, np.ndarray] | None = None,
              x_rms: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
        """The MLP's write; `override` replaces the kept readers' parts, `x_rms` gives the stream
        whose RMS divides the readers (x itself by default)."""
        x_rms = x if x_rms is None else x_rms
        rms = np.sqrt((x_rms**2).mean(1) + self.eps)[:, None]
        H = x @ self.G
        if override is not None:
            H[:, self.kept] = self.mp["mu"] + sum(override.values())
        h = H / rms
        ng = self.gi.size
        g = (h[:, :ng] * self.mb[:, self.gi]) @ self.Ug
        u = (h[:, ng:] * self.mb[:, self.ui]) @ self.Uu
        return (((g / (1 + np.exp(-g))) * u) @ self.Vd * self.mb[:, self.di]) @ self.Ud


def local_mlp(layer: int, base: np.ndarray, comps: dict[str, np.ndarray], gains: Any) -> LocalMLP:
    mp = load_map(f"L{layer}_mlp_eq")
    cols = np.flatnonzero(comps["layer"] == layer)
    kinds = comps["kind"][cols]
    names = np.array([f"L{layer}.{comps['kind'][c]}.c{comps['cidx'][c]}" for c in cols])
    V, U = load_uv(comps, cols)
    gi, ui, di = (np.flatnonzero(kinds == k) for k in ("gate", "up", "down"))
    gu = np.concatenate([gi, ui])
    m_all = np.load(ATLAS / "masks" / f"L{layer}.npy", mmap_mode="r")
    return LocalMLP(mp=mp, G=np.stack([V[i] for i in gu], 1) * gains["ln2"][layer][:, None],
                    kept=np.array([list(names[gu]).index(n) for n in mp["names"]]), gi=gi, ui=ui, di=di,
                    Ug=np.stack([U[i] for i in gi]), Uu=np.stack([U[i] for i in ui]),
                    Vd=np.stack([V[i] for i in di], 1), Ud=np.stack([U[i] for i in di]),
                    mb=np.asarray(m_all[base][:, 4], np.float64), eps=float(gains["eps"]))  # fmt: skip


def local() -> None:
    comps = comp_table()
    gains = np.load(RESID / "norms.npz")
    ix = np.load(DATASET / "index.npz")
    a_all, b_all, op_all = ix["a"].astype(int), ix["b"].astype(int), ix["op"].astype(int)
    rng = np.random.default_rng(2)
    base = np.sort(rng.choice(np.flatnonzero(op_all == 0), N_BASE, replace=False))
    a, b = a_all[base], b_all[base]
    other = lambda v: (v - 1 + rng.integers(1, 100, len(v))) % 100 + 1  # noqa: E731
    src = {"a": (other(a) - 1) * 100 + b - 1, "b": (a - 1) * 100 + other(b) - 1, "op": 10000 + base,
           "result": rng.choice(np.flatnonzero(op_all == 0), N_BASE)}  # fmt: skip
    atlas = json.loads((results(MODEL) / "atlas.json").read_text())
    resid = np.load(DATASET / MODEL / "resid.npy", mmap_mode="r")
    rows: list[dict[str, Any]] = []
    for layer in range(32):
        key = f"L{layer}_mlp_eq"
        if not (INTERV / "maps" / f"{key}.npz").exists():
            continue
        mlp = local_mlp(layer, base, comps, gains)
        mp = mlp.mp
        t = read_point_index(layer, "mlp")
        xb = np.asarray(resid[t][base][:, 4], np.float64)
        offs = np.concatenate([[0], np.cumsum(mp["dims"])]).astype(int)
        qs = [q for q in atlas["quantities"] if q["point"] == key]
        wb, pb = mlp.write(xb), mlp.parts(xb)
        for v, r in src.items():
            xs = np.asarray(resid[t][np.sort(r)][np.argsort(np.argsort(r)), 4], np.float64)
            ws, ps = mlp.write(xs), mlp.parts(xs)
            den = float(((wb - ws) ** 2).sum())
            rel = den / max(float(((wb - wb.mean(0)) ** 2).sum()), 1e-30)
            entry: dict[str, Any] = {"layer": layer, "src": v, "dependence": rel}
            if rel < 1e-3 or not qs:
                rows.append(entry)
                continue

            def iia(w: np.ndarray, ws: np.ndarray = ws, den: float = den) -> float:
                return 1 - float(((w - ws) ** 2).sum()) / den

            def swapped(
                F: np.ndarray, P: np.ndarray, xb: np.ndarray = xb, xs: np.ndarray = xs
            ) -> np.ndarray:
                return xb + ((xs - xb) @ F) @ np.linalg.solve(P.T @ F, P.T)

            def mixed(
                from_src: str, ps: dict[str, np.ndarray] = ps, pb: dict[str, np.ndarray] = pb
            ) -> dict[str, np.ndarray]:
                return {q: (ps if q in from_src else pb)[q] for q in "EUT"}

            groups: dict[str, list[int]] = {}
            single = []
            for q in qs:
                var = q["name"].split(",")[0].split(" ")[0]
                var = var if var in ("op", "a", "b") else "result"
                groups.setdefault(var, []).append(q["qi"])
                c = np.arange(int(offs[q["qi"]]), int(offs[q["qi"] + 1]))
                single.append({"qi": q["qi"], "name": q["name"], "var": var, "k": int(c.size),
                               "iia": iia(mlp.write(swapped(mp["F"][:, c], mp["P"][:, c]))),
                               "random": iia(mlp.write(swapped(mp["Fr"][:, : c.size], mp["Pr"][:, : c.size])))})  # fmt: skip
            grp = {}
            for var, qis in list(groups.items()) + [("all", [q["qi"] for q in qs])]:
                c = np.concatenate([np.arange(int(offs[i]), int(offs[i + 1])) for i in qis])
                grp[var] = {"k": int(c.size), "iia": iia(mlp.write(swapped(mp["F"][:, c], mp["P"][:, c]))),
                            "random": iia(mlp.write(swapped(mp["Fr"][:, : c.size], mp["Pr"][:, : c.size])))}  # fmt: skip
            reader = {"E": iia(mlp.write(xb, mixed("E"))), "U + T": iia(mlp.write(xb, mixed("UT"))),
                      "U": iia(mlp.write(xb, mixed("U"))), "rms": iia(mlp.write(xb, mixed(""), xs)),
                      "E + rms": iia(mlp.write(xb, mixed("E"), xs)), "U + T + rms": iia(mlp.write(xb, mixed("UT"), xs))}  # fmt: skip
            rows.append(entry | {"single": single, "groups": grp, "reader": reader})
            print(layer, v, f"dep {rel:.3f}", {k: round(g["iia"], 2) for k, g in grp.items()},
                  {k: round(x, 2) for k, x in reader.items()}, flush=True)  # fmt: skip
        (INTERV / "local.json").write_text(json.dumps(rows))


def result_code(Pc: np.ndarray, r: np.ndarray) -> dict[str, float]:
    """Shape of a part's dependence on the result (additions): the table of its mean per result, its
    participation ratio and the number of dimensions holding 90 % of its variance, and the correlation
    between the rows of results `lag` apart (1 = smooth in the result, 0 = unrelated rows)."""
    vals, inv = np.unique(r, return_inverse=True)
    cnt = np.bincount(inv)
    tab = (
        np.stack([np.bincount(inv, weights=Pc[:, j]) for j in range(Pc.shape[1])], 1) / cnt[:, None]
    )
    tab = tab - np.average(tab, axis=0, weights=cnt)
    ev = np.clip(np.linalg.eigvalsh((tab * cnt[:, None]).T @ tab), 0, None)[::-1]
    out = {"pr": float(ev.sum() ** 2 / max((ev**2).sum(), 1e-30)),
           "d90": int(np.searchsorted(np.cumsum(ev) / max(ev.sum(), 1e-30), 0.9) + 1)}  # fmt: skip
    rows = {int(v): tab[i] for i, v in enumerate(vals)}
    for lag in (1, 2, 5, 10, 50, 100):
        pairs = [(rows[v], rows[v + lag]) for v in rows if v + lag in rows]
        A, B = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        out[f"lag{lag}"] = float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))
    return out


def labels(layer: int) -> None:
    """What the explained part E and the unexplained part U of the readers depend on, after the fact:
    the share of each part's variance explained by the operation, a, b and the result (as categories,
    within each operation), by the best additive f_op(a) + g_op(b), and by a linear function of a and b.
    For additions, the shape of its dependence on the result (`result_code`)."""
    comps = comp_table()
    gains = np.load(RESID / "norms.npz")
    ix = np.load(DATASET / "index.npz")
    a, b, op = ix["a"].astype(int), ix["b"].astype(int), ix["op"].astype(int)
    res = np.where(op == 0, a + b, a - b)
    cols = np.flatnonzero(comps["layer"] == layer)
    resid = np.load(DATASET / MODEL / "resid.npy", mmap_mode="r")
    onehot = lambda v: (v[:, None] == np.unique(v)[None]).astype(np.float64)  # noqa: E731
    additive = np.concatenate(
        [
            np.concatenate([onehot(a) * (op == o)[:, None], onehot(b) * (op == o)[:, None]], 1)
            for o in (0, 1)
        ],
        1,
    )
    linear = np.stack(
        [np.ones(len(a)), op, a * (op == 0), b * (op == 0), a * (op == 1), b * (op == 1)], 1
    ).astype(np.float64)
    groups = {"op": op, "a": a, "b": b, "result": op * 1000 + res}
    out = []
    for site, kinds in SITES.items():
        sel = np.flatnonzero(np.isin(comps["kind"][cols], kinds))
        names = [f"L{layer}.{comps['kind'][cols[i]]}.c{comps['cidx'][cols[i]]}" for i in sel]
        V, _ = load_uv(comps, cols[sel])
        G_all = np.stack(V, 1) * gains["ln1" if site == "attn" else "ln2"][layer][:, None]
        for pos, pname in POSITIONS.items():
            key = f"L{layer}_{site}_{pname}".replace("=", "eq")
            if not (INTERV / "maps" / f"{key}.npz").exists():
                continue
            mp = load_map(key)
            G = G_all[:, [names.index(str(n)) for n in mp["names"]]]
            x = np.asarray(resid[read_point_index(layer, site), :, pos], np.float64)
            xc = x - mp["xbar"]
            E = (xc @ mp["F"]) @ mp["B"]
            top = (xc @ mp["FQ"]) @ mp["Q"].T
            parts = {"E": E, "U": top - E, "T": x @ G - mp["mu"] - top}
            tot = float(sum(float(v.var(0).sum()) for v in parts.values()))
            row: dict[str, Any] = {"key": key, "layer": layer, "site": site, "pos": pname}
            for pn, P in parts.items():
                Pc = P - P.mean(0)
                v = float((Pc**2).sum())
                eta = {}
                for gname, g in groups.items():
                    inv = np.unique(g, return_inverse=True)[1]
                    m = (
                        np.stack(
                            [np.bincount(inv, weights=Pc[:, j]) for j in range(Pc.shape[1])], 1
                        )
                        / np.bincount(inv)[:, None]
                    )
                    eta[gname] = float((m[inv] ** 2).sum() / max(v, 1e-30))
                coef = np.linalg.lstsq(additive, Pc, rcond=None)[0]
                eta["f(a) + g(b)"] = float(((additive @ coef) ** 2).sum() / max(v, 1e-30))
                coef = np.linalg.lstsq(linear, Pc, rcond=None)[0]
                eta["linear a, b"] = float(((linear @ coef) ** 2).sum() / max(v, 1e-30))
                row[pn] = {
                    "share": v / len(P) / max(tot, 1e-30),
                    "eta2": eta,
                    "result_code": result_code(Pc[op == 0], res[op == 0]),
                }
            out.append(row)
            print(
                key,
                {
                    pn: (
                        round(row[pn]["share"], 3),
                        {k: round(e, 2) for k, e in row[pn]["eta2"].items()},
                    )
                    for pn in "EU"
                },
                flush=True,
            )
    (INTERV / "labels").mkdir(parents=True, exist_ok=True)
    (INTERV / "labels" / f"L{layer}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "maps":
        maps(int(sys.argv[2]))
    elif cmd == "swap":
        swap()
    elif cmd == "ablate":
        ablate(sys.argv[2])
    elif cmd == "local":
        local()
    elif cmd == "labels":
        labels(int(sys.argv[2]))
    else:
        raise SystemExit(f"unknown command {cmd}")
