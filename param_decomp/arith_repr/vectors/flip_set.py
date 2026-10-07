"""The set of components that flips b on subtraction, found causally, and what happens to the flip when
they are removed with and without the weight delta.

    python -m param_decomp.arith_repr.vectors.flip_set search    # GPU -> OUT/flip/set.json
    python -m param_decomp.arith_repr.vectors.flip_set evaluate  # GPU -> OUT/flip/eval_<cond>.npz, eval.json

Flip measure. At `=`, the readers of block l (its gate and up V times the norm gain, G_l (d, n_l))
see the raw stream x before block l's MLP: r = x G_l. For op o and b in 1..100, ybar_o(b) is the mean
of r over the 100 prompts (o, b), with its mean over b and its linear trend in b removed. Under
b -> -b (mod 100, 100 = 0) ybar splits into a b-even part and a b-odd part (the sine axes, which a
mirror reverses). With A = the b-odd part on add and S on sub:
    Phi_l = sum_b |(A - S) / 2|^2   (b-odd, op-odd: the flip),
    Sig_l = sum_b |(A + S) / 2|^2   (b-odd, same on both ops),
    Ev_l  = the b-even part's energy, both ops.
A pure mirror has Sig = 0, no mirror Phi = 0; a code present on one op only has Phi = Sig.
The search target is M = mean_l Phi_l / Phi_l(base) + KEEP_W mean_l relu(|Ev_l / Ev_l(base) - 1| - BAND),
l in 16..31: the flip must go while b's b-even code (mostly the L15H13 copy) stays. (set.json, the
masked components model's set, was found with an earlier one-sided penalty; commit 56d9ec89f.)

Models. `comp`: the components-only model of the dataset (alive components, masks of `ATLAS/masks`,
no delta). `dense`: the original weights. A scale s[c, t] per alive component c and position t
multiplies the component's contribution; in `comp` the mask becomes m * s, in `dense` the site
computes x W^T - sum_c (1 - s[c, t]) (x V_c) U_c, i.e. removing c subtracts its rank-one term and
keeps the delta.

search: greedy on the components model. Each round computes dM/ds at the current s (two passes:
forward for the group sums, then one VJP per chunk) and removes the BATCH (component, position)
pairs with the largest predicted drop that are not yet removed; the round's M is measured exactly.
Stops when M <= TARGET. Then pruning: removed pairs whose restoration is predicted to raise M by
less than PRUNE x the margin are restored in batches, keeping a batch only if M stays <= TARGET.

evaluate: comp / dense, base / set removed (and comp / dense with only the set's `=` pairs removed).
Saves group sums at `=` of the stream before every MLP (by (op, b), (op, a), (op, (a + b) mod 100),
(op, (a - b) mod 100)), the group sums by (op, b) of every sublayer write and of every alive
component's scaled inner activation, and last-position log-probs metrics."""

import json
import sys
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.isa.components_model import (
    CHUNK,
    KINDS,
    ComponentsModel,
    _attention,
    kl,
)
from param_decomp.arith_repr.vectors.common import AUTOINTERP, RESID, comp_table
from param_decomp.arith_repr.vectors.flip_components import DIR

jax.config.update("jax_default_matmul_precision", "highest")

READ = tuple(range(16, 32))
PERM = (98 - np.arange(100)) % 100  # row of -b (mod 100) for the row of b (b = row + 1)
TARGET = 0.05
KEEP_W, BAND = 2.0, 0.1  # penalty when b's b-even code drifts more than BAND from base
BATCH = 8
PRUNE = 0.2
T = 5


def flip_stats(r: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """r (200, n): group SUMS by op * 100 + b - 1 -> (Phi, Sig, Ev) scalars."""
    y = r.reshape(2, 100, -1) / 100.0
    y = y - y.mean(1, keepdims=True)
    v = jnp.arange(1, 101, dtype=jnp.float32) - 50.5
    slope = jnp.einsum("b,obn->on", v, y) / (v**2).sum()
    y = y - slope[:, None] * v[None, :, None]
    odd = (y - y[:, PERM]) / 2
    even = (y + y[:, PERM]) / 2
    phi = (((odd[0] - odd[1]) / 2) ** 2).sum()
    sig = (((odd[0] + odd[1]) / 2) ** 2).sum()
    return phi, sig, (even**2).sum()


class Net:
    """Pure-function forward over chunks of prompts, components-only or dense, with scales s."""

    def __init__(self, dense: bool, masked_removal: bool = False, alive: bool = False) -> None:
        """masked_removal (dense only): removing a pair subtracts only the component's ACTIVE
        contribution m (x V_c) U_c (the dataset's mask m), leaving its inactive contribution and the
        delta in place. alive (components only): every alive component is on everywhere (m = 1)."""
        self.masked_removal = masked_removal
        self.alive = alive
        cm = ComponentsModel()
        self.cm = cm
        self.dense = dense
        self.n_cols = [int(np.load(_mask_path(li), mmap_mode="r").shape[2]) for li in range(32)]
        ln2 = cm.ln2
        self.G = {li: jnp.concatenate([cm.sites[li]["gate"].V, cm.sites[li]["up"].V], 1)
                  * jnp.asarray(ln2[li])[:, None] for li in READ}  # fmt: skip
        self.sites = [
            {k: (s.V, s.U, jnp.asarray(s.cols)) for k, s in cm.sites[li].items()}
            for li in range(32)
        ]
        self.ln = (jnp.asarray(cm.ln1), jnp.asarray(cm.ln2))
        self.W: list[dict[str, jax.Array]] = []
        if dense:
            w = Weights()
            for li in range(32):
                self.W.append({k: jnp.asarray(w.get(f"model.layers.{li}.{s}.weight"), jnp.bfloat16)
                               for k, s in KINDS.items()})  # fmt: skip
        # weights go in as arguments: a jit closing over them would embed them as constants
        self.P: dict[str, Any] = {"sites": self.sites, "W": self.W, "G": self.G, "ln": self.ln,
                                  "unembed": cm.unembed, "final": cm.final_j, "cos": cm.cos, "sin": cm.sin,
                                  "dU": [{} for _ in range(32)], "clamp": {}}  # fmt: skip
        self._light = jax.jit(lambda P, s, x0, m, oh: self._run(P, s, x0, m, oh, None)[0])
        self._grad = jax.jit(
            lambda P, s, x0, m, oh, cot: jax.vjp(
                lambda ss: self._run(P, ss, x0, m, oh, None)[0], s
            )[1](cot)[0]
        )
        self._full = jax.jit(lambda P, s, x0, m, oh, ohs: self._run(P, s, x0, m, oh, ohs))
        self._lp = jax.jit(lambda P, s, x0, m, oh: self._run(P, s, x0, m, oh, None, True)[1]["lp"])

    def scales_one(self) -> list[jax.Array]:
        return [jnp.ones((n, T), jnp.float32) for n in self.n_cols]

    def _run(
        self,
        P: dict[str, Any],
        s: list[jax.Array],
        x: jax.Array,
        m: list[jax.Array],
        oh: jax.Array,
        ohs: jax.Array | None,
        want_lp: bool = False,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Light captures: reader-view group sums `r<l>` (200, n_l). Full (ohs = (B, 800) one-hot of
        the four groupings): `x<l>` (800, d) before every MLP, `w<p>` (200, d) every sublayer write,
        `h<l>.<kind>` (200, n_site) scaled inners, `lp` last-position log-probs."""
        cm = self.cm
        B = x.shape[0]
        light: dict[str, jax.Array] = {}
        full: dict[str, Any] = {}

        def gsum(g: jax.Array, v: jax.Array) -> jax.Array:
            return jnp.tensordot(g, v, axes=(0, 0))

        for li in range(32):
            sc = s[li].T[None]  # (1, T, n_cols)
            mm = m[li] * sc

            def site(kind: str, xin: jax.Array) -> jax.Array:
                V, U, cols = P["sites"][li][kind]  # noqa: B023
                h = xin @ V
                if self.dense:
                    y = xin @ P["W"][li][kind].astype(jnp.float32).T  # noqa: B023
                    off_ = 1.0 - sc[:, :, cols]  # noqa: B023
                    if self.masked_removal:
                        off_ = off_ * m[li][:, :, cols]  # noqa: B023
                    y = y - (h * off_) @ U
                    hs = h * (1.0 - off_)
                else:
                    hs = h * mm[:, :, cols]  # noqa: B023
                    y = hs @ U
                if kind in P["dU"][li]:  # noqa: B023
                    j, dU = P["dU"][li][kind]  # noqa: B023
                    y = y + hs[:, :, j] @ dU
                if ohs is not None:
                    full[f"h{li}.{kind}"] = jnp.einsum("bg,btn->tgn", ohs, hs)  # noqa: B023
                    opoh = ohs[:, :200].reshape(-1, 2, 100).sum(-1)  # noqa: B023
                    full[f"q{li}.{kind}"] = jnp.einsum("bo,btn->otn", opoh, hs * hs)  # noqa: B023
                return y

            def attn(xin: jax.Array) -> jax.Array:
                q = site("q", xin).reshape(B, T, cm.n_head, cm.hd)
                k = site("k", xin).reshape(B, T, cm.n_kv, cm.hd)
                v = site("v", xin).reshape(B, T, cm.n_kv, cm.hd)
                return site("o", _attention(q, k, v, P["cos"], P["sin"]))

            def mlp(xin: jax.Array) -> jax.Array:
                return site("down", jax.nn.silu(site("gate", xin)) * site("up", xin))

            for off, fn in ((0, attn), (1, mlp)):
                if off == 1 and li in P["clamp"]:  # remove a fixed per-(op, b) vector at `=`
                    x = x.at[:, 4].add(-(oh @ P["clamp"][li]))
                if off == 1 and li in READ:
                    light[f"r{li}"] = gsum(oh, x[:, 4] @ P["G"][li])
                if off == 1 and ohs is not None:
                    full[f"x{li}"] = gsum(ohs, x[:, 4])
                rms = jnp.sqrt((x * x).mean(-1) + cm.eps)
                xin = x / rms[..., None] * P["ln"][off][li]
                out = jax.checkpoint(fn)(xin) if ohs is None else fn(xin)
                if ohs is not None:
                    full[f"w{2 * li + off}"] = gsum(oh, out[:, 4])
                x = x + out
        if ohs is not None or want_lp:
            xl = x[:, -1]
            xf = xl / jnp.sqrt((xl * xl).mean(-1, keepdims=True) + cm.eps) * P["final"]
            full["lp"] = jax.nn.log_softmax(xf @ P["unembed"].T, -1)
        return light, full

    def chunk_inputs(self, rows: np.ndarray, full: bool = False) -> tuple[Any, ...]:
        cm = self.cm
        x0 = jnp.asarray(np.stack([[cm.embed[int(t)] for t in toks] for toks in cm.tokens[rows]]), jnp.float32)  # fmt: skip
        if self.alive:
            m = [jnp.ones((len(rows), T, n), jnp.float32) for n in self.n_cols]
        else:
            m = [jnp.asarray(cm.masks(li, rows)) for li in range(32)]
        op, a, b = cm.op[rows], cm.a[rows], cm.b[rows]
        oh = jnp.asarray(np.eye(200, dtype=np.float32)[op * 100 + b - 1])
        if not full:
            return x0, m, oh
        grp = [
            op * 100 + b - 1,
            op * 100 + a - 1,
            op * 100 + (a + b) % 100,
            op * 100 + (a - b) % 100,
        ]
        ohs = np.concatenate([np.eye(200, dtype=np.float32)[g] for g in grp], 1)
        return x0, m, oh, jnp.asarray(ohs)

    def light(self, s: list[jax.Array]) -> dict[str, np.ndarray]:
        tot: dict[str, jax.Array] = {}
        for c0 in range(0, 20000, CHUNK):
            out = self._light(self.P, s, *self.chunk_inputs(np.arange(c0, c0 + CHUNK)))
            tot = {k: tot[k] + v if k in tot else v for k, v in out.items()}
        return {k: np.asarray(v) for k, v in tot.items()}

    def grad(self, s: list[jax.Array], cot: dict[str, jax.Array]) -> list[np.ndarray]:
        g: list[jax.Array] | None = None
        for c0 in range(0, 20000, CHUNK):
            gi = self._grad(self.P, s, *self.chunk_inputs(np.arange(c0, c0 + CHUNK)), cot)
            g = gi if g is None else [a + b for a, b in zip(g, gi, strict=True)]
        assert g is not None
        return [np.asarray(x) for x in g]

    def logprobs(self, s: list[jax.Array], rows: np.ndarray) -> np.ndarray:
        """Last-position log-probs of `rows` (a multiple of CHUNK), with the edits in P["dU"]:
        per layer, kind -> (columns j of the site, dU (len(j), d_out)) added to those components' U."""
        out = []
        for c0 in range(0, len(rows), CHUNK):
            out.append(np.asarray(self._lp(self.P, s, *self.chunk_inputs(rows[c0 : c0 + CHUNK]))))
        return np.concatenate(out)

    def full(self, s: list[jax.Array]) -> dict[str, np.ndarray]:
        tot: dict[str, Any] = {}
        lps = []
        for c0 in range(0, 20000, CHUNK):
            _, out = self._full(self.P, s, *self.chunk_inputs(np.arange(c0, c0 + CHUNK), full=True))
            lps.append(np.asarray(out.pop("lp")))
            tot = {k: tot[k] + v if k in tot else v for k, v in out.items()}
        res = {k: np.asarray(v) for k, v in tot.items()}
        res["lp"] = np.concatenate(lps)
        return res


def _mask_path(li: int):  # noqa: ANN202
    from param_decomp.arith_repr.isa.atlas import ATLAS

    return ATLAS / "masks" / f"L{li}.npy"


def measure(r: dict[str, np.ndarray]) -> dict[int, tuple[float, float, float]]:
    out = {}
    for li in READ:
        phi, sig, ev = flip_stats(jnp.asarray(r[f"r{li}"]))
        out[li] = (float(phi), float(sig), float(ev))
    return out


def objective(base: dict[int, tuple[float, float, float]]):  # noqa: ANN201
    phi0 = jnp.asarray([base[li][0] for li in READ])
    ev0 = jnp.asarray([base[li][2] for li in READ])

    def M(r: dict[str, jax.Array]) -> jax.Array:
        st = [flip_stats(r[f"r{li}"]) for li in READ]
        phi = jnp.stack([x[0] for x in st]) / phi0
        ev = jnp.stack([x[2] for x in st]) / ev0
        return phi.mean() + KEEP_W * jax.nn.relu(jnp.abs(ev - 1.0) - BAND).mean()

    return M


def pair_names(net: Net) -> list[list[str]]:
    """Per layer, the component name of each mask column."""
    out = []
    for li in range(32):
        names = [""] * net.n_cols[li]
        for s in net.cm.sites[li].values():
            for j, c in enumerate(s.cols):
                names[int(c)] = s.names[j]
        out.append(names)
    return out


def set_path(model: str):  # noqa: ANN201
    return DIR / ("set.json" if model == "comp" else f"set_{model}.json")


def search(model: str = "comp") -> None:
    net = Net(dense=False, alive=model == "alive")
    names = pair_names(net)
    s = net.scales_one()
    t0 = time.time()
    r = net.light(s)
    base = measure(r)
    M = objective(base)
    print("base Phi/Sig/Ev per reader layer:", {li: tuple(round(x, 3) for x in v) for li, v in base.items()}, flush=True)  # fmt: skip
    removed: list[tuple[int, int, int]] = []  # (layer, column, position)
    history = []
    Mv = float(M({k: jnp.asarray(v) for k, v in r.items()}))
    banned: set[tuple[int, int, int]] = set()
    batch = BATCH
    while Mv > TARGET:
        rj = {k: jnp.asarray(v) for k, v in r.items()}
        cot = jax.grad(M)(rj)
        g = net.grad(s, cot)
        # predicted change of M when a pair at s = 1 goes to 0 is -g
        cand = []
        for li in range(32):
            sl = np.asarray(s[li])
            gi = np.where(sl > 0.5, g[li], -np.inf)
            for col, pos in zip(
                *np.unravel_index(np.argsort(-gi, axis=None)[: 4 * BATCH], gi.shape), strict=True
            ):
                if (li, int(col), int(pos)) not in banned:
                    cand.append((float(gi[col, pos]), li, int(col), int(pos)))
        cand.sort(reverse=True)
        pick = [c for c in cand[:batch] if c[0] > 0]
        if not pick:
            print("no candidate predicted to lower M; stopping", flush=True)
            break
        s_try = list(s)
        for _, li, col, pos in pick:
            s_try[li] = s_try[li].at[col, pos].set(0.0)
        r_try = net.light(s_try)
        M_try = float(M({k: jnp.asarray(v) for k, v in r_try.items()}))
        if M_try >= Mv:  # the linear prediction failed: retry smaller, ban a single failing pick
            print(f"   rejected {len(pick)} picks (M {M_try:.4f} >= {Mv:.4f})", flush=True)
            if len(pick) == 1:
                banned.add(pick[0][1:])
            batch = max(1, len(pick) // 2)
            continue
        s, r, Mv = s_try, r_try, M_try
        removed.extend(c[1:] for c in pick)
        batch = BATCH
        history.append(
            {
                "n": len(removed),
                "M": Mv,
                "added": [(names[li][col], pos, gv) for gv, li, col, pos in pick],
            }
        )
        print(f"[{time.time() - t0:.0f}s] removed {len(removed)}: M {Mv:.4f}; added "
              + ", ".join(f"{names[li][col]}@{pos} ({gv:.3f})" for gv, li, col, pos in pick), flush=True)  # fmt: skip
        if len(removed) > 400:
            print("over 400 pairs; stopping", flush=True)
            break
    # pruning
    for rnd in range(4):
        rj = {k: jnp.asarray(v) for k, v in r.items()}
        g = net.grad(s, jax.grad(M)(rj))
        est = sorted(
            ((float(g[li][col, pos]), (li, col, pos)) for li, col, pos in removed),
            key=lambda z: z[0],
        )
        margin = TARGET - Mv
        batch = [p for gv, p in est if gv < PRUNE * margin][: max(1, len(removed) // 4)]
        if not batch:
            break
        s_try = [x for x in s]
        for li, col, pos in batch:
            s_try[li] = s_try[li].at[col, pos].set(1.0)
        r_try = net.light(s_try)
        M_try = float(M({k: jnp.asarray(v) for k, v in r_try.items()}))
        ok = M_try <= TARGET
        print(
            f"prune round {rnd}: restore {len(batch)} -> M {M_try:.4f} {'kept' if ok else 'rejected'}",
            flush=True,
        )
        if ok:
            s, r, Mv = s_try, r_try, M_try
            removed = [p for p in removed if p not in batch]
    final = measure(r)
    out = {
        "removed": [
            {"layer": li, "col": col, "pos": pos, "name": names[li][col]}
            for li, col, pos in removed
        ],
        "M": Mv,
        "base": {str(k): v for k, v in base.items()},
        "final": {str(k): v for k, v in final.items()},
        "history": history,
    }
    set_path(model).write_text(json.dumps(out, indent=1))
    print(f"final set: {len(removed)} pairs, M {Mv:.4f}", flush=True)


def evaluate(variant: str = "all") -> None:
    """The masked components model's set (set.json): components model, dense with the set's active
    contributions removed (masked), dense with its full rank-one terms removed. variant
    `masked_only` runs the masked one only."""
    S = json.loads((DIR / "set.json").read_text())["removed"]
    res: dict[str, Any] = {}
    for dense, masked in ((True, True), (False, False), (True, False)):
        if variant == "masked_only" and not masked:
            continue
        net = Net(dense=dense, masked_removal=masked)
        conds = {"base": [], "set": S, "set_eq": [p for p in S if p["pos"] == 4]}
        base_lp = None
        for cond, pairs in conds.items():
            s = net.scales_one()
            for p in pairs:
                s[p["layer"]] = s[p["layer"]].at[p["col"], p["pos"]].set(0.0)
            out = net.full(s)
            lp = out.pop("lp")
            if base_lp is None:
                base_lp = lp
            key = f"{'dense' if dense else 'comp'}{'_masked' if masked else ''}_{cond}"
            light = {f"r{li}": out[f"x{li}"][:200] @ np.asarray(net.G[li]) for li in READ}
            st = measure(light)
            top = lp.argmax(-1)
            cm = net.cm
            d = kl(base_lp, lp)
            mets: dict[str, Any] = {"stats": {str(k): v for k, v in st.items()}}
            for o, nm in ((0, "add"), (1, "sub")):
                sel = (cm.op == o) & ((o == 0) | (cm.a >= cm.b))
                mets[f"kl_{nm}"] = float(d[cm.op == o].mean())
                mets[f"acc_{nm}"] = float((top[sel] == cm.answer[sel]).mean())
            res[key] = mets
            np.savez(DIR / f"eval_{key}.npz", **cast(dict[str, Any], out))
            print(key, {k: v for k, v in mets.items() if k != "stats"},
                  {li: f"{v[0]:.3g}/{v[1]:.3g}/{v[2]:.3g}" for li, v in st.items()}, flush=True)  # fmt: skip
        del net
    prev = json.loads((DIR / "eval.json").read_text()) if (DIR / "eval.json").exists() else {}
    (DIR / "eval.json").write_text(json.dumps(prev | res, indent=1))


def flip_part(r: np.ndarray) -> np.ndarray:
    """(200, ...) group sums by (op, b) -> (100, ...) the b-odd, op-odd part (linear in r)."""
    y = r.reshape(2, 100, *r.shape[1:]) / 100.0
    y = y - y.mean(1, keepdims=True)
    v = (np.arange(1, 101, dtype=np.float64) - 50.5).reshape((1, 100) + (1,) * (r.ndim - 1))
    y = y - (v * y).sum(1, keepdims=True) / (v**2).sum() * v
    odd = (y - y[:, PERM]) / 2
    return (odd[0] - odd[1]) / 2


def line_energy(xg: np.ndarray, G: np.ndarray) -> np.ndarray:
    """(200, d) group sums of one grouping -> energy of its code in the readers' view, per op."""
    y = (xg / 100.0).reshape(2, 100, -1) @ G
    y = y - y.mean(1, keepdims=True)
    return (y**2).sum((1, 2))


def report() -> None:
    from param_decomp.arith_repr.vectors.load import W50, describe, load_reads, load_writes

    S = json.loads((DIR / "set.json").read_text())
    pairs = S["removed"]
    comps = comp_table()
    name_of = np.array(
        [
            f"L{lay}.{k}.c{c}"
            for lay, k, c in zip(comps["layer"], comps["kind"], comps["cidx"], strict=True)
        ]
    )
    col_of = {n: i for i, n in enumerate(name_of)}
    reads = load_reads()
    writes = load_writes()
    wrow = {int(c): i for i, c in enumerate(writes.cols)}
    pos_names = ("BOS", "a", "op", "b", "=")
    print(f"1. the set: {len(pairs)} (component, position) pairs, M {S['M']:.3f} (target {TARGET})")
    from collections import Counter

    cnt = Counter((p["layer"], p["name"].split(".")[1], pos_names[p["pos"]]) for p in pairs)
    by_layer = Counter(p["layer"] for p in pairs)
    print("   per layer:", dict(sorted(by_layer.items())))
    print("   per (layer, kind, position):", dict(sorted(cnt.items())))
    print("\n2. each pair, original-model inner at its position (vector auto-interp read spectra):"
          " flip share = b-odd op-odd energy / inner variance; op share = (add - sub mean)^2 / 4 / (var + that);"
          " top codes per op; mask on-rate at a / op / b / =")  # fmt: skip
    masks = {}
    for p in sorted(pairs, key=lambda q: (q["layer"], q["pos"], q["name"])):
        c = col_of[p["name"]]
        li = p["layer"]
        if li not in masks:
            masks[li] = np.load(_mask_path(li), mmap_mode="r")
        on = [float(np.asarray(masks[li][:, t, p["col"]], np.float32).mean()) for t in range(1, 5)]
        if p["pos"] == 0:
            print(f"   {p['name']} @BOS: on {on}")
            continue
        pi = p["pos"] - 1
        R = reads.R[c, pi]  # (2, 4, 50)
        var = reads.var[c, pi].mean()
        flip = float((W50 * ((R[0, 1].imag - R[1, 1].imag) / 2) ** 2).sum() / max(var, 1e-12))
        dm = (reads.mean[c, pi, 0] - reads.mean[c, pi, 1]) / 2
        opsh = float(dm**2 / (var + dm**2))
        txt = " | ".join(
            f"{('add', 'sub')[o]}: {describe(R[o], reads.energy[c, pi, o], 3)}" for o in (0, 1)
        )
        wtxt = ""
        if c in wrow:
            j = wrow[c]
            Wc = writes.W[j, pi]
            en = (
                np.abs(Wc) ** 2
                * W50
                / max(float((np.abs(Wc) ** 2 * W50).sum((-1, -2)).max()), 1e-12)
            )
            wtxt = " || writes " + " | ".join(
                f"{('add', 'sub')[o]}: {describe(Wc[o], en[o], 2)}" for o in (0, 1)
            )
        print(
            f"   {p['name']} @{pos_names[p['pos']]}: flip {flip:.2f}, op {opsh:.2f}, on {np.round(on, 2).tolist()}; {txt}{wtxt}"
        )

    print("\n3. evaluation: last-position KL vs the same model's base, accuracy (sub: a >= b);"
          " flip Phi / same Sig / even Ev per reader layer relative to base")  # fmt: skip
    ev = json.loads((DIR / "eval.json").read_text())
    for key, m in ev.items():
        mdl = key.split("_")[0]
        b0 = ev[f"{mdl}_base"]["stats"]
        rel = {
            li: tuple(m["stats"][li][i] / max(b0[li][i], 1e-12) for i in range(3))
            for li in m["stats"]
        }
        print(
            f"   {key}: KL add {m['kl_add']:.4f} sub {m['kl_sub']:.4f}, acc add {m['acc_add']:.3f} sub {m['acc_sub']:.3f}"
        )
        print(
            "      " + " ".join(f"L{li} {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for li, v in rel.items())
        )

    print("\n4. other codes at `=` in the readers' view (energy of the a / b / (a+b) / (a-b) group means,"
          " per op, relative to the same model's base), at L16 / L18 / L24 / L31")  # fmt: skip
    net_G = {}
    cm_uv = np.load(AUTOINTERP / "uv_alive.npz")
    ln2 = np.load(RESID / "norms.npz")["ln2"]
    for li in (16, 18, 24, 31):
        Vs = []
        for kind in ("gate", "up"):
            key = f"layers.{li}.mlp.{kind}_proj"
            Vs.append(cm_uv[key + ".V"])
        net_G[li] = np.concatenate(Vs, 1) * ln2[li][:, None]
    Z = {k: np.load(DIR / f"eval_{k}.npz") for k in ev}
    for key in ev:
        mdl = key.split("_")[0]
        row = []
        for li in (16, 18, 24, 31):
            x = Z[key][f"x{li}"]
            x0 = Z[f"{mdl}_base"][f"x{li}"]
            rel = [line_energy(x[g * 200 : (g + 1) * 200], net_G[li]) / line_energy(x0[g * 200 : (g + 1) * 200], net_G[li])
                   for g in (1, 0, 2, 3)]  # fmt: skip
            row.append(
                f"L{li} "
                + " ".join(
                    f"{nm} {r[0]:.2f}/{r[1]:.2f}"
                    for nm, r in zip(("a", "b", "a+b", "a-b"), rel, strict=True)
                )
            )
        print(f"   {key}: " + "; ".join(row))

    print("\n5. who writes the flip that remains: share of O (b-odd op-odd part in the readers' view) per"
          " sublayer, split into alive components and the delta (dense) ")  # fmt: skip
    for key in ev:
        z = Z[key]
        for li in (16, 18, 20, 24):
            if li not in net_G:
                Vs = [cm_uv[f"layers.{li}.mlp.{k}_proj.V"] for k in ("gate", "up")]
                net_G[li] = np.concatenate(Vs, 1) * ln2[li][:, None]
            Gl = net_G[li]
            Ofl = flip_part(z[f"x{li}"][:200] @ Gl)
            den = float((Ofl**2).sum())
            parts = []
            for p in range(2 * li + 1):
                w = z[f"w{p}"]
                lw, kind = p // 2, ("o" if p % 2 == 0 else "down")
                suffix = "self_attn.o_proj" if kind == "o" else "mlp.down_proj"
                U = cm_uv[f"layers.{lw}.{suffix}.U"]
                ids = cm_uv[f"layers.{lw}.{suffix}.ids"]
                h = z[f"h{lw}.{kind}"]
                h = h[4, :200] if h.ndim == 3 else h
                order = _site_order(lw, suffix, ids)
                comp_w = h @ U[order]
                sc = float((flip_part(comp_w @ Gl) * Ofl).sum() / den)
                sd = float((flip_part((w - comp_w) @ Gl) * Ofl).sum() / den)
                parts.append((f"L{lw}.{'attn' if kind == 'o' else 'mlp'}", sc, sd))
            parts.sort(key=lambda t: -abs(t[1]) - abs(t[2]))
            print(f"   {key} L{li} readers (|O|^2 {den:.3g}): "
                  + ", ".join(f"{n} comps {a:+.2f} delta {d:+.2f}" for n, a, d in parts[:5]))  # fmt: skip


def _site_order(layer: int, suffix: str, ids: np.ndarray) -> list[int]:
    """Rows of the site's alive U in the components model's column order (that of the h captures)."""

    comps = comp_table()
    lcols = np.flatnonzero(comps["layer"] == layer)
    sel = np.flatnonzero(comps["site"][lcols] == f"layers.{layer}.{suffix}")
    pos = {int(i): j for j, i in enumerate(ids)}
    return [pos[int(comps["cidx"][lcols[s]])] for s in sel]


if __name__ == "__main__":
    cast(Any, {"search": search, "evaluate": evaluate, "report": report})[sys.argv[1]](
        *sys.argv[2:3]
    )
