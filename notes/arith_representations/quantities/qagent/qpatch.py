"""Closed patching test (AGENT.md step 7) at token position t, in the alive-only model.

Each site's reader activations at t are replaced by their reconstruction from the accepted
quantities: Yhat(dom(i), r) * ||V_r|| / rhohat(dom(i)), with both the raw reads Yhat and the stream
RMS rhohat (the norm's divisor) reconstructed from the quantities (qloop.py), so the patched
activations are functions of (op, a, b) through the quantities only. The readers that are dead at
t (alive somewhere, but CI <= 0.01 at t on every prompt) are switched off at t in every run,
patched or not. With every site of the position patched, the stream at t then reaches the rest of
the network only through the reconstruction. Writers (o, down) stay on: they read module
internals, not the stream.
Variants: recon (in-sample fit), heldout (nested: each domain row predicted from quantities
selected and fit without its fold), mean (the readers' mean raw read over the domain divided by the
mean rho: the cost of losing what the site carries at t), exact (the stored raw reads and rho of
the unpatched alive-only model: a check of the wiring, KL = that of the dead-reader switch alone).
Sites one at a time (exact for the first 3 sites), then all together (all variants). KL at the last
position against the unpatched alive-only model with every alive reader on (`kl_*`) and against
the unpatched model with the dead-at-t readers off (`kl_vs_masked_*`); answer accuracy.

    python qpatch.py <t> [--tag name] [--n 2000] [--sites ...]
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from qdata import DATA, dom_index, load
from qprep import VW

from param_decomp.arith_repr.isa import components_model as cm
from param_decomp.arith_repr.rmsnorm.model import NormModel, _logprobs, _normed

KIND_OF = {"attn": ("q", "k", "v"), "mlp": ("gate", "up")}
READERS = ("q", "k", "v", "gate", "up")


class Model(NormModel):
    def __init__(self) -> None:
        super().__init__(dense=False)
        self.comp_layer = np.load(cm.DATASET / "index.npz")["comp_layer"]
        self.off: dict[
            tuple[int, str], jax.Array
        ] = {}  # (layer, kind) -> component indices off at t
        self.mask_on = False

    def mask_dead_readers(self, t: int) -> int:
        """Register the readers dead at t (CI <= 0.01 at t on every prompt); returns their number."""
        ci_max = np.load(VW / "ci_positions.npz")["ci_max"][:, t]
        n = 0
        for li in range(self.n_layer):
            layer_cols = np.flatnonzero(self.comp_layer == li)
            for kind in READERS:
                dead = np.flatnonzero(ci_max[layer_cols[self.sites[li][kind].cols]] <= 0.01)
                if len(dead):
                    self.off[(li, kind)] = jnp.asarray(dead)
                    n += len(dead)
        return n

    def forward(self, rows: np.ndarray, t: int, inner=None) -> np.ndarray:  # noqa: ANN001
        """Last-position log-probs; inner(li, kind, h, rms, rows) patches reader activations."""
        assert len(rows) == cm.CHUNK
        x = jnp.asarray(np.stack([[self.embed[int(tk)] for tk in toks] for toks in self.tokens[rows]]), jnp.float32)  # fmt: skip
        B = len(rows)
        for li in range(self.n_layer):
            for off in (0, 1):
                rms = cm._rms(x, self.eps)
                xin = _normed(x, rms, self.ln[off][li])

                def site(kind: str, xin_: jax.Array) -> jax.Array:
                    s = self.sites[li][kind]  # noqa: B023
                    h = cm._project(xin_, s.V)
                    if self.mask_on and (li, kind) in self.off:  # noqa: B023
                        h = h.at[:, t, self.off[(li, kind)]].set(0.0)  # noqa: B023
                    if inner is not None:
                        h = inner(li, kind, h, rms, rows)  # noqa: B023
                    return h @ s.U

                if off == 0:
                    q = site("q", xin).reshape(B, cm.T, self.n_head, self.hd)
                    k = site("k", xin).reshape(B, cm.T, self.n_kv, self.hd)
                    v = site("v", xin).reshape(B, cm.T, self.n_kv, self.hd)
                    out = site("o", cm._attention(q, k, v, self.cos, self.sin))
                else:
                    out = site("down", jax.nn.silu(site("gate", xin)) * site("up", xin))
                x = x + out
        xf = _normed(x[:, -1], cm._rms(x, self.eps)[:, -1], self.final_j)
        return np.asarray(_logprobs(xf, self.unembed))


def batched(model: Model, rows: np.ndarray, t: int, **kw) -> np.ndarray:  # noqa: ANN003
    lps = []
    for s in range(0, len(rows), cm.CHUNK):
        r = rows[s : s + cm.CHUNK]
        n = len(r)
        lps.append(model.forward(np.concatenate([r, np.full(cm.CHUNK - n, r[0])]), t, **kw)[:n])
    return np.concatenate(lps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("t", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--sites", default=None)
    args = ap.parse_args()
    t, tag = args.t, (f"_{args.tag}" if args.tag else "")
    meta = json.loads((DATA / f"qloop_t{t}{tag}.json").read_text())
    assert meta["nested"], (
        "run qloop.py --fold f (all folds) and --combine first: the held-out variant needs it"
    )
    runs = meta["sites"]
    accepted = {
        r["site"]: [a["name"] for a in r["accepted"]] for r in runs if not r.get("constant")
    }
    pos = load(t)
    model = Model()
    rows = np.sort(np.random.default_rng(0).choice(20000, args.n, replace=False))
    answer = model.answer[rows]
    dom_all = dom_index(t, model.op, model.a, model.b)  # domain row of every prompt
    vnorm = np.load(cm.DATASET / "index.npz")["comp_v_norm"]

    # reconstructions of the raw reads (D, n) per site
    R = np.load(DATA / f"recon_t{t}{tag}.npz")
    assert np.array_equal(R["rows"], pos.rows), (
        "recon file and site data cover different domain rows"
    )
    vals: dict[
        str, dict[str, np.ndarray]
    ] = {}  # site -> variant -> activations Y / rho (D, n), raw gauge
    for lab in accepted:
        assert np.array_equal(R[lab + "/cols"], pos.sites[lab].cols), (
            f"{lab}: recon and site data readers differ"
        )
        g = lambda k: R[lab + "/" + k].astype(np.float64)  # noqa: B023, E731
        Y, rho = g("Y"), g("rho")
        vals[lab] = {"recon": g("Yhat") / g("rhohat")[:, None], "heldout": g("Yhat_heldout") / g("rhohat_heldout")[:, None],
                     "mean": np.tile(Y.mean(0) / rho.mean(), (len(Y), 1)), "exact": Y / rho[:, None]}  # fmt: skip

    def hooks(sel: dict[str, str]) -> dict:  # site -> variant
        tab = {}
        for lab, variant in sel.items():
            li, pt = int(lab[1:].split(".")[0]), lab.split(".")[1]
            layer_cols = np.flatnonzero(model.comp_layer == li)
            cols = pos.sites[lab].cols
            for kind in KIND_OF[pt]:
                gl = layer_cols[model.sites[li][kind].cols]
                where = {int(c): j for j, c in enumerate(gl)}
                js = [where[int(c)] for c in cols if int(c) in where]
                rs = [r for r, c in enumerate(cols) if int(c) in where]
                if js:
                    V = vals[lab][variant][:, rs] * vnorm[cols[rs]][None, :]
                    tab[(li, kind)] = (np.array(js), jnp.asarray(V, jnp.float32))

        def inner(li, kind, h, rms, rws):  # noqa: ANN001, ANN202
            if (li, kind) not in tab:
                return h
            js, V = tab[(li, kind)]
            return h.at[:, t, js].set(
                V[jnp.asarray(dom_all[rws])]
            )  # not divided by the live rms: closed

        return {"inner": inner}

    lp0 = batched(model, rows, t)  # every alive reader on, unpatched
    n_dead = model.mask_dead_readers(t)
    model.mask_on = True
    lpm = batched(model, rows, t)  # dead-at-t readers off, unpatched

    def score(lp: np.ndarray) -> dict:
        kl = (np.exp(lp0) * (lp0 - lp)).sum(-1)
        klm = (np.exp(lpm) * (lpm - lp)).sum(-1)
        return {"kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)), "acc": float((lp.argmax(-1) == answer).mean()),
                "kl_vs_masked_mean": float(klm.mean()), "kl_vs_masked_median": float(np.median(klm))}  # fmt: skip

    only = set(args.sites.split(",")) if args.sites else set(accepted)
    res = {
        "t": t,
        "model": "alive",
        "n_prompts": args.n,
        "baseline_acc": float((lp0.argmax(-1) == answer).mean()),
        "dead_readers_off": n_dead,
        "masked_only": score(lpm),
        "sites": {},
    }
    print(f"{n_dead} dead-at-t readers off; unpatched:", res["masked_only"], flush=True)
    for i, lab in enumerate(accepted):
        if lab not in only:
            continue
        ent = {
            v: score(batched(model, rows, t, **hooks({lab: v})))
            for v in ("recon", "heldout", "mean")
        }
        if i < 3:
            ent["exact"] = score(batched(model, rows, t, **hooks({lab: "exact"})))
        res["sites"][lab] = ent
        print(lab, {k: round(v["kl_mean"], 4) for k, v in ent.items()}, flush=True)
    for v in ("exact", "recon", "heldout", "mean"):
        res["all_sites_" + v] = score(batched(model, rows, t, **hooks(dict.fromkeys(accepted, v))))
        print("all sites", v, res["all_sites_" + v], flush=True)
    Path(DATA / f"patch_t{t}_alive{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
