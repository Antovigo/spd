"""Patching test (AGENT.md step 7) at token position t, in the alive-only model or the original model.

alive-only: each site's reader activations at t are replaced by their reconstruction from the
  accepted quantities: Yhat(dom(i), r) * ||V_r|| / rho(i) (rho: the RMS of the stream the site
  reads, from the patched run itself).
original: the dense model has no reader components; at t the stream entering the site's norm is
  replaced by x + (Zhat(dom(i)) - x Q) Q^T (its reader-span component set to the reconstruction),
  with Zhat refit on the original model's own stream over the domain.
Variants: recon (in-sample fit), heldout (each domain row predicted by a fit without its fold),
mean (the readers' mean over the domain: the cost of losing what the site carries at t), exact
(the true values; a check, first 3 sites). Sites one at a time, then all together. KL at the last
position against the unpatched model of the same kind; answer accuracy.

    python qpatch.py <t> <alive|original> [--tag name] [--n 2000] [--sites ...] [--extra mod.fn]
"""

import argparse
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from qdata import DATA, dom_index, load
from qfeat import pool
from qfit import Fitter, folds, primary_scheme

from param_decomp.arith_repr.isa import components_model as cm
from param_decomp.arith_repr.rmsnorm.model import NormModel, _dense, _logprobs, _normed

KIND_OF = {"attn": ("q", "k", "v"), "mlp": ("gate", "up")}


class Model(NormModel):
    def __init__(self, dense: bool) -> None:
        super().__init__(dense=dense)
        self.dense = dense
        self.n_layer_cols = [int((self._comp_layer == li).sum()) for li in range(self.n_layer)]

    @property
    def _comp_layer(self) -> np.ndarray:
        return np.load(cm.DATASET / "index.npz")["comp_layer"]

    def masks(
        self, layer: int, rows: np.ndarray
    ) -> np.ndarray:  # alive-only: every alive component on
        return np.ones((len(rows), cm.T, self.n_layer_cols[layer]), np.float32)

    def forward(self, rows: np.ndarray, t: int, inner=None, stream=None, capture=None):  # noqa: ANN001, ANN201
        """Last-position log-probs; inner(li, kind, h, rms, rows) patches reader activations,
        stream(lpos, x, rows) the stream entering norm point lpos; capture: stream points to return at t."""
        assert len(rows) == cm.CHUNK
        x = jnp.asarray(np.stack([[self.embed[int(tk)] for tk in toks] for toks in self.tokens[rows]]), jnp.float32)  # fmt: skip
        B = len(rows)
        caps = {}
        for li in range(self.n_layer):
            for off in (0, 1):
                lpos = 2 * li + off
                if stream is not None:
                    x = stream(lpos, x, rows)
                if capture and lpos in capture:
                    caps[lpos] = np.asarray(x[:, t])
                rms = cm._rms(x, self.eps)
                xin = _normed(x, rms, self.ln[off][li])

                def site(kind: str, xin_: jax.Array) -> jax.Array:
                    if self.dense:
                        return _dense(xin_, self.W[li][kind])  # noqa: B023
                    s = self.sites[li][kind]  # noqa: B023
                    h = cm._project(xin_, s.V)
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
        return np.asarray(_logprobs(xf, self.unembed)), caps


def batched(model: Model, rows: np.ndarray, t: int, **kw) -> tuple[np.ndarray, dict]:  # noqa: ANN003
    lps, caps = [], {}
    for s in range(0, len(rows), cm.CHUNK):
        r = rows[s : s + cm.CHUNK]
        n = len(r)
        lp, cp = model.forward(np.concatenate([r, np.full(cm.CHUNK - n, r[0])]), t, **kw)
        lps.append(lp[:n])
        for key, v in cp.items():
            caps.setdefault(key, []).append(v[:n])
    return np.concatenate(lps), {key: np.concatenate(v) for key, v in caps.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("t", type=int)
    ap.add_argument("model", choices=["alive", "original"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--sites", default=None)
    ap.add_argument("--extra", default=None)
    args = ap.parse_args()
    t, tag = args.t, (f"_{args.tag}" if args.tag else "")
    runs = json.loads((DATA / f"qloop_t{t}{tag}.json").read_text())["sites"]
    accepted = {
        r["site"]: [a["name"] for a in r["accepted"]] for r in runs if not r.get("constant")
    }
    pos = load(t)
    model = Model(dense=args.model == "original")
    rows = np.sort(np.random.default_rng(0).choice(20000, args.n, replace=False))
    answer = model.answer[rows]
    dom_all = dom_index(t, model.op, model.a, model.b)  # domain row of every prompt
    vnorm = np.load(cm.DATASET / "index.npz")["comp_v_norm"]
    comp_layer = model._comp_layer

    # reconstructions in the space the patch acts on: alive -> raw reads (D, n); original -> Zhat (D, k)
    vals: dict[str, dict[str, np.ndarray]] = {}
    if args.model == "alive":
        R = np.load(DATA / f"recon_t{t}{tag}.npz")
        for lab in accepted:
            Y = R[lab + "/Y"].astype(np.float64)
            vals[lab] = {"recon": R[lab + "/Yhat"].astype(np.float64), "heldout": R[lab + "/Yhat_heldout"].astype(np.float64),
                         "mean": np.tile(Y.mean(0), (len(Y), 1)), "exact": Y}  # fmt: skip
    else:
        cands = pool(pos)
        if args.extra:
            mod, fn = args.extra.rsplit(".", 1)
            cands += getattr(importlib.import_module(mod), fn)(pos)
        points = {pos.sites[lab].stream_pos for lab in accepted}
        _, caps = batched(model, pos.rows, t, capture=points)
        fs = folds(pos, primary_scheme(t))
        for lab, A in accepted.items():
            s = pos.sites[lab]
            s_orig = type(s)(
                lab, caps[s.stream_pos].astype(np.float64) @ s.Q, s.W, s.CI, s.cols, s.Q
            )
            F = Fitter(s_orig, cands, fs)
            vals[lab] = {"recon": F.reconstruct(A), "heldout": F.reconstruct(A, heldout=True),
                         "mean": np.tile(s_orig.Z.mean(0), (pos.D, 1)), "exact": s_orig.Z,
                         "r2_stream": np.array(1 - ((s_orig.Z - F.reconstruct(A)) ** 2).sum() / ((s_orig.Z - s_orig.Z.mean(0)) ** 2).sum())}  # fmt: skip

    def hooks(sel: dict[str, str]) -> dict:  # site -> variant
        if args.model == "alive":
            tab = {}
            for lab, variant in sel.items():
                li, pt = int(lab[1:].split(".")[0]), lab.split(".")[1]
                layer_cols = np.flatnonzero(comp_layer == li)
                cols = pos.sites[lab].cols
                for kind in KIND_OF[pt]:
                    gl = layer_cols[model.sites[li][kind].cols]
                    where = {int(c): j for j, c in enumerate(gl)}
                    js = [where[int(c)] for c in cols if int(c) in where]
                    rs = [r for r, c in enumerate(cols) if int(c) in where]
                    if js:
                        tab[(li, kind)] = (
                            np.array(js),
                            jnp.asarray(
                                vals[lab][variant][:, rs] * vnorm[cols[rs]][None, :], jnp.float32
                            ),
                        )

            def inner(li, kind, h, rms, rws):  # noqa: ANN001, ANN202
                if (li, kind) not in tab:
                    return h
                js, V = tab[(li, kind)]
                return h.at[:, t, js].set(V[jnp.asarray(dom_all[rws])] / rms[:, t][:, None])

            return {"inner": inner}
        tabs = {pos.sites[lab].stream_pos: (jnp.asarray(pos.sites[lab].Q, jnp.float32), jnp.asarray(vals[lab][variant], jnp.float32))
                for lab, variant in sel.items()}  # fmt: skip

        def stream(lpos, x, rws):  # noqa: ANN001, ANN202
            if lpos not in tabs:
                return x
            Q, V = tabs[lpos]
            xt = x[:, t]
            return x.at[:, t].set(xt + (V[jnp.asarray(dom_all[rws])] - xt @ Q) @ Q.T)

        return {"stream": stream}

    lp0, _ = batched(model, rows, t)

    def score(lp: np.ndarray) -> dict:
        kl = (np.exp(lp0) * (lp0 - lp)).sum(-1)
        return {"kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)), "acc": float((lp.argmax(-1) == answer).mean())}  # fmt: skip

    only = set(args.sites.split(",")) if args.sites else set(accepted)
    res = {
        "t": t,
        "model": args.model,
        "n_prompts": args.n,
        "baseline_acc": float((lp0.argmax(-1) == answer).mean()),
        "sites": {},
    }
    for i, lab in enumerate(accepted):
        if lab not in only:
            continue
        ent = {
            v: score(batched(model, rows, t, **hooks({lab: v}))[0])
            for v in ("recon", "heldout", "mean")
        }
        if i < 3:
            ent["exact"] = score(batched(model, rows, t, **hooks({lab: "exact"}))[0])
        if "r2_stream" in vals[lab]:
            ent["r2_stream"] = float(vals[lab]["r2_stream"])
        res["sites"][lab] = ent
        print(
            lab,
            {
                k: (round(v["kl_mean"], 4) if isinstance(v, dict) else round(v, 3))
                for k, v in ent.items()
            },
            flush=True,
        )
    for v in ("recon", "heldout", "mean"):
        res["all_sites_" + v] = score(
            batched(model, rows, t, **hooks(dict.fromkeys(accepted, v)))[0]
        )
        print("all sites", v, res["all_sites_" + v], flush=True)
    Path(DATA / f"patch_t{t}_{args.model}{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
