"""Step 7 of PROTOCOL.md at token a: patch each site's reader activations at t = 1 with their
reconstruction from the accepted quantities (v3_run.py) and compare the alive-only model's
last-position output.

Patched value of reader r on prompt i: Yhat(a_i, r) * ||V_r|| / rho(i), with Yhat the raw-read
reconstruction (gain-folded unit read direction) and rho(i) the RMS of the stream the site reads
in the patched run itself (the components model's inner activation is x / rho * gamma . V_r).
References: the readers' mean raw read over a ("mean patch") and the exact raw read ("exact
patch", a check: KL should be ~0). Also all sites patched together.

    python v3_patch.py [n_prompts]
"""

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa import components_model as cm
from param_decomp.arith_repr.vectors.common import DATASET

HERE = Path(__file__).parent
KIND_OF = {"attn": ("q", "k", "v"), "mlp": ("gate", "up")}


class AliveOnly(cm.ComponentsModel):
    """Every alive component on (masks of ones; the per-prompt atlas masks are not needed)."""

    def masks(self, layer: int, rows: np.ndarray) -> np.ndarray:
        n = int((np.load(DATASET / "index.npz")["comp_layer"] == layer).sum())
        return np.ones((len(rows), cm.T, n), np.float32)


def main() -> None:
    n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    model = AliveOnly()
    ix = np.load(DATASET / "index.npz")
    vnorm = ix["comp_v_norm"]
    comp_layer = ix["comp_layer"]
    R = np.load(HERE / "v3_recon.npz")
    sites = sorted(
        {k.split("/")[0] for k in R.files},
        key=lambda s: (int(s[1:].split(".")[0]), s.endswith("mlp")),
    )
    rows = np.sort(np.random.default_rng(0).choice(20000, n_prompts, replace=False))
    answer = model.answer[rows]

    # (layer, kind) -> list of (column in the site's kind block, reader index in the site) per site
    def mapping(site: str) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
        li, pt = int(site[1:].split(".")[0]), site.split(".")[1]
        layer_cols = np.flatnonzero(comp_layer == li)
        cols = R[site + "/cols"]
        out = {}
        for kind in KIND_OF[pt]:
            gl = layer_cols[model.sites[li][kind].cols]
            pos = {int(c): j for j, c in enumerate(gl)}
            js, rs = [], []
            for r, c in enumerate(cols):
                if int(c) in pos:
                    js.append(pos[int(c)])
                    rs.append(r)
            if js:
                out[(li, kind)] = (np.array(js), np.array(rs))
        return out

    def run(
        patches: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """patches[(layer, kind)] = (columns in the kind block, values (100, m) raw reads, vnorm (m,))."""

        def hook(li, kind, h, rms, idx):  # noqa: ANN001, ANN202
            if (li, kind) not in patches:
                return h
            js, vals, vn = patches[(li, kind)]
            a_idx = model.a[rows[idx]] - 1
            new = jnp.asarray(vals[a_idx] * vn[None, :], h.dtype) / rms[:, 1][:, None]
            return h.at[:, 1, js].set(new)

        lp, _ = model.forward(rows, inner=hook)
        return lp

    lp0, _ = model.forward(rows)

    def score(lp: np.ndarray) -> dict:
        kl = (np.exp(lp0) * (lp0 - lp)).sum(-1)
        return {"kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
                "acc": float((lp.argmax(-1) == answer).mean())}  # fmt: skip

    res = {
        "n_prompts": n_prompts,
        "baseline_acc": float((lp0.argmax(-1) == answer).mean()),
        "sites": {},
    }
    joint = {"recon": {}, "mean": {}}
    for s_i, site in enumerate(sites):
        Yhat, Y = R[site + "/Yhat"], R[site + "/Y"]
        ent = {}
        for variant, vals in (
            ("recon", Yhat),
            ("mean", np.tile(Y.mean(0), (100, 1))),
            ("exact", Y),
        ):
            if variant == "exact" and s_i >= 4:
                continue
            p = {}
            for key, (js, rs) in mapping(site).items():
                cols = R[site + "/cols"][rs]
                p[key] = (js, vals[:, rs], vnorm[cols])
                if variant in joint:
                    joint[variant][key] = p[key]
            ent[variant] = score(run(p))
        res["sites"][site] = ent
        print(site, {k: round(v["kl_mean"], 4) for k, v in ent.items()}, flush=True)
    for variant, p in joint.items():
        res["all_sites_" + variant] = score(run(p))
        print("all sites", variant, res["all_sites_" + variant], flush=True)
    (HERE / "v3_patch.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
