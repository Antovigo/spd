"""Step 7 of PROTOCOL.md in the ORIGINAL model (dense Llama-3.1-8B) at token a.

The original model has no reader components to overwrite, so a site is patched in the residual
stream: at token position t = 1, the stream entering the site's norm, x, is replaced by
x + (Zhat(a) - x Q) Q^T, i.e. its component in the site's reader span S (orthonormal basis Q) is
set to the reconstruction Zhat from the site's accepted quantities (v3_token_a.json); the part of x
outside S is kept. Zhat is refit on the original model's own stream at token a (captured here on
the 100 prompts op = +, b = 1, a = 1..100). References: Zhat = the mean over a ("mean patch") and
Zhat = the original stream itself ("exact", a check). Sites one at a time, and all together.

    python v3_patch_full.py [n_prompts]
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from v2 import reader_dirs, span_basis
from v3 import Site, fit
from v3_run import pool

from param_decomp.arith_repr.isa.components_model import CHUNK, T, _attention, _rms
from param_decomp.arith_repr.rmsnorm.model import FINAL, NormModel, _dense, _logprobs, _normed

HERE = Path(__file__).parent
ROWS100 = np.arange(100) * 100  # op = +, b = 1, a = 1..100


class Dense(NormModel):
    def forward_patched(self, rows: np.ndarray, hook=None, capture: set[int] | None = None):  # noqa: ANN001, ANN201
        """Original model; hook(t, x, rows) may replace the stream entering norm point t."""
        assert len(rows) == CHUNK
        x = jnp.asarray(np.stack([[self.embed[int(tk)] for tk in toks] for toks in self.tokens[rows]]), jnp.float32)  # fmt: skip
        B = len(rows)
        caps = {}
        for li in range(self.n_layer):
            W = self.W[li]
            for off in (0, 1):
                t = 2 * li + off
                if hook is not None:
                    x = hook(t, x, rows)
                if capture and t in capture:
                    caps[t] = np.asarray(x[:, 1])
                xin = _normed(x, _rms(x, self.eps), self.ln[off][li])
                if off == 0:
                    q = _dense(xin, W["q"]).reshape(B, T, self.n_head, self.hd)
                    k = _dense(xin, W["k"]).reshape(B, T, self.n_kv, self.hd)
                    v = _dense(xin, W["v"]).reshape(B, T, self.n_kv, self.hd)
                    out = _dense(_attention(q, k, v, self.cos, self.sin), W["o"])
                else:
                    out = _dense(
                        jax.nn.silu(_dense(xin, W["gate"])) * _dense(xin, W["up"]), W["down"]
                    )
                x = x + out
        xf = _normed(x[:, -1], _rms(x, self.eps)[:, -1], self.final_j)
        return np.asarray(_logprobs(xf, self.unembed)), caps


def run(model: Dense, rows: np.ndarray, hook=None) -> np.ndarray:  # noqa: ANN001
    out = []
    for s in range(0, len(rows), CHUNK):
        r = rows[s : s + CHUNK]
        n = len(r)
        lp, _ = model.forward_patched(np.concatenate([r, np.full(CHUNK - n, r[0])]), hook)
        out.append(lp[:n])
    return np.concatenate(out)


def main() -> None:
    n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    model = Dense(dense=True)
    rows = np.sort(np.random.default_rng(0).choice(20000, n_prompts, replace=False))
    answer = model.answer[rows]

    # the original stream at token a on the 100 values of a, at every norm point
    pad = np.concatenate([ROWS100, np.full(CHUNK - 100, ROWS100[0])])
    _, caps = model.forward_patched(pad, capture=set(range(64)))
    X = {t: caps[t][:100].astype(np.float64) for t in caps}

    P = {c.name: c for c in pool()}
    runs = json.loads((HERE / "v3_token_a.json").read_text())
    patches: dict[str, dict] = {}
    for s in runs:
        b, pt = int(s["site"][1:].split(".")[0]), s["site"].split(".")[1]
        t = 2 * b + (pt == "mlp")
        Vt, names, _ = reader_dirs(b, pt)
        Qb = span_basis(Vt)
        Z = X[t] @ Qb
        site = Site(
            s["site"], Z, Qb.T @ Vt.T, names, np.zeros(len(names), int), np.zeros((100, len(names)))
        )
        Zh, _ = fit(site, [P[a["name"]] for a in s["accepted"]])
        patches[s["site"]] = {"t": t, "Q": Qb, "recon": Zh, "mean": np.tile(Z.mean(0), (100, 1)), "exact": Z,
                              "r2_stream": 1 - float(((Z - Zh) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum())}  # fmt: skip

    def make_hook(sel: dict[int, tuple[np.ndarray, np.ndarray]]):  # noqa: ANN202
        """sel[t] = (Q, values (100, k)): set the reader-span component at token a."""
        dev = {
            t: (jnp.asarray(Q, jnp.float32), jnp.asarray(V, jnp.float32))
            for t, (Q, V) in sel.items()
        }

        def hook(t, x, rws):  # noqa: ANN001, ANN202
            if t not in dev:
                return x
            Q, V = dev[t]
            a_idx = jnp.asarray(model.a[rws] - 1)
            x1 = x[:, 1]
            return x.at[:, 1].set(x1 + (V[a_idx] - x1 @ Q) @ Q.T)

        return hook

    lp0 = run(model, rows)

    def score(lp: np.ndarray) -> dict:
        kl = (np.exp(lp0) * (lp0 - lp)).sum(-1)
        return {"kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)), "acc": float((lp.argmax(-1) == answer).mean())}  # fmt: skip

    res = {
        "n_prompts": n_prompts,
        "baseline_acc": float((lp0.argmax(-1) == answer).mean()),
        "sites": {},
    }
    for label, p in patches.items():
        ent = {"r2_stream": p["r2_stream"]}
        for variant in ("recon", "mean"):
            ent[variant] = score(run(model, rows, make_hook({p["t"]: (p["Q"], p[variant])})))
        res["sites"][label] = ent
        print(
            label,
            f"stream R2 {p['r2_stream']:.3f}",
            {k: round(v["kl_mean"], 4) for k, v in ent.items() if k != "r2_stream"},
            flush=True,
        )
    for variant in ("exact", "recon", "mean"):
        sel = {p["t"]: (p["Q"], p[variant]) for p in patches.values()}
        res["all_sites_" + variant] = score(run(model, rows, make_hook(sel)))
        print("all sites", variant, res["all_sites_" + variant], flush=True)
    (HERE / "v3_patch_full.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    assert FINAL == 64
    main()
