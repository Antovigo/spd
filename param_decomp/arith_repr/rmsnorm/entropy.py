"""Temperature components vs Llama's own entropy neurons (follow-up to `temperature.py`).

    python -m param_decomp.arith_repr.rmsnorm.entropy neurons <shard> <n_shards>   # every L31 neuron, causal
    python -m param_decomp.arith_repr.rmsnorm.entropy components                   # per-position study + dumps
    python -m param_decomp.arith_repr.rmsnorm.entropy join                         # directions, connectivity

Neurons are mean-ablated on fineweb (Stolfo et al.'s protocol, mean over fineweb positions 1..63) and
zero-ablated on addsub `=`; a last-block neuron's ablation is exactly x_final += (m - h_n) W_down[:, n].
Position 0 of the fineweb rows is an attention sink (no BOS) and is excluded everywhere.

The temperature test here is a probability-weighted regression of the log-prob change on the base logits,
`log p_b - log p_a ~ c + (1 - beta) z_b`, weights (p_b + p_a) / 2: R2 = the share of the effect that a
temperature change plus a softmax-invisible shift describes; the residual is the rest."""

import functools
import operator
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.components_model import _rms
from param_decomp.arith_repr.rmsnorm.run import savez
from param_decomp.arith_repr.rmsnorm.temperature import (
    CANDIDATES,
    CONTROLS,
    EPS,
    FW,
    OFFSET,
    TEMP,
    Stats,
    TextModel,
    base_pass,
    comp_vectors,
    fineweb,
    parse,
    start_point,
)
from param_decomp.arith_repr.vectors.common import DATASET

ENT = TEMP / "entropy"


def log(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@jax.jit
def light(xb: jax.Array, xa: jax.Array, g: jax.Array, WU: jax.Array, target: jax.Array) -> Stats:
    """Per-position effect (…, d) -> (…): total / direct (final rms frozen) / norm-only KL, entropies, CE,
    and the weighted temperature regression R2 with its residual weighted variance."""
    rb, ra = _rms(xb, EPS)[..., None], _rms(xa, EPS)[..., None]
    zb = (xb / rb * g) @ WU.T
    lb = jax.nn.log_softmax(zb, -1)
    la = jax.nn.log_softmax((xa / ra * g) @ WU.T, -1)
    lf = jax.nn.log_softmax((xa / rb * g) @ WU.T, -1)
    ln = jax.nn.log_softmax((xb / ra * g) @ WU.T, -1)
    pb, pa = jnp.exp(lb), jnp.exp(la)
    kl = lambda lq: (pb * (lb - lq)).sum(-1)  # noqa: E731
    w = (pb + pa) / 2
    d = lb - la
    wm = lambda v: (w * v).sum(-1, keepdims=True)  # noqa: E731  (weights sum to 1)
    dc, zc = d - wm(d), zb - wm(zb)
    slope = wm(dc * zc) / wm(zc * zc)
    resid = dc - slope * zc
    var_d, var_r = wm(dc * dc)[..., 0], wm(resid * resid)[..., 0]
    tgt = jnp.maximum(target, 0)[..., None]
    ce = lambda lp: jnp.where(target >= 0, -jnp.take_along_axis(lp, tgt, -1)[..., 0], 0.0)  # noqa: E731
    return {"kl_total": kl(la), "kl_direct": kl(lf), "kl_norm": kl(ln), "H_b": -(pb * lb).sum(-1),
            "H_a": -(pa * la).sum(-1), "ce_b": ce(lb), "ce_a": ce(la), "var_d": var_d, "var_resid": var_r,
            "beta": 1 - slope[..., 0]}  # fmt: skip


def base_h(M: TextModel, x: jax.Array) -> jax.Array:
    """L31 MLP inner activations h given the stream entering block 31's MLP norm."""
    h = M.run(x, 63, {}, want_h=True)[2]
    assert h is not None
    return h


def addsub_tokens(n: int = 2000) -> np.ndarray:
    ix = np.load(DATASET / "index.npz")
    rows = np.sort(np.random.default_rng(0).choice(len(ix["tokens"]), n, replace=False))
    return ix["tokens"][rows].astype(np.int64)


def neurons(shard: int, n_shards: int, n_rows: int = 32, batch: int = 32) -> None:
    toks, target = fineweb(n_rows, batch)
    M = TextModel(toks.shape[1])
    Wd = M.W[31]["down"].astype(jnp.float32)  # (4096, 14336)
    fw = base_pass(M, toks, batch, {63})
    # fineweb means over 512 rows, positions 1..63
    mtoks, _ = fineweb(512, 32)
    mc = base_pass(M, mtoks, 32, {63})
    hsum = functools.reduce(operator.add, (base_h(M, x)[:, 1:].sum((0, 1)) for x in mc[63]))
    hmean = hsum / (512 * 63)
    del mc
    hb_fw = [base_h(M, x) for x in fw[63]]
    atoks = addsub_tokens()
    ad = base_pass(M, atoks, 1000, {63})
    hb_eq = [base_h(M, x)[:, -1] for x in ad[63]]
    xf_eq = [x[:, -1] for x in ad[64]]
    atgt = jnp.full((1000,), -1)
    ids = np.arange(Wd.shape[1])[shard::n_shards]
    keys = [
        "kl_total",
        "kl_direct",
        "kl_norm",
        "H_b",
        "H_a",
        "ce_b",
        "ce_a",
        "var_d",
        "var_resid",
        "beta",
    ]
    out = {f"{s}__{k}": np.zeros(len(ids)) for s in ("fw", "eq") for k in keys}
    t0 = time.time()
    for i, n_ in enumerate(ids):
        w = Wd[:, n_]
        for bi, xb in enumerate(fw[64]):
            xa = xb + (hmean[n_] - hb_fw[bi][..., n_])[..., None] * w
            m = light(xb[:, 1:], xa[:, 1:], M.g, M.unembed, jnp.asarray(target[:, 1:]))
            for k in keys:
                out[f"fw__{k}"][i] += float(m[k].mean()) / len(fw[64])
        for bi, xb in enumerate(xf_eq):
            xa = xb - hb_eq[bi][:, n_][:, None] * w
            m = light(xb, xa, M.g, M.unembed, atgt)
            for k in keys:
                out[f"eq__{k}"][i] += float(m[k].mean()) / len(xf_eq)
        if i % 200 == 0:
            log(f"neurons {shard}: {i}/{len(ids)} ({time.time() - t0:.0f}s)")
    ENT.mkdir(parents=True, exist_ok=True)
    savez(
        ENT / f"neurons_{shard}of{n_shards}.npz", ids=ids, h_mean_fw=np.asarray(hmean)[ids], **out
    )


def components(n_rows: int = 512, batch: int = 32, n_dump: int = 3000) -> None:
    """Per fineweb position (1..63) for candidates and controls: the inner activation and, for mean and
    zero ablation, the light metrics plus the write geometry (shares of |dx|^2 along the base stream, the
    uniform-logit direction and the bottom-40 null space). Dumps, on a shared position set (every CI-flagged
    position of any candidate + random ones) and on 2000 addsub `=` positions: each component's write dx
    and exact change of every L31 neuron's activation dh (gate/up components), and the base activations."""
    toks, target = fineweb(n_rows, batch)
    M = TextModel(toks.shape[1])
    uv = {li: dict(np.load(FW / f"uv_L{li}.npz")) for li in (30, 31)}
    ci = {li: np.load(FW / f"ci_L{li}.npy", mmap_mode="r") for li in (30, 31)}
    names = CANDIDATES + CONTROLS
    caps = base_pass(M, toks, batch, {60, 61, 62, 63})
    B40 = M.null[40]
    rng = np.random.default_rng(5)
    flagged = np.zeros((n_rows, toks.shape[1]), bool)
    for nm in CANDIDATES:
        li, kind, c = parse(nm)
        flagged |= np.asarray(ci[li][:n_rows, :, OFFSET[kind] + c]) > 0.01
    flagged[:, 0] = False
    extra = rng.choice(
        np.flatnonzero(~flagged[:, 1:].ravel()), max(n_dump - int(flagged.sum()), 0), replace=False
    )
    dump = flagged.copy()
    d2 = dump[:, 1:].copy().ravel()
    d2[extra] = True
    dump[:, 1:] = d2.reshape(n_rows, -1)
    out: dict[str, np.ndarray] = {"dump_mask": dump, "flagged": flagged}
    hb_all = [base_h(M, caps[63][bi]) for bi in range(len(caps[64]))]
    out["h_base_dump"] = np.concatenate(
        [
            np.asarray(h)[dump[s : s + batch]]
            for h, s in zip(hb_all, range(0, n_rows, batch), strict=False)
        ]
    )
    out["x_base_dump"] = np.concatenate(
        [
            np.asarray(x)[dump[s : s + batch]]
            for x, s in zip(caps[64], range(0, n_rows, batch), strict=False)
        ]
    )
    for nm in names:
        li, kind, c = parse(nm)
        V, U = comp_vectors(uv[li], kind, c)
        tp = start_point(li, kind)
        inner = jnp.concatenate(
            [M.inner(caps[tp][bi], li, kind, V[:, None])[..., 0] for bi in range(len(caps[64]))]
        )
        out[f"{nm}__inner"] = np.asarray(inner)
        mean_h = float(inner[:, 1:].mean())
        for abl, f in (("zero", jnp.zeros_like), ("mean", lambda h, m=mean_h: jnp.full_like(h, m))):
            rec: dict[str, list[np.ndarray]] = {}
            dxs: list[np.ndarray] = []
            dhs: list[np.ndarray] = []
            for bi, s in enumerate(range(0, n_rows, batch)):
                xa, _, ha = M.run(caps[tp][bi], tp, {(li, kind): (V, U, f)}, want_h=li == 31)
                xb = caps[64][bi]
                m = light(xb, xa, M.g, M.unembed, jnp.asarray(target[s : s + batch]))
                dx = xb - xa
                e = (dx * dx).sum(-1) + 1e-12
                xh = xb / jnp.linalg.norm(xb, axis=-1, keepdims=True)
                m |= {"sh_par": ((dx * xh).sum(-1)) ** 2 / e, "sh_uni": (dx @ M.uniform) ** 2 / e,
                      "sh_null40": ((dx @ B40) ** 2).sum(-1) / e, "e_dx": e,
                      "x_sh_uni": (xb @ M.uniform) ** 2 / (xb * xb).sum(-1),
                      "x_sh_null40": ((xb @ B40) ** 2).sum(-1) / (xb * xb).sum(-1)}  # fmt: skip
                for k, v in m.items():
                    rec.setdefault(k, []).append(np.asarray(v))
                if abl == "zero":
                    sel = dump[s : s + batch]
                    dxs.append(np.asarray(dx)[sel])
                    if ha is not None:
                        dhs.append(np.asarray(hb_all[bi] - ha)[sel])
            for k, v in rec.items():
                out[f"{nm}__{abl}__{k}"] = np.concatenate(v)
            if abl == "zero":
                out[f"{nm}__dx_dump"] = np.concatenate(dxs)
                if dhs:
                    out[f"{nm}__dh_dump"] = np.concatenate(dhs).astype(np.float16)
            log(
                nm,
                abl,
                {
                    k: round(float(np.mean(out[f"{nm}__{abl}__{k}"][:, 1:])), 4)
                    for k in (
                        "kl_total",
                        "kl_direct",
                        "var_d",
                        "var_resid",
                        "sh_par",
                        "sh_uni",
                        "sh_null40",
                    )
                },
            )
    ENT.mkdir(parents=True, exist_ok=True)
    savez(ENT / "components_fineweb.npz", tokens=toks, **out)
    del caps, hb_all
    # addsub `=`
    atoks = addsub_tokens()
    ad = base_pass(M, atoks, 1000, {60, 61, 62, 63})
    aout: dict[str, np.ndarray] = {}
    hb_eq = [base_h(M, x)[:, -1] for x in ad[63]]
    aout["h_base_eq"] = np.asarray(jnp.concatenate(hb_eq))
    aout["x_base_eq"] = np.asarray(jnp.concatenate([x[:, -1] for x in ad[64]]))
    for nm in names:
        li, kind, c = parse(nm)
        vv, uu = comp_vectors(uv[li], kind, c)
        tp = start_point(li, kind)
        dxs = []
        dhs = []
        for bi in range(len(ad[64])):
            xa, _, ha = M.run(
                ad[tp][bi], tp, {(li, kind): (vv, uu, jnp.zeros_like)}, want_h=li == 31
            )
            dxs.append(np.asarray((ad[64][bi] - xa)[:, -1]))
            if ha is not None:
                dhs.append(np.asarray(hb_eq[bi] - ha[:, -1]))
        aout[f"{nm}__dx_eq"] = np.concatenate(dxs)
        if dhs:
            aout[f"{nm}__dh_eq"] = np.concatenate(dhs).astype(np.float16)
    savez(ENT / "components_eq.npz", **aout)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "neurons":
        neurons(int(sys.argv[2]), int(sys.argv[3]))
    else:
        {"components": components}[cmd]()
