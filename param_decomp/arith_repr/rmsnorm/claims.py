"""Experiments behind notes/rmsnorm/entropy-components.md that `temperature.py` / `entropy.py` did not run.

    python -m param_decomp.arith_repr.rmsnorm.claims run

1. Subspace split of each component's write dx (zero ablation) at every analysed position:
   S0 = span(bottom-k singular directions of W_U diag(g), the uniform-logit direction u, the stream's own
   direction x_hat) — directions that cannot change the token ranking by themselves — and its complement S1.
   Removing only the S0 part / only the S1 part / only one of the three S0 pieces, measured with
   `entropy.light` (KL, temperature R2, final-norm mediation, entropy change), plus the direct logit
   profile of the S1 part: number-token offset, correlation with log unigram frequency, and the R2 of a
   regression of it on [is-number, log-frequency].
2. Neuron level (L31 gate/up components only): per-neuron static properties (number offset, frequency
   correlation, uniform share, bottom-40 share of W_eff w_n), the gate state of every neuron at the
   analysed positions (|silu(g)| for up components, |silu'(g) u| for gate components), each component's
   exact per-neuron activation change dh, and class-restricted ablations (the component's effect routed
   through one neuron class only).
3. Per fineweb position (1..63, 512 rows): activations of the entropy / number / frequency neuron classes,
   for the coupling test.

Positions: the 2000-prompt addsub subset at `=`, and fineweb positions flagged by the -05 CI for any
candidate (the `components_fineweb.npz` mask)."""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import snapshot
from param_decomp.arith_repr.isa.components_model import _rms
from param_decomp.arith_repr.rmsnorm.entropy import ENT, addsub_tokens, light
from param_decomp.arith_repr.rmsnorm.run import savez
from param_decomp.arith_repr.rmsnorm.temperature import (
    CANDIDATES,
    CONTROLS,
    EPS,
    FW,
    TEMP,
    Stats,
    TextModel,
    _normed,
    base_pass,
    comp_vectors,
    fineweb,
    parse,
    start_point,
)

OUTC = TEMP / "claims"
ENT6 = [1209, 2398, 2564, 3191, 5966, 6696]
KS = (40, 512)


def log(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def token_tables(nv: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """is-number (every pure-digit token), log unigram frequency (tokens seen in the fineweb sample), seen mask."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(snapshot() / "tokenizer.json"))
    isnum = np.zeros(nv, bool)
    for i in range(min(nv, tok.get_vocab_size())):
        s = tok.decode([i]).strip()
        isnum[i] = s.isdigit()
    counts = np.load(TEMP / "dose.npz")["unigram_counts"]
    seen = counts > 0
    lf = np.where(seen, np.log(np.maximum(counts, 1)), 0.0)
    return isnum, lf, seen


def logit_profile(dz: jax.Array, isnum: jax.Array, lf: jax.Array, seen: jax.Array) -> Stats:
    """dz (..., V) a direct logit change. Uniform share; over seen tokens: number offset (sd units), corr with
    log frequency, R2 of dz ~ a + b isnum + c logfreq."""
    s = seen.astype(jnp.float32)
    n = s.sum()
    mean = (dz * s).sum(-1, keepdims=True) / n
    c = (dz - mean) * s
    var = (c * c).sum(-1) / n
    sd = jnp.sqrt(var) + 1e-12
    xn = (isnum.astype(jnp.float32) - (isnum * s).sum() / n) * s
    xf = (lf - (lf * s).sum() / n) * s
    num_off = (
        (c * isnum).sum(-1) / jnp.maximum((isnum * s).sum(), 1)
        - (c * (1 - isnum) * s).sum(-1) / (n - (isnum * s).sum())
    ) / sd
    corr_f = (c * xf).sum(-1) / jnp.sqrt((c * c).sum(-1) * (xf * xf).sum() + 1e-12)
    X = jnp.stack([xn, xf], -1)  # (V, 2)
    XtX = X.T @ X
    beta = (c @ X) @ jnp.linalg.inv(XtX)  # XtX is symmetric
    fit = beta @ X.T
    r2 = 1 - ((c - fit) ** 2).sum(-1) / ((c * c).sum(-1) + 1e-12)
    uni = dz.mean(-1) ** 2 / (dz * dz).mean(-1)
    return {"num_off": num_off, "corr_freq": corr_f, "r2_numfreq": r2, "uniform_share": uni}


def run(batch: int = 16) -> None:
    OUTC.mkdir(parents=True, exist_ok=True)
    toks, target = fineweb(512, batch)
    M = TextModel(toks.shape[1])
    V = M.unembed.shape[0]
    isnum_np, lf_np, seen_np = token_tables(V)
    isnum, lf, seen = jnp.asarray(isnum_np), jnp.asarray(lf_np, jnp.float32), jnp.asarray(seen_np)
    ev, evec = np.linalg.eigh(np.asarray(M.Weff.T @ M.Weff, np.float64))
    B = {k: jnp.asarray(evec[:, :k], jnp.float32) for k in KS}
    u = M.uniform
    Wd = M.W[31]["down"].astype(jnp.float32)
    Wg, Wu = M.W[31]["gate"], M.W[31]["up"]
    uv = {li: dict(np.load(FW / f"uv_L{li}.npz")) for li in (30, 31)}
    out: dict[str, np.ndarray] = {}

    # ---- static neuron properties
    stat = {k: [] for k in ("num_off", "corr_freq", "r2_numfreq", "uniform_share")}
    for s in range(0, Wd.shape[1], 256):
        dz = (M.Weff @ Wd[:, s : s + 256]).T  # (256, V)
        p = logit_profile(dz, isnum, lf, seen)
        for k in stat:
            stat[k].append(np.asarray(p[k]))
    for k, v in stat.items():
        out[f"neuron_{k}"] = np.concatenate(v)
    out["neuron_norm"] = np.asarray(jnp.linalg.norm(Wd, axis=0))
    out["neuron_null40"] = np.asarray(((B[40].T @ Wd) ** 2).sum(0)) / out["neuron_norm"] ** 2
    out["eig"] = ev
    log("static neuron properties done")

    # ---- positions
    cf = np.load(ENT / "components_fineweb.npz")
    flagged = cf["flagged"]
    fw_caps = base_pass(M, toks, batch, {60, 61, 62, 63})
    atoks = addsub_tokens()
    ad = base_pass(M, atoks, 1000, {60, 61, 62, 63})

    def sets():
        """(tag, list of (captures-by-point, final, selector of positions, targets))"""
        fw = [({t: fw_caps[t][bi] for t in (60, 61, 62, 63)}, fw_caps[64][bi], jnp.asarray(flagged[s : s + batch]),
               jnp.asarray(target[s : s + batch])) for bi, s in enumerate(range(0, 512, batch))]  # fmt: skip
        eq_sel = np.zeros((1000, atoks.shape[1]), bool)
        eq_sel[:, -1] = True
        eq = [({t: ad[t][bi] for t in (60, 61, 62, 63)}, ad[64][bi], jnp.asarray(eq_sel), jnp.full((1000, atoks.shape[1]), -1))
              for bi in range(len(ad[64]))]  # fmt: skip
        return (("fw", fw), ("eq", eq))

    # ---- gate state of every neuron at the analysed positions
    for tag, batches in sets():
        acc: dict[str, jax.Array | float] = {"abs_silu_g": 0.0, "abs_dsilu_g_u": 0.0, "abs_h": 0.0}
        n = 0
        for cp, _, sel, _ in batches:
            xin = _normed(cp[63], M.ln[1][31])
            g = xin @ Wg.astype(jnp.float32).T
            uu = xin @ Wu.astype(jnp.float32).T
            sg = jax.nn.sigmoid(g)
            dsilu = sg * (1 + g * (1 - sg))
            m = sel[..., None].astype(jnp.float32)
            acc["abs_silu_g"] = acc["abs_silu_g"] + (jnp.abs(g * sg) * m).sum((0, 1))
            acc["abs_dsilu_g_u"] = acc["abs_dsilu_g_u"] + (jnp.abs(dsilu * uu) * m).sum((0, 1))
            acc["abs_h"] = acc["abs_h"] + (jnp.abs(g * sg * uu) * m).sum((0, 1))
            n += int(sel.sum())
        for k, v in acc.items():
            out[f"{tag}__state_{k}"] = np.asarray(v) / n
    log("gate states done")

    # neuron classes (for class-restricted ablations)
    scan = {}
    import glob

    for f in sorted(glob.glob(str(ENT / "neurons_*of5.npz"))):
        z = np.load(f)
        for k in z.files:
            scan.setdefault(k, []).append(z[k])
    scan = {k: np.concatenate(v) for k, v in scan.items()}
    o = np.argsort(scan["ids"])
    scan = {k: v[o] for k, v in scan.items()}
    ent_like = np.zeros(Wd.shape[1], bool)
    for s_ in ("fw", "eq"):
        TE, DE = scan[f"{s_}__kl_total"], scan[f"{s_}__kl_direct"]
        med = 1 - DE / np.maximum(TE, 1e-12)
        R2 = 1 - scan[f"{s_}__var_resid"] / np.maximum(scan[f"{s_}__var_d"], 1e-12)
        floor = 1e-5 if s_ == "fw" else 1e-4
        ent_like |= (med >= 0.5) & (R2 >= 0.5) & (floor <= TE)
    number = out["neuron_num_off"] >= np.percentile(out["neuron_num_off"], 99)
    freq = np.abs(out["neuron_corr_freq"]) >= np.percentile(np.abs(out["neuron_corr_freq"]), 99)
    ent6 = np.zeros(Wd.shape[1], bool)
    ent6[ENT6] = True
    classes = {
        "ent6": ent6,
        "entropy_like": ent_like,
        "number": number & ~ent_like,
        "freq": freq & ~ent_like & ~number,
    }
    classes["rest"] = ~(ent_like | number | freq)
    for k, v in classes.items():
        out[f"class_{k}"] = v
    log("classes", {k: int(v.sum()) for k, v in classes.items()})
    cls_j = {k: jnp.asarray(v, jnp.float32) for k, v in classes.items()}

    def pieces(dx: jax.Array, xb: jax.Array, k: int) -> dict[str, jax.Array]:
        """dx split into S0 pieces (bottom-k, uniform, stream direction; orthogonalised in that order) and S1."""
        Bk = B[k]
        p_null = (dx @ Bk) @ Bk.T
        u1 = u - Bk @ (Bk.T @ u)
        u1 = u1 / jnp.linalg.norm(u1)
        p_uni = (dx @ u1)[..., None] * u1
        xh = xb / jnp.linalg.norm(xb, axis=-1, keepdims=True)
        x1 = xh - (xh @ Bk) @ Bk.T - (xh @ u1)[..., None] * u1
        x1 = x1 / (jnp.linalg.norm(x1, axis=-1, keepdims=True) + 1e-12)
        p_par = (dx * x1).sum(-1, keepdims=True) * x1
        s0 = p_null + p_uni + p_par
        return {"S0": s0, "S1": dx - s0, "null": p_null, "uni": p_uni, "par": p_par}

    keys = ["kl_total", "kl_direct", "var_d", "var_resid", "H_a", "H_b", "beta"]
    for nm in CANDIDATES + CONTROLS:
        li, kind, c = parse(nm)
        Vc, Uc = comp_vectors(uv[li], kind, c)
        tp = start_point(li, kind)
        neuron_level = li == 31 and kind in ("gate", "up")
        for tag, batches in sets():
            rec: dict[str, list[float]] = {}
            share = jnp.zeros(Wd.shape[1])
            e_tot = 0.0
            for cp, xb, sel, tgt in batches:
                xa, _, ha = M.run(
                    cp[tp], tp, {(li, kind): (Vc, Uc, jnp.zeros_like)}, want_h=neuron_level
                )
                x63 = cp[63]
                if (
                    tag == "eq"
                ):  # only `=` is analysed: drop the other positions before any full-vocab op
                    xb, xa, sel, tgt, x63 = (
                        xb[:, -1:],
                        xa[:, -1:],
                        sel[:, -1:],
                        tgt[:, -1:],
                        x63[:, -1:],
                    )
                    ha = None if ha is None else ha[:, -1:]
                dx = xb - xa
                sel_f = sel.astype(jnp.float32)
                ns = sel.sum()

                def add(name: str, m: Stats) -> None:
                    for k2, v in m.items():
                        rec.setdefault(f"{name}__{k2}", []).append(float((v * sel_f).sum()))  # noqa: B023

                add("full", light(xb, xa, M.g, M.unembed, tgt))
                for k in KS:
                    pc = pieces(dx, xb, k)
                    for pn, pv in pc.items():
                        if k == 512 and pn not in ("S0", "S1"):
                            continue
                        add(
                            f"k{k}_{pn}",
                            {
                                kk: vv
                                for kk, vv in light(xb, xb - pv, M.g, M.unembed, tgt).items()
                                if kk in keys
                            },
                        )
                    rb = _rms(xb, EPS)[..., None]
                    prof = logit_profile(((pc["S1"] / rb) * M.g) @ M.unembed.T, isnum, lf, seen)
                    add(f"k{k}_S1prof", prof)
                    prof0 = logit_profile(((pc["S0"] / rb) * M.g) @ M.unembed.T, isnum, lf, seen)
                    add(f"k{k}_S0prof", prof0)
                prof_full = logit_profile(
                    ((dx / _rms(xb, EPS)[..., None]) * M.g) @ M.unembed.T, isnum, lf, seen
                )
                add("full_prof", prof_full)
                rec.setdefault("n", []).append(float(ns))
                if neuron_level:
                    xin = _normed(x63, M.ln[1][31])
                    hb = jax.nn.silu(xin @ Wg.astype(jnp.float32).T) * (
                        xin @ Wu.astype(jnp.float32).T
                    )
                    dh = hb - ha
                    share = share + ((dh * (dx @ Wd)) * sel_f[..., None]).sum((0, 1))
                    e_tot += float(((dx * dx).sum(-1) * sel_f).sum())
                    for cn, cm in cls_j.items():
                        xa_c = xb - (dh * cm) @ Wd.T
                        m = light(xb, xa_c, M.g, M.unembed, tgt)
                        add(f"class_{cn}", {kk: vv for kk, vv in m.items() if kk in keys})
                        pr = logit_profile(
                            (((xb - xa_c) / _rms(xb, EPS)[..., None]) * M.g) @ M.unembed.T,
                            isnum,
                            lf,
                            seen,
                        )
                        add(f"class_{cn}_prof", pr)
            ntot = sum(rec.pop("n"))
            for k2, v in rec.items():
                out[f"{nm}__{tag}__{k2}"] = np.array(sum(v) / ntot)
            if neuron_level:
                out[f"{nm}__{tag}__share"] = np.asarray(share) / e_tot
            log(
                nm,
                tag,
                f"full KL {out[f'{nm}__{tag}__full__kl_total']:.4f}",
                f"S0 KL {out[f'{nm}__{tag}__k512_S0__kl_total']:.4f}",
                f"S1 KL {out[f'{nm}__{tag}__k512_S1__kl_total']:.4f}",
            )
        if neuron_level:
            out[f"{nm}__U"] = uv[li][f"{kind}.U"][c]

    # ---- coupling: class activations at every fineweb position 1..63
    acts: dict[str, list[np.ndarray]] = {k: [] for k in ("ent6", "entropy_like", "number", "freq")}
    for bi in range(len(fw_caps[64])):
        xin = _normed(fw_caps[63][bi], M.ln[1][31])
        h = jax.nn.silu(xin @ Wg.astype(jnp.float32).T) * (xin @ Wu.astype(jnp.float32).T)
        h = h[:, 1:]
        for k in acts:
            # each class's summed write projected on its own effect: temperature classes by their write norm,
            # number class by its number offset
            wts = jnp.asarray(np.where(classes[k], 1.0, 0.0), jnp.float32)
            acts[k].append(np.asarray(h * wts)[..., classes[k]])
    for k, v in acts.items():
        out[f"acts_{k}"] = np.concatenate(v).astype(np.float16)
    out["acts_ids_" + "ent6"] = np.flatnonzero(classes["ent6"])
    for k in ("entropy_like", "number", "freq"):
        out[f"acts_ids_{k}"] = np.flatnonzero(classes[k])
    savez(OUTC / "claims.npz", **out)
    log("saved")


if __name__ == "__main__":
    {"run": run}[sys.argv[1]]()
