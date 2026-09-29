"""RMSNorm experiments on the -05 decomposition (the dataset's decomposed model: ceiling-filter masks
at CI > 0.01, alive components only, no delta). Outputs in `OUT`.

    python -m param_decomp.arith_repr.rmsnorm.run base      # rms of every norm, stream + read comparison
    python -m param_decomp.arith_repr.rmsnorm.run orig      # freezing norms in the original model
    python -m param_decomp.arith_repr.rmsnorm.run dec       # decomposed model with forced / frozen norms
    python -m param_decomp.arith_repr.rmsnorm.run hybrid    # original model, one block decomposed
    python -m param_decomp.arith_repr.rmsnorm.run ablate <shard> <n_shards>   # per-component, free vs frozen
    python -m param_decomp.arith_repr.rmsnorm.run groups    # per-site / per-layer ablations, free vs frozen
    python -m param_decomp.arith_repr.rmsnorm.run swap      # one norm at a time forced to the other model's rms
    python -m param_decomp.arith_repr.rmsnorm.run split <shard> <n_shards>    # per-component write vs norm share
    python -m param_decomp.arith_repr.rmsnorm.run probe31   # what the norm-acting last-block MLP components write

Frozen = the rms every norm computed in the unablated run of the same model on the same prompt."""

import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.components_model import T
from param_decomp.arith_repr.rmsnorm.model import (
    FINAL,
    N_NORM,
    NormHook,
    NormModel,
    forced,
    kl_rows,
)
from param_decomp.arith_repr.vectors.common import RUN

OUT = RUN / "analysis/rmsnorm"
N = 20000
CHUNK_N = 500
HookFactory = Callable[[np.ndarray], NormHook]
Conds = dict[str, tuple[Sequence[bool] | bool, HookFactory | None]]
READERS = {0: ("q", "k", "v"), 1: ("gate", "up")}


def savez(path: Path, **arrays: Any) -> None:  # noqa: ANN401
    """`np.savez` with **kwargs of arrays (numpy's stub types every kwarg as `allow_pickle: bool`)."""
    np.savez(path, **arrays)


def subset(n: int, seed: int = 0) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(N, n, replace=False))


def log(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Metrics:
    """Per-prompt KL to a reference, top-1 token and true-answer log-prob, per condition."""

    def __init__(self, names: list[str], n: int) -> None:
        self.d = {k: {m: np.zeros(n, np.float32) for m in ("kl", "lp_ans")} | {"top1": np.zeros(n, np.int32)} for k in names}  # fmt: skip

    def add(
        self, name: str, sl: slice, ref: jax.Array, lp: jax.Array, ans: np.ndarray, nreal: int
    ) -> None:
        d = self.d[name]
        d["kl"][sl] = np.asarray(kl_rows(ref, lp))[:nreal]
        d["top1"][sl] = np.asarray(jnp.argmax(lp, -1))[:nreal]
        d["lp_ans"][sl] = np.asarray(jnp.take_along_axis(lp, jnp.asarray(ans)[:, None], 1)[:, 0])[
            :nreal
        ]

    def save(
        self,
        path: Path,
        rows: np.ndarray,
        answer: np.ndarray,
        extra: dict[str, np.ndarray] | None = None,
    ) -> None:
        flat = {f"{k}__{m}": v for k, d in self.d.items() for m, v in d.items()}
        savez(path, rows=rows, answer=answer[rows], **flat, **(extra or {}))
        summ = {k: {"kl": float(d["kl"].mean()), "acc": float((d["top1"] == answer[rows]).mean()),
                    "lp_ans": float(d["lp_ans"].mean())} for k, d in self.d.items()}  # fmt: skip
        path.with_suffix(".json").write_text(json.dumps(summ, indent=1))
        for k, s in summ.items():
            log(f"{k:40s} KL {s['kl']:.4f} acc {s['acc']:.3f} lp_ans {s['lp_ans']:.3f}")


def base(M: NormModel) -> None:
    """Both models on the whole pool: rms of every norm, how the decomposed stream sits against the
    original one (x_d = alpha x_o + perp), where each stream's energy is, and how the alive readers'
    inner activations compare with the decomposed model's own rms and with the original rms."""
    rows_all = np.arange(N)
    pts = set(range(N_NORM))
    rms = {m: np.zeros((N_NORM, N, T), np.float32) for m in ("o", "d")}
    geo = {k: np.zeros((N_NORM, N, T), np.float32) for k in ("cos", "alpha", "perp")}
    energy = {k: np.zeros((N_NORM, T, 4096), np.float64) for k in ("o", "d", "r")}
    # reads[t, kind, pos, stat]: stats = |Ho|^2, |Hd|^2, <Ho,Hd>, |Hd-Ho|^2, |HdT-Ho|^2, |HdT|^2, <Ho,HdT>
    reads = np.zeros((N_NORM, 3, T, 7), np.float64)
    top = {m: np.zeros(N, np.int32) for m in ("o", "d")}
    kl = np.zeros(N, np.float32)
    s0 = 0
    for rows, n in M.chunks(rows_all):
        sl = slice(s0, s0 + n)
        lo, ro, co = M.forward_chunk(rows, False, capture=pts)
        ld, rd, cd = M.forward_chunk(rows, True, capture=pts)
        kl[sl] = np.asarray(kl_rows(lo, ld))[:n]
        top["o"][sl], top["d"][sl] = (
            np.asarray(jnp.argmax(lo, -1))[:n],
            np.asarray(jnp.argmax(ld, -1))[:n],
        )
        rms["o"][:, sl], rms["d"][:, sl] = np.asarray(ro)[:, :n], np.asarray(rd)[:, :n]
        for t in range(N_NORM):
            xo, xd = co[t][:n], cd[t][:n]
            oo, dd, od = (xo * xo).sum(-1), (xd * xd).sum(-1), (xo * xd).sum(-1)
            a = od / oo
            geo["cos"][t, sl] = np.asarray(od / jnp.sqrt(oo * dd))
            geo["alpha"][t, sl] = np.asarray(a)
            geo["perp"][t, sl] = np.asarray(jnp.sqrt(jnp.maximum(dd - a * od, 0) / oo))
            energy["o"][t] += np.asarray((xo * xo).sum(0), np.float64)
            energy["d"][t] += np.asarray((xd * xd).sum(0), np.float64)
            energy["r"][t] += np.asarray(((xd - xo) ** 2).sum(0), np.float64)
            if t == FINAL:
                continue
            li, off = divmod(t, 2)
            m = jnp.asarray(M.masks(li, rows[:n]))
            g = M.ln[off][li]
            no = xo / ro[t, :n, :, None] * g
            nd = xd / rd[t, :n, :, None] * g
            ndT = xd / ro[t, :n, :, None] * g
            for ki, kind in enumerate(READERS[off]):
                s = M.sites[li][kind]
                mk = m[:, :, s.cols]
                Ho, Hd, HdT = (y @ s.V * mk for y in (no, nd, ndT))
                st = [
                    (Ho * Ho),
                    (Hd * Hd),
                    (Ho * Hd),
                    (Hd - Ho) ** 2,
                    (HdT - Ho) ** 2,
                    (HdT * HdT),
                    (Ho * HdT),
                ]
                reads[t, ki] += np.stack([np.asarray(z.sum((0, 2)), np.float64) for z in st], -1)
        s0 += n
        log(f"base {s0}/{N}: KL so far {kl[:s0].mean():.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    savez(OUT / "base.npz", rms_o=rms["o"], rms_d=rms["d"], **geo, top_o=top["o"], top_d=top["d"], kl=kl,
             energy_o=energy["o"], energy_d=energy["d"], energy_r=energy["r"], reads=reads, answer=M.answer)  # fmt: skip


def load_base() -> dict[str, np.ndarray]:
    b = np.load(OUT / "base.npz")
    return {k: b[k] for k in ("rms_o", "rms_d", "top_o")}


def run_conditions(
    M: NormModel,
    rows_all: np.ndarray,
    conds: Mapping[str, tuple[Sequence[bool] | bool, HookFactory | None]],
    name: str,
    ref_dec: bool = False,
) -> None:
    """conds: name -> (dec, hook_factory(rows) or None). Reference = the original model (or the
    decomposed one if ref_dec) on the same prompts."""
    met = Metrics(list(conds), len(rows_all))
    s0 = 0
    t0 = time.time()
    for rows, n in M.chunks(rows_all):
        sl = slice(s0, s0 + n)
        ref, _, _ = M.forward_chunk(rows, ref_dec)
        for k, (dec, hf) in conds.items():
            lp, _, _ = M.forward_chunk(rows, dec, norm=None if hf is None else hf(rows))
            met.add(k, sl, ref, lp, M.answer[rows], n)
        s0 += n
        log(f"{name} {s0}/{len(rows_all)} ({time.time() - t0:.0f}s)")
    met.save(OUT / f"{name}.npz", rows_all, M.answer)


def pool_mean(r: np.ndarray) -> jax.Array:
    return jnp.asarray(r.mean(1, keepdims=True))


def per_prompt(r: np.ndarray) -> Callable[[np.ndarray], jax.Array]:
    return lambda rows: jnp.asarray(r[:, rows])


def orig(M: NormModel) -> None:
    b = load_base()
    ro = b["rms_o"]
    mean_o = pool_mean(ro)
    perm = np.random.default_rng(1).permutation(N)
    ALL: set[int] | None = None
    ATTN, MLP = set(range(0, 64, 2)), set(range(1, 64, 2))
    NONFINAL = set(range(64))

    def mean(points: set[int] | None = ALL, positions: Sequence[int] | None = None) -> HookFactory:
        return lambda rows: forced(mean_o, points, positions)

    conds: Conds = {
        "orig_free": (False, None),
        "freeze_all": (False, mean()),
        "freeze_all_shuffled": (False, lambda rows: forced(jnp.asarray(ro[:, perm[rows]]))),
        "freeze_attn_norms": (False, mean(ATTN)),
        "freeze_mlp_norms": (False, mean(MLP)),
        "freeze_final_only": (False, mean({FINAL})),
        "freeze_all_but_final": (False, mean(NONFINAL)),
        "freeze_all_at_eq_only": (False, mean(ALL, [4])),
        "freeze_all_before_eq": (False, mean(ALL, [0, 1, 2, 3])),
    }
    for w in range(4):
        conds[f"freeze_layers_{8 * w}-{8 * w + 7}"] = (False, mean(set(range(16 * w, 16 * w + 16))))
    run_conditions(M, np.arange(N), conds, "orig")
    single: Conds = {f"freeze_t{t}": (False, mean({t})) for t in range(N_NORM)}
    run_conditions(M, subset(4000), single, "orig_single")


def dec(M: NormModel) -> None:
    b = load_base()
    ro, rd = b["rms_o"], b["rms_d"]
    NONFINAL = set(range(64))
    conds: Conds = {
        "dec_free": (True, None),
        "dec_forced_orig_all": (True, lambda rows: forced(jnp.asarray(ro[:, rows]))),
        "dec_forced_orig_nonfinal": (True, lambda rows: forced(jnp.asarray(ro[:, rows]), NONFINAL)),
        "dec_forced_orig_final": (True, lambda rows: forced(jnp.asarray(ro[:, rows]), {FINAL})),
        "dec_frozen_own_mean": (True, lambda rows: forced(pool_mean(rd))),
        "dec_frozen_orig_mean": (True, lambda rows: forced(pool_mean(ro))),
        "orig_forced_dec_all": (False, lambda rows: forced(jnp.asarray(rd[:, rows]))),
        "orig_forced_dec_nonfinal": (
            False,
            lambda rows: forced(jnp.asarray(rd[:, rows]), NONFINAL),
        ),
        "orig_forced_dec_final": (False, lambda rows: forced(jnp.asarray(rd[:, rows]), {FINAL})),
    }
    run_conditions(M, np.arange(N), conds, "dec")


def hybrid(M: NormModel) -> None:
    """Original model with ONE block decomposed (its inactive components and delta ablated), norms
    free vs forced to the original model's rms; plus how much that block's ablation moves every
    later norm's rms."""
    b = load_base()
    ro = b["rms_o"]
    rows_all = subset(4000)
    conds: Conds = {}
    for li in range(M.n_layer):
        d = [i == li for i in range(M.n_layer)]
        conds[f"block{li}_free"] = (d, None)
        conds[f"block{li}_forced_orig"] = (d, lambda rows: forced(jnp.asarray(ro[:, rows])))
    met = Metrics(list(conds), len(rows_all))
    dlog = np.zeros((M.n_layer, N_NORM, T), np.float64)  # mean |log rms_hybrid / rms_orig|
    s0 = 0
    for rows, n in M.chunks(rows_all):
        sl = slice(s0, s0 + n)
        ref, r_ref, _ = M.forward_chunk(rows, False)
        for k, (d, hf) in conds.items():
            lp, r, _ = M.forward_chunk(rows, d, norm=None if hf is None else hf(rows))
            met.add(k, sl, ref, lp, M.answer[rows], n)
            if hf is None:
                li = int(k[5:].split("_")[0])
                dlog[li] += np.asarray(jnp.abs(jnp.log(r[:, :n] / r_ref[:, :n])).sum(1), np.float64)
        s0 += n
        log(f"hybrid {s0}/{len(rows_all)}")
    met.save(OUT / "hybrid.npz", rows_all, M.answer, {"dlog_rms": dlog / len(rows_all)})


def ablation_targets(M: NormModel) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for li in range(M.n_layer):
        for s in M.sites[li].values():
            out += [(li, int(c), nm) for c, nm in zip(s.cols, s.names, strict=False)]
    return out


def ablate(M: NormModel, shard: int, n_shards: int, n_prompts: int = 500) -> None:
    rows = subset(n_prompts, seed=2)
    targets = ablation_targets(M)[shard::n_shards]
    lo, _, _ = M.forward_chunk(rows, False)
    ld, rd, _ = M.forward_chunk(rows, True)
    frz = forced(rd)
    kl_d = kl_rows(lo, ld)
    ans = jnp.asarray(M.answer[rows])[:, None]
    lpa_d = jnp.take_along_axis(ld, ans, 1)[:, 0]
    act = {li: M.masks(li, rows) for li in range(M.n_layer)}
    res = {k: np.zeros((len(targets), n_prompts), np.float32)
           for k in ("kl_free", "kl_frozen", "dkl_orig_free", "dkl_orig_frozen", "dlp_free", "dlp_frozen")}  # fmt: skip
    n_act = np.zeros((len(targets), T), np.int32)
    drms = np.zeros((len(targets), N_NORM, T), np.float32)  # mean log(rms_ablated / rms_dec), free
    t0 = time.time()
    for i, (li, col, _) in enumerate(targets):
        n_act[i] = act[li][:, :, col].sum(0)
        if n_act[i].sum() == 0:
            continue
        ab = {li: np.array([col])}
        for cond, hook in (("free", None), ("frozen", frz)):
            lp, r, _ = M.forward_chunk(rows, True, norm=hook, ablate=ab)
            res[f"kl_{cond}"][i] = np.asarray(kl_rows(ld, lp))
            res[f"dkl_orig_{cond}"][i] = np.asarray(kl_rows(lo, lp) - kl_d)
            res[f"dlp_{cond}"][i] = np.asarray(jnp.take_along_axis(lp, ans, 1)[:, 0] - lpa_d)
            if hook is None:
                drms[i] = np.asarray(jnp.log(r / rd).mean(1))
        if i % 100 == 0:
            log(f"shard {shard}: {i}/{len(targets)} ({time.time() - t0:.0f}s)")
    (OUT / "ablate").mkdir(parents=True, exist_ok=True)
    savez(OUT / "ablate" / f"shard{shard}of{n_shards}.npz", rows=rows, names=np.array([t[2] for t in targets]), layer=np.array([t[0] for t in targets]),
             col=np.array([t[1] for t in targets]), n_act=n_act, drms=drms, **res)  # fmt: skip


def groups(M: NormModel, n_prompts: int = 2000) -> None:
    """Ablate every alive component of one site / one block at once (where it is active), free vs frozen."""
    rows_all = subset(n_prompts, seed=3)
    targets = ablation_targets(M)
    by_site: dict[str, tuple[int, list[int]]] = {}
    for li, col, nm in targets:
        key = ".".join(nm.split(".")[:2])
        by_site.setdefault(key, (li, []))[1].append(col)
    grp = {k: {v[0]: np.array(v[1])} for k, v in by_site.items()}
    for li in range(M.n_layer):
        grp[f"L{li}"] = {li: np.array([c for l2, c, _ in targets if l2 == li])}
    res = {k: np.zeros((len(grp), len(rows_all)), np.float32) for k in ("kl_free", "kl_frozen", "dkl_orig_free", "dkl_orig_frozen")}  # fmt: skip
    drms = np.zeros((len(grp), N_NORM, T), np.float64)
    s0 = 0
    for rows, n in M.chunks(rows_all):
        sl = slice(s0, s0 + n)
        lo, _, _ = M.forward_chunk(rows, False)
        ld, rd, _ = M.forward_chunk(rows, True)
        kl_d = kl_rows(lo, ld)
        frz = forced(rd)
        for gi, ab in enumerate(grp.values()):
            for cond, hook in (("free", None), ("frozen", frz)):
                lp, r, _ = M.forward_chunk(rows, True, norm=hook, ablate=ab)
                res[f"kl_{cond}"][gi, sl] = np.asarray(kl_rows(ld, lp))[:n]
                res[f"dkl_orig_{cond}"][gi, sl] = np.asarray(kl_rows(lo, lp) - kl_d)[:n]
                if hook is None:
                    drms[gi] += np.asarray(jnp.log(r[:, :n] / rd[:, :n]).sum(1), np.float64)
        s0 += n
        log(f"groups {s0}/{len(rows_all)}")
    savez(
        OUT / "groups.npz",
        rows=rows_all,
        names=np.array(list(grp)),
        drms=drms / len(rows_all),
        **res,
    )


def swap(M: NormModel) -> None:
    """One norm point at a time (every other norm free): the decomposed model forced to the original
    model's per-prompt rms there, the original model forced to the decomposed one's, and the original
    model frozen at its pool mean there. Reference: the original model."""
    b = load_base()
    ro, rd = b["rms_o"], b["rms_d"]
    conds = {}
    for t in range(N_NORM):
        conds[f"dec_forced_orig_t{t}"] = (
            True,
            lambda rows, t=t: forced(jnp.asarray(ro[:, rows]), {t}),
        )
        conds[f"orig_forced_dec_t{t}"] = (
            False,
            lambda rows, t=t: forced(jnp.asarray(rd[:, rows]), {t}),
        )
    run_conditions(M, subset(2000, seed=4), conds, "swap")


def split(M: NormModel, shard: int, n_shards: int, n_prompts: int = 500) -> None:
    """Each component's ablation effect split into what it writes into the stream (direct_only: write
    dropped, its energy kept in every later norm) and its share of the later norms (norm_only: write
    kept, its energy removed from every later norm). Same prompts as `ablate`."""
    rows = subset(n_prompts, seed=2)
    targets = ablation_targets(M)[shard::n_shards]
    ld, _, _ = M.forward_chunk(rows, True)
    act = {li: M.masks(li, rows) for li in range(M.n_layer)}
    res = {
        k: np.zeros((len(targets), n_prompts), np.float32)
        for k in ("kl_norm_only", "kl_direct_only")
    }
    t0 = time.time()
    for i, (li, col, nm) in enumerate(targets):
        if act[li][:, :, col].sum() == 0:
            continue
        off = 0 if nm.split(".")[1] in ("q", "k", "v", "o") else 1
        for mode in ("norm_only", "direct_only"):
            lp, _, _ = M.forward_chunk(rows, True, split=(li, off, np.array([col]), mode))
            res[f"kl_{mode}"][i] = np.asarray(kl_rows(ld, lp))
        if i % 100 == 0:
            log(f"split shard {shard}: {i}/{len(targets)} ({time.time() - t0:.0f}s)")
    (OUT / "split").mkdir(parents=True, exist_ok=True)
    savez(
        OUT / "split" / f"shard{shard}of{n_shards}.npz",
        rows=rows,
        names=np.array([t[2] for t in targets]),
        **res,
    )


PROBE = ["L31.gate.c14", "L31.gate.c238", "L31.gate.c288", "L31.up.c534", "L31.up.c36", "L31.up.c50",
         "L31.up.c16", "L31.down.c9", "L31.down.c22", "L31.up.c731", "L31.down.c18", "L31.down.c23"]  # fmt: skip


def probe31(M: NormModel) -> None:
    """For last-block MLP components: their write w into the final stream at `=` (the final residual
    minus the one with their write dropped), its size and angle to the rest of the stream, how much of
    the logit change it would make is a constant shift over the vocabulary (softmax-invisible), and the
    output entropy under ablation / norm-only / direct-only."""
    rows = subset(CHUNK_N, seed=2)
    ld, rd, caps = M.forward_chunk(rows, True, capture={FINAL})
    xf = caps[FINAL][:, -1]
    rms_f = rd[FINAL, :, -1]
    g = M.final_j

    def entropy(lp: jax.Array) -> np.ndarray:
        return np.asarray(-(jnp.exp(lp) * lp).sum(-1))

    out = {"H_dec": entropy(ld)}
    for nm in PROBE:
        li = int(nm.split(".")[0][1:])
        kind, j = M.column(li, nm)
        col = np.array([M.sites[li][kind].cols[j]])
        _, _, cap_d = M.forward_chunk(
            rows, True, capture={FINAL}, split=(li, 1, col, "direct_only")
        )
        w = xf - cap_d[FINAL][:, -1]
        rest = xf - w
        dlog = (
            (w / rms_f[:, None]) * g
        ) @ M.unembed.T  # (B, V) logit change the write makes at fixed rms
        const = dlog.mean(-1) ** 2 / (dlog**2).mean(-1)
        rec = {"w_rel": np.asarray(jnp.linalg.norm(w, axis=-1) / jnp.linalg.norm(xf, axis=-1)),
               "cos_rest": np.asarray((w * rest).sum(-1) / (jnp.linalg.norm(w, axis=-1) * jnp.linalg.norm(rest, axis=-1))),
               "const_share": np.asarray(const),
               "dlogit_mean": np.asarray(dlog.mean(-1)), "dlogit_sd": np.asarray(dlog.std(-1))}  # fmt: skip
        ab = {li: col}
        rec["H_ablate"] = entropy(M.forward_chunk(rows, True, ablate=ab)[0])
        for mode in ("norm_only", "direct_only"):
            rec[f"H_{mode}"] = entropy(M.forward_chunk(rows, True, split=(li, 1, col, mode))[0])
        out |= {f"{nm}__{k}": v for k, v in rec.items()}
        log(
            nm,
            " ".join(f"{k} {float(np.mean(v)):+.3f}" for k, v in rec.items()),
            f"H_dec {out['H_dec'].mean():.3f}",
        )
    savez(OUT / "probe31.npz", rows=rows, **out)


def probe_orig(M: NormModel) -> None:
    """The PROBE components ablated in the ORIGINAL model (their rank-one term subtracted from the dense
    weights, every prompt): ablate / norm_only / direct_only, and ablate with the final norm frozen."""
    rows = subset(CHUNK_N, seed=2)
    lo, ro, _ = M.forward_chunk(rows, False)

    def entropy(lp: jax.Array) -> float:
        return float(-(jnp.exp(lp) * lp).sum(-1).mean())

    log(f"orig H {entropy(lo):.3f}")
    out = {}
    for nm in PROBE:
        li = int(nm.split(".")[0][1:])
        kind, j = M.column(li, nm)
        col = np.array([M.sites[li][kind].cols[j]])
        res = {}
        for mode in ("ablate", "norm_only", "direct_only"):
            lp, _, _ = M.forward_chunk(rows, False, split=(li, 1, col, mode))
            res[mode] = (float(kl_rows(lo, lp).mean()), entropy(lp))
        lp, _, _ = M.forward_chunk(
            rows, False, split=(li, 1, col, "ablate"), norm=forced(ro, {FINAL})
        )
        res["ablate_final_frozen"] = (float(kl_rows(lo, lp).mean()), entropy(lp))
        out[nm] = res
        log(nm, " ".join(f"{k}: KL {v[0]:.4f} H {v[1]:.3f}" for k, v in res.items()))
    (OUT / "probe_orig.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    M = NormModel()
    log("model loaded")
    if cmd in ("ablate", "split"):
        {"ablate": ablate, "split": split}[cmd](M, int(sys.argv[2]), int(sys.argv[3]))
    else:
        {
            "base": base,
            "orig": orig,
            "dec": dec,
            "hybrid": hybrid,
            "groups": groups,
            "swap": swap,
            "probe31": probe31,
            "probe_orig": probe_orig,
        }[cmd](M)
