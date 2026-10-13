"""Which gate/up components (readers) each down component (writer) of one MLP block depends on.

Input: `extract_layer.py`'s OUT (inner.npy, masks.npy, comps.npz). Notation, at one position t:

    s_r = m_r a_r                      masked inner activation of reader r (gate or up)
    G   = sum_{r in gate} s_r U_r      gate pre-activation (d_mlp,)
    H   = sum_{r in up}   s_r U_r      up pre-activation (d_mlp,)
    y_w = V_w . (silu(G) * H)          inner activation of writer w

(1) Single-reader ablations, exact, on the fit half's prompts where w is on (m_w = 1):
    mean ablation: delta_mean(r, w) = y_w(s_r -> mean_s[w, r]) - y_w, mean_s[w, r] = E[s_r | m_w = 1]
    zero ablation: delta_zero(r, w) = y_w(s_r -> 0) - y_w
    (up readers enter y_w linearly, so their deltas are (target - s_r) * sum_n V_w[n] U_r[n]
    silu(G_n); gate readers are recomputed through silu). Scores: score(r, w) = E[delta_mean^2 | on]
    (the ranking), score_zero(r, w) = E[delta_zero^2 | on]. A reader with a small score but a large
    score_zero is an OFFSET reader of w: needed, but its variation does not matter.
(2) Validation: keep the k readers of largest score(r, w) (mean-ablation ranking), replace every other s_r by its mean over the
    fit half's writer-on points, recompute y_w; R^2_k on held-out writer-on points.
    Same with a weights-only ranking: rms(s_r) * ||V_w * U_r||_2. For k at R^2 >= 0.9 and 0.99,
    the left-out readers' mean contribution is stored as offset_gate / offset_up (W, d_mlp), so
    y_w can be predicted from its support readers alone.
(3) Greedy forward selection (best R^2 subset, fit half) for the writers carrying the most of
    the block's write, to measure how far the mean-ablation ranking is from the best subset.

    python attribute.py --layer 18 --positions 4 3
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

RUN_DIR = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
K_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 320)
N_EVAL = 20000
BATCH = 100
OFFSET_MEAN_MAX = 0.01
OFFSET_ZERO_MIN = 0.01
N_GREEDY_FIT = 1500
N_GREEDY_WRITERS = 30
K_GREEDY = 15


@jax.jit
def recompute(sg, su, Ug, Uu, Vd):  # noqa: ANN001, ANN201
    """y (B, W) from the readers' masked activations."""
    return (jax.nn.silu(sg @ Ug) * (su @ Uu)) @ Vd


@jax.jit
def ablation_sums(sg, su, Ug, Uu, Vd, mean_s, won):  # noqa: ANN001, ANN201
    """Writer-on weighted sums over the block of delta_mean^2 and delta_zero^2, (W, R), and y.
    sg (B, Rg), su (B, Ru), mean_s (W, R), won (B, W) in {0, 1}."""
    rg = sg.shape[1]
    G, H = sg @ Ug, su @ Uu
    sG = jax.nn.silu(G)
    y = (sG * H) @ Vd
    # up readers: exact linear deltas
    A = jnp.einsum("bn,nw,rn->bwr", sG, Vd, Uu)
    d_up_mean = (mean_s[None, :, rg:] - su[:, None, :]) * A
    d_up_zero = -su[:, None, :] * A

    def gate_one(r):  # noqa: ANN001, ANN202
        u = Ug[r]
        dm = mean_s[:, r][None, :] - sg[:, r][:, None]  # (B, W)
        Gp = G[:, None, :] + dm[..., None] * u  # (B, W, n)
        d_mean = jnp.einsum("bwn,bn,nw->bw", jax.nn.silu(Gp) - sG[:, None, :], H, Vd)
        Gz = G - sg[:, r][:, None] * u  # (B, n)
        d_zero = (jax.nn.silu(Gz) * H) @ Vd - y
        return d_mean, d_zero

    d_g_mean, d_g_zero = jax.lax.map(gate_one, jnp.arange(rg))  # (Rg, B, W)
    d_mean = jnp.concatenate([jnp.moveaxis(d_g_mean, 0, -1), d_up_mean], -1)
    d_zero = jnp.concatenate([jnp.moveaxis(d_g_zero, 0, -1), d_up_zero], -1)
    w = won[..., None]
    return jnp.sum(w * d_mean**2, 0), jnp.sum(w * d_zero**2, 0), y


@jax.jit
def topk_predict(dev, mean_w, sel, Ug, Uu, v):  # noqa: ANN001, ANN201
    """One writer: y_hat (B, nk) with readers outside `sel` (nk, R) at their means.
    dev (B, R) = s - mean_w, mean_w (R,), v (d_mlp,) = V_w."""
    rg = Ug.shape[0]
    G0, H0 = mean_w[:rg] @ Ug, mean_w[rg:] @ Uu
    G = G0 + jnp.einsum("br,kr,rn->bkn", dev[:, :rg], sel[:, :rg], Ug)
    H = H0 + jnp.einsum("br,kr,rn->bkn", dev[:, rg:], sel[:, rg:], Uu)
    return (jax.nn.silu(G) * H) @ v


def padded(fn, x, size, *args):  # noqa: ANN001, ANN002, ANN201
    """fn over row blocks of exactly `size` rows (zero-padded; one compile), concatenated."""
    outs = []
    for start in range(0, x.shape[0], size):
        blk = x[start : start + size]
        real = blk.shape[0]
        if real < size:
            blk = jnp.concatenate([blk, jnp.zeros((size - real, *blk.shape[1:]), blk.dtype)])
        outs.append(np.asarray(fn(blk, *args))[:real])
    return np.concatenate(outs)


def recompute_cols(x, rg, Ug, Uu, Vd):  # noqa: ANN001, ANN201
    return recompute(x[:, :rg], x[:, rg:], Ug, Uu, Vd)


def r2(y, yhat):  # noqa: ANN001, ANN201
    return 1 - np.mean((y - yhat) ** 2, 0) / np.var(y)


def greedy(dev, mean_w, Ug, Uu, v, y, k_max):  # noqa: ANN001, ANN201
    """Forward selection maximising R^2 on (dev, y); returns chosen readers and R^2 path."""
    rg, R = Ug.shape[0], dev.shape[1]
    U_all = jnp.concatenate([Ug, Uu])
    is_gate = jnp.arange(R) < rg
    G, H = (
        jnp.tile(mean_w[:rg] @ Ug, (dev.shape[0], 1)),
        jnp.tile(mean_w[rg:] @ Uu, (dev.shape[0], 1)),
    )

    @jax.jit
    def scores(G, H, cand):  # noqa: ANN001, ANN202
        upd = dev[:, cand, None] * U_all[cand][None]  # (B, C, n)
        g = is_gate[cand][None, :, None]
        Gc = G[:, None] + jnp.where(g, upd, 0)
        Hc = H[:, None] + jnp.where(g, 0, upd)
        yh = (jax.nn.silu(Gc) * Hc) @ v
        return jnp.mean((y[:, None] - yh) ** 2, 0)

    chosen, path, var = [], [], float(jnp.var(y))
    for _ in range(min(k_max, R)):
        rest = np.setdiff1d(np.arange(R), chosen)
        mse = []
        for start in range(0, rest.size, 24):
            c = rest[start : start + 24]
            cp = np.concatenate([c, np.full(24 - c.size, c[0])])
            mse.append(np.asarray(scores(G, H, jnp.asarray(cp)))[: c.size])
        mse = np.concatenate(mse)
        best = int(rest[np.argmin(mse)])
        chosen.append(best)
        path.append(1 - float(mse.min()) / var)
        upd = dev[:, best, None] * U_all[best][None]
        if best < rg:
            G = G + upd
        else:
            H = H + upd
    return chosen, path


def run_position(out: Path, t: int, rng: np.random.Generator) -> dict:
    comps = np.load(out / "comps.npz")
    kind = comps["kind"]
    gi, ui, di = (np.flatnonzero(kind == k) for k in ("gate", "up", "down"))
    inner = np.load(out / "inner.npy", mmap_mode="r")
    masks = np.load(out / "masks.npy", mmap_mode="r")
    t0 = time.time()
    a = np.asarray(inner[:, t], np.float32)
    m = np.asarray(masks[:, t])
    print(f"t={t}: loaded {a.shape} in {time.time() - t0:.0f}s", flush=True)
    rcols = np.concatenate([gi, ui])
    on_r = m[:, rcols].any(0)
    on_w = m[:, di].any(0)
    rcols, wcols = rcols[on_r], di[on_w]
    rg = int(on_r[: gi.size].sum())
    s = a[:, rcols] * m[:, rcols]
    y_store, mw = a[:, wcols], m[:, wcols]
    Ug = jnp.asarray(comps["U_gate"][on_r[: gi.size]])
    Uu = jnp.asarray(comps["U_up"][on_r[gi.size :]])
    Vd = jnp.asarray(comps["V_down"][:, on_w])
    R, W, N = s.shape[1], wcols.size, s.shape[0]
    print(f"t={t}: {rg} gate + {R - rg} up readers, {W} writers on somewhere", flush=True)

    perm = rng.permutation(N)
    fit, held = np.sort(perm[: N // 2]), perm[N // 2 :]
    ev = np.sort(held[:N_EVAL])

    # pass 1a (fit half): per-writer means and rms of the readers over writer-on prompts
    w8_fit = mw[fit].astype(np.float64)
    cnt = w8_fit.sum(0)
    cntc = np.maximum(cnt, 1)[:, None]
    s_fit = s[fit].astype(np.float64)
    mean_s = (w8_fit.T @ s_fit) / cntc
    rms_s = np.sqrt((w8_fit.T @ s_fit**2) / cntc)
    del s_fit
    # pass 1b (fit half): single-reader mean / zero ablations, y vs stored
    sum_mean2, sum_zero2 = np.zeros((W, R)), np.zeros((W, R))
    sum_y, sum_y2 = np.zeros(W), np.zeros(W)
    err_store, var_store = 0.0, 0.0
    mean_j = jnp.asarray(mean_s, jnp.float32)
    for start in range(0, fit.size, BATCH):
        c = fit[start : start + BATCH]
        real = c.size
        c = np.concatenate([c, fit[: BATCH - real]])  # pad to one shape; padded rows weigh 0
        sc = jnp.asarray(s[c])
        w8 = mw[c].astype(np.float32)
        w8[real:] = 0
        dm2, dz2, y = ablation_sums(sc[:, :rg], sc[:, rg:], Ug, Uu, Vd, mean_j, jnp.asarray(w8))
        y = np.asarray(y, np.float64)
        sum_mean2 += np.asarray(dm2, np.float64)
        sum_zero2 += np.asarray(dz2, np.float64)
        sum_y += np.sum(w8 * y, 0)
        sum_y2 += np.sum(w8 * y**2, 0)
        err_store += float(np.sum(w8 * (y - y_store[c]) ** 2))
        var_store += float(np.sum(w8 * y_store[c] ** 2))
        if start // BATCH % 200 == 0:
            print(f"t={t}: pass 1b {start}/{fit.size} {time.time() - t0:.0f}s", flush=True)
    score = sum_mean2 / cntc
    score_zero = sum_zero2 / cntc
    msq_fit = sum_y2 / np.maximum(cnt, 1)
    var_fit = msq_fit - (sum_y / np.maximum(cnt, 1)) ** 2
    # offset readers: holding them at their mean costs < OFFSET_MEAN_MAX of Var(y_w), switching
    # them off costs > OFFSET_ZERO_MIN of E[y_w^2]
    frac_mean = score / np.maximum(var_fit, 1e-30)[:, None]
    frac_zero = score_zero / np.maximum(msq_fit, 1e-30)[:, None]
    offset = (frac_mean < OFFSET_MEAN_MAX) & (frac_zero > OFFSET_ZERO_MIN)
    print(f"t={t}: offset reader-writer pairs {int(offset.sum())}; writers with >= 1: "
          f"{int(offset.any(1).sum())}/{W}", flush=True)  # fmt: skip
    vu = np.asarray(jnp.concatenate([Ug, Uu]))  # (R, n)
    Vn = np.asarray(Vd)  # (n, W)
    weight_score = rms_s * np.sqrt((Vn.T**2) @ (vu.T**2))  # ||V_w * U_r||_2, (W, R)
    print(f"t={t}: pass 1 {time.time() - t0:.0f}s; "
          f"recomputed vs stored y (writer-on) rel sq err {err_store / var_store:.2e}", flush=True)  # fmt: skip

    # pass 2 (held-out): R^2 of top-k readers, mean-ablation ranking and weights-only ranking
    ks = np.asarray([k for k in K_GRID if k < R] + [R])
    rank_abl = np.argsort(-score, 1)
    rank_wt = np.argsort(-weight_score, 1)
    r2_abl, r2_wt = np.full((W, ks.size), np.nan), np.full((W, ks.size), np.nan)
    y_rec = np.zeros((ev.size, W), np.float32)
    s_ev = s[ev]
    on_ev = mw[ev]
    sc_ev = jnp.asarray(s_ev)
    y_rec[:] = padded(recompute_cols, sc_ev, 500, rg, Ug, Uu, Vd)
    for w in range(W):
        rows = np.flatnonzero(on_ev[:, w])
        if rows.size < 20:
            continue
        dev = jnp.asarray(s_ev[rows] - mean_s[w], jnp.float32)
        yt = y_rec[rows, w]
        for rank, dest in ((rank_abl, r2_abl), (rank_wt, r2_wt)):
            sel = np.zeros((ks.size, R), np.float32)
            for j, k in enumerate(ks):
                sel[j, rank[w, :k]] = 1
            mean_w, selj, v = jnp.asarray(mean_s[w], jnp.float32), jnp.asarray(sel), Vd[:, w]
            yh = padded(topk_predict, dev, 400, mean_w, selj, Ug, Uu, v)
            dest[w] = r2(yt[:, None], yh)
    print(f"t={t}: pass 2 {time.time() - t0:.0f}s", flush=True)

    # per-writer predictor at the smallest k reaching R^2 >= target (mean-ablation ranking): the support
    # (first k of rank_abl) and the offsets the left-out readers contribute at their means,
    #   offset_gate[w] = sum_{gate r not in support} mean_s[w, r] U_r   (d_mlp,), same for up,
    # so y_w ~ V_w . (silu(offset_gate[w] + sum_{gate r in S} s_r U_r) * (offset_up[w] + ...)).
    predictors = {}
    Ug_np, Uu_np = np.asarray(Ug), np.asarray(Uu)
    for target in (0.9, 0.99):
        hit = r2_abl >= target
        k_at = np.where(hit.any(1), ks[np.argmax(hit, 1)], -1)
        off_g = np.zeros((W, Ug_np.shape[1]), np.float32)
        off_u = np.zeros((W, Uu_np.shape[1]), np.float32)
        for w in range(W):
            out_mask = np.ones(R, bool)
            out_mask[rank_abl[w, : max(k_at[w], 0)]] = False
            ms = np.where(out_mask, mean_s[w], 0.0)
            off_g[w], off_u[w] = ms[:rg] @ Ug_np, ms[rg:] @ Uu_np
        tag = f"r2_{round(target * 100)}"
        predictors[f"k_{tag}"] = k_at
        predictors[f"offset_gate_{tag}"] = off_g
        predictors[f"offset_up_{tag}"] = off_u

    # writer weight: share of the block's write, E[y_w^2 | on] P(on) ||U_w||^2 (U_w not stored:
    # use the writer's own output norm from the run's index, else 1)
    p_on = mw.mean(0)
    var_on = np.asarray(
        [np.var(y_rec[on_ev[:, w], w]) if on_ev[:, w].sum() > 1 else 0.0 for w in range(W)]
    )
    msq_on = np.asarray(
        [np.mean(y_rec[on_ev[:, w], w] ** 2) if on_ev[:, w].any() else 0.0 for w in range(W)]
    )
    U_down = comps["U_down"][on_w] if "U_down" in comps.files else np.ones((W, 1))
    write = p_on * msq_on * (U_down**2).sum(1)

    # pass 3: greedy on the biggest writers (fit half subset), evaluated held-out
    greedy_res = []
    order = np.argsort(-write)
    for w in order[:N_GREEDY_WRITERS]:
        rows_fit = fit[mw[fit, w]]
        rows_fit = rng.choice(rows_fit, min(N_GREEDY_FIT, rows_fit.size), replace=False)
        rows_ev = np.flatnonzero(on_ev[:, w])
        if rows_fit.size < 50 or rows_ev.size < 20:
            continue
        sf = jnp.asarray(s[rows_fit])
        yf = recompute(sf[:, :rg], sf[:, rg:], Ug, Uu, Vd[:, w : w + 1])[:, 0]
        mean_w = jnp.asarray(mean_s[w], jnp.float32)
        chosen, path = greedy(sf - mean_w, mean_w, Ug, Uu, Vd[:, w], yf, K_GREEDY)
        sel = np.zeros((len(chosen), R), np.float32)
        for j in range(len(chosen)):
            sel[j, chosen[: j + 1]] = 1
        dev = jnp.asarray(s_ev[rows_ev] - mean_s[w], jnp.float32)
        selj, v = jnp.asarray(sel), Vd[:, w]
        yh = padded(topk_predict, dev, 400, mean_w, selj, Ug, Uu, v)
        greedy_res.append({"writer": int(w), "chosen": chosen, "r2_fit": path,
                           "r2_held": r2(y_rec[rows_ev, w][:, None], yh).tolist()})  # fmt: skip
    print(f"t={t}: pass 3 {time.time() - t0:.0f}s", flush=True)

    np.savez(
        out / f"attr_pos{t}.npz",
        reader_cols=rcols, writer_cols=wcols, n_gate=rg, score=score, score_zero=score_zero, frac_mean=frac_mean, frac_zero=frac_zero, offset=offset, var_fit=var_fit, msq_fit=msq_fit,
        mean_s=mean_s, rms_s=rms_s, weight_score=weight_score, ks=ks, r2_abl=r2_abl, r2_wt=r2_wt,
        p_on=p_on, var_on=var_on, msq_on=msq_on, write=write, n_on_fit=cnt,
        reader_index=comps["index"][rcols], writer_index=comps["index"][wcols],
        rank_abl=rank_abl, **predictors,
    )  # fmt: skip
    (out / f"greedy_pos{t}.json").write_text(json.dumps(greedy_res))
    return {"t": t, "n_gate": rg, "n_up": R - rg, "n_writers": W,
            "recomputed_vs_stored_rel_sq_err": err_store / var_store}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--positions", type=int, nargs="+", default=[4])
    args = ap.parse_args()
    out = RUN_DIR / "analysis" / "reader_writer" / f"L{args.layer}"
    rng = np.random.default_rng(0)
    summary = [run_position(out, t, rng) for t in args.positions]
    (out / "attribute_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
