"""Which gate/up components (readers) each down component (writer) of one MLP block depends on.

Input: `extract_layer.py`'s OUT (inner.npy, masks.npy, comps.npz). Notation, at one position t:

    s_r = m_r a_r                      masked inner activation of reader r (gate or up)
    G   = sum_{r in gate} s_r U_r      gate pre-activation (d_mlp,)
    H   = sum_{r in up}   s_r U_r      up pre-activation (d_mlp,)
    y_w = V_w . (silu(G) * H)          inner activation of writer w

(1) Exact attribution (integrated gradients from the all-readers-off point, where y_w = 0):
    phi_r(w) for up r   = s_r sum_n V_w[n] U_r[n] P(G_n),        P(x) = int_0^1 silu(alpha x) dalpha
    phi_r(w) for gate r = s_r sum_n V_w[n] U_r[n] H_n Q(G_n),    Q(x) = int_0^1 alpha silu'(alpha x) dalpha
    and sum_r phi_r(w) = y_w on every point (completeness; checked). Score of reader r for writer
    w: score(r, w) = E[phi_r(w)^2 | m_w = 1] on the fit half of the prompts.
(2) Validation: keep the k readers of largest score(r, w), replace every other s_r by its mean over the
    fit half's writer-on points, recompute y_w; R^2_k on held-out writer-on points.
    Same with a weights-only ranking: rms(s_r) * ||V_w * U_r||_2. For k at R^2 >= 0.9 and 0.99,
    the left-out readers' mean contribution is stored as offset_gate / offset_up (W, d_mlp), so
    y_w can be predicted from its support readers alone.
(3) Greedy forward selection (best R^2 subset, fit half) for the writers carrying the most of
    the block's write, to measure how far the IG ranking is from the best subset.

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
N_GREEDY_FIT = 1500
N_GREEDY_WRITERS = 30
K_GREEDY = 15
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(24)
ALPHA = jnp.asarray((GL_NODES + 1) / 2, jnp.float32)
ALPHA_W = jnp.asarray(GL_WEIGHTS / 2, jnp.float32)


def silu_prime(x):  # noqa: ANN001, ANN201
    sig = jax.nn.sigmoid(x)
    return sig * (1 + x * (1 - sig))


@jax.jit
def pq(x):  # noqa: ANN001, ANN201
    """P(x), Q(x) by 24-point Gauss-Legendre on alpha in [0, 1]."""
    ax = x[..., None] * ALPHA
    p = jnp.sum(jax.nn.silu(ax) * ALPHA_W, -1)
    q = jnp.sum(ALPHA * silu_prime(ax) * ALPHA_W, -1)
    return p, q


@jax.jit
def attribution(sg, su, Ug, Uu, Vd):  # noqa: ANN001, ANN201
    """phi (B, W, Rg + Ru) and the recomputed y (B, W)."""
    G, H = sg @ Ug, su @ Uu
    p, q = pq(G)
    phi_g = jnp.einsum("bn,nw,rn->bwr", H * q, Vd, Ug) * sg[:, None, :]
    phi_u = jnp.einsum("bn,nw,rn->bwr", p, Vd, Uu) * su[:, None, :]
    y = (jax.nn.silu(G) * H) @ Vd
    return jnp.concatenate([phi_g, phi_u], -1), y


@jax.jit
def topk_predict(dev, mean_w, sel, Ug, Uu, v):  # noqa: ANN001, ANN201
    """One writer: y_hat (B, nk) with readers outside `sel` (nk, R) at their means.
    dev (B, R) = s - mean_w, mean_w (R,), v (d_mlp,) = V_w."""
    rg = Ug.shape[0]
    G0, H0 = mean_w[:rg] @ Ug, mean_w[rg:] @ Uu
    G = G0 + jnp.einsum("br,kr,rn->bkn", dev[:, :rg], sel[:, :rg], Ug)
    H = H0 + jnp.einsum("br,kr,rn->bkn", dev[:, rg:], sel[:, rg:], Uu)
    return (jax.nn.silu(G) * H) @ v


@jax.jit
def attribution_sums(sg, su, Ug, Uu, Vd, won):  # noqa: ANN001, ANN201
    """Writer-on weighted sums of phi^2 and phi (W, R), recomputed y (B, W), and the
    completeness residual max |sum_r phi_r - y|."""
    phi, y = attribution(sg, su, Ug, Uu, Vd)
    return (jnp.einsum("bw,bwr->wr", won, phi**2), jnp.einsum("bw,bwr->wr", won, phi), y,
            jnp.max(jnp.abs(phi.sum(-1) - y)))  # fmt: skip


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


def recompute(x, rg, Ug, Uu, Vd):  # noqa: ANN001, ANN201
    return attribution(x[:, :rg], x[:, rg:], Ug, Uu, Vd)[1]


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
    for _ in range(k_max):
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

    # pass 1 (fit half): attribution sums, means, completeness
    sum_phi2 = np.zeros((W, R))
    sum_phi = np.zeros((W, R))
    sum_s = np.zeros((W, R))
    sum_s2 = np.zeros((W, R))
    cnt = np.zeros(W)
    err_compl, err_store, var_store = 0.0, 0.0, 0.0
    max_abs_y = 0.0
    for start in range(0, fit.size, 500):
        c = fit[start : start + 500]
        real = c.size
        c = np.concatenate([c, fit[: 500 - real]])  # pad to one shape; padded rows weigh 0
        sc = jnp.asarray(s[c])
        w8 = mw[c].astype(np.float32)
        w8[real:] = 0
        p2, p1, y, res = attribution_sums(sc[:, :rg], sc[:, rg:], Ug, Uu, Vd, jnp.asarray(w8))
        y = np.asarray(y, np.float64)
        sum_phi2 += np.asarray(p2, np.float64)
        sum_phi += np.asarray(p1, np.float64)
        sc64 = s[c].astype(np.float64)
        sum_s += w8.T.astype(np.float64) @ sc64
        sum_s2 += w8.T.astype(np.float64) @ sc64**2
        cnt += w8.sum(0)
        err_compl = max(err_compl, float(res))
        max_abs_y = max(max_abs_y, float(np.abs(y).max()))
        err_store += float(np.sum(w8 * (y - y_store[c]) ** 2))
        var_store += float(np.sum(w8 * y_store[c] ** 2))
    err_compl /= max_abs_y
    cntc = np.maximum(cnt, 1)[:, None]
    score = sum_phi2 / cntc
    mean_s = sum_s / cntc
    rms_s = np.sqrt(sum_s2 / cntc)
    vu = np.asarray(jnp.concatenate([Ug, Uu]))  # (R, n)
    Vn = np.asarray(Vd)  # (n, W)
    weight_score = rms_s * np.sqrt((Vn.T**2) @ (vu.T**2))  # ||V_w * U_r||_2, (W, R)
    print(f"t={t}: pass 1 {time.time() - t0:.0f}s; completeness max rel err {err_compl:.2e}; "
          f"recomputed vs stored y (writer-on) rel sq err {err_store / var_store:.2e}", flush=True)  # fmt: skip

    # pass 2 (held-out): R^2 of top-k readers, IG ranking and weights-only ranking
    ks = np.asarray([k for k in K_GRID if k < R] + [R])
    rank_ig = np.argsort(-score, 1)
    rank_wt = np.argsort(-weight_score, 1)
    r2_ig, r2_wt = np.full((W, ks.size), np.nan), np.full((W, ks.size), np.nan)
    y_rec = np.zeros((ev.size, W), np.float32)
    s_ev = s[ev]
    on_ev = mw[ev]
    sc_ev = jnp.asarray(s_ev)
    y_rec[:] = padded(recompute, sc_ev, 500, rg, Ug, Uu, Vd)
    for w in range(W):
        rows = np.flatnonzero(on_ev[:, w])
        if rows.size < 20:
            continue
        dev = jnp.asarray(s_ev[rows] - mean_s[w], jnp.float32)
        yt = y_rec[rows, w]
        for rank, dest in ((rank_ig, r2_ig), (rank_wt, r2_wt)):
            sel = np.zeros((ks.size, R), np.float32)
            for j, k in enumerate(ks):
                sel[j, rank[w, :k]] = 1
            mean_w, selj, v = jnp.asarray(mean_s[w], jnp.float32), jnp.asarray(sel), Vd[:, w]
            yh = padded(topk_predict, dev, 400, mean_w, selj, Ug, Uu, v)
            dest[w] = r2(yt[:, None], yh)
    print(f"t={t}: pass 2 {time.time() - t0:.0f}s", flush=True)

    # per-writer predictor at the smallest k reaching R^2 >= target (IG ranking): the support
    # (first k of rank_ig) and the offsets the left-out readers contribute at their means,
    #   offset_gate[w] = sum_{gate r not in support} mean_s[w, r] U_r   (d_mlp,), same for up,
    # so y_w ~ V_w . (silu(offset_gate[w] + sum_{gate r in S} s_r U_r) * (offset_up[w] + ...)).
    predictors = {}
    Ug_np, Uu_np = np.asarray(Ug), np.asarray(Uu)
    for target in (0.9, 0.99):
        hit = r2_ig >= target
        k_at = np.where(hit.any(1), ks[np.argmax(hit, 1)], -1)
        off_g = np.zeros((W, Ug_np.shape[1]), np.float32)
        off_u = np.zeros((W, Uu_np.shape[1]), np.float32)
        for w in range(W):
            out_mask = np.ones(R, bool)
            out_mask[rank_ig[w, : max(k_at[w], 0)]] = False
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
        yf = attribution(sf[:, :rg], sf[:, rg:], Ug, Uu, Vd[:, w : w + 1])[1][:, 0]
        mean_w = jnp.asarray(mean_s[w], jnp.float32)
        chosen, path = greedy(sf - mean_w, mean_w, Ug, Uu, Vd[:, w], yf, K_GREEDY)
        sel = np.zeros((K_GREEDY, R), np.float32)
        for j in range(K_GREEDY):
            sel[j, chosen[: j + 1]] = 1
        dev = jnp.asarray(s_ev[rows_ev] - mean_s[w], jnp.float32)
        selj, v = jnp.asarray(sel), Vd[:, w]
        yh = padded(topk_predict, dev, 400, mean_w, selj, Ug, Uu, v)
        greedy_res.append({"writer": int(w), "chosen": chosen, "r2_fit": path,
                           "r2_held": r2(y_rec[rows_ev, w][:, None], yh).tolist()})  # fmt: skip
    print(f"t={t}: pass 3 {time.time() - t0:.0f}s", flush=True)

    np.savez(
        out / f"attr_pos{t}.npz",
        reader_cols=rcols, writer_cols=wcols, n_gate=rg, I=score, mean_phi=sum_phi / cntc,
        mean_s=mean_s, rms_s=rms_s, weight_score=weight_score, ks=ks, r2_ig=r2_ig, r2_wt=r2_wt,
        p_on=p_on, var_on=var_on, msq_on=msq_on, write=write, n_on_fit=cnt,
        reader_index=comps["index"][rcols], writer_index=comps["index"][wcols],
        rank_ig=rank_ig, **predictors,
    )  # fmt: skip
    (out / f"greedy_pos{t}.json").write_text(json.dumps(greedy_res))
    return {"t": t, "n_gate": rg, "n_up": R - rg, "n_writers": W,
            "completeness_max_rel_err": err_compl,
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
