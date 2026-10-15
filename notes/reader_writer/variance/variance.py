"""Exact variance split of every MLP writer over its up and gate readers (`../plan_variance.md`).

Reads the alive-only activations of `extract_alive_only.py` and, for one block and each position
tau in 1..4 (the plan's sections 6-8 and 10), computes:

* pass 1 (whole pool): the gate means t̄_g, h̄_g and the gate-weighted up references
  r_{c,n} = E[omega_n y_{u,c,n}] / E[omega_n], omega_n = SiLU(h_{g,n})^2;
* pass 2 (whole pool): the per-reader terms M^u, M^{u,0}, M^g of every writer and their covariances
  with the writer's activation t_{d,w}, from which shares phi and weights a follow;
* pass 3 (random 20,000-prompt subsample): the routing tensor J_{w,c,c'} and its covariances,
  with M^u and M^{u,0} again on the same prompts (identity (5) is checked there).

t_{d,w} is recomputed from the stored readers by equation (1) in float32 (plan §10); the stored
writer activation only serves check 1. Accumulators are float32 sums per flush, added into float64
on the host; they are saved to disk every SAVE_SECONDS and a restarted job resumes from them.

    python variance.py --layer 0                      # full analysis of block 0, positions 1..4
    python variance.py --layer 0 --pilot              # the plan's pilot (writes pilot.json)
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")  # float32, TF32 off (plan §10)

SRC = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/variance")
POSITIONS = (1, 2, 3, 4)
"""Token positions stored by `extract_alive_only.py` (column p of inner.npy is position p + 1)."""
N_SUB = 20_000
N_POOL = 320_000
"""The full pool, for the pilot's projection."""
SEED = 0
EPS_SECANT = 1e-3
EXCLUDE_REL = 1e-6
FLAG_FRAC = 0.01
SAVE_SECONDS = 600.0


# ---------------------------------------------------------------- the math (pure, dtype-agnostic)


def silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def dsilu(x: jax.Array) -> jax.Array:
    s = jax.nn.sigmoid(x)
    return s * (1 + x * (1 - s))


def d2silu(x: jax.Array) -> jax.Array:
    s = jax.nn.sigmoid(x)
    return s * (1 - s) * (2 + x * (1 - 2 * s))


def secant(h: jax.Array, hbar: jax.Array, eps: float = EPS_SECANT) -> jax.Array:
    """rho = (SiLU(h) - SiLU(hbar)) / (h - hbar); second-order Taylor in h - hbar when it is small."""
    d = h - hbar
    small = jnp.abs(d) < eps
    quotient = (silu(h) - silu(hbar)) / jnp.where(small, 1.0, d)
    return jnp.where(small, dsilu(hbar) + 0.5 * d2silu(hbar) * d, quotient)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Weights:
    """One block's alive MLP components: gate / up write vectors (G, N) / (U, N), writer read
    vectors (N, D), and the writers' squared write-vector norms (D,)."""

    Ug: jax.Array
    Uu: jax.Array
    Vd: jax.Array
    ud2: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Refs:
    """The references of plan §6: t̄_g (G,), h̄_g and SiLU(h̄_g) (N,), r_{c,n} (U, N), r_n (N,),
    and the prompt-independent pieces W0 = V^T diag(SiLU(h̄_g)) U_u^T, W0r = V^T diag(SiLU(h̄_g)) R^T
    (D, U) and t^ref (D,)."""

    tbar_g: jax.Array
    hbar_g: jax.Array
    sbar: jax.Array
    R: jax.Array
    r: jax.Array
    W0: jax.Array
    W0r: jax.Array
    tref: jax.Array


def pass1_sums(tg: jax.Array, tu: jax.Array, W: Weights) -> dict[str, jax.Array]:
    """Batch sums for the references: sum t_g (G,), sum omega_n (N,), sum omega_n t_{u,c} (U, N)."""
    omega = silu(tg @ W.Ug) ** 2
    return {"tg": tg.sum(0), "w": omega.sum(0), "wt": tu.T @ omega}


def make_refs(sums: dict[str, np.ndarray], n: int, W: Weights) -> Refs:
    tbar_g = jnp.asarray(sums["tg"] / n, W.Ug.dtype)
    w = jnp.asarray(sums["w"], W.Ug.dtype)
    wt = jnp.asarray(sums["wt"], W.Ug.dtype)
    ttilde = jnp.where(w > 0, wt / jnp.where(w > 0, w, 1.0), 0.0)  # (U, N); 0 where omega == 0
    R = ttilde * W.Uu
    hbar_g = tbar_g @ W.Ug
    sbar = silu(hbar_g)
    VdT_s = W.Vd.T * sbar[None, :]  # (D, N)
    return Refs(
        tbar_g=tbar_g,
        hbar_g=hbar_g,
        sbar=sbar,
        R=R,
        r=R.sum(0),
        W0=VdT_s @ W.Uu.T,
        W0r=VdT_s @ R.T,
        tref=VdT_s @ R.sum(0),
    )


def writer_activation(tg: jax.Array, tu: jax.Array, W: Weights) -> jax.Array:
    """Equation (1): t_{d,w} = sum_n V_{d,w,n} SiLU(h_{g,n}) h_{u,n}, (B, D)."""
    return (silu(tg @ W.Ug) * (tu @ W.Uu)) @ W.Vd


def terms(tg: jax.Array, tu: jax.Array, refs: Refs, W: Weights) -> dict[str, jax.Array]:
    """Equation (4) and the mean-gate part of (5) on a batch: t_d (B, D), M^u and M^{u,0} (B, D, U),
    M^g (B, D, G)."""
    hg = tg @ W.Ug
    s = silu(hg)
    t_d = (s * (tu @ W.Uu)) @ W.Vd
    rho = secant(hg, refs.hbar_g[None, :])
    # M^u_{w,c} = t_c sum_n V s U_u[c,n] - sum_n V s R[c,n]
    both = jnp.concatenate([W.Uu, refs.R], axis=0)  # (2U, N)
    X = s[:, :, None] * W.Vd[None]  # (B, N, D)
    P_ = jnp.einsum("bnw,cn->bwc", X, both)
    U = W.Uu.shape[0]
    Mu = tu[:, None, :] * P_[..., :U] - P_[..., U:]
    Mu0 = tu[:, None, :] * refs.W0[None] - refs.W0r[None]
    # M^g_{w,c'} = (t_{g,c'} - t̄_{g,c'}) sum_n V rho r_n U_g[c',n]
    Y = (rho * refs.r[None, :])[:, :, None] * W.Vd[None]  # (B, N, D)
    Mg = (tg - refs.tbar_g)[:, None, :] * jnp.einsum("bnw,cn->bwc", Y, W.Ug)
    return {"t_d": t_d, "Mu": Mu, "Mu0": Mu0, "Mg": Mg}


def routing(
    tg: jax.Array, tu: jax.Array, refs: Refs, W: Weights, t_d: jax.Array, wc: int,
    valid: jax.Array | None = None,
) -> dict[str, jax.Array]:  # fmt: skip
    """Equation (5)'s J_{w,c,c'} = sum_n V_{d,w,n} rho_n eps_{u,c,n} delta_{g,c',n} on a batch,
    looping over chunks of `wc` writers; returns the batch sums sum_x J and sum_x J t_d,
    (D_pad, U, G), and sum_{c'} J (B, D_pad, U) for identity (5). Rows with `valid == 0`
    (padding) get J = 0."""
    B, U, G, D = tu.shape[0], W.Uu.shape[0], W.Ug.shape[0], W.Vd.shape[1]
    n_chunk = -(-D // wc)
    Vp = jnp.pad(W.Vd, ((0, 0), (0, n_chunk * wc - D)))
    tdp = jnp.pad(t_d, ((0, 0), (0, n_chunk * wc - D)))
    hg = tg @ W.Ug
    rho = secant(hg, refs.hbar_g[None, :])
    eps = tu[:, :, None] * W.Uu[None] - refs.R[None]  # (B, U, N)
    dg = tg - refs.tbar_g  # (B, G)
    if valid is not None:
        dg = dg * valid[:, None]
    rho_eps = rho[:, None, :] * eps  # (B, U, N)

    def chunk(j: jax.Array, acc: tuple[jax.Array, jax.Array, jax.Array]):  # noqa: ANN202
        S, St, Ssum = acc
        Vc = jax.lax.dynamic_slice_in_dim(Vp, j * wc, wc, axis=1)  # (N, wc)
        X = Vc.T[None, :, None, :] * rho_eps[:, None, :, :]  # (B, wc, U, N)
        J = jnp.einsum("bwcn,dn->bwcd", X, W.Ug) * dg[:, None, None, :]  # (B, wc, U, G)
        tc = jax.lax.dynamic_slice_in_dim(tdp, j * wc, wc, axis=1)  # (B, wc)
        S = jax.lax.dynamic_update_slice_in_dim(S, J.sum(0), j * wc, axis=0)
        St = jax.lax.dynamic_update_slice_in_dim(St, jnp.einsum("bwcd,bw->wcd", J, tc), j * wc, 0)
        Ssum = jax.lax.dynamic_update_slice_in_dim(Ssum, J.sum(-1), j * wc, axis=1)
        return S, St, Ssum

    zeros = jnp.zeros((n_chunk * wc, U, G), tu.dtype)
    S, St, Ssum = jax.lax.fori_loop(
        0, n_chunk, chunk, (zeros, zeros, jnp.zeros((B, n_chunk * wc, U), tu.dtype))
    )
    return {"J": S, "Jt": St, "J_sum_gates": Ssum}


def cov_sums(t_d: jax.Array, X: jax.Array) -> tuple[jax.Array, jax.Array]:
    """sum_x X and sum_x X t_d for a term X (B, D, K)."""
    return X.sum(0), jnp.einsum("bwk,bw->wk", X, t_d)


def finalize(
    n: int, S_t: np.ndarray, S_tt: np.ndarray, sums: dict[str, tuple[np.ndarray, np.ndarray]],
    ud2: np.ndarray,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Shares phi = Cov(Q, t) / Var(t) (NaN for excluded writers), weights a = Cov(Q, t) ||U_d||^2,
    and the writer statistics of plan §7-8. `sums[name] = (sum Q, sum Q t)`, float64."""
    mean_t = S_t / n
    e_t2 = S_tt / n
    var_t = e_t2 - mean_t**2
    excluded = var_t <= EXCLUDE_REL * e_t2
    resid_var = var_t * ud2
    med = np.median(resid_var[~excluded]) if (~excluded).any() else 0.0
    out = {"n": np.asarray(n), "mean_t": mean_t, "e_t2": e_t2, "var_t": var_t,
           "resid_var": resid_var, "excluded": excluded,
           "flagged": (resid_var < FLAG_FRAC * med) & ~excluded}  # fmt: skip
    shape_ext = (slice(None),) + (None,) * 2
    for name, (S, St) in sums.items():
        cov = St / n - (S / n) * mean_t[shape_ext[: S.ndim]]
        denom = np.where(excluded, np.nan, var_t)
        out[f"phi_{name}"] = cov / denom[shape_ext[: S.ndim]]
        out[f"a_{name}"] = cov * ud2[shape_ext[: S.ndim]]
    return out


def analyse_in_memory(
    tg: jax.Array, tu: jax.Array, W: Weights, wc: int = 3
) -> tuple[dict[str, np.ndarray], dict[str, jax.Array], Refs]:
    """Every pass on one in-memory batch (tests): finalize() of Mu, Mu0, Mg and J, the raw terms
    with J's gate sums, and the references."""
    n = tg.shape[0]
    refs = make_refs({k: np.asarray(v) for k, v in pass1_sums(tg, tu, W).items()}, n, W)
    tm = terms(tg, tu, refs, W)
    rt = routing(tg, tu, refs, W, tm["t_d"], wc)
    D = W.Vd.shape[1]
    t_d = np.asarray(tm["t_d"], np.float64)
    sums = {k: tuple(np.asarray(a, np.float64) for a in cov_sums(tm["t_d"], tm[k]))
            for k in ("Mu", "Mu0", "Mg")}  # fmt: skip
    sums["J"] = (np.asarray(rt["J"][:D], np.float64), np.asarray(rt["Jt"][:D], np.float64))
    res = finalize(n, t_d.sum(0), (t_d**2).sum(0), sums, np.asarray(W.ud2, np.float64))
    return res, {**tm, "J_sum_gates": rt["J_sum_gates"][:, :D]}, refs


# ---------------------------------------------------------------- I/O, passes, checkpoints


def load_block(src: Path, layer: int) -> tuple[Weights, np.ndarray, dict[str, np.ndarray]]:
    """The block's weights, its stored activations (N, 4, K) float16 in RAM, and its columns."""
    comps = np.load(src / "alive_only" / f"L{layer}" / "comps.npz")
    kind = comps["kind"]
    cols = {k: np.flatnonzero(kind == k) for k in ("gate", "up", "down")}
    W = Weights(
        Ug=jnp.asarray(comps["U_gate"]),
        Uu=jnp.asarray(comps["U_up"]),
        Vd=jnp.asarray(comps["V_down"]),
        ud2=jnp.asarray((comps["U_down"].astype(np.float64) ** 2).sum(1), jnp.float32),
    )
    assert W.Ug.shape == (cols["gate"].size, W.Vd.shape[0]), (W.Ug.shape, W.Vd.shape)
    assert W.Uu.shape == (cols["up"].size, W.Vd.shape[0]), W.Uu.shape
    inner = np.load(src / "alive_only" / f"L{layer}" / "inner.npy", mmap_mode="r")
    t0 = time.time()
    inner = np.ascontiguousarray(inner)
    print(f"L{layer}: inner {inner.shape} loaded in {time.time() - t0:.0f}s; "
          f"G={cols['gate'].size} U={cols['up'].size} D={cols['down'].size}", flush=True)  # fmt: skip
    return W, inner, cols


class Accum:
    """Host float64 sums over prompt batches, saved atomically every SAVE_SECONDS for resume."""

    def __init__(self, path: Path, tag: str) -> None:
        self.path, self.tag = path, tag
        self.sums: dict[str, np.ndarray] = {}
        self.next = 0
        self.last_save = time.time()
        if path.exists():
            z = np.load(path)
            assert str(z["tag"]) == tag, (str(z["tag"]), tag)
            self.next = int(z["next"])
            self.sums = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
            print(f"resume {path.name} at batch {self.next}", flush=True)

    def add(self, batch: dict[str, jax.Array]) -> None:
        for k, v in batch.items():
            v = np.asarray(v, np.float64)
            self.sums[k] = self.sums[k] + v if k in self.sums else v

    def add_max(self, name: str, value: jax.Array) -> None:
        v = np.asarray(value, np.float64)
        self.sums[name] = np.maximum(self.sums[name], v) if name in self.sums else v

    def maybe_save(self, next_batch: int, force: bool = False) -> None:
        self.next = next_batch
        if force or time.time() - self.last_save > SAVE_SECONDS:
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez(
                tmp, tag=self.tag, next=next_batch, **{f"s_{k}": v for k, v in self.sums.items()}
            )
            os.replace(tmp, self.path)
            self.last_save = time.time()


def batches(rows: np.ndarray, size: int) -> list[np.ndarray]:
    """Consecutive batches of row indices; the last one may be short (`padded` fills it)."""
    return [rows[i : i + size] for i in range(0, rows.size, size)]


def padded(inner: np.ndarray, idx: np.ndarray, p: int, cols: np.ndarray, size: int) -> np.ndarray:
    """(size, len(cols)) float32 of rows idx at stored position p, zero-padded; the caller masks
    the padding rows out of every sum with `valid`."""
    out = np.zeros((size, cols.size), np.float32)
    out[: idx.size] = inner[idx, p][:, cols]
    return out


@jax.jit
def _pass1(tg, tu, W):  # noqa: ANN001, ANN202
    return pass1_sums(tg, tu, W)


@jax.jit
def _pass2(tg, tu, td_stored, valid, refs, W):  # noqa: ANN001, ANN202
    tm = terms(tg, tu, refs, W)
    t_d = tm["t_d"] * valid[:, None]
    out = {"t": t_d.sum(0), "tt": (t_d**2).sum(0),
           "c1_diff2": ((t_d - td_stored * valid[:, None]) ** 2).sum(0),
           "c1_ref2": ((td_stored * valid[:, None]) ** 2).sum(0)}  # fmt: skip
    for k in ("Mu", "Mu0", "Mg"):
        X = tm[k] * valid[:, None, None]
        out[f"{k}"], out[f"{k}t"] = cov_sums(t_d, X)
    total = refs.tref[None] + tm["Mu"].sum(-1) + tm["Mg"].sum(-1)
    err = jnp.abs(total - tm["t_d"]) * valid[:, None]
    scale = jnp.maximum(jnp.max(jnp.abs(tm["t_d"])), 1e-30)
    return out, jnp.max(err) / scale


def _pass3(wc: int):  # noqa: ANN202
    @jax.jit
    def run(tg, tu, valid, refs, W):  # noqa: ANN001, ANN202
        tm = terms(tg, tu, refs, W)
        t_d = tm["t_d"] * valid[:, None]
        rt = routing(tg, tu, refs, W, t_d, wc, valid)
        D = W.Vd.shape[1]
        out = {"t": t_d.sum(0), "tt": (t_d**2).sum(0), "J": rt["J"][:D], "Jt": rt["Jt"][:D]}
        for k in ("Mu", "Mu0"):
            out[k], out[f"{k}t"] = cov_sums(t_d, tm[k] * valid[:, None, None])
        diff = (rt["J_sum_gates"][:, :D] - (tm["Mu"] - tm["Mu0"])) * valid[:, None, None]
        scale = jnp.maximum(jnp.max(jnp.abs(tm["Mu"]) * valid[:, None, None]), 1e-30)
        return out, jnp.max(jnp.abs(diff)) / scale

    return run


def flops_per_prompt(N: int, G: int, U: int, D: int) -> dict[str, float]:
    """Multiply-adds x 2 of each pass, per prompt (the dominant products)."""
    pass2 = 2 * N * D * (2 * U + G) + 2 * N * (G + U + D)
    return {
        "pass1": 2 * N * (G + U),
        "pass2": pass2,
        "pass3": pass2 + 2 * D * U * G * N + D * U * N,
    }


def run_position(
    layer: int, p: int, W: Weights, inner: np.ndarray, cols: dict[str, np.ndarray], out: Path,
    n: int, sub: np.ndarray, B: int, jB: int, wc: int,
) -> dict[str, object]:  # fmt: skip
    """All three passes at stored position p (token position p + 1); writes variance_pos<tau>.npz
    and J_pos<tau>.npz into `out`, returns timings and checks."""
    tau = POSITIONS[p]
    N_, G, U, D = W.Vd.shape[0], W.Ug.shape[0], W.Uu.shape[0], W.Vd.shape[1]
    fl = flops_per_prompt(N_, G, U, D)
    info: dict[str, object] = {"layer": layer, "tau": tau, "N": N_, "G": G, "U": U, "D": D}
    rows = np.arange(n)

    def timed(name: str, acc: Accum, todo: list[np.ndarray], size: int, step) -> None:  # noqa: ANN001
        t_first = t_rest = 0.0
        n_rest = 0
        for i in range(acc.next, len(todo)):
            idx = todo[i]
            t0 = time.time()
            step(idx, size, acc)
            dt = time.time() - t0
            if i == acc.next and t_first == 0.0:
                t_first = dt
            else:
                t_rest += dt
                n_rest += idx.size
            acc.maybe_save(i + 1)
        acc.maybe_save(len(todo), force=True)
        if n_rest:
            info[f"{name}_seconds"] = t_first + t_rest
            info[f"{name}_flops_per_s"] = fl[name] * n_rest / t_rest
        print(f"L{layer} tau={tau} {name}: {info.get(f'{name}_seconds', 0):.0f}s, "
              f"{info.get(f'{name}_flops_per_s', 0) / 1e12:.1f} TFLOP/s", flush=True)  # fmt: skip

    def get(idx: np.ndarray, kind: str, size: int) -> jax.Array:
        return jnp.asarray(padded(inner, idx, p, cols[kind], size))

    def valid_of(idx: np.ndarray, size: int) -> jax.Array:
        v = np.zeros(size, np.float32)
        v[: idx.size] = 1.0
        return jnp.asarray(v)

    # pass 1: references (zero padding adds nothing: SiLU(0) = 0 and t = 0)
    acc1 = Accum(out / f"state_p1_pos{tau}.npz", f"p1 n={n} B={B}")
    timed("pass1", acc1, batches(rows, B), B,
          lambda idx, size, acc: acc.add(_pass1(get(idx, "gate", size), get(idx, "up", size), W)))  # fmt: skip
    refs = make_refs(acc1.sums, n, W)

    # pass 2: per-reader terms on the whole pool
    acc2 = Accum(out / f"state_p2_pos{tau}.npz", f"p2 n={n} B={B}")

    def step2(idx: np.ndarray, size: int, acc: Accum) -> None:
        got, ident = _pass2(get(idx, "gate", size), get(idx, "up", size), get(idx, "down", size),
                            valid_of(idx, size), refs, W)  # fmt: skip
        acc.add(got)
        acc.add_max("ident4", ident)

    timed("pass2", acc2, batches(rows, B), B, step2)
    s2 = acc2.sums
    ud2 = np.asarray(W.ud2, np.float64)
    res = finalize(
        n, s2["t"], s2["tt"], {k: (s2[k], s2[f"{k}t"]) for k in ("Mu", "Mu0", "Mg")}, ud2
    )
    res["check1_rel"] = np.sqrt(s2["c1_diff2"] / np.maximum(s2["c1_ref2"], 1e-300))
    res["ident4_max"] = s2["ident4"]
    np.savez(out / f"variance_pos{tau}.npz", **res,
             tbar_g=np.asarray(refs.tbar_g), hbar_g=np.asarray(refs.hbar_g),
             r=np.asarray(refs.r), R=np.asarray(refs.R), tref=np.asarray(refs.tref))  # fmt: skip
    share_sum = np.nansum(res["phi_Mu"], 1) + np.nansum(res["phi_Mg"], 1)
    ok = ~res["excluded"]
    info.update(ident4_max=float(s2["ident4"]), check1_rel_max=float(res["check1_rel"].max()),
                share_sum_err_max=float(np.abs(share_sum[ok] - 1).max()) if ok.any() else 0.0,
                n_excluded=int(res["excluded"].sum()), n_flagged=int(res["flagged"].sum()))  # fmt: skip

    # pass 3: routing tensor on the subsample
    acc3 = Accum(out / f"state_p3_pos{tau}.npz", f"p3 n={sub.size} jB={jB} wc={wc}")
    run3 = _pass3(wc)

    def step3(idx: np.ndarray, size: int, acc: Accum) -> None:
        got, ident = run3(
            get(idx, "gate", size), get(idx, "up", size), valid_of(idx, size), refs, W
        )
        acc.add(got)
        acc.add_max("ident5", ident)

    timed("pass3", acc3, batches(sub, jB), jB, step3)
    s3 = acc3.sums
    res3 = finalize(
        sub.size, s3["t"], s3["tt"], {k: (s3[k], s3[f"{k}t"]) for k in ("J", "Mu", "Mu0")}, ud2
    )
    gap = res3["phi_Mu"] - res3["phi_Mu0"] - np.nansum(res3["phi_J"], 2)
    info.update(ident5_max=float(s3["ident5"]),
                share5_err_max=float(np.nanmax(np.abs(gap))) if np.isfinite(gap).any() else 0.0)  # fmt: skip
    np.savez(out / f"J_pos{tau}.npz", sub=sub,
             **{k: (v.astype(np.float32) if v.ndim == 3 else v) for k, v in res3.items()})  # fmt: skip
    for acc in (acc1, acc2, acc3):
        acc.path.unlink(missing_ok=True)
    return info


def all_dims(src: Path) -> list[tuple[int, int, int, int]]:
    out = []
    for layer in range(32):
        kind = np.load(src / "alive_only" / f"L{layer}" / "comps.npz")["kind"]
        out.append((14336, int((kind == "gate").sum()), int((kind == "up").sum()),
                    int((kind == "down").sum())))  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--positions", type=int, nargs="+", default=list(POSITIONS))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--jbatch", type=int, default=16)
    ap.add_argument("--wc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="prompts for passes 1-2 (default all)")
    ap.add_argument("--j_limit", type=int, default=None, help="subsample prompts for pass 3")
    ap.add_argument("--pilot", action="store_true", help="tau = 4 only, pass 3 on 500 prompts")
    args = ap.parse_args()
    if args.pilot:
        args.positions = [4]
        args.j_limit = args.j_limit or 500
    W, inner, cols = load_block(args.src, args.layer)
    n_pool = inner.shape[0]
    n = n_pool if args.limit is None else min(args.limit, n_pool)
    sub = np.sort(np.random.default_rng(SEED).choice(n_pool, min(N_SUB, n_pool), replace=False))
    if args.j_limit is not None:
        sub = sub[: args.j_limit]
    out = args.src / ("pilot" if args.pilot else "results") / f"L{args.layer}"
    out.mkdir(parents=True, exist_ok=True)
    infos = []
    for tau in args.positions:
        if (out / f"J_pos{tau}.npz").exists():
            print(f"L{args.layer} tau={tau}: done already, skipped", flush=True)
            continue
        infos.append(run_position(args.layer, POSITIONS.index(tau), W, inner, cols, out, n, sub,
                                  args.batch, args.jbatch, args.wc))  # fmt: skip
    stats = jax.local_devices()[0].memory_stats() or {}
    summary = {"infos": infos, "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
               "device": str(jax.local_devices()[0]), "batch": args.batch,
               "jbatch": args.jbatch, "wc": args.wc}  # fmt: skip
    if args.pilot:
        info = infos[0]
        thr = {k: info[f"{k}_flops_per_s"] for k in ("pass1", "pass2", "pass3")}
        dims = all_dims(args.src)
        per_reader = sum(
            N_POOL
            * (
                flops_per_prompt(*d)["pass1"] / thr["pass1"]
                + flops_per_prompt(*d)["pass2"] / thr["pass2"]
            )
            for d in dims
        )
        dense_j = sum(N_SUB * flops_per_prompt(*d)["pass3"] / thr["pass3"] for d in dims)
        n_pos = len(POSITIONS)
        summary["projection_gpu_hours"] = {
            "passes_1_2_all_blocks_positions": per_reader * n_pos / 3600,
            "pass_3_dense_all_blocks_positions": dense_j * n_pos / 3600,
            "flops_pass3_total": sum(N_SUB * flops_per_prompt(*d)["pass3"] for d in dims) * n_pos,
        }
        summary["go_dense"] = (per_reader + dense_j) * n_pos / 3600 <= 24.0
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
