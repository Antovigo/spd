"""Why does the ORIGINAL pattern hurt the decomposed model at blocks 15-16? Tests on the four heads
L15H13, L15H3, L16H21, L16H3 (`HEADS`), each with a FOCUS key f = the non-<BOS> key its decomposed
`=` row weighs most (b for the L15 heads, a for the L16 heads).

Notation (per head, prompt, at the `=` query): P_o(j), P_d(j) the original / decomposed weights on
key j; dP(j) = P_o(j) - P_d(j); u_o(j), u_d(j) the vector the head would add to the residual stream
at `=` if it attended only to key j (its value at j through its o-projection), in the original /
decomposed model. The head's write is sum_j P(j) u(j), linear in the weights, so a pattern change
splits exactly into per-key pieces dP(j) u(j).

* `values` mode -> OUT/copy_values.npz: per block (15, 16), head, key: sums over the 20000 prompts of
  |u_o|^2, |u_d - u_o|^2, |u_l - u_o|^2 (u_l: decomposed components on the ORIGINAL stream) and
  u_o . u_d, for every head of the two blocks (test 2: are the values at the keys the decomposed
  head ignores reconstructed?).
* `patch` mode -> OUT/copy_patch.npz: KL(original || variant) at the last position on every
  STRIDE-th prompt, for
  - the DECOMPOSED model plus, at `=`, sum over the four heads of sum_{j in S} dP(j) u(j) (test 3):
    S = all keys / the focus key / the other keys / <BOS> only / the other non-<BOS> keys, with u = u_d
    (= the decomposed model's own values, i.e. exactly a pattern change) or u = u_o (the original
    model's values at those keys, from the stored original stream: the change the original head
    would make, injected without any mis-reconstruction);
  - the ORIGINAL model with those heads' `=` rows changed (test 1): to the decomposed row; the other
    keys removed (keep P_o(f) only); the other non-<BOS> keys removed; the focus weight alone set to
    P_d(f). Plus both models with the four heads' full patterns swapped on every query row."""

import sys
import time
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.attn_patterns.extract import OUT, mix, norm, pattern
from param_decomp.arith_repr.attn_patterns.patch import STRIDE, PatchModel
from param_decomp.arith_repr.isa.components_model import T, _logprobs, _project, _rms, _write
from param_decomp.arith_repr.rmsnorm.model import kl_rows
from param_decomp.arith_repr.vectors.common import DATASET

HEADS = ((15, 13), (15, 3), (16, 21), (16, 3))
EQ, BOS = 4, 0
H, HD, D = 32, 128, 4096


def focus_keys() -> dict[tuple[int, int], int]:
    S = np.load(OUT / "summary.npz")
    return {(li, h): int(1 + np.argmax(S["meanP_dec"][li, h, EQ, 1:])) for li, h in HEADS}


def head_writes_dense(v: jax.Array, Wo: jax.Array, hs: list[int]) -> jax.Array:
    """v (B, T, KV*HD) -> u (B, T_key, len(hs), D): head h's write at any query attending only to key j."""
    B = v.shape[0]
    hs_ = np.asarray(hs)
    vr = jnp.repeat(v.reshape(B, T, 8, HD), H // 8, axis=2)[:, :, hs_]
    return jnp.einsum("bkhd,ehd->bkhe", vr, Wo.reshape(D, H, HD)[:, hs_])


def head_writes_comp(
    v: jax.Array, Vo: jax.Array, m_eq: jax.Array, Uo: jax.Array, hs: list[int]
) -> jax.Array:
    """Same through the masked o components, masked as at the `=` query (m_eq (B, C))."""
    B = v.shape[0]
    hs_ = np.asarray(hs)
    vr = jnp.repeat(v.reshape(B, T, 8, HD), H // 8, axis=2)[:, :, hs_]
    inner = jnp.einsum("bkhd,hdc->bkhc", vr, Vo.reshape(H, HD, -1)[hs_])
    return jnp.einsum("bkhc,bc,ce->bkhe", inner, m_eq, Uo)


# A variant: (model, swap, add). `swap` = True replaces the four heads' patterns on every row by the
# other model's; `add(layer, P_o, P_d) -> {head: (coef (B, T_key), source)}` gives the extra write at `=`.
Add = Callable[[int, np.ndarray, np.ndarray], dict[int, tuple[np.ndarray, str]]]


class CopyModel(PatchModel):
    def __init__(self) -> None:
        super().__init__()
        self.dec_P = np.load(OUT / "dec_P.npy", mmap_mode="r")
        self.resid_o = np.load(DATASET / "original/resid.npy", mmap_mode="r")
        self.focus = focus_keys()

    def v_dense(self, li: int, x: jax.Array) -> jax.Array:
        return _rms_norm(x, self.eps, self.ln[0][li]) @ self.W[li]["v"].astype(jnp.float32).T

    def run_copy(self, rows: np.ndarray, dense: bool, swap: bool, add: Add | None) -> jax.Array:
        x = jnp.asarray(np.stack([[self.embed[int(t)] for t in toks] for toks in self.tokens[rows]]), jnp.float32)  # fmt: skip
        B = len(rows)
        layers = {li for li, _ in HEADS}
        for li in range(self.n_layer):
            m = jnp.asarray(self.masks(li, rows))

            def site(kind: str, xin: jax.Array) -> jax.Array:
                if dense:
                    return xin @ self.W[li][kind].astype(jnp.float32).T  # noqa: B023
                s = self.sites[li][kind]  # noqa: B023
                return _write(_project(xin, s.V), m[:, :, s.cols], s.U)  # noqa: B023

            xin = x / _rms(x, self.eps)[..., None] * self.ln[0][li]
            q, k, v = site("q", xin), site("k", xin), site("v", xin)
            pat = pattern(q, k)
            hs = [h for l2, h in HEADS if l2 == li]
            P_o = P_d = np.zeros(0, np.float32)
            if li in layers:
                P_o = np.asarray(self.orig_P[li, rows], np.float32)
                P_d = np.asarray(self.dec_P[li, rows], np.float32)
                if swap:
                    other = P_d if dense else P_o
                    pat = pat.at[:, np.asarray(hs)].set(jnp.asarray(other[:, hs]))
            out = site("o", mix(pat, v).reshape(B, T, -1))
            if li in layers and add is not None:
                extra = add(li, P_o, P_d)
                if extra:
                    so = self.sites[li]["o"]
                    u_run = (head_writes_dense(v, self.W[li]["o"].astype(jnp.float32), hs) if dense
                             else head_writes_comp(v, so.V, m[:, EQ, so.cols], so.U, hs))  # fmt: skip
                    u_orig = None
                    for i, h in enumerate(hs):
                        if h not in extra:
                            continue
                        coef, src = extra[h]
                        if src == "orig" and u_orig is None:
                            xo = jnp.asarray(self.resid_o[2 * li, rows], jnp.float32)
                            u_orig = head_writes_dense(
                                self.v_dense(li, xo), self.W[li]["o"].astype(jnp.float32), hs
                            )
                        u = u_orig if src == "orig" else u_run
                        assert u is not None
                        out = out.at[:, EQ].add(
                            jnp.einsum("bk,bke->be", jnp.asarray(coef), u[:, :, i])
                        )
            x = x + out
            xin = x / _rms(x, self.eps)[..., None] * self.ln[1][li]
            x = x + site("down", jax.nn.silu(site("gate", xin)) * site("up", xin))
        return _logprobs(x[:, -1], self.final_j, self.unembed)


def _rms_norm(x: jax.Array, eps: float, g: jax.Array) -> jax.Array:
    return norm(x, g, eps)


def keyset(name: str, f: int) -> np.ndarray:
    """Indicator over the 5 keys of the subset `name` for focus key f."""
    sel = np.zeros(T, np.float32)
    if name == "all":
        sel[:] = 1
    elif name == "focus":
        sel[f] = 1
    elif name == "other":
        sel[:] = 1
        sel[f] = 0
    elif name == "bos":
        sel[BOS] = 1
    elif name == "rest":
        sel[:] = 1
        sel[[f, BOS]] = 0
    return sel


def dec_add(
    subset: str, src: str, heads: tuple[tuple[int, int], ...], focus: dict[tuple[int, int], int]
) -> Add:
    """Decomposed model + sum_{j in subset} dP(j) u(j) at `=`, dP = P_o - P_d."""

    def add(li: int, P_o: np.ndarray, P_d: np.ndarray) -> dict[int, tuple[np.ndarray, str]]:
        return {h: ((P_o[:, h, EQ] - P_d[:, h, EQ]) * keyset(subset, focus[li, h]), src)
                for l2, h in heads if l2 == li}  # fmt: skip

    return add


def orig_add(kind: str, focus: dict[tuple[int, int], int]) -> Add:
    """Original model with the four heads' `=` rows changed (coefficients on its own values)."""

    def add(li: int, P_o: np.ndarray, P_d: np.ndarray) -> dict[int, tuple[np.ndarray, str]]:
        res = {}
        for l2, h in HEADS:
            if l2 != li:
                continue
            f = focus[li, h]
            po, pd = P_o[:, h, EQ], P_d[:, h, EQ]
            if kind == "dec_row":
                c = pd - po
            elif kind == "drop_other":
                c = -po * keyset("other", f)
            elif kind == "drop_rest":
                c = -po * keyset("rest", f)
            else:  # "focus_amp"
                c = (pd - po) * keyset("focus", f)
            res[h] = (c, "run")
        return res

    return add


def patch_mode() -> None:
    cm = CopyModel()
    fk = cm.focus
    print("focus keys", fk, flush=True)
    rows_all = np.arange(0, len(cm.tokens), STRIDE)
    variants: dict[str, tuple[bool, bool, Add | None]] = {
        "orig/none": (True, False, None),
        "dec/none": (False, False, None),
        "dec/swap_all_rows": (False, True, None),
        "orig/swap_all_rows": (True, True, None),
    }
    for sub in ("all", "focus", "other", "bos", "rest"):
        for src in ("run", "orig"):
            variants[f"dec/{sub}/{'dec_values' if src == 'run' else 'orig_values'}"] = (
                False,
                False,
                dec_add(sub, src, HEADS, fk),
            )
    for hd in HEADS:
        for sub in ("focus", "other"):
            variants[f"dec/L{hd[0]}H{hd[1]}/{sub}/dec_values"] = (
                False,
                False,
                dec_add(sub, "run", (hd,), fk),
            )
    for kind in ("dec_row", "drop_other", "drop_rest", "focus_amp"):
        variants[f"orig/{kind}"] = (True, False, orig_add(kind, fk))
    ref = []
    for r, n in cm.chunks(rows_all):
        lp, _, _ = cm.forward_chunk(r, dec=False)
        ref.append(np.asarray(lp[:n]))
    print("reference done", flush=True)
    res: dict[str, np.ndarray] = {"rows": rows_all}
    t0 = time.time()
    for name, (dense, swap, add) in variants.items():
        kls = []
        for (r, n), o in zip(cm.chunks(rows_all), ref, strict=True):
            kls.append(np.asarray(kl_rows(jnp.asarray(o), cm.run_copy(r, dense, swap, add)[:n])))
        res[name] = np.concatenate(kls)
        print(f"{name} KL={res[name].mean():.5f} {time.time() - t0:.0f}s", flush=True)
    np.savez(OUT / "copy_patch.npz", **cast(dict[str, Any], res))


def values_mode() -> None:
    """Test 2 on every head of blocks 15 and 16, all prompts."""
    cm = CopyModel()
    N = len(cm.tokens)
    resid_d = np.load(DATASET / "decomposed/resid.npy", mmap_mode="r")
    acc = {k: np.zeros((2, H, T)) for k in ("oo", "dd", "err_d", "err_l", "od")}
    hs = list(range(H))
    for bi, li in enumerate((15, 16)):
        so, sv = cm.sites[li]["o"], cm.sites[li]["v"]
        Wo = cm.W[li]["o"].astype(jnp.float32)
        for s in range(0, N, 500):
            rows = np.arange(s, min(s + 500, N))
            m = jnp.asarray(cm.masks(li, rows))
            xo = _rms_norm(jnp.asarray(cm.resid_o[2 * li, rows], jnp.float32), cm.eps, cm.ln[0][li])
            xd = _rms_norm(jnp.asarray(resid_d[2 * li, rows], jnp.float32), cm.eps, cm.ln[0][li])
            v_o = xo @ cm.W[li]["v"].astype(jnp.float32).T
            v_d = _write(_project(xd, sv.V), m[:, :, sv.cols], sv.U)
            v_l = _write(_project(xo, sv.V), m[:, :, sv.cols], sv.U)
            m_eq = m[:, EQ, so.cols]
            for h0 in range(0, H, 8):
                hh = hs[h0 : h0 + 8]
                uo = head_writes_dense(v_o, Wo, hh)
                ud = head_writes_comp(v_d, so.V, m_eq, so.U, hh)
                ul = head_writes_comp(v_l, so.V, m_eq, so.U, hh)
                sl = slice(h0, h0 + 8)
                acc["oo"][bi, sl] += np.asarray((uo**2).sum((0, 3)).T)
                acc["dd"][bi, sl] += np.asarray((ud**2).sum((0, 3)).T)
                acc["err_d"][bi, sl] += np.asarray(((ud - uo) ** 2).sum((0, 3)).T)
                acc["err_l"][bi, sl] += np.asarray(((ul - uo) ** 2).sum((0, 3)).T)
                acc["od"][bi, sl] += np.asarray((uo * ud).sum((0, 3)).T)
            print(li, s, flush=True)
    np.savez(OUT / "copy_values.npz", n=N, **cast(dict[str, Any], acc))


if __name__ == "__main__":
    {"patch": patch_mode, "values": values_mode}[sys.argv[1]]()
    sys.exit(0)
