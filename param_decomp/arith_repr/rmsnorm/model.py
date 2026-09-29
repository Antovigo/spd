"""Llama-3.1-8B on the addsub pool with every RMSNorm's scale exposed, layer by layer either the
ORIGINAL model (dense weights) or the DECOMPOSED one (the dataset's masked alive components, no
delta; `ComponentsModel`).

Norm points t = 2l (block l's attention norm), 2l + 1 (its MLP norm), 64 (the final norm). Each
norm computes `x / rms(x) * gain`; the hook `norm(t, rms, idx) -> rms` may replace the (B, T)
per-token rms before it divides, which is how a norm is frozen or teacher-forced. `forward_chunk`
also returns the rms every norm actually computed (before the hook), so a later run can force it."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.isa.components_model import (
    CHUNK,
    KINDS,
    ComponentsModel,
    T,
    _attention,
    _project,
    _rms,
    _write,
)

jax.config.update("jax_default_matmul_precision", "highest")

N_NORM = 65
FINAL = 64
NormHook = Callable[[int, jax.Array, np.ndarray], jax.Array]


@jax.jit
def _dense(x: jax.Array, W: jax.Array) -> jax.Array:
    return x @ W.astype(jnp.float32).T


@jax.jit
def _normed(x: jax.Array, rms: jax.Array, gain: jax.Array) -> jax.Array:
    return x / rms[..., None] * gain


@jax.jit
def _logprobs(xf: jax.Array, unembed: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(xf @ unembed.T, -1)


@jax.jit
def kl_rows(p_log: jax.Array, q_log: jax.Array) -> jax.Array:
    """KL(p || q) per row."""
    return (jnp.exp(p_log) * (p_log - q_log)).sum(-1)


class NormModel(ComponentsModel):
    def __init__(self, dense: bool = True) -> None:
        super().__init__()
        self.ln = [jnp.asarray(self.ln1), jnp.asarray(self.ln2)]
        self.W: list[dict[str, jax.Array]] = []
        if dense:
            w = Weights()
            for li in range(self.n_layer):
                self.W.append({k: jnp.asarray(w.get(f"model.layers.{li}.{s}.weight"), jnp.bfloat16)
                               for k, s in KINDS.items()})  # fmt: skip

    def chunks(self, rows: np.ndarray):
        """(rows of the chunk padded to CHUNK, number of real rows) over `rows`."""
        for s in range(0, len(rows), CHUNK):
            r = rows[s : s + CHUNK]
            n = len(r)
            yield np.concatenate([r, np.full(CHUNK - n, r[0])]), n

    def forward_chunk(
        self,
        rows: np.ndarray,
        dec: Sequence[bool] | bool = False,
        norm: NormHook | None = None,
        ablate: dict[int, np.ndarray] | None = None,
        capture: set[int] | None = None,
        split: tuple[int, int, np.ndarray, str] | None = None,
    ) -> tuple[jax.Array, jax.Array, dict[int, jax.Array]]:
        """Last-position log-probs (B, V), every norm's own rms (65, B, T) before the hook, and the raw
        stream entering the norm points in `capture`. `dec[l]`: block l runs its masked components
        (else the dense weights); `ablate[l]`: mask columns of block l forced off.

        `split = (layer, off, cols, mode)` separates what the columns `cols` of that decomposed block
        (off 0 attention, 1 MLP) write from their share of every later norm: with w = the block's
        output minus its output with `cols` off, "norm_only" keeps the write in the stream but every
        later norm divides by rms(x - w); "direct_only" drops the write but divides by rms(x + w); "ablate"
        drops it. In a dense (original) block the columns' rank-one terms are subtracted from the weights."""
        assert len(rows) == CHUNK
        decs = [dec] * self.n_layer if isinstance(dec, bool) else list(dec)
        idx = rows
        x = jnp.asarray(np.stack([[self.embed[int(t)] for t in toks] for toks in self.tokens[rows]]), jnp.float32)  # fmt: skip
        B = len(rows)
        rms_all, caps = [], {}
        shift = None  # (sign, w): later norms see rms(x + sign * w)

        def own_rms(x: jax.Array) -> jax.Array:
            return _rms(x, self.eps) if shift is None else _rms(x + shift[0] * shift[1], self.eps)

        for li in range(self.n_layer):
            if decs[li]:
                mnp = self.masks(li, rows)
                if ablate and li in ablate:
                    mnp = mnp.copy()
                    mnp[:, :, ablate[li]] = 0
                m = jnp.asarray(mnp)
                m_off: jax.Array | str = m
                if split is not None and split[0] == li:
                    mnp = mnp.copy()
                    mnp[:, :, split[2]] = 0
                    m_off = jnp.asarray(mnp)

                def site(kind: str, xin: jax.Array, mm: jax.Array | str | None) -> jax.Array:
                    assert isinstance(mm, jax.Array)
                    s = self.sites[li][kind]  # noqa: B023
                    return _write(_project(xin, s.V), mm[:, :, s.cols], s.U)
            else:
                m, m_off = None, "off"

                def site(kind: str, xin: jax.Array, mm: jax.Array | str | None) -> jax.Array:
                    y = _dense(xin, self.W[li][kind])  # noqa: B023
                    if (
                        isinstance(mm, str) and split is not None
                    ):  # the dense weight minus the split components
                        s = self.sites[li][kind]  # noqa: B023
                        j = np.flatnonzero(np.isin(s.cols, split[2]))
                        if len(j):
                            y = y - _write(
                                _project(xin, s.V[:, j]), jnp.ones((1, 1, len(j))), s.U[j]
                            )
                    return y

            def block(off: int, xin: jax.Array, mm: jax.Array | str | None) -> jax.Array:
                if off == 0:
                    q = site("q", xin, mm).reshape(B, T, self.n_head, self.hd)
                    k = site("k", xin, mm).reshape(B, T, self.n_kv, self.hd)
                    v = site("v", xin, mm).reshape(B, T, self.n_kv, self.hd)
                    return site("o", _attention(q, k, v, self.cos, self.sin), mm)
                return site("down", jax.nn.silu(site("gate", xin, mm)) * site("up", xin, mm), mm)

            for off in (0, 1):
                t = 2 * li + off
                if capture and t in capture:
                    caps[t] = x
                rms_all.append(_rms(x, self.eps))
                rms = own_rms(x)
                if norm is not None:
                    rms = norm(t, rms, idx)
                xin = _normed(x, rms, self.ln[off][li])
                out = block(off, xin, m)
                if split is not None and (split[0], split[1]) == (li, off):
                    out_off = block(off, xin, m_off)
                    if split[3] == "ablate":
                        out = out_off
                    elif split[3] == "norm_only":
                        shift = (-1.0, out - out_off)
                    else:
                        shift = (1.0, out - out_off)
                        out = out_off
                x = x + out
        if capture and FINAL in capture:
            caps[FINAL] = x
        rms = own_rms(x)
        rms_all.append(_rms(x, self.eps))
        if norm is not None:
            rms = norm(FINAL, rms, idx)
        xf = _normed(x[:, -1], rms[:, -1], self.final_j)
        return _logprobs(xf, self.unembed), jnp.stack(rms_all), caps


def forced(
    values: jax.Array, points: set[int] | None = None, positions: Sequence[int] | None = None
) -> NormHook:
    """Hook replacing the rms at `points` (default all) and `positions` (default all) by `values`
    (65, B, T) or (65, 1, T) — per-prompt teacher forcing or a pool mean."""
    pos = None if positions is None else jnp.asarray(np.isin(np.arange(T), positions))

    def hook(t: int, rms: jax.Array, _idx: np.ndarray) -> jax.Array:
        if points is not None and t not in points:
            return rms
        v = jnp.broadcast_to(values[t], rms.shape)
        return v if pos is None else jnp.where(pos[None], v, rms)

    return hook
