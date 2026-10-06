"""The components-only model with L15 MLP ablations and per-(op, b) captures at `=`.

Ablation spec (`FlipModel.spec`), all at L15:
* comps: list of (kind, column, neuron mask (14336,), position mask (T,)); removes that component's
  contribution through the masked neurons (gate / up: its write into g / u; down: its read of act);
* neurons: (neuron mask, position mask) zeroed in act.

Captures (`FlipModel.cap`, when `capture_on` is set): sums over the prompts of each group
`op * 100 + (b - 1)` (200 groups) of position `=` quantities:
* `x<p>`: the raw stream at point p (p = 2 l before block l's attention, 2 l + 1 before its MLP,
  64 after the last block), (200, d);
* `oh<l>`: the masked o-site inner activations of block l split by head, (200, H, n_o), so that head
  h's write is `oh[:, h] @ U_o`;
* `act<l>`: the MLP's act = silu(g) u, (200, 14336); `hd<l>`: the masked down inner activations,
  (200, n_down), so that the MLP write is `hd @ U_down`."""

from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.components_model import (
    ComponentsModel,
    InnerHook,
    ResidHook,
    _attention,
    _logprobs,
    _project,
    _rms,
    _write,
)

L = 15
FLIP = np.array([12769, 6456, 9205, 9057, 13193, 7446, 11305])  # the b-flip neurons of L15
CAP_LAYERS = range(12, 22)


class FlipModel(ComponentsModel):
    def __init__(self) -> None:
        super().__init__()
        self.spec: dict[str, Any] = {"comps": [], "neurons": None}
        self.capture_on = False
        self.cap: dict[str, jax.Array] = {}

    def reset_capture(self) -> None:
        self.cap = {}

    def _acc(self, key: str, onehot: jax.Array, x: jax.Array) -> None:
        s = jnp.tensordot(onehot, x, axes=(0, 0))
        self.cap[key] = self.cap[key] + s if key in self.cap else s

    @override
    def _chunk(self, rows: np.ndarray, idx: np.ndarray, resid: ResidHook | None, inner: InnerHook | None,
               capture: set[int] | None) -> tuple[np.ndarray, dict[int, np.ndarray]]:  # fmt: skip
        if self.capture_on:
            assert len(np.unique(rows)) == len(rows), "captures need unpadded chunks"
        x = jnp.asarray(
            np.stack([[self.embed[int(t)] for t in toks] for toks in self.tokens[rows]]),
            jnp.float32,
        )
        B = len(rows)
        grp = self.op[rows] * 100 + self.b[rows] - 1
        onehot = jnp.asarray(np.eye(200, dtype=np.float32)[grp])
        cap = self.capture_on
        for li in range(self.n_layer):
            m = jnp.asarray(self.masks(li, rows))
            for t_off in (0, 1):
                if cap:
                    self._acc(f"x{2 * li + t_off}", onehot, x[:, 4])
                rms = _rms(x, self.eps)
                xin = x / rms[..., None] * jnp.asarray(self.ln1[li] if t_off == 0 else self.ln2[li])
                s = self.sites[li]
                if t_off == 0:
                    q = _write(_project(xin, s["q"].V), m[:, :, s["q"].cols], s["q"].U)
                    k = _write(_project(xin, s["k"].V), m[:, :, s["k"].cols], s["k"].U)
                    v = _write(_project(xin, s["v"].V), m[:, :, s["v"].cols], s["v"].U)
                    att = _attention(q.reshape(B, 5, self.n_head, self.hd),
                                     k.reshape(B, 5, self.n_kv, self.hd),
                                     v.reshape(B, 5, self.n_kv, self.hd), self.cos, self.sin)  # fmt: skip
                    o = s["o"]
                    if cap and li in CAP_LAYERS:
                        ah = att[:, 4].reshape(B, self.n_head, self.hd)
                        Vh = o.V.reshape(self.n_head, self.hd, -1)
                        oh = jnp.einsum("bhd,hdn->bhn", ah, Vh) * m[:, 4, o.cols][:, None]
                        self._acc(f"oh{li}", onehot, oh)
                    x = x + _write(_project(att, o.V), m[:, :, o.cols], o.U)
                else:
                    x = x + self._mlp(li, xin, m, onehot)
        if cap:
            self._acc("x64", onehot, x[:, 4])
        return np.asarray(_logprobs(x[:, -1], self.final_j, self.unembed)), {}

    def _mlp(self, li: int, xin: jax.Array, m: jax.Array, onehot: jax.Array) -> jax.Array:
        s = self.sites[li]
        hg = _project(xin, s["gate"].V) * m[:, :, s["gate"].cols]
        hu = _project(xin, s["up"].V) * m[:, :, s["up"].cols]
        g = hg @ s["gate"].U
        u = hu @ s["up"].U
        down_specs = []
        if li == L:
            for kind, c, nmask, pmask in self.spec["comps"]:
                pm = jnp.asarray(pmask, jnp.float32)[None, :, None]
                nm = jnp.asarray(nmask, jnp.float32)[None, None]
                if kind == "gate":
                    g = g - pm * hg[:, :, c : c + 1] * (s["gate"].U[c] * nm)
                elif kind == "up":
                    u = u - pm * hu[:, :, c : c + 1] * (s["up"].U[c] * nm)
                else:
                    down_specs.append((c, nm, pm))
        act = jax.nn.silu(g) * u
        if li == L and self.spec["neurons"] is not None:
            nmask, pmask = self.spec["neurons"]
            act = act * (1 - jnp.asarray(pmask, jnp.float32)[None, :, None]
                         * jnp.asarray(nmask, jnp.float32)[None, None])  # fmt: skip
        hd = _project(act, s["down"].V)
        for c, nm, pm in down_specs:
            hd = hd.at[:, :, c].add(-pm[:, :, 0] * ((act * nm) @ s["down"].V[:, c]))
        hd = hd * m[:, :, s["down"].cols]
        if self.capture_on and li in CAP_LAYERS:
            self._acc(f"act{li}", onehot, act[:, 4])
            self._acc(f"hd{li}", onehot, hd[:, 4])
        return hd @ s["down"].U
