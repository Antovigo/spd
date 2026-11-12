"""The integer-only components' activation dataset over the whole pool.

For every integer-only component and every position `t` where it is EVER on (its original CI
exceeds the threshold at `t` on some pool prompt, `max_ci.npz`), over all `N` prompts: (a) its
inner activation `x V_c` in the INTEGER-ONLY model (exactly the integer-only components on, the
weight delta off, `model.factored_inner`), float16; (b) its binarized original CI (CI on the
original model > threshold), bit-packed. Layout in `DATASET_README`."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from param_decomp.integer_only.config import Precision
from param_decomp.integer_only.model import (
    KIND_ID,
    Arch,
    Factors,
    Mask,
    Shared,
    Slots,
    factored_inner,
)
from param_decomp.targets.transformer import KIND_ORDER

FP16_MAX = 65504.0


@dataclass(frozen=True)
class Columns:
    """The dataset's columns: one row per (position, component) pair where the component is
    ever on, ordered by position, then the canonical component order (sites sorted by name,
    component index ascending)."""

    position: np.ndarray
    """`(K,)` int: the position `t`."""
    component: np.ndarray
    """`(K,)` int: the component's index `j` in `Slots` (integer-only canonical order)."""

    def at(self, t: int) -> np.ndarray:
        """Slot indices `j` of the columns of position `t`, in file-column order."""
        return self.component[self.position == t]


def ever_on_columns(
    slots: Slots, max_ci: Mapping[str, np.ndarray], threshold: float, seq_len: int
) -> Columns:
    """The (t, j) pairs with the original max CI over the pool above `threshold`."""
    positions, comps = [], []
    for t in range(seq_len):
        on = np.asarray(
            [
                max_ci[site][t, c] > threshold
                for site, c in zip(slots.sites, slots.index, strict=True)
            ]
        )
        idx = np.flatnonzero(on)
        positions.append(np.full(idx.size, t, np.int64))
        comps.append(idx)
    return Columns(position=np.concatenate(positions), component=np.concatenate(comps))


def gather_index(slots: Slots) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per kind: (canonical `j`, block, slot) of its components."""
    out = {}
    for kind in KIND_ORDER:
        sel = np.flatnonzero(slots.kind == KIND_ID[kind])
        out[kind] = (sel, slots.layer[sel], slots.slot[sel])
    return out


def make_inner_columns(slots: Slots) -> Callable[..., Array]:
    """A jitted `(B, T, n)` inner activation of every slot, canonical order."""
    index = gather_index(slots)
    order = np.argsort(np.concatenate([index[k][0] for k in KIND_ORDER]))

    @eqx.filter_jit
    def inner_columns(
        arch: Arch,
        shared: Shared,
        factors: Factors,
        mask: Mask,
        tokens: Int[Array, "B T"],
        precision: Precision,
    ) -> Float[Array, "B T n"]:
        inner = factored_inner(arch, shared, factors, mask, tokens, precision)
        parts = [
            inner[kind][jnp.asarray(index[kind][1]), :, :, jnp.asarray(index[kind][2])]
            for kind in KIND_ORDER
        ]  # each (k, B, T)
        stacked = jnp.concatenate(parts, axis=0)[jnp.asarray(order)]
        return jnp.transpose(stacked, (1, 2, 0))

    return inner_columns


def pack_rows(bits: np.ndarray) -> np.ndarray:
    """`(B, k)` bool -> `(B / 8, k)` uint8 packed along the prompt axis (`B` a multiple of 8)."""
    assert bits.shape[0] % 8 == 0, bits.shape
    return np.packbits(bits, axis=0)


def unpack_column(packed: np.ndarray, n_prompts: int) -> np.ndarray:
    """One column of a packed `ci_on_pos<t>.npy` (`(ceil(N/8),)` uint8) -> `(N,)` bool."""
    return np.unpackbits(packed, count=n_prompts).astype(bool)


DATASET_README = """\
# Integer-only components: inner activations and binarized CI

Run `{run_dir}`, checkpoint step {step}, commit `{commit}`. Component list:
`../integer-only.json` ({n_components} components). A component is a site name (a module such
as `layers.26.mlp.down_proj`) plus its index within the site's decomposition.

Notation: `N` = {n_prompts} prompts; row `i` (integer, 0 <= i < N) is line `i` of the prompts
file `{prompts}`, counting from 0 (the same rows as `../pool.npz`). `t` (integer,
0 <= t < T = {seq_len}) is a token position: {labels}. A component is ON at `t` on a prompt when
its causal importance (CI) there exceeds {threshold}: the output head's lower-leaky CI of the
run's own CI function, evaluated on the ORIGINAL model's activations. A component's inner
activation is `x V_c` (real): the input `x` of its site (a `d_in`-vector) times its input vector
`V_c` (the column `c` of the site's `V`); when on, the component writes `(x V_c) U_c` into the
site's output, `U_c` its output vector.

Only the (position, component) pairs where the component is ON on at least one prompt are
stored: {n_columns} pairs over the {seq_len} positions ({per_position}). Every other pair is
never ON.

## Files

- `index.npz`:
  - `tokens` (N, T) int32; `op` (N,) int8, index into `ops` ({ops}); `a`, `b` (N,) int64 the
    integer operands, `r` (N,) int64 the true answer; `position_labels` (T,).
  - `col_position`, `col_site`, `col_component` (K,): the K = {n_columns} stored pairs, sorted by
    position, then site name, then component index. Position `t`'s files hold, as columns, the
    rows with `col_position == t`, in this order.
  - `col_v_norm` (K,) float32: `||V_c||`, the Euclidean norm of the component's input vector
    (the applet divides inner activations by it).
- `inner_pos<t>.npy` (N, K_t) float16: the inner activation of position `t`'s K_t columns in the
  INTEGER-ONLY model: exactly the components of `../integer-only.json` on at every position of
  every prompt, every other component and the weight delta off, computed in {precision}.
- `ci_on_pos<t>.npy` (ceil(N/8), K_t) uint8: whether each column is ON (binarized original
  CI), bit-packed along the prompt axis (numpy `packbits`, big bit order).

```python
import numpy as np

index = np.load("index.npz")
t = {last}
cols = np.flatnonzero(index["col_position"] == t)
j = 0  # the first column of position t: component index["col_component"][cols[j]] of index["col_site"][cols[j]]
inner = np.load(f"inner_pos{{t}}.npy", mmap_mode="r")[:, j].astype(np.float32)  # (N,)
on = np.unpackbits(np.load(f"ci_on_pos{{t}}.npy", mmap_mode="r")[:, j], count=len(index["op"])).astype(bool)  # (N,)
```
"""
