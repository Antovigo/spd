"""The integer-only components' activation dataset over the whole pool.

For every integer-only component and every position `t` where it is EVER on (its original CI
exceeds the threshold at `t` on some pool prompt, in stage 1's pass or in this stage's own),
over all `N` prompts: (a) its
inner activation `x V_c` in the INTEGER-ONLY model (exactly the integer-only components on, the
weight delta off, `model.factored_inner`), float16; (b) its binarized original CI (CI on the
original model > threshold), bit-packed. Layout: `index.npz` and one folder per dataset
(`CI_DIR`, `INNER_DIR`), each with its README (`dataset_readmes`). `collect_dataset` writes
it for any (component set, CI fn, constraints): `integer_only.adv` reads its fine-tuned CI fn
under the ceiling."""

import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.ci_filter.step import CIConstraints, FilterCIFn, Prepared, index_batches
from param_decomp.core.log import logger
from param_decomp.experiments.lm.load_run import PlacedTransformer
from param_decomp.integer_only.ci_pass import chunk_ci_columns
from param_decomp.integer_only.config import Precision
from param_decomp.integer_only.layout import CI_DIR, INNER_DIR
from param_decomp.integer_only.model import (
    KIND_ID,
    Arch,
    Factors,
    Mask,
    Shared,
    Slots,
    factored_inner,
)
from param_decomp.integer_only.pool import PromptPool
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


def ever_on_max_ci(
    slots: Slots, max_ci: Mapping[str, np.ndarray], threshold: float, seq_len: int
) -> np.ndarray:
    """`(T, n)` bool: the stored components' max original CI over the pool (`max_ci.npz`, stage
    1) above `threshold` at position `t`."""
    return np.stack(
        [
            np.asarray(
                [
                    max_ci[s][t, c] > threshold
                    for s, c in zip(slots.sites, slots.index, strict=True)
                ],
                bool,
            )
            for t in range(seq_len)
        ]
    ).reshape(seq_len, slots.n)


def ever_on_bits(bits: np.ndarray) -> np.ndarray:
    """`(T, n)` bool: ON on at least one prompt, from `(ceil(N/8), T, n)` packed ON bits (a
    packed byte is nonzero iff one of its 8 prompts is ON)."""
    return np.asarray(bits.any(axis=0))


def columns_of(ever: np.ndarray) -> Columns:
    """The `(T, n)` mask's true pairs as `Columns` (position, then component, ascending)."""
    t, j = np.nonzero(ever)
    return Columns(position=t.astype(np.int64), component=j.astype(np.int64))


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


def collect_dataset(
    placed: PlacedTransformer,
    prepared: Prepared,
    v_norms: Mapping[str, Array],
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    mesh: jax.sharding.Mesh,
    arch: Arch,
    shared: Shared,
    factors: Factors,
    slots: Slots,
    pool: PromptPool,
    max_ci: Mapping[str, np.ndarray],
    threshold: float,
    ci_batch: int,
    precision: Precision,
    destination: Path,
    readme: Callable[[int, str, int, int], dict[str, str]],
) -> dict[str, object]:
    """Write the dataset of the stored components `slots` (every one on, weight delta off) into
    `destination` (replaced atomically). Pass 1 evaluates `ci_fn` under `constraints` over the
    pool (batches of `ci_batch`) for every stored component's ON bit; the stored pairs are those
    ON there or in the earlier pass `max_ci` (`{site: (T, C)}` max CI). Pass 2 records the inner
    activations in `precision`. `readme(n_columns, per_position, n_first_only, n_second_only)`
    gives the README texts by path relative to `destination` (`dataset_readmes`)."""
    assert ci_batch % 8 == 0, f"ci_batch {ci_batch} must be a multiple of 8"
    by_site = slots.components()
    site_cols = {s: jnp.asarray(np.asarray(by_site.get(s, []), np.int32)) for s in sorted(max_ci)}
    n_packed = -(-pool.n // 8)
    # Pass 1, the CI: every stored component's ON bit at every position, packed along prompts,
    # `(ceil(N/8), T, n)` uint8 in host memory.
    bits = np.zeros((n_packed, pool.seq_len, slots.n), np.uint8)
    t0 = time.time()
    batches = index_batches(pool.n, ci_batch)
    with jax.set_mesh(mesh):
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        for i, (idx, real) in enumerate(batches):
            ci = np.asarray(
                chunk_ci_columns(
                    placed, prepared, ci_fn, constraints, tokens_all, jnp.asarray(idx), site_cols
                )
            )
            on = ci > threshold
            on[real:] = False
            start = int(idx[0])
            n_rows = -(-real // 8)
            bits[start // 8 : start // 8 + n_rows] = pack_rows(on)[:n_rows]
            if i % 50 == 0 or i == len(batches) - 1:
                logger.info(f"collect CI: batch {i + 1}/{len(batches)} ({time.time() - t0:.0f}s)")
    seconds_ci = time.time() - t0
    on_here = ever_on_bits(bits)
    on_first = ever_on_max_ci(slots, max_ci, threshold, pool.seq_len)
    columns = columns_of(on_here | on_first)
    per_t = [columns.at(t) for t in range(pool.seq_len)]
    n_first_only = int((on_first & ~on_here).sum())
    n_second_only = int((on_here & ~on_first).sum())
    out = destination.with_name(destination.name + ".partial")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / CI_DIR).mkdir()
    (out / INNER_DIR).mkdir()
    for t, cols in enumerate(per_t):
        np.save(out / CI_DIR / f"ci_on_pos{t}.npy", np.ascontiguousarray(bits[:, t][:, cols]))
    n_on = int(np.unpackbits(bits).sum())
    del bits
    # Pass 2, the stored components' model: the inner activations of the stored columns.
    inner_mm = [
        np.lib.format.open_memmap(out / INNER_DIR / f"inner_pos{t}.npy", mode="w+", dtype=np.float16,
                                  shape=(pool.n, c.size))
        for t, c in enumerate(per_t)
    ]  # fmt: skip
    on_mask = slots.mask(np.ones(slots.n, bool), arch.n_layer)
    inner_columns = make_inner_columns(slots)
    inner_absmax = 0.0
    t1 = time.time()
    for i, (idx, real) in enumerate(batches):
        inner = np.asarray(
            inner_columns(arch, shared, factors, on_mask, jnp.asarray(pool.tokens[idx]), precision)
        )
        start = int(idx[0])
        for t, cols in enumerate(per_t):
            values = inner[:real, t][:, cols]
            if values.size:
                inner_absmax = max(inner_absmax, float(np.abs(values).max()))
            inner_mm[t][start : start + real] = values.astype(np.float16)
        assert inner_absmax < FP16_MAX, f"inner overflows float16 ({inner_absmax})"
        if i % 50 == 0 or i == len(batches) - 1:
            logger.info(f"collect inner: batch {i + 1}/{len(batches)} ({time.time() - t1:.0f}s)")
    for mm in inner_mm:
        mm.flush()
    del inner_mm
    pos, comp = columns.position, columns.component
    v_norm = {s: np.asarray(v) for s, v in v_norms.items()}
    np.savez(
        out / "index.npz", tokens=pool.tokens, op=pool.op, ops=np.asarray(pool.ops), a=pool.a,
        b=pool.b, r=pool.r, position_labels=np.asarray(pool.position_labels),
        col_position=pos, col_site=np.asarray([slots.sites[j] for j in comp]),
        col_component=slots.index[comp],
        col_v_norm=np.asarray([v_norm[slots.sites[j]][slots.index[j]] for j in comp], np.float32),
    )  # fmt: skip
    texts = readme(
        int(pos.size),
        ", ".join(f"t={t}: {c.size}" for t, c in enumerate(per_t)),
        n_first_only,
        n_second_only,
    )
    for relative, text in texts.items():
        (out / relative).write_text(text)
    if destination.exists():
        shutil.rmtree(destination)
    out.rename(destination)
    never_on = np.setdiff1d(np.arange(slots.n), columns.component)
    return {
        "n_columns": int(pos.size),
        "columns_per_position": [int(c.size) for c in per_t],
        "n_on_events": n_on,
        "integer_only_without_column": [f"{slots.sites[j]}:{slots.index[j]}" for j in never_on],
        "pairs_on_only_in_stage_1": n_first_only,
        "pairs_on_only_here": n_second_only,
        "integer_only_never_on_here": int((~on_here.any(axis=0)).sum()),
        "inner_absmax": inner_absmax,
        "seconds_ci_pass": round(seconds_ci, 1),
        "seconds_inner_pass": round(time.time() - t1, 1),
    }


def dataset_readmes(fields: Mapping[str, object]) -> dict[str, str]:
    """The three README texts of a collection, by path relative to it: the index
    (`README.md`) and one per dataset folder. `fields` fills `INDEX_README`, `CI_README` and
    `INNER_README`; their `{root}` is the relative path from each README to the pipeline
    folder."""
    return {
        "README.md": INDEX_README.format(root="..", **fields),
        f"{CI_DIR}/README.md": CI_README.format(root="../..", **fields),
        f"{INNER_DIR}/README.md": INNER_README.format(root="../..", **fields),
    }


_COMMON = """\
Run `{run_dir}`, checkpoint step {step}, commit `{commit}`. Component list:
`{root}/{component_list}` ({n_components} components). A component is a site name (a module
such as `layers.26.mlp.down_proj`) plus its index within the site's decomposition.

Notation: `N` = {n_prompts} prompts; row `i` (integer, 0 <= i < N) is row `i` of
`{root}/{pool_file}` (line `i` of the prompts file `{prompts}` in the repository, counting
from 0). `t` (integer, 0 <= t < T = {seq_len}) is a token position: {labels}. A component is ON
at `t` on a prompt when its {ci_name} there exceeds {threshold}: {ci_definition}

Only the (position, component) pairs where the component is ON on at least one prompt are
stored: K = {n_columns} pairs over the {seq_len} positions ({per_position}); `K_t` (integer) is
the number at position `t`. Every other pair is never ON. Both datasets store the same columns,
in the order of `{index}`'s column table (`col_position`, `col_site`, `col_component`, sorted by
position, then site name, then component index): position `t`'s file holds, as columns, the
rows with `col_position == t`, in that order.
"""

INDEX_README = (
    """\
# {title} components: the collected datasets

"""
    + _COMMON.replace("{index}", "index.npz")
    + """
## Contents

- `index.npz`, shared by both datasets:
  - `tokens` (N, T) int32; `op` (N,) int8, index into `ops` ({ops}); `a`, `b` (N,) int64 the
    integer operands, `r` (N,) int64 the true answer; `position_labels` (T,).
  - `col_position`, `col_site`, `col_component` (K,): the stored pairs.
  - `col_v_norm` (K,) float32: `||V_c||`, the Euclidean norm of the component's input vector
    (the applet divides inner activations by it).
- `causal_importances/`: whether each stored pair is ON on each prompt (its README).
- `inner_activations/`: each stored pair's inner activation in the {model_name} on each
  prompt (its README).
"""
)

CI_README = (
    """\
# {title} components: binarized causal importances

"""
    + _COMMON.replace("{index}", "../index.npz")
    + """
The ON bits come from a second evaluation of the CI function over the pool (a separately
compiled program, in bf16), which can differ at the threshold from the first evaluation
(`{root}/max_ci.npz`, {first_eval}): {n_first_only} stored pairs are ON only in the first
evaluation (their stored bits are all 0) and {n_second_only} only here.

## Files

- `ci_on_pos<t>.npy` (ceil(N/8), K_t) uint8, `ceil(N/8)` = N/8 rounded up: whether each column
  is ON ({ci_name} > {threshold}), bit-packed along the prompt axis (numpy `packbits`, big bit
  order).

```python
import numpy as np

index = np.load("../index.npz")
t = {last}
cols = np.flatnonzero(index["col_position"] == t)
j = 0  # position t's first column: component index["col_component"][cols[j]] of index["col_site"][cols[j]]
on = np.unpackbits(np.load(f"ci_on_pos{{t}}.npy", mmap_mode="r")[:, j], count=len(index["op"])).astype(bool)  # (N,)
```
"""
)

INNER_README = (
    """\
# {title} components: inner activations

"""
    + _COMMON.replace("{index}", "../index.npz")
    + """
A component's inner activation is `x V_c` (real): the input `x` of its site (a `d_in`-vector)
times its input vector `V_c` (column `c` of the site's `V`, `c` the component's index); when on,
the component writes `(x V_c) U_c` into the site's output, `U_c` its output vector.

## Files

- `inner_pos<t>.npy` (N, K_t) float16: the inner activation of position `t`'s columns in the
  {model_name}: {model_description} Computed in {precision}.

```python
import numpy as np

index = np.load("../index.npz")
t = {last}
cols = np.flatnonzero(index["col_position"] == t)
j = 0  # position t's first column: component index["col_component"][cols[j]] of index["col_site"][cols[j]]
inner = np.load(f"inner_pos{{t}}.npy", mmap_mode="r")[:, j].astype(np.float32)  # (N,)
```
"""
)
