"""Single-component screens: phase A on the shared sample `R` (cached residuals, partial forwards),
phase B on each component's `H_c` (grouped full forwards), and the stratified combination.

A screen scores a list of components against a global mask `S`: each one is FLIPPED (removed if
on, added if off) and `δ_p = f(S without c) − f(S with c)` is recorded per prompt. The full
("rich") screen of `S₀` writes resumable shards of every per-prompt quantity and integer-shift
vector; the round screens keep only the `CE` estimate."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.core.log import logger
from param_decomp.pruning.engine import (
    SampleCache,
    flip_on_groups,
    flip_on_sample,
    make_sample_cache,
    target_logp,
)
from param_decomp.pruning.model import Arch, DenseWeights, Factors, Mask, Precision, Shared, Slots
from param_decomp.pruning.problem import Problem
from param_decomp.pruning.stats import QUANTITIES, stratified

CE = QUANTITIES.index("ce")
SHARD = 256
"""Components per resumable shard of the rich screen."""


@dataclass(frozen=True)
class Precisions:
    site: Precision
    head: Precision
    target: Precision


@dataclass
class Context:
    """Everything a screen reads: the model, the prompts, the strata."""

    arch: Arch
    shared: Shared
    factors: Factors
    dense: DenseWeights
    slots: Slots
    problem: Problem
    sample: np.ndarray
    """`(|R|,)` indices into `problem` of the shared sample, sorted."""
    top: np.ndarray
    """`(A, k)` indices into `problem` of each component's `H_c`."""
    precisions: Precisions
    batch_size: int
    group_size: int
    int_ids: jax.Array = field(init=False)
    position_in_sample: np.ndarray = field(init=False)
    """`(N,)` each prompt's row in `sample`, -1 outside it."""

    def __post_init__(self) -> None:
        self.int_ids = jnp.asarray(self.problem.int_ids)
        self.position_in_sample = np.full(self.problem.n, -1, np.int64)
        self.position_in_sample[self.sample] = np.arange(self.sample.size)

    def chunks(self) -> list[np.ndarray]:
        return [
            self.sample[s : s + self.batch_size]
            for s in range(0, self.sample.size, self.batch_size)
        ]


def target_on_sample(ctx: Context) -> list[jax.Array]:
    """The target model's last-position log-probs on each sample chunk (mask-independent)."""
    p = ctx.precisions
    return [
        target_logp(
            ctx.arch,
            ctx.shared,
            ctx.dense,
            jnp.asarray(ctx.problem.tokens[chunk]),
            p.target,
            p.target,
        )
        for chunk in ctx.chunks()
    ]


def sample_caches(ctx: Context, mask: Mask, targets: list[jax.Array]) -> list[SampleCache]:
    p = ctx.precisions
    out = []
    for chunk, logp_t in zip(ctx.chunks(), targets, strict=True):
        out.append(
            make_sample_cache(
                ctx.arch,
                ctx.shared,
                ctx.factors,
                mask,
                logp_t,
                jnp.asarray(ctx.problem.tokens[chunk]),
                jnp.asarray(ctx.problem.y[chunk]),
                jnp.asarray(ctx.problem.r[chunk]),
                ctx.int_ids,
                p.site,
                p.head,
            )  # fmt: skip
        )
    return out


def _sign(on: np.ndarray, a: int) -> float:
    return -1.0 if on[a] else 1.0


def phase_a(
    ctx: Context, mask: Mask, on: np.ndarray, caches: list[SampleCache], a: int
) -> dict[str, np.ndarray]:
    """Component `a` flipped on every sample chunk: per-prompt `without` and `delta` `(Q, |R|)`,
    and the integer-shift sums over `R`."""
    p = ctx.precisions
    without, delta, dq_x, dq_rel = [], [], 0.0, 0.0
    for cache in caches:
        res = flip_on_sample(
            ctx.arch, ctx.shared, ctx.factors, mask, cache,
            jnp.asarray(ctx.slots.kind[a], jnp.int32), jnp.asarray(ctx.slots.layer[a], jnp.int32),
            jnp.asarray(ctx.slots.slot[a], jnp.int32), jnp.asarray(_sign(on, a), jnp.float32),
            ctx.int_ids, p.site, p.head,
        )  # fmt: skip
        without.append(np.asarray(res.without))
        delta.append(np.asarray(res.delta))
        dq_x = dq_x + np.asarray(res.dq_x)
        dq_rel = dq_rel + np.asarray(res.dq_rel)
    return {
        "without": np.concatenate(without, axis=1),
        "delta": np.concatenate(delta, axis=1),
        "dq_x": np.asarray(dq_x),
        "dq_rel": np.asarray(dq_rel),
    }


def phase_b(
    ctx: Context, mask: Mask, on: np.ndarray, comps: np.ndarray, with_target: bool
) -> dict[str, np.ndarray]:
    """Components `comps` (at most `group_size`) each flipped on its own `H_c`."""
    p = ctx.precisions
    k = comps.size
    pad = ctx.group_size - k
    padded = np.concatenate([comps, np.repeat(comps[-1:], pad)])
    prompts = ctx.top[padded]  # (G, n)
    tokens = jnp.asarray(ctx.problem.tokens[prompts])
    res = flip_on_groups(
        ctx.arch, ctx.shared, ctx.factors, ctx.dense if with_target else None, mask, tokens,
        jnp.asarray(ctx.problem.y[prompts]), jnp.asarray(ctx.problem.r[prompts]),
        jnp.asarray((ctx.position_in_sample[prompts] >= 0).astype(np.float32)),
        jnp.asarray(ctx.slots.kind[padded], jnp.int32), jnp.asarray(ctx.slots.layer[padded], jnp.int32),
        jnp.asarray(ctx.slots.slot[padded], jnp.int32),
        jnp.asarray([_sign(on, int(a)) for a in padded], jnp.float32),
        ctx.int_ids, p.site, p.head, p.target,
    )  # fmt: skip
    return {
        "without": np.asarray(res.without)[:k],
        "delta": np.asarray(res.delta)[:k],
        "dq_x": np.asarray(res.dq_x)[:k],
        "dq_rel": np.asarray(res.dq_rel)[:k],
    }


def _powers(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.float64)
    return np.stack([np.sum(v**k, axis=-1) for k in (1, 2, 3)], axis=-1)


def ce_estimate(
    ctx: Context, a: int, delta_r: np.ndarray, delta_h: np.ndarray
) -> tuple[float, float]:
    """`(ΔĈE_a, SE_a)` from the per-prompt deltas of one quantity on `R` and on `H_a`."""
    pos = ctx.position_in_sample[ctx.top[a]]
    in_r = pos[pos >= 0]
    mean, se = stratified(
        ctx.problem.n,
        _powers(delta_h),
        _powers(delta_r),
        _powers(delta_r[in_r]),
        ctx.top.shape[1],
        np.asarray(ctx.sample.size - in_r.size),
    )
    return float(mean), float(se)


def lean_screen(
    ctx: Context,
    mask: Mask,
    on: np.ndarray,
    comps: np.ndarray,
    caches: list[SampleCache],
    quantity: int = CE,
) -> tuple[np.ndarray, np.ndarray]:
    """`(Δ̂, SE)` of one quantity (default `CE`) for each component in `comps` against `mask`
    (removal if on, addition if off)."""
    t0 = time.time()
    delta_r = np.stack([phase_a(ctx, mask, on, caches, int(a))["delta"][quantity] for a in comps])
    delta_h = np.concatenate(
        [
            phase_b(ctx, mask, on, comps[s : s + ctx.group_size], with_target=False)["delta"][
                :, quantity
            ]
            for s in range(0, comps.size, ctx.group_size)
        ]
    )
    est = [ce_estimate(ctx, int(a), delta_r[i], delta_h[i]) for i, a in enumerate(comps)]
    logger.info(f"lean screen of {comps.size} components: {time.time() - t0:.0f}s")
    return np.asarray([e[0] for e in est]), np.asarray([e[1] for e in est])


# ----------------------------------- the rich screen of S₀ -----------------------------------


def rich_screen(
    ctx: Context, mask: Mask, on: np.ndarray, out_dir: Path, caches: list[SampleCache]
) -> None:
    """Phase A then phase B over every component of `S₀`, one shard file per `SHARD`
    components per phase; rerunning skips the shards already written."""
    comps = np.arange(ctx.slots.n)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = [comps[s : s + SHARD] for s in range(0, comps.size, SHARD)]
    t0 = time.time()
    for i, shard in enumerate(shards):
        path = out_dir / f"a_{i:04d}.npz"
        if path.exists():
            continue
        parts = [phase_a(ctx, mask, on, caches, int(a)) for a in shard]
        _save(
            path,
            comps=shard,
            delta_ce=np.stack([q["delta"][CE] for q in parts]).astype(np.float32),
            delta=np.stack([q["delta"] for q in parts]).astype(np.float16),
            without=np.stack([q["without"] for q in parts]).astype(np.float16),
            dq_x=np.stack([q["dq_x"] for q in parts]).astype(np.float32),
            dq_rel=np.stack([q["dq_rel"] for q in parts]).astype(np.float32),
        )
        _progress("phase A", i, len(shards), t0)
    t0 = time.time()
    for i, shard in enumerate(shards):
        path = out_dir / f"b_{i:04d}.npz"
        if path.exists():
            continue
        parts = [
            phase_b(ctx, mask, on, shard[s : s + ctx.group_size], with_target=True)
            for s in range(0, shard.size, ctx.group_size)
        ]
        _save(
            path,
            comps=shard,
            delta=np.concatenate([q["delta"] for q in parts]).astype(np.float32),
            without=np.concatenate([q["without"] for q in parts]).astype(np.float32),
            dq_x=np.concatenate([q["dq_x"] for q in parts]).astype(np.float32),
            dq_rel=np.concatenate([q["dq_rel"] for q in parts]).astype(np.float32),
        )
        _progress("phase B", i, len(shards), t0)


def _save(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
    tmp.rename(path)


def _progress(what: str, i: int, n: int, t0: float) -> None:
    rate = (time.time() - t0) / (i + 1)
    logger.info(
        f"{what}: shard {i + 1}/{n} ({rate:.0f}s/shard, {(n - i - 1) * rate / 3600:.2f}h left)"
    )


def load_shards(out_dir: Path, phase: str) -> dict[str, np.ndarray]:
    files = sorted(out_dir.glob(f"{phase}_*.npz"))
    assert files, f"no {phase} shards in {out_dir}"
    loaded = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([d[k] for d in loaded]) for k in loaded[0]}
