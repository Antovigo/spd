"""The pruning rounds (plan §4) over a fixed check set `Q ⊂ P` and the screens of `screen.py`.

State: `on` `(A,)` the current set `S_j`; `est`, `se` `(A,)` the latest `ΔĈE` / `SE` of every
component (removal effect if on, addition effect if off, both signed "positive = helps").

Round `j`: remove the most negative candidates with `est + z·se ≤ ε` (at most a fraction of
`|S_j|`), accepted by an exact `CE` check on `Q` against the per-round and the global (vs `S₀`)
tolerance, halving the batch on failure; then the mask gradient on `Q`, direct re-measurement of
the components whose first-order estimate disagrees with `est`, and the re-addition of pruned
components whose gradient and direct ablation-in say they help. A stalled round triggers a fresh
screen of every component; a fresh screen that changes nothing ends the run."""

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import jax.numpy as jnp
import numpy as np

from param_decomp.core.log import logger
from param_decomp.pruning.config import PruneConfig
from param_decomp.pruning.engine import GradObjective, mask_gradient, score_prompts
from param_decomp.pruning.model import Mask, Precision
from param_decomp.pruning.screen import (
    Context,
    Precisions,
    lean_screen,
    sample_caches,
    target_on_sample,
)
from param_decomp.pruning.stats import QUANTITIES


def score_set(
    ctx: Context, mask: Mask, prompts: np.ndarray, precision: Precision, with_target: bool
) -> np.ndarray:
    """`(Q, n)` per-prompt quantities of `M_S` on `prompts`."""
    out = []
    p = ctx.problem
    for s in range(0, prompts.size, ctx.batch_size):
        chunk = prompts[s : s + ctx.batch_size]
        pad = ctx.batch_size - chunk.size
        rows = np.concatenate([chunk, np.repeat(chunk[-1:], pad)])
        scores = score_prompts(
            ctx.arch, ctx.shared, ctx.factors, mask, ctx.dense if with_target else None,
            jnp.asarray(p.tokens[rows]), jnp.asarray(p.y[rows]), jnp.asarray(p.r[rows]),
            ctx.int_ids, precision, ctx.precisions.target,
        )  # fmt: skip
        out.append(np.asarray(scores)[:, : chunk.size])
    return np.concatenate(out, axis=1)


def gradient(
    ctx: Context,
    mask: Mask,
    prompts: np.ndarray,
    batch: int,
    precision: Precision,
    objective: GradObjective,
    reference: Mask | None = None,
) -> np.ndarray:
    """`(A,)` `∂ mean f / ∂ m_c` over `prompts` at `mask`, `f` the `objective` of
    `engine.mask_gradient` (`reference`, the `kl_int` reference mask, defaults to `mask`)."""
    p = ctx.problem
    total: dict[str, np.ndarray] | None = None
    for s in range(0, prompts.size, batch):
        chunk = prompts[s : s + batch]
        pad = batch - chunk.size
        rows = np.concatenate([chunk, np.repeat(chunk[-1:], pad)])
        weight = np.concatenate([np.ones(chunk.size), np.zeros(pad)]).astype(np.float32)
        g = mask_gradient(
            ctx.arch, ctx.shared, ctx.factors, mask, mask if reference is None else reference,
            jnp.asarray(p.tokens[rows]), jnp.asarray(p.y[rows]), jnp.asarray(p.r[rows]),
            ctx.int_ids, jnp.asarray(weight), precision, objective,
        )  # fmt: skip
        g_np = {k: np.asarray(v, np.float64) for k, v in g.items()}
        total = g_np if total is None else {k: total[k] + g_np[k] for k in total}
    assert total is not None
    return ctx.slots.gather(total) / prompts.size


@dataclass
class PruneState:
    on: np.ndarray
    est: np.ndarray
    se: np.ndarray
    readded: np.ndarray
    """`(A,)` round of the last re-addition (-1 never)."""
    removed: np.ndarray
    """`(A,)` round of the last removal (-1 never)."""
    next_round: int = 0
    """Rounds already run (a resumed run continues the numbering; `max_rounds` caps the total)."""
    fresh_screens: int = 0


def run_rounds(
    config: PruneConfig,
    ctx: Context,
    state: PruneState,
    check: np.ndarray,
    ce_s0: float,
    out_dir: Path,
) -> PruneState:
    assert config.epsilon is not None, "set `epsilon` from the screen's SE distribution first"
    eps, z, theta = config.epsilon, config.z, config.theta
    slots, n_layer = ctx.slots, ctx.arch.n_layer
    full = config.full_precision
    objective = QUANTITIES.index(config.objective)
    targets = target_on_sample(ctx)
    (out_dir / "sets").mkdir(parents=True, exist_ok=True)
    log = (out_dir / "rounds.jsonl").open("a")

    def mean_ce(on: np.ndarray) -> float:
        return float(
            np.mean(score_set(ctx, slots.mask(on, n_layer), check, full, False)[objective])
        )

    ce_now = mean_ce(state.on)
    for j in range(state.next_round, config.max_rounds):
        t0 = time.time()
        record: dict[str, object] = {"round": j, "size_before": int(state.on.sum())}
        # 1-2. removal batch and its exact check on Q
        cand = np.nonzero(state.on & (state.est + z * state.se <= eps) & (state.readded != j))[0]
        cand = cand[np.argsort(state.est[cand])]
        cap = max(1, int(config.max_batch_fraction * state.on.sum()))
        batch = cand[:cap]
        tries: list[dict[str, float]] = []
        while batch.size:
            trial = state.on.copy()
            trial[batch] = False
            ce_trial = mean_ce(trial)
            tries.append({"size": int(batch.size), "ce": ce_trial})
            if (
                ce_trial - ce_now <= config.round_tolerance
                and ce_trial - ce_s0 <= config.global_budget
            ):
                break
            batch = batch[: batch.size // 2]  # put back the less-negative half
        accepted = batch.size > 0
        if accepted:
            state.on[batch] = False
            state.removed[batch] = j
            ce_now = tries[-1]["ce"]
        record.update(candidates=int(cand.size), removed=int(batch.size), tries=tries)
        mask = slots.mask(state.on, n_layer)
        caches = sample_caches(ctx, mask, targets)
        # 3. gradient pass on Q
        g = gradient(ctx, mask, check, config.grad_batch_size, full, config.objective)
        # 4. disagreement re-ablations of the on components
        lin = -g
        gap = np.abs(lin - state.est) - np.maximum(config.disagreement_sigma * state.se, eps)
        dis = np.nonzero(state.on & (gap > 0))[0]
        dis = dis[np.argsort(-gap[dis])][: config.max_disagreements]
        if dis.size:
            state.est[dis], state.se[dis] = lean_screen(ctx, mask, state.on, dis, caches, objective)
        # 5. re-additions
        add = np.nonzero(~state.on & (g < -theta))[0]
        add = add[np.argsort(g[add])][: config.max_disagreements]
        readded = np.zeros(0, np.int64)
        if add.size:
            est_add, se_add = lean_screen(ctx, mask, state.on, add, caches, objective)
            state.est[add], state.se[add] = est_add, se_add
            readded = add[est_add - z * se_add > theta]
            state.on[readded] = True
            state.readded[readded] = j
            if readded.size:
                ce_now = mean_ce(state.on)
        record.update(
            disagreements=int(dis.size),
            readd_candidates=int(add.size),
            readded=int(readded.size),
            size_after=int(state.on.sum()),
            ce_check=ce_now,
            seconds=time.time() - t0,
        )
        del caches  # rebuilt on the next round's set; two would not fit beside the model
        stalled = not accepted and readded.size == 0
        if stalled:
            if state.fresh_screens >= config.max_fresh_screens:
                record["stop"] = "fresh-screen cap"
                _write(log, record, out_dir, j, state, g)
                break
            state.fresh_screens += 1
            mask = slots.mask(state.on, n_layer)
            caches = sample_caches(ctx, mask, targets)
            everything = np.arange(slots.n)
            state.est, state.se = lean_screen(ctx, mask, state.on, everything, caches, objective)
            back = np.nonzero(~state.on & (state.est - z * state.se > theta))[0]
            state.on[back] = True
            state.readded[back] = j
            del caches
            removable = state.on & (state.est + z * state.se <= eps)
            record.update(fresh_screen=state.fresh_screens, fresh_readded=int(back.size),
                          fresh_removable=int(removable.sum()))  # fmt: skip
            if back.size:
                ce_now = mean_ce(state.on)
            if not back.size and not removable.any():
                record["stop"] = "fresh screen: no change"
                _write(log, record, out_dir, j, state, g)
                break
        _write(log, record, out_dir, j, state, g)
    log.close()
    return state


def _write(
    log: TextIO, record: dict[str, object], out_dir: Path, j: int, state: PruneState, g: np.ndarray
) -> None:
    state.next_round = j + 1
    logger.info(f"round {j}: {record}")
    log.write(json.dumps(record) + "\n")
    log.flush()
    np.savez(
        out_dir / "sets" / f"S_{j + 1:02d}.npz",
        on=state.on,
        est=state.est,
        se=state.se,
        readded=state.readded,
        removed=state.removed,
        grad=g,
        next_round=np.asarray(state.next_round),
        fresh_screens=np.asarray(state.fresh_screens),
    )


def run_blockwise(
    config: PruneConfig,
    ctx: Context,
    state: PruneState,
    check: np.ndarray,
    ce_s0: float,
    out_dir: Path,
) -> PruneState:
    """Prune block by block, L0 first. For block `l`: screen every surviving component of the
    block on the current set, remove the most negative candidates (`est + z·se ≤ ε`, at most
    `max_batch_fraction` of the block's survivors) under the exact check on `Q`, re-screen the
    block's remaining candidates after each accepted batch (a survivor above the threshold is
    not re-measured until the block's re-addition pass), stop when none qualifies (or a batch
    of one fails the check), then test the block's removed components for re-addition. Lower
    blocks are never revisited. `state.next_round` counts the blocks done; the state is also
    saved after every accepted batch (`sets/partial.npz`), and a resumed block starts from it."""
    assert config.epsilon is not None, "set `epsilon` from the screen's SE distribution first"
    eps, z, theta = config.epsilon, config.z, config.theta
    slots, n_layer = ctx.slots, ctx.arch.n_layer
    full = config.full_precision
    objective = QUANTITIES.index(config.objective)
    targets = target_on_sample(ctx)
    if config.round_precision is not None:
        ctx = dataclasses.replace(
            ctx,
            precisions=Precisions(
                config.round_precision, config.round_precision, ctx.precisions.target
            ),
        )
    (out_dir / "sets").mkdir(parents=True, exist_ok=True)
    partial = out_dir / "sets" / "partial.npz"
    log = (out_dir / "rounds.jsonl").open("a")

    def mean_obj(on: np.ndarray) -> float:
        return float(
            np.mean(score_set(ctx, slots.mask(on, n_layer), check, full, False)[objective])
        )

    def screen(comps: np.ndarray) -> None:
        mask = slots.mask(state.on, n_layer)
        caches = sample_caches(ctx, mask, targets)
        state.est[comps], state.se[comps] = lean_screen(
            ctx, mask, state.on, comps, caches, objective
        )
        del caches

    ce_now = mean_obj(state.on)
    for layer in range(state.next_round, n_layer):
        t0 = time.time()
        block = slots.layer == layer
        record: dict[str, object] = {
            "block": layer,
            "size_before": int(state.on.sum()),
            "block_before": int((block & state.on).sum()),
            "ce_before": ce_now,
        }
        batches: list[dict[str, object]] = []
        resumed = partial.exists() and int(np.load(partial)["layer"]) == layer
        if resumed:
            saved = np.load(partial)
            state.on, state.est, state.se = (
                saved["on"].copy(),
                saved["est"].copy(),
                saved["se"].copy(),
            )
            state.removed = saved["removed"].copy()
            ce_now = mean_obj(state.on)
            logger.info(f"block {layer}: resumed mid-block, {int((block & state.on).sum())} left")
        else:
            screen(np.nonzero(block & state.on)[0])
        for _ in range(config.max_rounds):
            cand = np.nonzero(block & state.on & (state.est + z * state.se <= eps))[0]
            if not cand.size:
                break
            cand = cand[np.argsort(state.est[cand])]
            cap = max(1, int(config.max_batch_fraction * (block & state.on).sum()))
            batch = cand[:cap]
            tries: list[dict[str, float]] = []
            while batch.size:
                trial = state.on.copy()
                trial[batch] = False
                ce_trial = mean_obj(trial)
                tries.append({"size": int(batch.size), "ce": ce_trial})
                if (
                    ce_trial - ce_now <= config.round_tolerance
                    and ce_trial - ce_s0 <= config.global_budget
                ):
                    break
                batch = batch[: batch.size // 2]
            batches.append({"candidates": int(cand.size), "tries": tries})
            if not batch.size:
                break
            state.on[batch] = False
            state.removed[batch] = layer
            ce_now = tries[-1]["ce"]
            rest = np.setdiff1d(cand, batch)
            if rest.size:
                screen(rest)
            np.savez(partial, layer=np.asarray(layer), on=state.on, est=state.est, se=state.se,
                     removed=state.removed)  # fmt: skip
        gone = np.nonzero(block & ~state.on)[0]
        readded = np.zeros(0, np.int64)
        if gone.size:
            screen(gone)
            readded = gone[state.est[gone] - z * state.se[gone] > theta]
            if readded.size:
                state.on[readded] = True
                state.readded[readded] = layer
                ce_now = mean_obj(state.on)
        record.update(
            batches=batches,
            removed=int(gone.size - readded.size),
            readded=int(readded.size),
            block_after=int((block & state.on).sum()),
            size_after=int(state.on.sum()),
            ce_after=ce_now,
            seconds=time.time() - t0,
        )
        _write(log, record, out_dir, layer, state, np.zeros(slots.n))
        partial.unlink(missing_ok=True)
    log.close()
    return state
