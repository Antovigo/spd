"""The integer-only-adv fine-tune: the run's CI fn trained against `KL_int` at the last position
under the run's merged stochastic-subset persistent-PGD term (weight delta ablated as in the run's target
stream), its components
frozen, every coefficient constant at the run's final-step value (`coefficients.py`).

Each Adam step `k` (integer, `0 <= k < steps`) draws `batch` distinct rows of `P \\ Q`
(`ci_filter.step.sample_batch`, a pure function of the seed and `k`) and runs
`ci_filter.ppgd.PPGDStep`: the warmup ascents of the persistent adversary, the CI fn's
gradient over the microbatches (recon coefficient times mean `KL_int`, plus the imp-min
activity and frequency terms on the capped upper-leaky CI) and the final ascent. The CI read
everywhere is the ceiling's: the entrywise minimum of the trained fn's and the run's own CI.

The state (CI fn, its Adam state, the adversary, `k`) is saved every `save_every` steps as
`training/state_<k>/`, named by `training/latest_state.json` once complete; a rerun resumes
there and continues the same trajectory (batches and keys depend on `k` only). The finished CI
fn goes to `ci_fn/` and the state folders are removed."""

import json
import os
import shutil
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from param_decomp.ci_filter.checkpoint import restore_tree, save_tree
from param_decomp.ci_filter.config import CIFilterConfig
from param_decomp.ci_filter.ppgd import Adversary, PPGDStep
from param_decomp.ci_filter.step import (
    CIConstraints,
    FilterCIFn,
    Prepared,
    lr_schedule,
    make_apply_update,
    make_optimizer,
    sample_batch,
    trainable,
)
from param_decomp.core.log import logger
from param_decomp.experiments.lm.load_run import PlacedTransformer
from param_decomp.integer_only.adv.config import IntegerOnlyAdvConfig
from param_decomp.integer_only.adv.layout import AdvLayout


@dataclass(frozen=True)
class Problem:
    """The pool on device and the training rows."""

    tokens_all: jax.Array
    """`(N, T)` int32 pool tokens, replicated."""
    answer_ids: jax.Array
    """`(1000,)` int32: the tokens of the integers 0..999 in integer order (`V_int`)."""
    targets: jax.Array
    """`(N,)` int32: each pool row's true answer `r`, an index into `answer_ids`."""
    train: np.ndarray
    """`P \\ Q`, ascending pool rows."""


@dataclass(frozen=True)
class State:
    ci_fn: FilterCIFn
    opt_state: optax.OptState
    adversary: Adversary
    step: int


def batch_rows(problem: Problem, config: IntegerOnlyAdvConfig, step: int) -> np.ndarray:
    """Step `step`'s pool rows."""
    picks = sample_batch(problem.train.size, config.batch, config.seed, step)
    return problem.train[picks].astype(np.int32)


def save_state(layout: AdvLayout, state: State) -> None:
    """Write `state_<k>/`, point `latest_state.json` at it, remove the older states."""
    final = layout.state(state.step)
    partial = final.with_name(final.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    save_tree(partial / "ci_fn", state.ci_fn)
    save_tree(partial / "opt_state", state.opt_state)
    save_tree(partial / "adversary", state.adversary)
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    pointer = layout.latest_state.with_suffix(".partial.json")
    pointer.write_text(json.dumps({"step": state.step}) + "\n")
    os.replace(pointer, layout.latest_state)
    for old in layout.training.glob("state_*"):
        if old != final:
            shutil.rmtree(old)


def load_state(layout: AdvLayout, like: State) -> State | None:
    if not layout.latest_state.exists():
        return None
    step = int(json.loads(layout.latest_state.read_text())["step"])
    d = layout.state(step)
    return State(
        ci_fn=restore_tree(d / "ci_fn", like.ci_fn),
        opt_state=restore_tree(d / "opt_state", like.opt_state),
        adversary=restore_tree(d / "adversary", like.adversary),
        step=step,
    )


def train(
    config: IntegerOnlyAdvConfig,
    filter_config: CIFilterConfig,
    layout: AdvLayout,
    placed: PlacedTransformer,
    prepared: Prepared,
    constraints: CIConstraints,
    problem: Problem,
    start: State,
) -> tuple[FilterCIFn, dict[str, object]]:
    """Run (or resume, from the newest saved state) the fine-tune from `start`; returns the
    trained CI fn and the run's facts. Call under the run's mesh."""
    optimizer = make_optimizer(filter_config)
    apply_update = make_apply_update(optimizer)
    lr = lr_schedule(filter_config)
    ppgd_step = PPGDStep.build(filter_config)
    resumed = load_state(layout, start)
    state = start if resumed is None else resumed
    del start, resumed
    if state.step:
        logger.info(f"integer-only-adv: resume at step {state.step}")
    step_key = jax.random.PRNGKey(config.seed + 3)
    train_frac = jnp.float32(1.0)  # every schedule is constant
    ci_fn, opt_state, adversary = state.ci_fn, state.opt_state, state.adversary
    t0 = time.time()
    t_log = t0
    seconds_steps: list[float] = []
    with layout.metrics.open("a") as sink:
        if state.step:
            sink.write(json.dumps({"resumed_at": state.step}) + "\n")
        for k in range(state.step, config.steps):
            t_step = time.time()
            grads, metrics, schedules, adversary = ppgd_step(
                placed,
                prepared,
                ci_fn,
                constraints,
                problem.tokens_all,
                batch_rows(problem, config, k),
                problem.answer_ids,
                problem.targets,
                train_frac,
                adversary,
                jax.random.fold_in(step_key, k),
                config.microbatch,
                config.accumulate_on_host,
            )
            ci_fn, opt_state, grad_norm = apply_update(ci_fn, opt_state, grads)
            del grads
            done = k + 1
            if k % config.log_every == 0 or done == config.steps:
                now = time.time()
                record = {
                    "step": k,
                    **{name: float(v) for name, v in (metrics | schedules).items()},
                    "grad_norm": float(grad_norm),
                    "lr": float(lr(k)),
                    "adversary_ascents": float(adversary.opt_state.step_count),
                    "elapsed_s": round(now - t0, 1),
                    "s_per_step": (now - t_log) / (config.log_every if k > state.step else 1),
                }
                t_log = now
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                logger.info(
                    f"step {k}: loss {record['loss']:.5f} KL_int {record['recon']:.5f} "
                    f"act {record['imp_activity']:.0f} L0 {record['l0']:.0f} "
                    f"L0>{config.threshold} {record['l0_alive']:.0f} "
                    f"{record['s_per_step']:.2f}s/step"
                )
            seconds_steps.append(time.time() - t_step)
            if done % config.save_every == 0 and done < config.steps:
                t_save = time.time()
                save_state(layout, State(ci_fn, opt_state, adversary, done))
                logger.info(f"state saved at step {done} ({time.time() - t_save:.0f}s)")
    steps_run = config.steps - state.step
    timed = seconds_steps[1:] or seconds_steps
    return ci_fn, {
        "steps_run": steps_run,
        "resumed_at": state.step,
        "seconds_training": round(time.time() - t0, 1),
        "seconds_per_step_after_first": round(float(np.mean(timed)), 3) if timed else None,
        "seconds_first_step": round(seconds_steps[0], 1) if seconds_steps else None,
        "adversary_ascents": float(adversary.opt_state.step_count),
    }


def fresh_state(ci_fn: FilterCIFn, filter_config: CIFilterConfig, adversary: Adversary) -> State:
    return State(
        ci_fn=ci_fn,
        opt_state=make_optimizer(filter_config).init(trainable(ci_fn)),
        adversary=adversary,
        step=0,
    )
