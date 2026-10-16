"""Learned global mask: a λ sweep of straight-through mask training over `S₀` (plan:
`notes/pruning/learned_mask.md`), the comparison point for the blockwise screen-and-prune set.

Each `S₀` component `c` has a real logit `ℓ_c`; the forward uses the hard mask `m_c = 1[ℓ_c > 0]`
(zero ablation, delta off, one mask for every prompt) and the backward the surrogate
`∂m_c/∂ℓ_c = σ'(ℓ_c / T) / T` (`σ` the logistic function, `T` the surrogate temperature). The
loss is `mean_B ce_int + λ · Σ_c σ(ℓ_c / T)` over a minibatch `B` of `P \\ Q`, so `ℓ_c` drifts
down when the first-order cost of dropping `c`, `−∂ ce_int / ∂ m_c`, is below `λ` (nats per
component). Adam on `ℓ`; the mask at the last step is the result, scored on all of `Q`.

Writes `<prune_dir>/learn/lam_<λ>/{learn_config.json, log.jsonl, state.npz, final.json}`
(`learn_config.json` pins the settings of `TRAINING_FIELDS`; a resume under other settings fails) and
`<prune_dir>/learn/baseline.json` (`S₀` and the newest blockwise set on `Q`)."""

import json
import time
from pathlib import Path

import numpy as np

from param_decomp.core.log import logger
from param_decomp.pruning.config import PruneConfig
from param_decomp.pruning.prune import gradient, score_set
from param_decomp.pruning.screen import Context
from param_decomp.pruning.stats import QUANTITIES

CE_INT = QUANTITIES.index("ce_int")
TRAINING_FIELDS = (
    "seed", "full_precision", "learn_steps", "learn_batch", "learn_lr", "learn_init",
    "learn_temperature", "learn_eval_every", "learn_eval_size",
)  # fmt: skip
"""Config fields that shape one λ's outputs: pinned per λ in `learn_config.json`."""
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def summary(ctx: Context, scores: np.ndarray, prompts: np.ndarray) -> dict[str, object]:
    """Mean of every quantity over `prompts`, all and per operation."""
    op = ctx.problem.op[prompts]
    out: dict[str, object] = {}
    for name, sel in (("all", np.ones(op.size, bool)), ("add", op == 0), ("sub", op == 1)):
        out[name] = {q: float(scores[i, sel].mean()) for i, q in enumerate(QUANTITIES)} | {
            "n": int(sel.sum())
        }
    return out


def evaluate(
    config: PruneConfig, ctx: Context, on: np.ndarray, prompts: np.ndarray
) -> dict[str, object]:
    mask = ctx.slots.mask(on, ctx.arch.n_layer)
    scores = score_set(ctx, mask, prompts, config.full_precision, True)
    return {"size": int(on.sum()), "prompts": int(prompts.size)} | summary(ctx, scores, prompts)


def baseline(config: PruneConfig, ctx: Context, check: np.ndarray, out: Path) -> None:
    path = out / "learn" / "baseline.json"
    if path.exists():
        return
    sets = sorted((out / "sets").glob("S_*.npz"))
    rows = {"s0": evaluate(config, ctx, np.ones(ctx.slots.n, bool), check)}
    if sets:
        rows[sets[-1].stem] = evaluate(config, ctx, np.load(sets[-1])["on"], check)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    logger.info(f"learn baseline -> {path}")


def learn_mask(
    config: PruneConfig, ctx: Context, check: np.ndarray, lam: float, out: Path
) -> np.ndarray:
    """Train one mask at sparsity weight `lam`; returns `on` `(A,)`. Resumes from `state.npz`."""
    d = out / "learn" / f"lam_{lam:g}"
    d.mkdir(parents=True, exist_ok=True)
    pinned = {k: getattr(config, k) for k in TRAINING_FIELDS}
    pin_path = d / "learn_config.json"
    if pin_path.exists():
        old = json.loads(pin_path.read_text())
        assert old == pinned, f"{d}: learn settings differ from {pin_path}: {old} vs {pinned}"
    else:
        assert not (d / "state.npz").exists(), f"{d}: state.npz without {pin_path}"
        pin_path.write_text(json.dumps(pinned, indent=1))
    if (d / "final.json").exists():
        return np.load(d / "state.npz")["logit"] > 0
    temp = config.learn_temperature
    n_comp = ctx.slots.n
    train = np.setdiff1d(np.arange(ctx.problem.n), check)
    probe = check[: config.learn_eval_size]
    if (d / "state.npz").exists():
        st = np.load(d / "state.npz")
        logit, m1, m2, step0 = st["logit"], st["m1"], st["m2"], int(st["step"])
        logger.info(f"learn λ={lam:g}: resume at step {step0}")
    else:
        logit = np.full(n_comp, config.learn_init, np.float64)
        m1, m2, step0 = np.zeros(n_comp), np.zeros(n_comp), 0
    t0 = time.time()
    with (d / "log.jsonl").open("a") as log:
        for step in range(step0, config.learn_steps):
            rng = np.random.default_rng(np.random.SeedSequence((config.seed, 31, step)))
            batch = np.sort(rng.choice(train, config.learn_batch, replace=False))
            on = logit > 0
            mask = ctx.slots.mask(on, ctx.arch.n_layer)
            g_mask = gradient(ctx, mask, batch, config.grad_batch_size, config.full_precision, True)
            s = sigmoid(logit / temp)
            g = (g_mask + lam) * s * (1 - s) / temp
            m1 = BETA1 * m1 + (1 - BETA1) * g
            m2 = BETA2 * m2 + (1 - BETA2) * g**2
            k = step + 1
            logit = logit - config.learn_lr * (m1 / (1 - BETA1**k)) / (
                np.sqrt(m2 / (1 - BETA2**k)) + ADAM_EPS
            )
            if k % config.learn_eval_every == 0 or k == config.learn_steps:
                on = logit > 0
                mask = ctx.slots.mask(on, ctx.arch.n_layer)
                ce_int = float(
                    score_set(ctx, mask, probe, config.full_precision, False)[CE_INT].mean()
                )
                row = {"step": k, "size": int(on.sum()), "ce_int_probe": ce_int,
                       "sec": round(time.time() - t0, 1)}  # fmt: skip
                log.write(json.dumps(row) + "\n")
                log.flush()
                np.savez(d / "state.npz", logit=logit, m1=m1, m2=m2, step=k)
                logger.info(f"learn λ={lam:g}: {row}")
    on = logit > 0
    final = {"lambda": lam} | evaluate(config, ctx, on, check)
    (d / "final.json").write_text(json.dumps(final, indent=1))
    logger.info(f"learn λ={lam:g}: size {final['size']} -> {d / 'final.json'}")
    return on


def run_learn(config: PruneConfig, ctx: Context, check: np.ndarray, out: Path) -> None:
    baseline(config, ctx, check, out)
    for lam in config.learn_lambdas:
        learn_mask(config, ctx, check, lam, out)
