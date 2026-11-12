"""One learned global mask at one sparsity weight `λ` (nats per kept component), over the
candidates (`model.Slots`, canonical order `j = 0..n-1`).

Each candidate `j` has a real logit `ℓ_j`. Every Adam step `k` (0-based, `0 <= k < steps`)
draws ONE mask for the whole batch, `m_j ~ Bernoulli(σ(ℓ_j / T))` (`σ` the logistic function,
`T` the surrogate temperature), and minimizes

    L(ℓ) = mean_{b ∈ B} KL_int(p_target ‖ p_m)(b) + λ Σ_j σ(ℓ_j / T)

over a batch `B` of `P \\ Q`, with the straight-through gradient `∂L/∂ℓ_j = (∂ mean KL_int /
∂m_j + λ) · σ'(ℓ_j / T) / T` (`σ' = σ (1 − σ)`, the mask derivative taken at the drawn mask).
The result is the hard mask `1[ℓ_j > 0]` after the last step. Writes, into one `lam_<λ>/`
folder, `learn_config.json` (`LEARN_FIELDS`, `λ` and the candidate count; a resume under other
settings fails), `log.jsonl` (the probe every `eval_every` steps) and `state.npz` (`logit`, the
Adam moments `m1`, `m2`, and `step`, saved with each probe)."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from param_decomp.core.log import logger
from param_decomp.integer_only.config import LEARN_FIELDS, IntegerOnlyConfig
from param_decomp.integer_only.model import Arch, Factors, Shared, Slots
from param_decomp.integer_only.objective import mask_gradient, probe_scores

BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Context:
    """The factored model over the candidates and the problem's host arrays (pool rows)."""

    arch: Arch
    shared: Shared
    factors: Factors
    slots: Slots
    tokens: np.ndarray
    """`(N, T)` int32 pool tokens."""
    r: np.ndarray
    """`(N,)` true answers (only rows of `P` are read)."""
    log_q_ref: np.ndarray
    """`(N, 1000)` float32: the target's `log q` per pool row."""
    int_ids: jnp.ndarray
    train: np.ndarray
    """`P \\ Q`, ascending pool rows."""
    check: np.ndarray
    """`Q`, pool rows in draw order."""


def lr_at(config: IntegerOnlyConfig, step: int) -> float:
    match config.schedule:
        case "constant":
            return config.lr
        case "cosine":
            return config.lr * (1 + np.cos(np.pi * step / config.steps)) / 2


def gradient(
    ctx: Context, on: np.ndarray, rows: np.ndarray, config: IntegerOnlyConfig
) -> np.ndarray:
    """`(n,)` `∂ mean_{rows} KL_int / ∂m_j` at the mask `on`."""
    mask = ctx.slots.mask(on, ctx.arch.n_layer)
    total: dict[str, np.ndarray] | None = None
    for s in range(0, rows.size, config.microbatch):
        chunk = rows[s : s + config.microbatch]
        g = mask_gradient(
            ctx.arch, ctx.shared, ctx.factors, mask, jnp.asarray(ctx.tokens[chunk]),
            jnp.asarray(ctx.log_q_ref[chunk]), jnp.ones((chunk.size,), jnp.float32), ctx.int_ids,
            config.precision,
        )  # fmt: skip
        g_np = {k: np.asarray(v, np.float64) for k, v in g.items()}
        total = g_np if total is None else {k: total[k] + g_np[k] for k in total}
    assert total is not None
    return ctx.slots.gather(total) / rows.size


def probe(
    ctx: Context, on: np.ndarray, rows: np.ndarray, config: IntegerOnlyConfig
) -> tuple[float, float]:
    """Mean `KL_int` to the target and mean `ce_int` of the hard mask `on` over `rows`."""
    mask = ctx.slots.mask(on, ctx.arch.n_layer)
    sums = np.zeros(2)
    for s in range(0, rows.size, config.score_batch):
        chunk = rows[s : s + config.score_batch]
        pad = np.concatenate([chunk, np.repeat(chunk[-1:], config.score_batch - chunk.size)])
        got = probe_scores(
            ctx.arch, ctx.shared, ctx.factors, mask, jnp.asarray(ctx.tokens[pad]),
            jnp.asarray(ctx.log_q_ref[pad]), jnp.asarray(ctx.r[pad]), ctx.int_ids,
            config.precision,
        )  # fmt: skip
        sums += np.asarray(got, np.float64)[:, : chunk.size].sum(axis=1)
    kl, ce = sums / rows.size
    return float(kl), float(ce)


def pinned_settings(config: IntegerOnlyConfig, lam: float, n_candidates: int) -> dict[str, object]:
    pinned: dict[str, object] = {k: getattr(config, k) for k in LEARN_FIELDS}
    return pinned | {"lambda": lam, "n_candidates": n_candidates}


def is_finished(d: Path, config: IntegerOnlyConfig) -> bool:
    state = d / "state.npz"
    return state.exists() and int(np.load(state)["step"]) == config.steps


def learn_mask(config: IntegerOnlyConfig, ctx: Context, lam: float, d: Path) -> dict[str, object]:
    """Train (or resume, from `state.npz`) the mask of sparsity weight `lam` in folder `d`;
    returns the run's facts."""
    d.mkdir(parents=True, exist_ok=True)
    n = ctx.slots.n
    pinned = pinned_settings(config, lam, n)
    pin_path = d / "learn_config.json"
    if pin_path.exists():
        old = json.loads(pin_path.read_text())
        assert old == json.loads(json.dumps(pinned)), (
            f"{d}: learn settings differ from {pin_path}: {old} vs {pinned}"
        )
    else:
        assert not (d / "state.npz").exists(), f"{d}: state.npz without {pin_path}"
        pin_path.write_text(json.dumps(pinned, indent=1) + "\n")
    temp = config.temperature
    probe_rows = ctx.check[: config.eval_size]
    if (d / "state.npz").exists():
        st = np.load(d / "state.npz")
        logit, m1, m2, step0 = st["logit"], st["m1"], st["m2"], int(st["step"])
        assert logit.shape == (n,), (logit.shape, n)
        logger.info(f"learn λ={lam:g}: resume at step {step0}")
    else:
        logit = np.full(n, config.init, np.float64)
        m1, m2, step0 = np.zeros(n), np.zeros(n), 0
    t0 = time.time()
    seconds_grad = 0.0
    with (d / "log.jsonl").open("a") as log:
        for step in range(step0, config.steps):
            rng = np.random.default_rng(np.random.SeedSequence((config.seed, 31, step)))
            batch = np.sort(rng.choice(ctx.train, config.batch, replace=False))
            draw = np.random.default_rng(np.random.SeedSequence((config.seed, 37, step)))
            on = draw.random(n) < sigmoid(logit / temp)
            tg = time.time()
            g_mask = gradient(ctx, on, batch, config)
            seconds_grad += time.time() - tg
            s = sigmoid(logit / temp)
            g = (g_mask + lam) * s * (1 - s) / temp
            m1 = BETA1 * m1 + (1 - BETA1) * g
            m2 = BETA2 * m2 + (1 - BETA2) * g**2
            k = step + 1
            lr = lr_at(config, step)
            logit = logit - lr * (m1 / (1 - BETA1**k)) / (np.sqrt(m2 / (1 - BETA2**k)) + ADAM_EPS)
            if k % config.eval_every == 0 or k == config.steps:
                hard = logit > 0
                kl, ce = probe(ctx, hard, probe_rows, config)
                row = {"step": k, "size": int(hard.sum()), "kl_int_probe": kl,
                       "ce_int_probe": ce, "sec": round(time.time() - t0, 1),
                       "sec_gradient": round(seconds_grad, 1)}  # fmt: skip
                log.write(json.dumps(row) + "\n")
                log.flush()
                np.savez(d / "state.npz", logit=logit, m1=m1, m2=m2, step=k)
                logger.info(f"learn λ={lam:g}: {row}")
    hard = logit > 0
    steps_run = config.steps - step0
    return {
        "lambda": lam,
        "size": int(hard.sum()),
        "n_candidates": n,
        "steps_run": steps_run,
        "seconds": round(time.time() - t0, 1),
        "seconds_per_step_gradient": round(seconds_grad / max(steps_run, 1), 4),
    }
