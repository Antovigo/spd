"""Learned global mask: a λ sweep of straight-through mask training over `S₀` (plan:
`notes/pruning/learned_mask.md`), the comparison point for the blockwise screen-and-prune set.

Each `S₀` component `c` has a real logit `ℓ_c`; the forward uses the hard mask `m_c = 1[ℓ_c > 0]`
(zero ablation, delta off, one mask for every prompt) and the backward the surrogate
`∂m_c/∂ℓ_c = σ'(ℓ_c / T) / T` (`σ` the logistic function, `T` the surrogate temperature). The
loss is `mean_B f + λ · Σ_c σ(ℓ_c / T)` over a minibatch `B` of `P \\ Q`, `f` the
`learn_objective` (`ce_int`, or `kl_int`, the KL from `S₀` over the integer tokens), so `ℓ_c`
drifts down when the first-order cost of dropping `c`, `−∂ f / ∂ m_c`, is below `λ` (nats per
component). With `learn_sampling` the forward mask of each step is instead a random draw
`m_c ~ Bernoulli(σ(ℓ_c / T))`. Adam on `ℓ`; the hard mask `1[ℓ_c > 0]` at the last step is the
result, scored on all of `Q`.

Writes `<prune_dir>/learn/<learn_name>/lam_<λ>/{learn_config.json, log.jsonl, state.npz,
final.json}` (`learn_config.json` pins the settings of `TRAINING_FIELDS`; a resume under other
settings fails) and `<prune_dir>/learn/<learn_name>/baseline.json` (`S₀`, the newest blockwise
set and the other sweeps' masks on `Q`)."""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from param_decomp.core.log import logger
from param_decomp.pruning.config import PruneConfig
from param_decomp.pruning.engine import factored_logp, kl_four, kl_int_prompts, target_logp
from param_decomp.pruning.model import Mask
from param_decomp.pruning.prune import gradient, score_set
from param_decomp.pruning.screen import Context
from param_decomp.pruning.stats import QUANTITIES

CE_INT = QUANTITIES.index("ce_int")
TRAINING_FIELDS = (
    "seed", "full_precision", "learn_steps", "learn_batch", "learn_lr", "learn_init",
    "learn_temperature", "learn_eval_every", "learn_eval_size", "learn_schedule",
    "learn_objective", "learn_sampling", "learn_max_screen_d_ce_int",
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


def kl_int_set(config: PruneConfig, ctx: Context, mask: Mask, prompts: np.ndarray) -> np.ndarray:
    """`(n,)` `KL(q_S₀ ‖ q_mask)` over the integer tokens per prompt (`q` renormalized over
    the integer tokens)."""
    reference = ctx.slots.mask(np.ones(ctx.slots.n, bool), ctx.arch.n_layer)
    out = []
    for s in range(0, prompts.size, ctx.batch_size):
        chunk = prompts[s : s + ctx.batch_size]
        rows = np.concatenate([chunk, np.repeat(chunk[-1:], ctx.batch_size - chunk.size)])
        kl = kl_int_prompts(
            ctx.arch, ctx.shared, ctx.factors, mask, reference,
            jnp.asarray(ctx.problem.tokens[rows]), ctx.int_ids, config.full_precision,
        )  # fmt: skip
        out.append(np.asarray(kl)[: chunk.size])
    return np.concatenate(out)


def evaluate(
    config: PruneConfig, ctx: Context, on: np.ndarray, prompts: np.ndarray
) -> dict[str, object]:
    """Every quantity of `summary` plus `kl_int` (KL from `S₀` over the integer tokens)."""
    mask = ctx.slots.mask(on, ctx.arch.n_layer)
    scores = score_set(ctx, mask, prompts, config.full_precision, True)
    out = {"size": int(on.sum()), "prompts": int(prompts.size)} | summary(ctx, scores, prompts)
    kl = kl_int_set(config, ctx, mask, prompts)
    op = ctx.problem.op[prompts]
    for name, sel in (("all", np.ones(op.size, bool)), ("add", op == 0), ("sub", op == 1)):
        out[name]["kl_int"] = float(kl[sel].mean())  # pyright: ignore[reportIndexIssue]
    return out


def baseline(config: PruneConfig, ctx: Context, check: np.ndarray, out: Path) -> None:
    """`learn/<learn_name>/baseline.json`: `S₀`, the newest blockwise set and every learned mask
    of the other sweeps, scored on `Q` like this sweep's masks."""
    path = out / "learn" / config.learn_name / "baseline.json"
    if path.exists():
        return
    sets = sorted((out / "sets").glob("S_*.npz"))
    rows = {"s0": evaluate(config, ctx, np.ones(ctx.slots.n, bool), check)}
    if sets:
        rows[sets[-1].stem] = evaluate(config, ctx, np.load(sets[-1])["on"], check)
    here = out / "learn" / config.learn_name
    for state in sorted((out / "learn").glob("**/lam_*/state.npz")):
        if state.parent.parent != here and (state.parent / "final.json").exists():
            name = str(state.parent.relative_to(out / "learn"))
            rows[name] = evaluate(config, ctx, np.load(state)["logit"] > 0, check)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    logger.info(f"learn baseline -> {path}")


def learn_mask(
    config: PruneConfig, ctx: Context, check: np.ndarray, lam: float, out: Path
) -> np.ndarray:
    """Train one mask at sparsity weight `lam`; returns `on` `(A,)`. Resumes from `state.npz`."""
    d = out / "learn" / config.learn_name / f"lam_{lam:g}"
    d.mkdir(parents=True, exist_ok=True)
    pinned = {k: getattr(config, k) for k in TRAINING_FIELDS}
    pin_path = d / "learn_config.json"
    if pin_path.exists():
        # Pins written before `learn_objective` / `learn_sampling` existed are hard-mask `ce_int`
        # sweeps.
        old = {"learn_objective": "ce_int", "learn_sampling": False,
               "learn_max_screen_d_ce_int": None}  # fmt: skip
        old.update(json.loads(pin_path.read_text()))
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
        if config.learn_max_screen_d_ce_int is not None:
            # Non-candidates sit at ℓ = +∞: always on, zero surrogate gradient, no Adam step.
            table = np.genfromtxt(
                out / "components.tsv", delimiter="\t", names=True, dtype=None, encoding=None
            )
            frozen = table["d_ce_int"].astype(float) > config.learn_max_screen_d_ce_int
            assert frozen.size == n_comp
            logit[frozen] = np.inf
            logger.info(f"learn λ={lam:g}: {int((~frozen).sum())} candidates of {n_comp}")
        m1, m2, step0 = np.zeros(n_comp), np.zeros(n_comp), 0
    reference = ctx.slots.mask(np.ones(n_comp, bool), ctx.arch.n_layer)
    t0 = time.time()
    with (d / "log.jsonl").open("a") as log:
        for step in range(step0, config.learn_steps):
            rng = np.random.default_rng(np.random.SeedSequence((config.seed, 31, step)))
            batch = np.sort(rng.choice(train, config.learn_batch, replace=False))
            if config.learn_sampling:
                draw = np.random.default_rng(np.random.SeedSequence((config.seed, 37, step)))
                on = draw.random(n_comp) < sigmoid(logit / temp)
            else:
                on = logit > 0
            mask = ctx.slots.mask(on, ctx.arch.n_layer)
            g_mask = gradient(
                ctx, mask, batch, config.grad_batch_size, config.full_precision,
                config.learn_objective, reference,
            )  # fmt: skip
            s = sigmoid(logit / temp)
            g = (g_mask + lam) * s * (1 - s) / temp
            m1 = BETA1 * m1 + (1 - BETA1) * g
            m2 = BETA2 * m2 + (1 - BETA2) * g**2
            k = step + 1
            match config.learn_schedule:
                case "constant":
                    lr = config.learn_lr
                case "cosine":
                    lr = config.learn_lr * (1 + np.cos(np.pi * step / config.learn_steps)) / 2
            logit = logit - lr * (m1 / (1 - BETA1**k)) / (np.sqrt(m2 / (1 - BETA2**k)) + ADAM_EPS)
            if k % config.learn_eval_every == 0 or k == config.learn_steps:
                on = logit > 0
                mask = ctx.slots.mask(on, ctx.arch.n_layer)
                ce_int = float(
                    score_set(ctx, mask, probe, config.full_precision, False)[CE_INT].mean()
                )
                kl = float(kl_int_set(config, ctx, mask, probe).mean())
                row = {"step": k, "size": int(on.sum()), "ce_int_probe": ce_int,
                       "kl_int_probe": kl, "sec": round(time.time() - t0, 1)}  # fmt: skip
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


KL_COLUMNS = ("kl_int_s0", "kl_full_s0", "kl_int_llama", "kl_full_llama")


def kl_table(config: PruneConfig, ctx: Context, check: np.ndarray, out: Path) -> None:
    """`learn/kl_table.json`: for `S₀`, the newest blockwise set and every finished learned
    mask, the mean over `Q` of `KL(ref ‖ p_mask)` over the integer tokens (renormalized) and
    over the full vocabulary, `ref` = `S₀` (the alive-only model) and Llama."""
    masks = {"s0": np.ones(ctx.slots.n, bool)}
    sets = sorted((out / "sets").glob("S_*.npz"))
    if sets:
        masks[sets[-1].stem] = np.load(sets[-1])["on"]
    for state in sorted((out / "learn").glob("**/lam_*/state.npz")):
        if (state.parent / "final.json").exists():
            masks[str(state.parent.relative_to(out / "learn"))] = np.load(state)["logit"] > 0
    jmasks = {k: ctx.slots.mask(on, ctx.arch.n_layer) for k, on in masks.items()}
    sums = {k: np.zeros(len(KL_COLUMNS)) for k in masks}
    p, prec = ctx.problem, config.full_precision
    t0 = time.time()
    for s in range(0, check.size, ctx.batch_size):
        chunk = check[s : s + ctx.batch_size]
        rows = np.concatenate([chunk, np.repeat(chunk[-1:], ctx.batch_size - chunk.size)])
        tokens = jnp.asarray(p.tokens[rows])
        s0 = factored_logp(ctx.arch, ctx.shared, ctx.factors, jmasks["s0"], tokens, prec)
        llama = target_logp(
            ctx.arch, ctx.shared, ctx.dense, tokens, config.target_precision,
            config.target_precision,
        )  # fmt: skip
        for k, m in jmasks.items():
            logp = factored_logp(ctx.arch, ctx.shared, ctx.factors, m, tokens, prec)
            sums[k] += np.asarray(kl_four(logp, s0, llama, ctx.int_ids))[:, : chunk.size].sum(1)
        logger.info(f"kl_table: {s + chunk.size}/{check.size} ({time.time() - t0:.0f}s)")
    table = {
        k: {"size": int(masks[k].sum())}
        | dict(zip(KL_COLUMNS, (v / check.size).tolist(), strict=True))
        for k, v in sums.items()
    }
    (out / "learn" / "kl_table.json").write_text(json.dumps(table, indent=1))
    logger.info(f"kl_table -> {out / 'learn' / 'kl_table.json'}")
