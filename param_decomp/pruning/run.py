"""Entry point: every stage of the pruning plan in one process, each resumable from its outputs.

    python -m param_decomp.pruning.run --config <yaml> --data_root <root> --name <prune-id>
        [--stages select,validate,references,screen,combine,estimator_check,prune,report]

Writes `<run_dir>/analysis/pruning/step_<step>/<name>/` (layout in `notes/pruning/plan.md` §6):
`config.yaml`, `hc.npz`, `strata.npz`, `validation.json`, `references.json` (+ `.npz`),
`screen/{a,b}_*.npz`, `components.tsv`, `vectors.npz`, `estimator_check.json`, `rounds.jsonl`,
`sets/S_<j>.npz`, `report.json`. A stage whose output exists is skipped, except `prune`, which
resumes from the newest `sets/S_<j>.npz`."""

import argparse
import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from param_decomp.ci_filter.checkpoint import trained_ci
from param_decomp.ci_filter.config import resolve_run_dir
from param_decomp.ci_filter.paths import CIFilterOutputs
from param_decomp.ci_filter.pool import load_tokenizer
from param_decomp.ci_filter.step import (
    all_logits,
    lm_batch,
    masked_output,
    prepare_components,
    remove_components,
)
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.experiments.lm.load_run import PlacedTransformer, restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.pruning import model as pm
from param_decomp.pruning.config import PruneConfig
from param_decomp.pruning.engine import score_target
from param_decomp.pruning.problem import Problem, build_problem
from param_decomp.pruning.prune import PruneState, run_blockwise, run_rounds, score_set
from param_decomp.pruning.screen import (
    CE,
    Context,
    Precisions,
    load_shards,
    phase_a,
    rich_screen,
    sample_caches,
    target_on_sample,
)
from param_decomp.pruning.stats import (
    QUANTITIES,
    kl,
    log_softmax32,
    stratified,
    weighted_moments,
    weighted_quantiles,
)
from param_decomp.targets.transformer import KIND_ORDER, TransformerDecomposedModel

STAGES = (
    "select",
    "validate",
    "references",
    "screen",
    "combine",
    "estimator_check",
    "prune",
    "report",
)
PERIODS = (2, 5, 10, 20, 50, 100)
ROUND_KNOBS = (
    "z",
    "epsilon",
    "theta",
    "round_tolerance",
    "global_budget",
    "max_batch_fraction",
    "disagreement_sigma",
    "max_disagreements",
    "max_rounds",
    "max_fresh_screens",
    "objective",
    "mode",
    "round_precision",
)
R_BIN = 100


def pruning_dir(run_dir: Path, step: int, name: str) -> Path:
    return run_dir / "analysis" / "pruning" / f"step_{step}" / name


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=1) + "\n")


def run(config: PruneConfig, data_root: Path, name: str, stages: tuple[str, ...]) -> None:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    out = pruning_dir(run_dir, config.step, name)
    out.mkdir(parents=True, exist_ok=True)
    pinned = out / "config.yaml"
    dumped = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    if pinned.exists():
        old = PruneConfig.model_validate(yaml.safe_load(pinned.read_text()))
        # Every field shapes some persisted output, except the round knobs (read only by
        # `prune`, which resumes) and the batch sizes; the screen precisions may still change
        # until the screen has started (the validation measures every precision path).
        mutable = set(ROUND_KNOBS) | {"batch_size", "hc_components_per_batch", "select_batch_size",
                                      "grad_batch_size", "compilation_cache_dir"}  # fmt: skip
        if not (out / "screen").exists():
            mutable.update({"site_precision", "head_precision"})
        for key in PruneConfig.model_fields:
            if key not in mutable:
                assert getattr(old, key) == getattr(config, key), f"{key} differs from {pinned}"
    pinned.write_text(dumped)

    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh = restored.placed, restored.mesh
    problem = build_problem(config.pool, load_tokenizer(target.model_name))
    logger.info(f"P: {problem.n} prompts (answer >= 1), T = {problem.tokens.shape[1]}")

    filt = CIFilterOutputs.for_run(run_dir, config.step, config.filter_id)
    alive_ids = json.loads(filt.alive.read_text())["components"]
    alive = {}
    for site in placed.model.sites:
        keep = np.zeros(site.C, bool)
        keep[np.asarray(alive_ids.get(site.name, []), np.int64)] = True
        alive[site.name] = keep
    logger.info(f"S0: {sum(int(v.sum()) for v in alive.values())} components")

    with jax.set_mesh(mesh):
        ci_fn, constraints = trained_ci(run_dir, config.step, config.filter_id, restored.ci_fn)
        components = (
            restored.components
            if constraints.kept is None
            else remove_components(restored.components, constraints.kept)
        )
        prepared, _ = prepare_components(placed, components)
        assert isinstance(placed.model, TransformerDecomposedModel)
        arch, shared, factors, dense, slots = pm.build(placed.model, restored.components, alive)
        del components

        hc_path = out / "hc.npz"
        if "select" in stages and not hc_path.exists():
            from param_decomp.pruning.select import select_top_ci

            t0 = time.time()
            top, values = select_top_ci(
                placed, prepared, ci_fn, constraints, problem.tokens, slots,
                config.top_ci, config.select_batch_size,
            )  # fmt: skip
            np.savez(hc_path, top=top, values=values)
            logger.info(f"select: {time.time() - t0:.0f}s -> {hc_path}")
        del ci_fn, constraints, restored
        top = np.load(hc_path)["top"]

        strata_path = out / "strata.npz"
        if not strata_path.exists():
            rng = np.random.default_rng(np.random.SeedSequence((config.seed, 17)))
            sample = np.sort(rng.choice(problem.n, config.sample_size, replace=False))
            check = np.sort(rng.choice(problem.n, min(config.check_size, problem.n), replace=False))
            np.savez(strata_path, sample=sample, check=check)
        strata = np.load(strata_path)
        sample, check = strata["sample"], strata["check"]

        ctx = Context(
            arch=arch, shared=shared, factors=factors, dense=dense, slots=slots, problem=problem,
            sample=sample, top=top,
            precisions=Precisions(config.site_precision, config.head_precision, config.target_precision),
            batch_size=config.batch_size, group_size=config.hc_components_per_batch,
        )  # fmt: skip
        on0 = np.ones(slots.n, bool)
        mask0 = slots.mask(on0, arch.n_layer)

        def native_logp(on: np.ndarray, prompts: np.ndarray) -> np.ndarray:
            """The decomposition's own bf16 masked forward (global mask, delta off)."""
            keep = {s: np.zeros(c.size, np.float32) for s, c in alive.items()}
            for a in np.nonzero(on)[0]:
                keep[slots.sites[a]][slots.index[a]] = 1.0
            outs = []
            for s in range(0, prompts.size, 512):
                tokens = jnp.asarray(problem.tokens[prompts[s : s + 512]])
                masks = {
                    site: jnp.broadcast_to(jnp.asarray(k, COMPUTE_DT), (*tokens.shape, k.size))
                    for site, k in keep.items()
                }
                logits = all_logits(
                    masked_output(
                        placed, prepared, lm_batch(tokens),
                        MaterializedMasking(component_masks=masks), remat=False,
                    )
                )[:, -1, :]  # fmt: skip
                outs.append(np.asarray(log_softmax32(logits)))
            return np.concatenate(outs)

        if "validate" in stages and not (out / "validation.json").exists():
            _json(out / "validation.json", validate(config, ctx, mask0, on0, native_logp, placed))

        if "references" in stages and not (out / "references.json").exists():
            everything = np.arange(problem.n)
            llama = _score_target(ctx, everything, config.full_precision)
            s0 = score_set(ctx, mask0, everything, config.full_precision, True)
            np.savez(out / "references.npz", llama=llama, s0=s0)
            _json(
                out / "references.json",
                {"llama": _summary(problem, llama), "s0": _summary(problem, s0)},
            )

        if "screen" in stages:
            caches = sample_caches(ctx, mask0, target_on_sample(ctx))
            rich_screen(ctx, mask0, on0, out / "screen", caches)
            del caches

        if "combine" in stages and not (out / "components.tsv").exists():
            combine(ctx, out)

        if "estimator_check" in stages and not (out / "estimator_check.json").exists():
            _json(
                out / "estimator_check.json", estimator_check(config, ctx, mask0, on0, check, out)
            )

        if "prune" in stages:
            state = _resume(out, slots.n, config.objective)
            objective = QUANTITIES.index(config.objective)
            ce_s0 = float(
                np.mean(score_set(ctx, mask0, check, config.full_precision, False)[objective])
            )
            match config.mode:
                case "global":
                    run_rounds(config, ctx, state, check, ce_s0, out)
                case "blockwise":
                    run_blockwise(config, ctx, state, check, ce_s0, out)

        if "report" in stages:
            final = _resume(out, slots.n, config.objective)
            everything = np.arange(problem.n)
            refs = np.load(out / "references.npz")
            mask_f = slots.mask(final.on, arch.n_layer)
            sf = score_set(ctx, mask_f, everything, config.full_precision, True)
            native = {}
            for label, on in (("s0", on0), ("final", final.on)):
                lp = native_logp(on, check)
                ours = np.stack(
                    [
                        score_set(
                            ctx, slots.mask(on, arch.n_layer), check, config.full_precision, False
                        )[CE]
                    ]
                )
                y = problem.y[check]
                native[label] = {
                    "ce_native": float(np.mean(-lp[np.arange(check.size), y])),
                    "ce_ours": float(ours.mean()),
                }
            by_layer: dict[str, int] = {}
            for a in np.nonzero(final.on)[0]:
                key = f"{slots.layer[a]}.{KIND_ORDER[slots.kind[a]]}"
                by_layer[key] = by_layer.get(key, 0) + 1
            _json(
                out / "report.json",
                {
                    "size": {"s0": int(slots.n), "final": int(final.on.sum())},
                    "llama": _summary(problem, refs["llama"]),
                    "s0": _summary(problem, refs["s0"]),
                    "final": _summary(problem, sf),
                    "native_on_check_set": native,
                    "final_by_layer_kind": by_layer,
                },
            )
    logger.info(f"done -> {out}")


def _score_target(ctx: Context, prompts: np.ndarray, precision: pm.Precision) -> np.ndarray:
    p = ctx.problem
    out = []
    for s in range(0, prompts.size, ctx.batch_size):
        chunk = prompts[s : s + ctx.batch_size]
        rows = np.concatenate([chunk, np.repeat(chunk[-1:], ctx.batch_size - chunk.size)])
        scores = score_target(
            ctx.arch, ctx.shared, ctx.dense, jnp.asarray(p.tokens[rows]), jnp.asarray(p.y[rows]),
            jnp.asarray(p.r[rows]), ctx.int_ids, precision,
        )  # fmt: skip
        out.append(np.asarray(scores)[:, : chunk.size])
    return np.concatenate(out, axis=1)


def _summary(problem: Problem, scores: np.ndarray) -> dict[str, object]:
    """Means of every quantity over `P`, by operation and by answer bin."""

    def means(sel: np.ndarray) -> dict[str, float]:
        return {q: float(scores[i, sel].mean()) for i, q in enumerate(QUANTITIES)} | {
            "n": int(sel.sum())
        }

    out: dict[str, object] = {"all": means(np.ones(problem.n, bool))}
    for op_name, op_id in (("add", 0), ("sub", 1)):
        out[op_name] = means(problem.op == op_id)
    bins = (problem.r - 1) // R_BIN
    for op_name, op_id in (("add", 0), ("sub", 1)):
        for b in np.unique(bins[problem.op == op_id]):
            lo = int(b) * R_BIN + 1
            out[f"{op_name}_r{lo}-{lo + R_BIN - 1}"] = means((problem.op == op_id) & (bins == b))
    return out


def _resume(out: Path, n: int, objective: str) -> PruneState:
    sets = sorted((out / "sets").glob("S_*.npz")) if (out / "sets").exists() else []
    if sets:
        d = np.load(sets[-1])
        logger.info(f"resuming from {sets[-1].name}: |S| = {int(d['on'].sum())}")
        return PruneState(
            on=d["on"].copy(),
            est=d["est"].copy(),
            se=d["se"].copy(),
            readded=d["readded"].copy(),
            removed=d["removed"].copy(),
            next_round=int(d["next_round"]),
            fresh_screens=int(d["fresh_screens"]),
        )
    table = np.genfromtxt(
        out / "components.tsv", delimiter="\t", names=True, dtype=None, encoding=None
    )
    return PruneState(
        on=np.ones(n, bool),
        est=np.asarray(table[f"d_{objective}"], np.float64),
        se=np.asarray(table[f"se_{objective}"], np.float64),
        readded=np.full(n, -1),
        removed=np.full(n, -1),
    )


# ----------------------------------- validation -----------------------------------


def validate(
    config: PruneConfig,
    ctx: Context,
    mask0: pm.Mask,
    on0: np.ndarray,
    native_logp: Callable[[np.ndarray, np.ndarray], np.ndarray],
    placed: PlacedTransformer,
) -> dict[str, object]:
    out: dict[str, object] = {}
    prompts = ctx.sample[: config.validation_prompts]
    tokens = jnp.asarray(ctx.problem.tokens[prompts])
    ours = []
    tgt = []
    clean = []
    for s in range(0, prompts.size, 512):
        tk = tokens[s : s + 512]
        ours.append(np.asarray(log_softmax32(pm.factored_logits(
            ctx.arch, ctx.shared, ctx.factors, mask0, pm.Flips.none(tk.shape[0]), tk, "fp32", "fp32"))))  # fmt: skip
        tgt.append(
            np.asarray(
                log_softmax32(pm.dense_logits(ctx.arch, ctx.shared, ctx.dense, tk, "bf16", "bf16"))
            )
        )
        clean.append(
            np.asarray(
                log_softmax32(all_logits(placed.clean_forward(lm_batch(tk)).output)[:, -1, :])
            )
        )
    ours_a, tgt_a, clean_a = (np.concatenate(x) for x in (ours, tgt, clean))
    native = native_logp(on0, prompts)
    k_native = np.asarray(kl(jnp.asarray(native), jnp.asarray(ours_a)))
    k_target = np.asarray(kl(jnp.asarray(clean_a), jnp.asarray(tgt_a)))
    k_s0 = np.asarray(kl(jnp.asarray(tgt_a), jnp.asarray(ours_a)))
    out["kl_native_vs_ours_s0"] = {"mean": float(k_native.mean()), "max": float(k_native.max())}
    out["kl_clean_vs_ours_target"] = {"mean": float(k_target.mean()), "max": float(k_target.max())}
    out["kl_target_vs_s0"] = {"mean": float(k_s0.mean())}

    # Partial forwards from several blocks reproduce the full forward.
    tk = tokens[:256]
    inputs, final = pm.block_inputs(ctx.arch, ctx.shared, ctx.factors, mask0, tk, "fp32")
    full = pm.last_logits(ctx.arch, ctx.shared, final, "fp32")
    gaps = {}
    for start in (0, 8, 16, 24, ctx.arch.n_layer - 1):
        resid = pm.run_blocks(
            ctx.arch,
            ctx.shared,
            ctx.factors,
            mask0,
            pm.Flips.none(256),
            inputs[start],
            start,
            "fp32",
        )
        gaps[str(start)] = float(
            jnp.max(jnp.abs(pm.last_logits(ctx.arch, ctx.shared, resid, "fp32") - full))
        )
    out["partial_forward_max_abs_logit_gap"] = gaps

    # Noise floor of the fast precision paths on the screen statistic.
    rng = np.random.default_rng(np.random.SeedSequence((config.seed, 23)))
    comps = np.sort(rng.choice(ctx.slots.n, config.noise_check_components, replace=False))
    targets = target_on_sample(ctx)
    deltas: dict[str, np.ndarray] = {}
    for mode in ("fp32", "tf32", "bf16"):
        c = dataclasses.replace(ctx, precisions=Precisions(mode, mode, ctx.precisions.target))
        caches = sample_caches(c, mask0, targets)
        t0 = time.time()
        deltas[mode] = np.stack(
            [phase_a(c, mask0, on0, caches, int(a))["delta"][CE] for a in comps]
        )
        out[f"seconds_per_component_{mode}"] = (time.time() - t0) / comps.size
        del caches
    ref = deltas["fp32"]
    se = ref.std(axis=1, ddof=1) / np.sqrt(ref.shape[1])
    for mode in ("tf32", "bf16"):
        ratio = np.abs(deltas[mode].mean(axis=1) - ref.mean(axis=1)) / np.maximum(se, 1e-12)
        out[f"noise_{mode}_vs_fp32"] = {
            "frac_below_0.1_se": float((ratio < 0.1).mean()),
            "median_ratio": float(np.median(ratio)),
            "max_ratio": float(ratio.max()),
        }
    out["noise_components"] = comps.tolist()
    logger.info(f"validation: {json.dumps(out)[:2000]}")
    return out


# ----------------------------------- the per-component table -----------------------------------


def combine(ctx: Context, out: Path) -> None:
    """`components.tsv` and `vectors.npz` from the screen shards (plan §3)."""
    a = load_shards(out / "screen", "a")
    b = load_shards(out / "screen", "b")
    assert (a["comps"] == np.arange(ctx.slots.n)).all() and (b["comps"] == a["comps"]).all()
    n_total, k = ctx.problem.n, ctx.top.shape[1]
    pos = ctx.position_in_sample[ctx.top]  # (A, k)
    in_r = pos >= 0
    m = ctx.sample.size - in_r.sum(axis=1)
    w = (n_total - k) / m

    def pw(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float64)
        return np.stack([np.sum(v**p, axis=-1) for p in (1, 2, 3)], axis=-1)

    def sums_rh(values_r: np.ndarray) -> np.ndarray:
        """`(A, ..., 3)` power sums over R ∩ H_a of `(A, ..., |R|)` values."""
        out_ = np.zeros((*values_r.shape[:-1], 3))
        for i in range(values_r.shape[0]):
            out_[i] = pw(values_r[i][..., pos[i][in_r[i]]])
        return out_

    cols: dict[str, np.ndarray] = {}
    delta_r = a["delta"].astype(np.float64)
    delta_r[:, CE] = a["delta_ce"]
    for label, r_vals, h_vals in (
        ("d", delta_r, b["delta"]),
        ("without", a["without"], b["without"]),
    ):
        sh, sr, srh = pw(h_vals), pw(r_vals), sums_rh(r_vals)
        mean, se = stratified(n_total, sh, sr, srh, k, m[:, None])
        mu, sd, skew = weighted_moments(n_total, sh, sr, srh, k, m[:, None])
        for i, q in enumerate(QUANTITIES):
            cols[f"{label}_{q}"] = mean[:, i]
        cols[f"{label}_ce_sd"], cols[f"{label}_ce_skew"] = sd[:, CE], skew[:, CE]
        if label == "d":
            for i, q in enumerate(QUANTITIES):
                cols[f"se_{q}"] = se[:, i]
            del mu
    cols["d_ce"] = cols.pop("d_ce")
    dce_r, dce_h = a["delta_ce"], b["delta"][:, CE]
    for name, thr in (("frac_removal_hurts", 0.1), ("frac_removal_helps", -0.1)):
        ind_r = (dce_r > thr) if thr > 0 else (dce_r < thr)
        ind_h = (dce_h > thr) if thr > 0 else (dce_h < thr)
        cols[name], _ = stratified(n_total, pw(ind_h), pw(ind_r), sums_rh(ind_r), k, m)
    qs = np.zeros((ctx.slots.n, 3))
    for i in range(ctx.slots.n):
        rest = np.delete(dce_r[i], pos[i][in_r[i]])
        qs[i] = weighted_quantiles(dce_h[i], rest, float(w[i]), (0.01, 0.5, 0.99))
    cols["d_ce_q01"], cols["d_ce_q50"], cols["d_ce_q99"] = qs.T

    def vec(r_sum: np.ndarray, h_both: np.ndarray) -> np.ndarray:
        return (h_both[:, 0] + w[:, None] * (r_sum - h_both[:, 1])) / n_total

    dq_x = vec(a["dq_x"], b["dq_x"])
    dq_rel = vec(a["dq_rel"], b["dq_rel"])
    residues: dict[str, np.ndarray] = {}
    d_values = np.arange(-999, 1000)
    for tau in PERIODS:
        residues[f"x_mod_{tau}"] = dq_x.reshape(ctx.slots.n, -1, tau).sum(axis=1)
        residues[f"rel_mod_{tau}"] = np.stack(
            [dq_rel[:, (d_values % tau) == res].sum(axis=1) for res in range(tau)], axis=1
        )
    np.savez(out / "vectors.npz", dq_x=dq_x, dq_rel=dq_rel, **residues)  # pyright: ignore[reportArgumentType]

    s = ctx.slots
    header = ["site", "layer", "kind", "component", *cols]
    lines = ["\t".join(header)]
    for i in range(s.n):
        lines.append(
            "\t".join(
                [s.sites[i], str(s.layer[i]), KIND_ORDER[s.kind[i]], str(s.index[i])]
                + [f"{cols[c][i]:.6g}" for c in cols]
            )
        )
    (out / "components.tsv").write_text("\n".join(lines) + "\n")
    logger.info(f"combine: {s.n} components -> {out / 'components.tsv'}")


def estimator_check(
    config: PruneConfig,
    ctx: Context,
    mask0: pm.Mask,
    on0: np.ndarray,
    check: np.ndarray,
    out: Path,
) -> dict[str, object]:
    """Stratified estimates against the paired mean over the independent check set."""
    table = np.genfromtxt(
        out / "components.tsv", delimiter="\t", names=True, dtype=None, encoding=None
    )
    est, se = np.asarray(table["d_ce"]), np.asarray(table["se_ce"])
    z = est / np.maximum(se, 1e-12)
    order = np.argsort(z)
    picks = order[np.linspace(0, order.size - 1, config.estimator_check_components).astype(int)]
    base = score_set(ctx, mask0, check, config.full_precision, False)[CE]
    rows = []
    for a in picks:
        without = on0.copy()
        without[a] = False
        d = (
            score_set(
                ctx, ctx.slots.mask(without, ctx.arch.n_layer), check, config.full_precision, False
            )[CE]
            - base
        )
        se_check = d.std(ddof=1) / np.sqrt(d.size)
        rows.append({
            "component": int(a), "estimate": float(est[a]), "se": float(se[a]),
            "check_mean": float(d.mean()), "check_se": float(se_check),
            "z": float((est[a] - d.mean()) / np.hypot(se[a], se_check)),
        })  # fmt: skip
    zs = np.asarray([r["z"] for r in rows])
    return {"rows": rows, "z_sd": float(zs.std()), "z_mean": float(zs.mean())}


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    stages = tuple(args.stages.split(","))
    assert set(stages) <= set(STAGES), stages
    config = PruneConfig.model_validate(yaml.safe_load(args.config.read_text()))
    run(config, args.data_root, args.name, stages)


if __name__ == "__main__":
    main()
