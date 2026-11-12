"""Entry point: one subcommand per stage of the integer-only pipeline (plan:
`notes/integer_only/plan.md`; layout: `layout.py`).

    python -m param_decomp.integer_only.run <stage> --run_dir <run> --data_root <root>
        --config <yaml> [--lam λ] [--compilation_cache_dir <dir>]

Stages, in order: `candidates` (original CI over the pool), `problem` (`P`, `Q`, the target's
reference distributions, factored-model checks), `learn --lam λ` (one mask; run one process per
`λ` to sweep in parallel), `pick` (score every mask on `Q`, apply the rule, write
`integer-only.json`), `collect` (the dataset), `applet`. The first stage pins `--config` as
`config.yaml`; every later stage requires the same settings (batch sizes may differ). A stage
whose `status/<stage>.done` exists is skipped; `learn` resumes from its `state.npz`. Each stage
writes `status/<stage>.done` (facts) or `status/<stage>.failed` (the error)."""

import argparse
import json
import shutil
import subprocess
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.pool import load_tokenizer
from param_decomp.ci_filter.step import (
    all_logits,
    index_batches,
    lm_batch,
    masked_output,
    prepare_components,
)
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.experiments.lm.load_run import RestoredRun, restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.integer_only import applet as ap
from param_decomp.integer_only.ci_pass import chunk_ci_columns, chunk_max_ci
from param_decomp.integer_only.collect import (
    DATASET_README,
    FP16_MAX,
    columns_from_bits,
    ever_on_columns,
    make_inner_columns,
    pack_rows,
)
from param_decomp.integer_only.config import IntegerOnlyConfig
from param_decomp.integer_only.layout import README, STAGES, Layout, lam_name
from param_decomp.integer_only.learn import Context, is_finished, learn_mask
from param_decomp.integer_only.model import (
    Arch,
    DenseWeights,
    Factors,
    Shared,
    Slots,
    build_factors,
    build_target,
    factored_logits,
    slots_of,
)
from param_decomp.integer_only.objective import log_softmax32, target_logp
from param_decomp.integer_only.pick import (
    CANDIDATES,
    KL_INT,
    TARGET,
    choose,
    mask_json,
    read_mask_json,
    score_masks,
    summary,
    write_summary,
)
from param_decomp.integer_only.pool import (
    PromptPool,
    build_pool,
    check_answer_tokens,
    draw_check_set,
    file_sha256,
    integer_token_ids,
)
from param_decomp.targets.transformer import TransformerDecomposedModel

MUTABLE = {"ci_batch", "score_batch", "microbatch"}
"""Config fields a later stage may change: batch sizes only."""
N_VALIDATION = 256


def commit() -> str:
    """The HEAD of the git checkout the command runs in (the repository root)."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def pin_config(layout: Layout, config: IntegerOnlyConfig) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    if layout.config.exists():
        old = IntegerOnlyConfig.from_file(layout.config)
        for key in IntegerOnlyConfig.model_fields:
            if key not in MUTABLE:
                assert getattr(old, key) == getattr(config, key), (
                    f"{key} differs from {layout.config}"
                )
        return
    layout.config.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


def restore(run_dir: Path, config: IntegerOnlyConfig, data_root: Path) -> RestoredRun:
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    assert restored.step == config.step, (restored.step, config.step)
    assert isinstance(restored.placed.model, TransformerDecomposedModel)
    return restored


def load_pool(layout: Layout, config: IntegerOnlyConfig) -> PromptPool:
    pool = PromptPool.load(layout.pool)
    assert pool.sha256 == file_sha256(config.prompts), f"{config.prompts} changed since stage 1"
    return pool


def site_sizes(restored: RestoredRun) -> dict[str, int]:
    return {s.name: s.C for s in restored.placed.model.sites}


def keep_of(components: dict[str, list[int]], sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """`{site: (C,) bool}` of a component list over every site."""
    keep = {site: np.zeros(c, bool) for site, c in sizes.items()}
    for site, ids in components.items():
        keep[site][np.asarray(ids, np.int64)] = True
    return keep


# ----------------------------------- stages -----------------------------------


def stage_candidates(
    config: IntegerOnlyConfig, layout: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    restored = restore(run_dir, config, data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    tokenizer = load_tokenizer(target.model_name)
    pool = build_pool(config.prompts, config.templates, tokenizer)
    rows = pool.problem_rows()
    int_ids = integer_token_ids(tokenizer)
    check_answer_tokens(config.prompts, pool, rows, int_ids, tokenizer)
    pool.save(layout.pool)
    logger.info(f"pool: {pool.n} prompts, T = {pool.seq_len}, labels {pool.position_labels}")
    placed = restored.placed
    t0 = time.time()
    with jax.set_mesh(restored.mesh):
        prepared, _ = prepare_components(placed, restored.components)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        running: dict[str, np.ndarray] = {}
        batches = index_batches(pool.n, config.ci_batch)
        for i, (idx, real) in enumerate(batches):
            got = chunk_max_ci(
                placed, prepared, restored.ci_fn, tokens_all, jnp.asarray(idx), jnp.asarray(real)
            )
            for site, value in got.items():
                v = np.asarray(value)
                running[site] = np.maximum(running[site], v) if site in running else v
            if i % 50 == 0 or i == len(batches) - 1:
                logger.info(f"candidates: batch {i + 1}/{len(batches)} ({time.time() - t0:.0f}s)")
    np.savez(layout.max_ci, **running)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
    keep = {site: np.asarray((v > config.threshold).any(axis=0)) for site, v in running.items()}
    slots = slots_of(keep)
    doc = mask_json(
        slots, np.ones(slots.n, bool), site_sizes(restored),
        rule=f"original output-head lower-leaky CI > {config.threshold} at some position of some "
        "pool prompt (the run's own CI fn on the original model)",
        source=str(layout.max_ci), threshold=config.threshold,
    )  # fmt: skip
    layout.candidates.write_text(json.dumps(doc) + "\n")
    per_position = [
        int(sum(int((v[t] > config.threshold).sum()) for v in running.values()))
        for t in range(pool.seq_len)
    ]
    return {
        "n_prompts": pool.n,
        "n_problem": int(rows.size),
        "seq_len": pool.seq_len,
        "position_labels": list(pool.position_labels),
        "n_candidates": slots.n,
        "n_components": int(sum(site_sizes(restored).values())),
        "on_per_position": per_position,
        "candidate_widths": slots.widths,
        "seconds_ci_pass": round(time.time() - t0, 1),
    }


def stage_problem(
    config: IntegerOnlyConfig, layout: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    pool = load_pool(layout, config)
    rows = pool.problem_rows()
    check = draw_check_set(rows, config.check_size, config.seed)
    restored = restore(run_dir, config, data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    int_ids = jnp.asarray(integer_token_ids(load_tokenizer(target.model_name)))
    model = restored.placed.model
    assert isinstance(model, TransformerDecomposedModel)
    sizes = site_sizes(restored)
    keep = keep_of(read_mask_json(layout.candidates), sizes)
    with jax.set_mesh(restored.mesh):
        arch, shared, dense = build_target(model)
        factors, slots = build_factors(restored.components, keep, arch.n_layer)
        checks = factored_checks(restored, keep, arch, shared, factors, dense, slots, pool, check)
    del restored
    logger.info(f"factored-model checks: {checks}")
    t0 = time.time()
    partial = layout.reference.with_suffix(".partial.npy")
    ref = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(pool.n, 1000))
    for idx, real in index_batches(pool.n, config.score_batch):
        logp = target_logp(
            arch, shared, dense, jnp.asarray(pool.tokens[idx]), config.target_precision
        )
        lp_int = jnp.take(logp, int_ids, axis=-1)
        log_q = lp_int - jax.nn.logsumexp(lp_int, axis=-1, keepdims=True)
        ref[idx[:real]] = np.asarray(log_q)[:real]
    ref.flush()
    del ref
    partial.rename(layout.reference)
    np.save(layout.check_set, check)
    ops = np.asarray(pool.op[check])
    return {
        "n_problem": int(rows.size),
        "n_check": int(check.size),
        "check_per_operation": {op: int((ops == k).sum()) for k, op in enumerate(pool.ops)},
        "seconds_reference": round(time.time() - t0, 1),
        "checks": checks,
    }


def factored_checks(
    restored: RestoredRun,
    keep: dict[str, np.ndarray],
    arch: Arch,
    shared: Shared,
    factors: Factors,
    dense: DenseWeights,
    slots: Slots,
    pool: PromptPool,
    check: np.ndarray,
) -> dict[str, float]:
    """Last-position KLs on `N_VALIDATION` prompts of `Q`: the factored all-candidates model
    (fp32) against the decomposition's own masked forward with the same global mask (weight
    delta off, the run's compute precision), and the dense target (fp32 and the configured
    precision) against the run's clean forward."""
    placed = restored.placed
    rows = check[:N_VALIDATION]
    tokens = jnp.asarray(pool.tokens[rows])
    prepared, _ = prepare_components(placed, restored.components)
    masks = {
        site: jnp.broadcast_to(jnp.asarray(k, COMPUTE_DT), (*tokens.shape, k.size))
        for site, k in keep.items()
    }
    native = log_softmax32(
        all_logits(
            masked_output(
                placed, prepared, lm_batch(tokens), MaterializedMasking(component_masks=masks),
                remat=False,
            )
        )[:, -1, :]
    )  # fmt: skip
    clean = log_softmax32(all_logits(placed.clean_forward(lm_batch(tokens)).output)[:, -1, :])
    native, clean = (
        jax.device_put(jax.sharding.reshard(x, P()), jax.devices()[0]) for x in (native, clean)
    )
    on = slots.mask(np.ones(slots.n, bool), arch.n_layer)
    ours = log_softmax32(factored_logits(arch, shared, factors, on, tokens, "fp32", "fp32"))
    dense32 = target_logp(arch, shared, dense, tokens, "fp32")

    def kl(p: jax.Array, q: jax.Array) -> np.ndarray:
        return np.asarray(jnp.sum(jnp.exp(p) * (p - q), axis=-1))

    out = {
        "kl_native_masked_vs_factored_mean": float(kl(native, ours).mean()),
        "kl_native_masked_vs_factored_max": float(kl(native, ours).max()),
        "kl_clean_vs_dense_fp32_mean": float(kl(clean, dense32).mean()),
        "kl_clean_vs_dense_fp32_max": float(kl(clean, dense32).max()),
        "kl_dense_vs_candidates_mean": float(kl(dense32, ours).mean()),
    }
    return out


def learn_context(
    config: IntegerOnlyConfig, layout: Layout, run_dir: Path, data_root: Path
) -> tuple[Context, PromptPool, dict[str, int]]:
    pool = load_pool(layout, config)
    check = np.load(layout.check_set)
    restored = restore(run_dir, config, data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    int_ids = jnp.asarray(integer_token_ids(load_tokenizer(target.model_name)))
    model = restored.placed.model
    assert isinstance(model, TransformerDecomposedModel)
    sizes = site_sizes(restored)
    keep = keep_of(read_mask_json(layout.candidates), sizes)
    with jax.set_mesh(restored.mesh):
        arch, shared, dense = build_target(model)
        factors, slots = build_factors(restored.components, keep, arch.n_layer)
    del restored, model, dense  # the learner reads the target only through its cached log q
    ctx = Context(
        arch=arch, shared=shared, factors=factors, slots=slots, tokens=pool.tokens, r=pool.r,
        log_q_ref=np.load(layout.reference), int_ids=int_ids,
        train=np.setdiff1d(pool.problem_rows(), check), check=check,
    )  # fmt: skip
    return ctx, pool, sizes


def stage_learn(
    config: IntegerOnlyConfig, layout: Layout, run_dir: Path, data_root: Path, lam: float
) -> dict[str, object]:
    assert lam in config.lambdas, (lam, config.lambdas)
    ctx, _, _ = learn_context(config, layout, run_dir, data_root)
    return learn_mask(config, ctx, lam, layout.lam_dir(lam))


def stage_pick(
    config: IntegerOnlyConfig, layout: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    unfinished = [lam for lam in config.lambdas if not is_finished(layout.lam_dir(lam), config)]
    assert not unfinished, f"learn not finished for λ = {unfinished}"
    pool = load_pool(layout, config)
    check = np.load(layout.check_set)
    restored = restore(run_dir, config, data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    int_ids = integer_token_ids(load_tokenizer(target.model_name))
    model = restored.placed.model
    assert isinstance(model, TransformerDecomposedModel)
    sizes = site_sizes(restored)
    keep = keep_of(read_mask_json(layout.candidates), sizes)
    with jax.set_mesh(restored.mesh):
        arch, shared, dense = build_target(model)
        factors, slots = build_factors(restored.components, keep, arch.n_layer)
    del restored
    masks = {CANDIDATES: np.ones(slots.n, bool)}
    for lam in config.lambdas:
        logit = np.load(layout.lam_dir(lam) / "state.npz")["logit"]
        assert logit.shape == (slots.n,), (logit.shape, slots.n)
        masks[lam_name(lam)] = logit > 0
    t0 = time.time()
    scored = score_masks(arch, shared, factors, dense, slots, masks, pool, check, int_ids, config)
    ops = pool.op[check]
    rows: dict[str, tuple[int | None, dict[str, object]]] = {
        TARGET: (None, summary(scored[TARGET], ops, pool.ops))
    }
    for name, on in masks.items():
        rows[name] = (int(on.sum()), summary(scored[name], ops, pool.ops))
    kl_int: dict[float, float] = {}
    for lam in config.lambdas:
        size, summ = rows[lam_name(lam)]
        final = {"lambda": lam, "size": size, "prompts": int(check.size)} | summ
        (layout.lam_dir(lam) / "final.json").write_text(json.dumps(final, indent=1) + "\n")
        kl_int[lam] = float(scored[lam_name(lam)][KL_INT].mean())
    chosen = choose(kl_int, config.reference_kl_int)
    write_summary(layout.summary, rows, None if chosen is None else lam_name(chosen))
    facts: dict[str, object] = {
        "kl_int": {lam_name(k): v for k, v in kl_int.items()},
        "size": {name: rows[name][0] for name in rows},
        "reference_kl_int": config.reference_kl_int,
        "seconds_scoring": round(time.time() - t0, 1),
    }
    if chosen is None:
        raise NoQualifyingLambda(
            f"no λ has mean KL_int < {config.reference_kl_int} on Q: {facts['kl_int']}"
        )
    doc = mask_json(
        slots, masks[lam_name(chosen)], sizes,
        rule=f"subset of candidates.json kept by a learned sampled mask (logit > 0 in state.npz), "
        f"objective KL_int to the original model, sparsity weight {chosen:g}: the highest λ of "
        f"{list(config.lambdas)} with mean KL_int on Q below {config.reference_kl_int} "
        f"({config.reference_source}); threshold is the candidates'",
        source=str(layout.lam_dir(chosen)), threshold=config.threshold,
    )  # fmt: skip
    layout.integer_only.write_text(json.dumps(doc) + "\n")
    return facts | {"chosen": chosen, "n_integer_only": doc["n_alive"]}


class NoQualifyingLambda(RuntimeError):
    pass


def stage_collect(
    config: IntegerOnlyConfig, layout: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    assert config.ci_batch % 8 == 0, f"ci_batch {config.ci_batch} must be a multiple of 8"
    pool = load_pool(layout, config)
    restored = restore(run_dir, config, data_root)
    model = restored.placed.model
    assert isinstance(model, TransformerDecomposedModel)
    sizes = site_sizes(restored)
    keep = keep_of(read_mask_json(layout.integer_only), sizes)
    with np.load(layout.max_ci) as z:
        max_ci = {site: z[site] for site in z.files}
    placed = restored.placed
    with jax.set_mesh(restored.mesh):
        arch, shared, _dense = build_target(model)
        factors, slots = build_factors(restored.components, keep, arch.n_layer)
        prepared, v_norms = prepare_components(placed, restored.components)
    by_site = slots.components()
    site_cols = {s: jnp.asarray(np.asarray(by_site.get(s, []), np.int32)) for s in sorted(sizes)}
    n_packed = -(-pool.n // 8)
    # Pass 1, the original CI: every integer-only component's ON bit at every position, packed
    # along prompts, `(ceil(N/8), T, n)` uint8 in host memory.
    bits = np.zeros((n_packed, pool.seq_len, slots.n), np.uint8)
    t0 = time.time()
    batches = index_batches(pool.n, config.ci_batch)
    with jax.set_mesh(restored.mesh):
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        for i, (idx, real) in enumerate(batches):
            ci = np.asarray(
                chunk_ci_columns(placed, prepared, restored.ci_fn, tokens_all, jnp.asarray(idx), site_cols)
            )  # fmt: skip
            on = ci > config.threshold
            on[real:] = False
            start = int(idx[0])
            n_rows = -(-real // 8)
            bits[start // 8 : start // 8 + n_rows] = pack_rows(on)[:n_rows]
            if i % 50 == 0 or i == len(batches) - 1:
                logger.info(f"collect CI: batch {i + 1}/{len(batches)} ({time.time() - t0:.0f}s)")
    seconds_ci = time.time() - t0
    columns = columns_from_bits(bits)
    per_t = [columns.at(t) for t in range(pool.seq_len)]
    stage1 = ever_on_columns(slots, max_ci, config.threshold, pool.seq_len)
    here = set(zip(columns.position.tolist(), columns.component.tolist(), strict=True))
    there = set(zip(stage1.position.tolist(), stage1.component.tolist(), strict=True))
    out = layout.datasets.with_name(layout.datasets.name + ".partial")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for t, cols in enumerate(per_t):
        np.save(out / f"ci_on_pos{t}.npy", np.ascontiguousarray(bits[:, t][:, cols]))
    n_on = int(np.unpackbits(bits).sum())
    del bits
    # Pass 2, the integer-only model: the inner activations of the stored columns.
    inner_mm = [
        np.lib.format.open_memmap(out / f"inner_pos{t}.npy", mode="w+", dtype=np.float16,
                                  shape=(pool.n, c.size))
        for t, c in enumerate(per_t)
    ]  # fmt: skip
    on_mask = slots.mask(np.ones(slots.n, bool), arch.n_layer)
    inner_columns = make_inner_columns(slots)
    inner_absmax = 0.0
    t1 = time.time()
    for i, (idx, real) in enumerate(batches):
        inner = np.asarray(
            inner_columns(arch, shared, factors, on_mask, jnp.asarray(pool.tokens[idx]),
                          config.collect_precision)
        )  # fmt: skip
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
    (out / "README.md").write_text(
        DATASET_README.format(
            run_dir=run_dir, step=config.step, commit=commit(), n_components=slots.n,
            n_prompts=pool.n, prompts=config.prompts, seq_len=pool.seq_len,
            labels=labels_text(pool), threshold=config.threshold, n_columns=int(pos.size),
            per_position=", ".join(f"t={t}: {c.size}" for t, c in enumerate(per_t)),
            ops=", ".join(f"{k} = {o}" for k, o in enumerate(pool.ops)),
            precision=config.collect_precision, last=pool.seq_len - 1,
        )
    )  # fmt: skip
    if layout.datasets.exists():
        shutil.rmtree(layout.datasets)
    out.rename(layout.datasets)
    never_on = np.setdiff1d(np.arange(slots.n), columns.component)
    return {
        "n_integer_only": slots.n,
        "n_columns": int(pos.size),
        "columns_per_position": [int(c.size) for c in per_t],
        "n_on_events": n_on,
        "integer_only_never_on": [f"{slots.sites[j]}:{slots.index[j]}" for j in never_on],
        "pairs_on_here_not_in_max_ci": len(here - there),
        "pairs_in_max_ci_not_on_here": len(there - here),
        "inner_absmax": inner_absmax,
        "seconds_ci_pass": round(seconds_ci, 1),
        "seconds_inner_pass": round(time.time() - t1, 1),
    }


def labels_text(pool: PromptPool) -> str:
    return ", ".join(f"{t} = `{label}`" for t, label in enumerate(pool.position_labels))


def stage_applet(
    config: IntegerOnlyConfig, layout: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    del run_dir, data_root
    d = layout.datasets
    pool = load_pool(layout, config)
    index = dict(np.load(d / "index.npz"))
    n, ops, labels = pool.n, pool.ops, pool.position_labels
    assert np.array_equal(index["op"], pool.op), "the dataset is not this pool's"
    sizes = {site: int(v.shape[1]) for site, v in np.load(layout.max_ci).items()}
    out = layout.applet_data.with_name(layout.applet_data.name + ".partial")
    if out.exists():
        shutil.rmtree(out)
    t0 = time.time()
    counts: dict[str, int] = {}
    for t in range(len(labels)):
        cols = np.flatnonzero(index["col_position"] == t)
        inner_all = np.load(d / f"inner_pos{t}.npy").astype(np.float32)
        inner_all /= np.maximum(index["col_v_norm"][cols], 1e-12)[None, :]
        packed = np.load(d / f"ci_on_pos{t}.npy")
        on_all = np.unpackbits(packed, axis=0, count=n).astype(bool)
        sites = index["col_site"][cols]
        for k, op in enumerate(ops):
            rows = np.flatnonzero(pool.op == k)
            grid, cell = ap.operation_grid(pool.a[rows], pool.b[rows], op)
            on_op, inner_op = on_all[rows], inner_all[rows]
            site_columns = {}
            for site in np.unique(sites):
                sel = np.flatnonzero(sites == site)
                ids = index["col_component"][cols[sel]]
                site_columns[str(site)] = (ids, on_op[:, sel], inner_op[:, sel])
            counts[f"{op}_pos{t}"] = ap.write_operation_slice(
                out, op, t, len(labels), config.step, grid, cell, site_columns, sizes
            )
        logger.info(f"applet: position {t} ({time.time() - t0:.0f}s)")
    ap.write_manifest(
        out, f"integer-only components, step {config.step}", ops, labels, config.threshold
    )
    if layout.applet_data.exists():
        shutil.rmtree(layout.applet_data)
    out.rename(layout.applet_data)
    layout.applet_page.write_text(ap.applet_page(layout.applet_data.name))
    return {"grids_per_slice": counts, "seconds": round(time.time() - t0, 1)}


# ----------------------------------- driver -----------------------------------


def write_readme(layout: Layout, config: IntegerOnlyConfig, run_dir: Path) -> None:
    first = layout.facts("candidates") or {}
    pick = layout.facts("pick")
    chosen = (
        f"Chosen: λ = {pick['chosen']:g}, {pick['n_integer_only']} components."
        if pick is not None and isinstance(pick.get("chosen"), float)
        else "Not chosen yet."
    )
    labels = first.get("position_labels")
    stage_lines = []
    for stage in STAGES:
        markers = (
            [f"learn_{lam_name(lam)}" for lam in config.lambdas] if stage == "learn" else [stage]
        )
        for m in markers:
            state = (
                "done"
                if layout.done(m).exists()
                else "FAILED"
                if layout.failed(m).exists()
                else "not run"
            )
            stage_lines.append(f"- `{m}`: {state}")
    layout.readme.write_text(
        README.format(
            run_dir=run_dir, step=config.step, commit=commit(),
            seq_len=first.get("seq_len", "?"),
            labels=", ".join(f"{t} = `{x}`" for t, x in enumerate(labels))
            if isinstance(labels, list) else "?",
            n_prompts=first.get("n_prompts", "?"), prompts=config.prompts,
            threshold=config.threshold, n_candidates=first.get("n_candidates", "?"),
            check_size=config.check_size, eval_size=config.eval_size,
            eval_every=config.eval_every, reference_kl_int=config.reference_kl_int,
            reference_source=config.reference_source, chosen=chosen,
            stages="\n".join(stage_lines),
        )
    )  # fmt: skip


def peak_device_gib() -> list[float | None]:
    """Per local device, the allocator's peak bytes in use so far (GiB; None where the backend
    keeps no statistics)."""
    out: list[float | None] = []
    for device in jax.local_devices():
        stats = device.memory_stats()
        peak = None if stats is None else stats.get("peak_bytes_in_use")
        out.append(None if peak is None else round(peak / 2**30, 2))
    return out


def run_stage(layout: Layout, marker: str, fn: Callable[[], dict[str, object]]) -> None:
    if layout.done(marker).exists():
        logger.info(f"{marker}: already done ({layout.done(marker)})")
        return
    layout.status.mkdir(parents=True, exist_ok=True)
    layout.failed(marker).unlink(missing_ok=True)
    t0 = time.time()
    try:
        facts = fn()
    except BaseException:
        layout.failed(marker).write_text(traceback.format_exc())
        raise
    facts = facts | {
        "peak_device_gib": peak_device_gib(),
        "seconds_total": round(time.time() - t0, 1),
        "commit": commit(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    layout.done(marker).write_text(json.dumps(facts, indent=1) + "\n")
    logger.info(f"{marker}: done {json.dumps(facts)[:2000]}")


def main() -> None:
    setup_console_logger()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="IntegerOnlyConfig YAML")
    parser.add_argument("--lam", type=float, default=None, help="learn: the sparsity weight λ")
    parser.add_argument("--compilation_cache_dir", type=Path, default=None)
    args = parser.parse_args()
    if args.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(args.compilation_cache_dir)
    config = IntegerOnlyConfig.from_file(args.config)
    layout = Layout.of_run(args.run_dir)
    pin_config(layout, config)
    common = (config, layout, args.run_dir, args.data_root)
    match args.stage:
        case "candidates":
            run_stage(layout, "candidates", lambda: stage_candidates(*common))
        case "problem":
            run_stage(layout, "problem", lambda: stage_problem(*common))
        case "learn":
            assert args.lam is not None, "learn needs --lam"
            run_stage(layout, f"learn_{lam_name(args.lam)}", lambda: stage_learn(*common, args.lam))
        case "pick":
            run_stage(layout, "pick", lambda: stage_pick(*common))
        case "collect":
            run_stage(layout, "collect", lambda: stage_collect(*common))
        case "applet":
            run_stage(layout, "applet", lambda: stage_applet(*common))
        case other:
            raise AssertionError(other)
    write_readme(layout, config, args.run_dir)


if __name__ == "__main__":
    main()
