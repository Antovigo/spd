"""Harvest the neuron-ranking artifact `initialization: neuron_aligned_targeted` starts
from (SPEC T13) — the write side of `neuron_ranks`.

Reads a targeted run YAML for its TARGET (`target`, `decomposition.sites`) and its PROMPT
POOL (`prompts`, tokenized through the run's own tokenizer) only; every other section is
ignored, so one artifact serves every decomposition config over that (model, pool) pair —
any C, any seed, any subset of the harvested layers. Sweeps the whole pool exactly once
through the frozen model, capturing each harvested block's MLP hidden, q/k projection
outputs and attention-core output, and ranks every coordinate of every ranking family
(`targets.neuron_alignment.UNIT_KINDS`) by its energy score.

Single process by design (the sweep is seconds); devices within the process form a pure
data-parallel mesh, and the reductions land fully replicated. Blocks are captured in
groups under a byte budget — the forward retains one fp32 `[B, T, width]` slot per
requested block.

Output: `<out_dir>/neuron_ranks.npz` (`rank_<kind>_<b>` int32, `score_<kind>_<b>` float32
per block and family) + `meta.json` (provenance: target, pool fingerprint, statistic,
layers, unit counts). Publish the dir into the store as `<data_root>/neuron_ranks/<name>`
and reference it as `decomposition.sites.neuron_ranks: {kind: name, name: <name>}`, or
point `{kind: dir, dir: <out_dir>}` at it.

Run: `python -m param_decomp.experiments.lm.harvest_neuron_ranks --config <run.yaml>
--data_root <root> --out_dir <abs> --local_device_count N [--layers all|"18,19"]
[--batch_size 128]`
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from param_decomp.core.sharding import hsdp_mesh, initialize_topology
from param_decomp.experiments.lm.config import (
    LMTargetedExperimentConfig,
    resolve_decomposition,
)
from param_decomp.experiments.lm.eval_operations import global_token_batch
from param_decomp.experiments.lm.load_run import build_target
from param_decomp.experiments.lm.neuron_ranks import target_identity
from param_decomp.experiments.lm.targeted_data import build_prompt_pool
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.experiments.lm.training_targeted import pool_tokenizer
from param_decomp.infra.dataset_store import read_dataset_meta, resolve_dataset_ref
from param_decomp.targets.glu_transformer import GLUDecomposedModel
from param_decomp.targets.neuron_alignment import (
    STATISTIC,
    UNIT_KINDS,
    NeuronRanksMeta,
    UnitKind,
    accumulate_neuron_moments,
    block_scores,
    capture_groups,
    make_moments_step,
    pool_slices,
    pool_tokens_sha256,
    rank_units,
    tap_widths,
    unit_counts,
    write_neuron_ranks,
)


def _layers(spec: str | int | list[int] | tuple[int, ...], n_layer: int) -> tuple[int, ...]:
    """`all`, one int, or a comma list / sequence of ints — sorted, distinct, in range."""
    match spec:
        case "all":
            layers = tuple(range(n_layer))
        case int(single):
            layers = (single,)
        case str(text):
            layers = tuple(sorted({int(part) for part in text.split(",") if part.strip()}))
        case _:
            layers = tuple(sorted({int(layer) for layer in spec}))
    assert layers and all(0 <= layer < n_layer for layer in layers), (layers, n_layer)
    return layers


def harvest(
    config: Path,
    data_root: Path,
    out_dir: Path,
    local_device_count: int,
    layers: str | int | list[int] = "all",
    batch_size: int = 128,
    capture_budget_bytes: int = 1 << 30,
) -> None:
    config, data_root, out_dir = Path(config), Path(data_root), Path(out_dir)
    cfg = LMTargetedExperimentConfig.model_validate(yaml.safe_load(config.read_text()))
    runtime = cfg.runtime
    target = resolve_decomposition(cfg.target, cfg.decomposition, data_root).target

    initialize_topology(local_device_count, local_device_count)
    assert jax.process_count() == 1, "the harvest is a single-process tool"
    assert batch_size % local_device_count == 0, (batch_size, local_device_count)
    mesh = hsdp_mesh(local_device_count, 1, 1)
    jax.set_mesh(mesh)
    enable_persistent_compilation_cache(runtime.compilation_cache_dir)

    train_dir = resolve_dataset_ref(cfg.data.train, data_root)
    tokenizer = pool_tokenizer(target, read_dataset_meta(train_dir).tokenizer_name)
    pool = build_prompt_pool(cfg.prompts, tokenizer)
    n_prompts, prompt_len = pool.tokens.shape

    model = build_target(target, mesh, data_root, runtime.sharding)
    glu = model.model
    assert isinstance(glu, GLUDecomposedModel), type(glu)
    blocks = _layers(layers, glu.n_layer)
    counts = unit_counts(glu)
    widths = tap_widths(counts)
    groups = capture_groups(
        blocks, batch_size, prompt_len, sum(widths.values()), capture_budget_bytes
    )
    print(
        f"harvest: target={target_identity(target)} pool={n_prompts}x{prompt_len} "
        f"layers={blocks} units={counts} batch={batch_size} groups={[len(g) for g in groups]}",
        flush=True,
    )

    rank: dict[UnitKind, dict[int, np.ndarray]] = {kind: {} for kind in UNIT_KINDS}
    score: dict[UnitKind, dict[int, np.ndarray]] = {kind: {} for kind in UNIT_KINDS}
    for group in groups:
        t0 = time.time()
        step = make_moments_step(model, group)
        placed = (
            (global_token_batch(rows, mesh, batch_size), jnp.asarray(mask))
            for rows, mask in pool_slices(pool.tokens, batch_size)
        )
        totals = accumulate_neuron_moments(step, group, widths, placed)
        for block in group:
            scores = block_scores(glu, block, totals[block])
            summary: list[str] = []
            for kind in UNIT_KINDS:
                block_rank = rank_units(scores[kind])
                rank[kind][block] = block_rank
                score[kind][block] = scores[kind][block_rank].astype(np.float32)
                cum = np.cumsum(score[kind][block].astype(np.float64)) / score[kind][block].sum()
                n = counts[kind]
                at = {k: cum[min(k, n) - 1] for k in (64, 256, 512)}
                summary.append(f"{kind}: " + " ".join(f"cov@{k}={v:.3f}" for k, v in at.items()))
            print(
                f"  block {block}: tokens={totals[block].n_tokens:.0f} " + " | ".join(summary),
                flush=True,
            )
        print(f"  group {group[0]}..{group[-1]}: {time.time() - t0:.1f}s", flush=True)

    meta = NeuronRanksMeta(
        target=target_identity(target),
        tokens_sha256=pool_tokens_sha256(pool.tokens),
        n_prompts=int(n_prompts),
        prompt_len=int(prompt_len),
        statistic=STATISTIC,
        layers=list(blocks),
        n_units=counts,
    )
    write_neuron_ranks(out_dir, meta, rank, score)
    print(f"wrote {out_dir} ({len(blocks)} blocks)", flush=True)
