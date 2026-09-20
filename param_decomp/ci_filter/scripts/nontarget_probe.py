"""Probe what candidate arithmetic-interference components do on GENERAL text.

    python -m param_decomp.ci_filter.scripts.nontarget_probe --config <yaml> --data_root <root>
        --table <components.tsv> [--min_layer 13] [--max_layer 17] [--min_impairs 0.8]
        [--dataset fineweb_llama_tok_64_eval] [--rows 8192] [--batch_size 64] [--top_k 40]

Selects components from the classification table (`scripts/component_table.py`), then subtracts
each one from the FROZEN MODEL over pre-tokenized non-target text (weight delta on, every other
component on), recording per position the KL against the model and both argmax tokens
(`param_decomp/ci_filter/nontarget.py`).

Writes `<run_dir>/analysis/ci_filter/step_<step>/nontarget_probe/`: `summary.tsv` (one row per
component: how broadly it acts), `examples.jsonl` (its `top_k` highest-KL positions with the
decoded context and the token swap) and `heatmap.npz` + `heatmap_tokens.json` (per-token KL of
the first `heatmap_rows` texts, for the token-coloured figures)."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.config import CIFilterConfig, resolve_run_dir
from param_decomp.ci_filter.nontarget import make_probe_baseline, make_probe_component
from param_decomp.ci_filter.pool import Tokenizer, load_tokenizer
from param_decomp.ci_filter.step import prepare_components
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.infra.dataset_store import dataset_dir

SUMMARY_COLUMNS = (
    "site",
    "layer",
    "kind",
    "component",
    "impairs_frac_run",
    "swept_danswer_ce_run",
    "mean_kl",
    "median_kl",
    "p99_kl",
    "max_kl",
    "frac_kl_over_1e-3",
    "frac_kl_over_1e-2",
    "top1_flip_frac",
)


def _select(
    table: Path, min_layer: int, max_layer: int, min_impairs: float
) -> list[dict[str, str]]:
    lines = table.read_text().splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]
    chosen = [
        row
        for row in rows
        if min_layer <= int(row["layer"]) <= max_layer
        and row["impairs_frac_run"]
        and float(row["impairs_frac_run"]) >= min_impairs
    ]
    return sorted(chosen, key=lambda row: float(row["swept_danswer_ce_run"] or 0.0))


def _text_rows(data_root: Path, dataset: str, rows: int) -> np.ndarray:
    """`(rows, seq_len)` int32 of the pre-tokenized store dataset, first shard first."""
    directory = dataset_dir(data_root, dataset)
    shards = sorted(directory.glob("shard_*.parquet"))
    assert shards, f"no shards in {directory}"
    collected: list[np.ndarray] = []
    total = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["input_ids"])
        block = np.stack([np.asarray(x, dtype=np.int32) for x in table["input_ids"].to_pylist()])
        collected.append(block)
        total += block.shape[0]
        if total >= rows:
            break
    return np.concatenate(collected)[:rows]


def nontarget_probe(
    config: CIFilterConfig,
    data_root: Path,
    table: Path,
    dataset: str,
    rows: int,
    batch_size: int,
    top_k: int,
    heatmap_rows: int,
    min_layer: int,
    max_layer: int,
    min_impairs: float,
) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    out_dir = run_dir / "analysis" / "ci_filter" / f"step_{step}" / "nontarget_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = _select(table, min_layer, max_layer, min_impairs)
    assert chosen, (
        f"no components in layers {min_layer}-{max_layer} with impairs_frac >= {min_impairs}"
    )
    tokenizer = load_tokenizer(target.model_name)
    text = _text_rows(data_root, dataset, rows)
    logger.info(
        f"probing {len(chosen)} components on {text.shape[0]}x{text.shape[1]} tokens of "
        f"{dataset} -> {out_dir}"
    )

    site_c = {site.name: site.C for site in placed.sites}
    summary: list[dict[str, str | float]] = []
    examples = (out_dir / "examples.jsonl").open("w")
    with jax.set_mesh(mesh):
        prepared, _ = prepare_components(placed, restored.components)
        del restored
        baseline_of = make_probe_baseline()
        probe = make_probe_component()
        ones = {
            name: jax.sharding.reshard(jnp.ones(c, jnp.float32), P()) for name, c in site_c.items()
        }

        batches = [text[start : start + batch_size] for start in range(0, len(text), batch_size)]
        batches = [b for b in batches if b.shape[0] == batch_size]
        keys = [(row["site"], int(row["component"])) for row in chosen]
        heatmap: dict[tuple[str, int], np.ndarray] = {}
        kls: dict[tuple[str, int], list[np.ndarray]] = {k: [] for k in keys}
        flips: dict[tuple[str, int], list[np.ndarray]] = {k: [] for k in keys}
        best: dict[tuple[str, int], list[tuple[float, int, int, int, int]]] = {k: [] for k in keys}
        base_kls: list[float] = []

        # Batch OUTSIDE, components inside: one batch's reference logits (several GB) live at a
        # time, and the baseline forward still runs once per batch rather than once per pair.
        t0 = time.time()
        for batch_index, block in enumerate(batches):
            tokens = jnp.asarray(block)
            baseline = baseline_of(placed, prepared, tokens, ones)
            base_top1 = np.asarray(baseline.clean_top1)
            base_kls.append(float(np.mean(np.asarray(baseline.subtract_nothing_kl))))
            for site, component in keys:
                keep = dict(ones)
                keep[site] = ones[site].at[component].set(0.0)
                got = probe(placed, prepared, tokens, baseline, keep)
                kl = np.asarray(got["kl"])
                top1 = np.asarray(got["top1"])
                if batch_index == 0 and heatmap_rows:
                    heatmap[site, component] = kl[:heatmap_rows].copy()
                kls[site, component].append(kl.ravel())
                flips[site, component].append((top1 != base_top1).ravel())
                flat = kl.ravel()
                for index in np.argsort(flat)[-top_k:]:
                    r, position = divmod(int(index), kl.shape[1])
                    best[site, component].append(
                        (
                            float(flat[index]),
                            batch_index * batch_size + r,  # the row in `text`
                            position,
                            int(base_top1[r, position]),
                            int(top1[r, position]),
                        )
                    )
            del baseline
            logger.info(
                f"batch {batch_index + 1}/{len(batches)} x {len(keys)} components "
                f"({time.time() - t0:.0f}s elapsed)"
            )
        base_kl = float(np.mean(base_kls))
        logger.info(f"subtract-nothing KL (should be ~0, bf16 noise): {base_kl:.2e}")
        assert base_kl < 1e-2, (
            f"subtracting no component should reproduce the model, got KL {base_kl:.3f}"
        )

        for row in chosen:
            site, component = row["site"], int(row["component"])
            flat_kl = np.concatenate(kls[site, component])
            flat_flip = np.concatenate(flips[site, component])
            summary.append(
                {
                    "site": site,
                    "layer": row["layer"],
                    "kind": row["kind"],
                    "component": component,
                    "impairs_frac_run": row["impairs_frac_run"],
                    "swept_danswer_ce_run": row["swept_danswer_ce_run"],
                    "mean_kl": float(flat_kl.mean()),
                    "median_kl": float(np.median(flat_kl)),
                    "p99_kl": float(np.quantile(flat_kl, 0.99)),
                    "max_kl": float(flat_kl.max()),
                    "frac_kl_over_1e-3": float((flat_kl > 1e-3).mean()),
                    "frac_kl_over_1e-2": float((flat_kl > 1e-2).mean()),
                    "top1_flip_frac": float(flat_flip.mean()),
                }
            )
            seen: set[tuple[int, int]] = set()
            for kl_value, r, position, base_id, ablated_id in sorted(
                best[site, component], reverse=True
            ):
                if (r, position) in seen or len(seen) >= top_k:
                    continue
                seen.add((r, position))
                examples.write(
                    json.dumps(
                        {
                            "site": site,
                            "component": component,
                            "kl": kl_value,
                            "position": position,
                            "context": _decode(tokenizer, text[r, : position + 1]),
                            "model_top1": tokenizer.decode([base_id]),
                            "ablated_top1": tokenizer.decode([ablated_id]),
                        }
                    )
                    + "\n"
                )
            examples.flush()
            logger.info(
                f"{site} c{component}: mean KL {summary[-1]['mean_kl']:.2e}, "
                f"p99 {summary[-1]['p99_kl']:.2e}, top-1 flips "
                f"{summary[-1]['top1_flip_frac']:.3%}"
            )
    examples.close()
    if heatmap_rows:
        arrays = {
            f"{site}|{component}": np.stack([entry[2] for entry in ranked])
            for (site, component), ranked in heatmap.items()
            if ranked
        }
        np.savez(out_dir / "heatmap.npz", **arrays)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
        (out_dir / "heatmap_tokens.json").write_text(
            json.dumps(
                {
                    f"{site}|{component}": [
                        {
                            "row": entry[1],
                            "max_kl": entry[0],
                            "tokens": [tokenizer.decode([int(t)]) for t in text[entry[1]]],
                        }
                        for entry in ranked
                    ]
                    for (site, component), ranked in heatmap.items()
                    if ranked
                }
            )
        )
    out = out_dir / "summary.tsv"
    out.write_text(
        "\n".join(
            ["\t".join(SUMMARY_COLUMNS)]
            + ["\t".join(str(row[c]) for c in SUMMARY_COLUMNS) for row in summary]
        )
        + "\n"
    )
    (out_dir / "baseline.json").write_text(
        json.dumps({"dataset": dataset, "rows": int(text.shape[0]), "subtract_nothing_kl": base_kl})
    )
    logger.info(f"-> {out}")
    return out


def _decode(tokenizer: Tokenizer, ids: np.ndarray) -> str:
    return tokenizer.decode([int(t) for t in ids])


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True, help="components.tsv")
    ap.add_argument("--dataset", default="fineweb_llama_tok_64_eval")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument(
        "--heatmap_rows",
        type=int,
        default=8,
        help="per component, the top rows by max KL whose per-token KL is saved",
    )
    ap.add_argument("--min_layer", type=int, default=13)
    ap.add_argument("--max_layer", type=int, default=17)
    ap.add_argument("--min_impairs", type=float, default=0.8)
    args = ap.parse_args()
    nontarget_probe(
        CIFilterConfig.from_file(args.config),
        args.data_root,
        args.table,
        args.dataset,
        args.rows,
        args.batch_size,
        args.top_k,
        args.heatmap_rows,
        args.min_layer,
        args.max_layer,
        args.min_impairs,
    )


if __name__ == "__main__":
    main()
