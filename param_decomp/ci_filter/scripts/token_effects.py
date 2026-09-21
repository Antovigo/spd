"""What the components dropped by the INTEGER objective do: subtract each from the model and see
which non-integer tokens move.

    python -m param_decomp.ci_filter.scripts.token_effects --config <yaml> --data_root <root>
        --alive_in addsub-05-filter-last-pos-ceiling --dead_in addsub-05-filter-integers
        [--prompts 2048] [--top_tokens 5]

Selects every component alive (max output CI over the pool > `alive_threshold`) at the end of
`--alive_in` and dead at the end of `--dead_in` — for the default pair, the components the
full-vocabulary objective needs and the integer-only objective does not. Each is subtracted from
the MODEL (every other component on, weight delta on, `nontarget.subtracting`) on the arithmetic
pool, and scored at the last position against the subtract-nothing forward (same kernel path,
so bf16 noise cancels):

- full-vocabulary KL and integer-renormalized KL (the second should be ~0 if the component only
  moves non-integer mass);
- the non-integer tokens whose mean probability changes most, with before/after probabilities.

Group rows come first (`group: ALL`, a same-size `CONTROL` drawn from the components both
filters keep, then layer bands): a single component is often below the bf16 floor of this
subtraction, so the joint ablation — against its control — is the robust readout.

Writes `<run_dir>/analysis/ablations/step_<step>/non_integer_components.tsv`."""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.ci_filter.config import CIFilterConfig, resolve_run_dir
from param_decomp.ci_filter.nontarget import subtracting
from param_decomp.ci_filter.objective import kl_rows, restrict
from param_decomp.ci_filter.paths import CIFilterOutputs, ablations_dir
from param_decomp.ci_filter.pool import answer_token_ids, build_pool, load_tokenizer
from param_decomp.ci_filter.step import Prepared, gather_rows, prepare_components
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import PlacedModel
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache

type LastLogits = Float[Array, "B V"]


def make_last_logits() -> Callable[..., LastLogits]:
    @eqx.filter_jit
    def last_logits(
        placed: PlacedModel,
        prepared: Prepared,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        keep: dict[str, Array],
    ) -> LastLogits:
        tokens = gather_rows(tokens_all, idx)
        logits = subtracting(placed, prepared, tokens, keep)[:, -1, :].astype(jnp.float32)
        return jax.sharding.reshard(logits, P())

    return last_logits


def make_effects() -> Callable[..., dict[str, Array]]:
    """Per batch: summed full and integer KL of the ablated against the reference logits, and
    the summed probability of every token under the ablated forward."""

    @eqx.filter_jit
    def effects(
        reference: LastLogits, ablated: LastLogits, answer_ids: Int[Array, " K"]
    ) -> dict[str, Array]:
        return {
            "kl": jnp.sum(kl_rows(ablated, reference)),
            "integer_kl": jnp.sum(
                kl_rows(restrict(ablated, answer_ids), restrict(reference, answer_ids))
            ),
            "probs": jnp.sum(jax.nn.softmax(ablated, axis=-1), axis=0),
        }

    return effects


def _alive_in_both(
    run_dir: Path, step: int, first: str, second: str, threshold: float
) -> list[tuple[str, int]]:
    """Components alive at the end of BOTH filters: the control population."""

    def max_ci(filter_id: str) -> dict[str, np.ndarray]:
        with np.load(CIFilterOutputs.for_run(run_dir, step, filter_id).max_ci) as saved:
            return {site: saved[site] for site in saved.files}

    a, b = max_ci(first), max_ci(second)
    return [
        (site, int(c))
        for site in sorted(a)
        for c in np.nonzero((a[site] > threshold) & (b[site] > threshold))[0]
    ]


def _chosen(
    run_dir: Path, step: int, alive_in: str, dead_in: str, threshold: float
) -> list[tuple[str, int, float]]:
    """`(site, component, max CI in alive_in)` of the components alive in one filter, dead in the other."""

    def max_ci(filter_id: str) -> dict[str, np.ndarray]:
        with np.load(CIFilterOutputs.for_run(run_dir, step, filter_id).max_ci) as saved:
            return {site: saved[site] for site in saved.files}

    alive, dead = max_ci(alive_in), max_ci(dead_in)
    return [
        (site, int(c), float(alive[site][c]))
        for site in sorted(alive)
        for c in np.nonzero((alive[site] > threshold) & (dead[site] <= threshold))[0]
    ]


def token_effects(
    config: CIFilterConfig,
    data_root: Path,
    alive_in: str,
    dead_in: str,
    prompts: int,
    batch_size: int,
    top_tokens: int,
    groups_only: bool,
) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    out = ablations_dir(run_dir, step) / "non_integer_components.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(target.model_name)
    pool = build_pool(config.pool, tokenizer)
    answers = answer_token_ids(tokenizer, include_minus=True)
    chosen = _chosen(run_dir, step, alive_in, dead_in, config.alive_threshold)
    logger.info(
        f"{len(chosen)} components alive in {alive_in} and dead in {dead_in}; "
        f"{min(prompts, pool.n_prompts)} prompts -> {out}"
    )
    is_integer = np.zeros(len(tokenizer), bool)
    is_integer[answers] = True

    with jax.set_mesh(mesh):
        prepared, _ = prepare_components(placed, restored.components)
        del restored
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        answer_ids = jax.sharding.reshard(jnp.asarray(answers), P())
        ones = {
            site.name: jax.sharding.reshard(jnp.ones(site.C, jnp.float32), P())
            for site in placed.sites
        }
        rng = np.random.default_rng(np.random.SeedSequence((config.seed, 19)))
        subset = np.sort(
            rng.choice(pool.n_prompts, size=min(prompts, pool.n_prompts), replace=False)
        ).astype(np.int32)
        batches = [subset[s : s + batch_size] for s in range(0, subset.size, batch_size)]
        batches = [b for b in batches if b.size == batch_size]
        n_scored = len(batches) * batch_size

        last_logits = make_last_logits()
        effects = make_effects()
        references = [
            last_logits(placed, prepared, tokens_all, jnp.asarray(b), ones) for b in batches
        ]
        base_probs = np.zeros(len(tokenizer), np.float64)
        for reference in references:
            base_probs += np.asarray(jnp.sum(jax.nn.softmax(reference, axis=-1), axis=0))
        base_probs /= n_scored

        header = [
            "site",
            "layer",
            "kind",
            "component",
            "max_ci_alive_in",
            "kl_full",
            "kl_integer",
            "integer_share_of_kl",
            "top_token",
            "p_before",
            "p_after",
            "top_tokens",
        ]
        rows = ["\t".join(header)]
        # Group ablations first: one component is often below the bf16 floor of this subtraction
        # (it reshuffles rounding in the residual stream), so the WHOLE set and layer bands are
        # the robust readout; the per-component rows follow, marked by their own address.
        bands = [(0, 7), (8, 15), (16, 23), (24, 31)]
        both = _alive_in_both(run_dir, step, alive_in, dead_in, config.alive_threshold)
        pick = np.random.default_rng(np.random.SeedSequence((config.seed, 23))).choice(
            len(both), size=min(len(chosen), len(both)), replace=False
        )
        groups: list[tuple[str, list[tuple[str, int]]]] = [
            ("ALL", [(site, c) for site, c, _ in chosen]),
            # Same size, drawn from the components BOTH objectives keep: what an arbitrary
            # joint ablation of this many components does.
            ("CONTROL (alive in both)", [both[int(i)] for i in pick]),
            *[
                (
                    f"layers {lo}-{hi}",
                    [(site, c) for site, c, _ in chosen if lo <= int(site.split(".")[1]) <= hi],
                )
                for lo, hi in bands
            ],
        ]
        units: list[tuple[str, str, str, str, float, list[tuple[str, int]]]] = [
            (f"group: {name}", "", "group", str(len(members)), float("nan"), members)
            for name, members in groups
            if members
        ]
        units += [
            (
                site,
                site.split(".", 2)[1],
                site.split(".", 2)[2],
                str(component),
                max_ci,
                [(site, component)],
            )
            for site, component, max_ci in (chosen if not groups_only else [])
        ]
        t0 = time.time()
        for index, (site, layer, kind, component_label, max_ci, members) in enumerate(units):
            keep = dict(ones)
            for member_site, member in members:
                keep[member_site] = keep[member_site].at[member].set(0.0)
            kl = integer_kl = 0.0
            probs = np.zeros_like(base_probs)
            for block, reference in zip(batches, references, strict=True):
                ablated = last_logits(placed, prepared, tokens_all, jnp.asarray(block), keep)
                got = effects(reference, ablated, answer_ids)
                kl += float(got["kl"])
                integer_kl += float(got["integer_kl"])
                probs += np.asarray(got["probs"])
            kl, integer_kl, probs = kl / n_scored, integer_kl / n_scored, probs / n_scored
            change = np.where(is_integer, 0.0, np.abs(probs - base_probs))
            ranked = np.argsort(change)[::-1][:top_tokens]
            top = int(ranked[0])
            rows.append(
                "\t".join(
                    [
                        site,
                        layer,
                        kind,
                        component_label,
                        "" if max_ci != max_ci else f"{max_ci:.4g}",
                        f"{kl:.4g}",
                        f"{integer_kl:.4g}",
                        f"{integer_kl / kl:.3g}" if kl > 0 else "",
                        repr(tokenizer.decode([top])),
                        f"{base_probs[top]:.4g}",
                        f"{probs[top]:.4g}",
                        "; ".join(
                            f"{tokenizer.decode([int(t)])!r} {base_probs[t]:.3g}->{probs[t]:.3g}"
                            for t in ranked
                        ),
                    ]
                )
            )
            if index % 50 == 0 or kind == "group":
                logger.info(
                    f"{index}/{len(units)} ({time.time() - t0:.0f}s): {site} {component_label} "
                    f"KL {kl:.2e}, integer KL {integer_kl:.2e}, top {tokenizer.decode([top])!r}"
                )
    out.write_text("\n".join(rows) + "\n")
    logger.info(f"-> {out}")
    return out


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="any filter config (run, pool)")
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--alive_in", default="addsub-05-filter-last-pos-ceiling")
    ap.add_argument("--dead_in", default="addsub-05-filter-integers")
    ap.add_argument("--prompts", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--top_tokens", type=int, default=5)
    ap.add_argument(
        "--groups_only", action="store_true", help="only the whole set and the layer bands"
    )
    args = ap.parse_args()
    token_effects(
        CIFilterConfig.from_file(args.config),
        args.data_root,
        args.alive_in,
        args.dead_in,
        args.prompts,
        args.batch_size,
        args.top_tokens,
        args.groups_only,
    )


if __name__ == "__main__":
    main()
