"""In-loop eval pass: scalar parity with the torch eval metrics.

Implements independent pure JAX kernels for the scalar core of the torch reference
`eval:` block: `CEandKLLosses` (six masking variants), `CI_L0`, and fresh PGD
reconstruction. Each authored operation compiles only its own kernel; `make_eval_step`
composes them only for the fixed arithmetic probe and parity tests.
Plot-type metrics (CI histograms, activation density, per-component means, the
permutation/UV figures) ride the in-loop SLOW tier instead — natively in JAX
(`slow_eval.py`, SPEC S28; in-loop only, no offline CLI).

Variant semantics mirror `param_decomp/eval_metrics/ce_and_kl_losses.py`: each
variant is a masked forward with ALL sites live and no routing; only `stoch_masked`
carries a weight-delta mask (torch `make_mask_infos` without weight deltas drops the
delta term — delta mask 0 here). CE is next-token cross-entropy with the first label
ignored; KL is per-position vs the clean (frozen) logits.

Cross-batch aggregation (the multi-`n_steps` eval pass in `run.py`): every key this
function returns is a per-BATCH scalar that the caller averages uniformly over the
eval batches. This is mean-safe against the torch reference — i.e. it matches torch's
accumulate-then-`compute()` to within float reassociation — only because every emitted
key is itself a per-batch reduction that torch *also* averages across batches, and the
eval batches are uniform `(B, T)`. The S8/D2 Jensen trap (a nonlinearity applied AFTER
the cross-batch reduction, so mean-of-batch-results ≠ result-of-global-batch) does NOT
arise here, because no emitted key wraps the cross-batch axis in a nonlinearity:

- `ce_kl/kl_<variant>`: torch `CEandKLLosses` accumulates `kl * n_positions` and divides
  by total positions (token-weighted mean of a per-batch mean). Uniform `(B, T)` makes
  token-weighting equal to the uniform `1/n_steps` average here.
- `ce_kl/ce_difference_<variant>` = `ce_v - ce_target`: torch averages this per-batch
  DIFFERENCE (computed inside `_calc_ce_and_kl_losses`), not a difference of grand means.
  Linear, so uniform-average parity holds.
- `l0/<threshold>_<site|group>`: torch `CI_L0` collects per-batch L0 and averages them
  uniformly (`sum / count`); group L0 is a per-batch sum of member L0s. Linear.
- `loss/PGDReconLoss`: torch `PGDReconLoss` accumulates `kl * n` over batches and divides
  by total `n` (example-weighted mean of a per-batch mean KL); equals the uniform average
  under uniform `(B, T)`.

At `eval.n_steps: 1` the cross-batch average is a no-op; the parity argument above is
what keeps it correct when `n_steps` is raised.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.core.ci_fn import CIFn
from param_decomp.core.ci_l0_eval import ci_l0_scalars, resolve_l0_groups
from param_decomp.core.components import ComponentStacks
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.model import DecomposedModel
from param_decomp.core.recon_eval import FreshPGDReconEval, fresh_pgd_recon_loss
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.core.train import COMPUTE_DT, cast_floating
from param_decomp.targets.losses import kl_per_position

type ScalarStep = Callable[
    [DecomposedModel, ComponentStacks, CIFn, Array, PRNGKeyArray], Mapping[str, Array]
]


def next_token_cross_entropy(
    logits: Float[Array, "B T vocab"], token_ids: Int[Array, "B T"]
) -> Array:
    """Mean fp32 CE of positions 0..T-2 predicting tokens 1..T-1 (torch: labels with
    the first position set to ignore_index)."""
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    label_log_probs = jnp.take_along_axis(log_probs[:, :-1], token_ids[:, 1:, None], axis=-1)[
        ..., 0
    ]
    return -label_log_probs.mean()


def _row_masked_mean(per_position: Float[Array, "B ..."], row_mask: Float[Array, " B"]) -> Array:
    """Mean of `per_position` over the rows where `row_mask` is 1 (all positions of a masked
    row weigh 0). `per_position` is fp32 `(B, *positions)`."""
    positions_per_row = math.prod(per_position.shape[1:])
    mask = row_mask.reshape(row_mask.shape[0], *((1,) * (per_position.ndim - 1)))
    return jnp.sum(per_position * mask) / (jnp.sum(row_mask) * positions_per_row)


def _row_masked_kl(
    masked_output: Float[Array, "B T vocab"],
    clean_output: Float[Array, "B T vocab"],
    row_mask: Float[Array, " B"],
) -> Array:
    """`kl_per_position` restricted to the rows where `row_mask` is 1 (same fp32 math,
    per-position KL weighted before the mean)."""
    log_q = jax.nn.log_softmax(masked_output.astype(jnp.float32), axis=-1)
    log_p = jax.nn.log_softmax(clean_output.astype(jnp.float32), axis=-1)
    p = jnp.exp(log_p)
    return _row_masked_mean(jnp.sum(p * (log_p - log_q), axis=-1), row_mask)


def _row_masked_cross_entropy(
    logits: Float[Array, "B T vocab"], token_ids: Int[Array, "B T"], row_mask: Float[Array, " B"]
) -> Array:
    """`next_token_cross_entropy` restricted to the rows where `row_mask` is 1."""
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    label_log_probs = jnp.take_along_axis(log_probs[:, :-1], token_ids[:, 1:, None], axis=-1)[
        ..., 0
    ]
    return _row_masked_mean(-label_log_probs, row_mask)


@dataclass(frozen=True)
class _PreparedLMBatch:
    tokens: Array
    clean_output: Array
    prepared_weights: Any
    ci: dict[str, Array]
    row_mask: Array | None


def _prepare_lm_batch(
    model: DecomposedModel,
    components: ComponentStacks,
    ci_fn: Any,
    token_ids: Int[Array, "B T"],
    mesh: Mesh | None,
    n_valid_rows: int | None,
) -> _PreparedLMBatch:
    """Pure shared preparation required by independent LM metric kernels."""
    tokens = batch_shard_leading(token_ids, mesh)
    clean_output, taps = model.clean_output_and_activations(tokens, ci_fn.input_names)
    clean_output = batch_shard_leading(clean_output, mesh)
    ci = cast_floating(ci_fn, COMPUTE_DT)(taps, remat=False).lower
    if mesh is not None:
        sharding = NamedSharding(
            mesh, P(("replicate", "fsdp"), *((None,) * (tokens.ndim - 1)), None)
        )
        ci = {site: jax.lax.with_sharding_constraint(value, sharding) for site, value in ci.items()}
    row_mask = None
    if n_valid_rows is not None:
        assert n_valid_rows <= tokens.shape[0], (n_valid_rows, tokens.shape)
        row_mask = (jnp.arange(tokens.shape[0]) < n_valid_rows).astype(jnp.float32)
    return _PreparedLMBatch(
        tokens=tokens,
        clean_output=clean_output,
        prepared_weights=model.prepare_compute_weights(cast_floating(components, COMPUTE_DT)),
        ci=ci,
        row_mask=row_mask,
    )


def _masked_forward(
    model: DecomposedModel,
    batch: _PreparedLMBatch,
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    mesh: Mesh | None,
) -> Array:
    return batch_shard_leading(
        model.masked_output(
            batch.prepared_weights,
            batch.tokens,
            masks,
            delta_masks,
            None,
            model.site_names,
            True,
            remat=False,
        ),
        mesh,
    )


def _kl(batch: _PreparedLMBatch, logits: Array) -> Array:
    if batch.row_mask is None:
        return kl_per_position(logits, batch.clean_output)
    return _row_masked_kl(logits, batch.clean_output, batch.row_mask)


def _ce(batch: _PreparedLMBatch, logits: Array) -> Array:
    if batch.row_mask is None:
        return next_token_cross_entropy(logits, batch.tokens)
    return _row_masked_cross_entropy(logits, batch.tokens, batch.row_mask)


def make_ce_kl_step(
    model_static: DecomposedModel,
    rounding_threshold: float,
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str],
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep:
    """Build the single-purpose CE/KL evaluator."""
    assert model_static.has_position_axis, "CEandKLLosses is LM-only and requires a position axis"

    def eval_step(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: CIFn,
        token_ids: Array,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        batch = _prepare_lm_batch(model, components, ci_fn, token_ids, mesh, n_valid_rows)
        zeros_delta = {site: jnp.zeros(batch.tokens.shape, COMPUTE_DT) for site in model.site_names}
        stoch_key, random_key, _ = random.split(key, 3)
        stochastic_masks: dict[str, Array] = {}
        stochastic_deltas: dict[str, Array] = {}
        for site_idx, site in enumerate(model.site_names):
            ci = batch.ci[site]
            source = random.uniform(random.fold_in(stoch_key, site_idx), ci.shape, COMPUTE_DT)
            stochastic_masks[site] = ci + (1.0 - ci) * source
            stochastic_deltas[site] = random.uniform(
                random.fold_in(stoch_key, len(model.site_names) + site_idx),
                batch.tokens.shape,
                COMPUTE_DT,
            )
        variants = {
            "ci_masked": (batch.ci, zeros_delta),
            "unmasked": (
                {site: jnp.ones_like(batch.ci[site]) for site in model.site_names},
                zeros_delta,
            ),
            "stoch_masked": (stochastic_masks, stochastic_deltas),
            "random_masked": (
                {
                    site: random.uniform(
                        random.fold_in(random_key, site_idx), batch.ci[site].shape, COMPUTE_DT
                    )
                    for site_idx, site in enumerate(model.site_names)
                },
                zeros_delta,
            ),
            "rounded_masked": (
                {
                    site: (batch.ci[site] > rounding_threshold).astype(COMPUTE_DT)
                    for site in model.site_names
                },
                zeros_delta,
            ),
            "zero_masked": (
                {site: jnp.zeros_like(batch.ci[site]) for site in model.site_names},
                zeros_delta,
            ),
        }
        variant_logits = {
            name: _masked_forward(model, batch, masks, deltas, mesh)
            for name, (masks, deltas) in variants.items()
        }
        target_ce = _ce(batch, batch.clean_output)
        out = {f"ce_kl/kl_{name}": _kl(batch, logits) for name, logits in variant_logits.items()}
        out.update(
            {
                f"ce_kl/ce_difference_{name}": _ce(batch, variant_logits[name]) - target_ce
                for name in variants
                if name != "zero_masked"
            }
        )
        return out

    return filter_jit(eval_step, compiler_options=compiler_options)


def make_ci_l0_step(
    model_static: DecomposedModel,
    ci_alive_threshold: float,
    groups: dict[str, tuple[str, ...]] | None,
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str],
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep:
    """Bind the generic `CI_L0` arithmetic (`core.ci_l0_eval`) to the LM batch: the shared
    `_prepare_lm_batch` CI and, for the padded arithmetic probes, the row-masked mean."""
    assert model_static.has_position_axis, "CI_L0 is LM-only and requires a position axis"
    resolved_groups = resolve_l0_groups(model_static.site_names, groups)

    def eval_step(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: CIFn,
        token_ids: Array,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        del key
        batch = _prepare_lm_batch(model, components, ci_fn, token_ids, mesh, n_valid_rows)

        def mean(value: Array) -> Array:
            if batch.row_mask is None:
                return value.mean()
            return _row_masked_mean(value, batch.row_mask)

        return ci_l0_scalars(batch.ci, model.site_names, ci_alive_threshold, resolved_groups, mean)

    return filter_jit(eval_step, compiler_options=compiler_options)


def make_fresh_pgd_step(
    model_static: DecomposedModel,
    fresh_pgd: FreshPGDReconEval,
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str],
    *,
    n_valid_rows: int | None = None,
) -> ScalarStep:
    """Build the single-purpose fresh-mask PGD reconstruction evaluator."""
    assert model_static.has_position_axis, "LM PGDReconLoss requires a position axis"

    def eval_step(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: CIFn,
        token_ids: Array,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        batch = _prepare_lm_batch(model, components, ci_fn, token_ids, mesh, n_valid_rows)
        _, _, pgd_key = random.split(key, 3)

        def loss_at_masks(masks: dict[str, Array], delta_masks: dict[str, Array]) -> Array:
            return _kl(batch, _masked_forward(model, batch, masks, delta_masks, mesh))

        loss = fresh_pgd_recon_loss(
            model.sites,
            batch.ci,
            batch.tokens.shape,
            pgd_key,
            fresh_pgd,
            loss_at_masks,
        )
        return {f"loss/{fresh_pgd.name}": loss}

    return filter_jit(eval_step, compiler_options=compiler_options)


def make_eval_step(
    model_static: DecomposedModel,
    rounding_threshold: float,
    ci_alive_threshold: float,
    l0_group_patterns: dict[str, tuple[str, ...]] | None,
    fresh_pgd: FreshPGDReconEval | None,
    mesh: Mesh | None,
    *,
    n_valid_rows: int | None,
    compiler_options: dict[str, bool | int | str],
) -> ScalarStep:
    """Compose independent metric kernels for arithmetic probes and parity tests."""
    ce_kl = make_ce_kl_step(
        model_static,
        rounding_threshold,
        mesh,
        compiler_options,
        n_valid_rows=n_valid_rows,
    )
    ci_l0 = make_ci_l0_step(
        model_static,
        ci_alive_threshold,
        l0_group_patterns,
        mesh,
        compiler_options,
        n_valid_rows=n_valid_rows,
    )
    pgd = (
        make_fresh_pgd_step(
            model_static,
            fresh_pgd,
            mesh,
            compiler_options,
            n_valid_rows=n_valid_rows,
        )
        if fresh_pgd is not None
        else None
    )

    def evaluate(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: CIFn,
        token_ids: Array,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        record = dict(ce_kl(model, components, ci_fn, token_ids, key))
        record.update(ci_l0(model, components, ci_fn, token_ids, key))
        if pgd is not None:
            record.update(pgd(model, components, ci_fn, token_ids, key))
        return record

    return evaluate
