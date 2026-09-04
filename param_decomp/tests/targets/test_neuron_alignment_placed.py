"""The harvest's PLACED path (SPEC T13): a placed model on a real mesh, batch-sharded token
slices, and the moments step declaring its outputs fully replicated — the exact call shape
`neuron_ranks_harvest` makes, so a GPU job is not the first place it runs."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.components import SiteC
from param_decomp.core.model import BATCH_AXES, PlacedModel
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import single_device_mesh
from param_decomp.targets.glu_transformer import canonical_site_cs, glu_site_specs, site_name
from param_decomp.targets.neuron_alignment import (
    accumulate_neuron_moments,
    make_moments_step,
    pool_slices,
    tap_widths,
    unit_counts,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm


def test_placed_moments_match_the_unplaced_sweep():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs((SiteC(site_name(3, "gate"), 2), SiteC(site_name(3, "down"), 2))),
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(11, 4), dtype=np.int32)
    blocks = (0, 3)
    widths = tap_widths(unit_counts(model))

    unplaced = accumulate_neuron_moments(
        make_moments_step(PlacedModel(model=model, placement=None), blocks),
        blocks,
        widths,
        ((jnp.asarray(r), jnp.asarray(m)) for r, m in pool_slices(tokens, 4)),
    )

    mesh = single_device_mesh()
    rules = from_config("ddp", mesh, sites)
    placed_model = PlacedModel(model=model, placement=rules)
    with jax.set_mesh(mesh):
        step = make_moments_step(placed_model, blocks)
        sharding = NamedSharding(mesh, P(BATCH_AXES))

        def slices():
            for rows, mask in pool_slices(tokens, 4):
                yield (
                    jax.make_array_from_process_local_data(sharding, rows, rows.shape),
                    jnp.asarray(mask),
                )

        placed = accumulate_neuron_moments(step, blocks, widths, slices())
    for block in blocks:
        assert placed[block].n_tokens == unplaced[block].n_tokens == 11 * 4
        for tap in widths:
            assert np.allclose(
                placed[block].sum_sq[tap], unplaced[block].sum_sq[tap], rtol=1e-4, atol=1e-6
            ), tap
