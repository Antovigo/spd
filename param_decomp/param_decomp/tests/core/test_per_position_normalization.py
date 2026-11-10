"""The per-position form of the relative squared error (local amendment 2026-09-11): one
ratio per (batch, position) entry against its own clean scale, averaged — so no position can
silence another — with a floor that is a fraction of the MEDIAN per-entry clean energy."""

import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from param_decomp.core.configs import (
    BatchHiddenActsNormalization,
    CaptureReconstruction,
    PerPositionHiddenActsNormalization,
)
from param_decomp.core.losses import relative_squared_error

PER_POSITION = PerPositionHiddenActsNormalization(floor_fraction=0.01)


def test_per_position_normalization_hand_computed_matches_the_per_entry_mean():
    """The fixture of `..._pools_energy_over_batch_and_sequence`, read the other way: the
    per-position mode returns exactly the mean of per-entry ratios that the batch mode does
    not. Floor set negligible (median clean energy here is 6.5, floor 0.065) and folded in."""
    clean = jnp.array([[[1.0], [3.0]], [[2.0], [4.0]]])
    masked = jnp.array([[[2.0], [3.0]], [[2.0], [0.0]]])
    floor = 0.01 * 6.5  # median of {1, 9, 4, 16}
    expected = (1.0 / (1.0 + floor) + 0.0 + 0.0 + 16.0 / (16.0 + floor)) / 4
    got = float(relative_squared_error(masked, clean, normalization=PER_POSITION))
    assert got == pytest.approx(expected, rel=1e-6)
    batch = float(
        relative_squared_error(masked, clean, normalization=BatchHiddenActsNormalization())
    )
    assert batch == pytest.approx((1.0 + 16.0) / (1.0 + 9.0 + 4.0 + 16.0))
    assert got != pytest.approx(batch)


def test_per_position_normalization_does_not_let_one_position_silence_the_others():
    """The failure this mode exists for (Llama's BOS massive activation): one position holds
    ~all of the clean energy and is reproduced exactly, every other position is badly wrong.
    Batch mode reads ~0 — the point is dead; per-position mode reads the other positions'
    actual relative error."""
    b, t, d = 4, 5, 8
    clean = jnp.ones((b, t, d))
    clean = clean.at[:, 0, :].set(1000.0)  # position 0: 1e6 energy per entry vs 8 elsewhere
    masked = clean.at[:, 1:, :].set(0.0)  # BOS perfect, every other position fully lost
    batch = float(relative_squared_error(masked, clean))
    per_position = float(relative_squared_error(masked, clean, normalization=PER_POSITION))
    assert batch < 1e-4  # 4*4*8 / (4*8e6 + 4*4*8) ~ 4e-6: the point is silent
    # 4 of 5 positions at ratio 8/(8 + 0.01*8) = 1/1.01, BOS at 0; median energy is 8.
    assert per_position == pytest.approx(0.8 / 1.01, rel=1e-5)


def test_per_position_normalization_floor_is_the_median_not_the_mean():
    """A silent position (zero clean energy) would divide by zero without the floor; the floor
    is a fraction of the MEDIAN per-entry energy, so the outlier position cannot inflate it."""
    clean = jnp.array([[[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [100.0, 0.0]]])  # energies 0,2,2,1e4
    masked = clean.at[0, 0, :].set(1.0)  # error 2 at the silent position, 0 elsewhere
    # median of {0, 2, 2, 1e4} is 2 -> floor 0.02; the silent entry scores 2/0.02 = 100.
    got = float(relative_squared_error(masked, clean, normalization=PER_POSITION))
    assert got == pytest.approx(100.0 / 4, rel=1e-5)
    assert jnp.isfinite(got)


def test_per_position_normalization_stays_finite_when_the_median_energy_is_zero():
    """Sparse / ReLU taps: when at least half the valid entries are silent the median is
    exactly zero, and a floor built on it alone would vanish — `0/0` at the silent entries.
    The floor falls back to the MEAN clean energy, and an entirely silent point still divides
    by fp32 tiny rather than by zero."""
    clean = jnp.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.0]]])  # energies 0,0,0,4
    masked = clean.at[0, 0, :].set(1.0)  # error 2 at one silent entry, 0 elsewhere
    got = float(relative_squared_error(masked, clean, normalization=PER_POSITION))
    # median 0 -> scale = mean = 1 -> floor 0.01: the silent entry scores 2/0.01 = 200.
    assert got == pytest.approx(200.0 / 4, rel=1e-4)
    assert jnp.isfinite(got)
    silent = jnp.zeros((1, 4, 2))
    got = float(relative_squared_error(silent, silent, normalization=PER_POSITION))
    assert got == 0.0  # 0 / tiny, not 0 / 0
    got = float(
        relative_squared_error(silent.at[0, 0, 0].set(1.0), silent, normalization=PER_POSITION)
    )
    assert jnp.isfinite(got)


def test_per_position_normalization_respects_the_valid_row_mask():
    """A padded row must not enter the mean, the sums, or the median."""
    clean = jnp.ones((2, 3, 4))
    masked = clean.at[0].set(0.0)  # row 0 fully wrong, row 1 perfect
    valid = jnp.array([1.0, 0.0])
    got = float(
        relative_squared_error(masked, clean, valid_row_mask=valid, normalization=PER_POSITION)
    )
    assert got == pytest.approx(1.0 / 1.01, rel=1e-6)
    valid = jnp.array([0.0, 1.0])
    got = float(
        relative_squared_error(masked, clean, valid_row_mask=valid, normalization=PER_POSITION)
    )
    assert got == 0.0


def test_per_position_normalization_is_the_batch_mode_when_energies_are_uniform():
    """With equal clean energy at every entry the two reductions agree up to the floor, which
    pins the per-position mean to `1/(B·T)` per token — the KL's and imp-min's normalization."""
    clean = jnp.full((3, 4, 8), 2.0)
    masked = clean * 1.25
    batch = float(relative_squared_error(masked, clean))
    per_position = float(relative_squared_error(masked, clean, normalization=PER_POSITION))
    assert batch == 0.0625
    assert per_position == pytest.approx(0.0625 / 1.01, rel=1e-6)


def test_capture_normalization_parses_and_refuses_a_zero_floor_or_a_non_rse_distance():
    """Parsed the way YAML arrives: the `kind` discriminator selects the form, the floor
    defaults to 0.01, the default form is the original `batch`, a zero floor is refused, and
    only relative squared error takes a normalization."""
    cfg = CaptureReconstruction.model_validate(
        {
            "capture": "resid.1",
            "distance": "relative_squared_error",
            "normalization": {"kind": "per_position"},
        }
    )
    assert isinstance(cfg.normalization, PerPositionHiddenActsNormalization)
    assert cfg.normalization.floor_fraction == 0.01
    default = CaptureReconstruction(capture="resid.1", distance="relative_squared_error")
    assert isinstance(default.normalization, BatchHiddenActsNormalization)
    with pytest.raises(ValidationError):
        CaptureReconstruction.model_validate(
            {
                "capture": "resid.1",
                "distance": "relative_squared_error",
                "normalization": {"kind": "per_position", "floor_fraction": 0.0},
            }
        )
    with pytest.raises(ValidationError):
        CaptureReconstruction.model_validate(
            {
                "capture": "logits",
                "distance": "categorical_kl_from_logits",
                "normalization": {"kind": "per_position"},
            }
        )


def test_per_position_normalization_runs_with_a_batch_sharded_argument():
    """The median lowers to `lax.sort`, which refuses a sharded sort dimension. The hidden
    activations arrive batch-sharded on every real mesh, so without replicating the per-entry
    energies first this mode raises `ShardingTypeError` at trace time and the per-position
    recipe cannot run at all (measured on the 4x H100 seat, 2026-09-11). Run this file at
    `XLA_FLAGS=--xla_force_host_platform_device_count=4` for the sharded arm to be real."""
    n = jax.device_count()
    if n < 2:
        pytest.skip(f"needs >= 2 devices to shard the batch, have {n}")
    mesh = jax.make_mesh((n,), ("batch",), axis_types=(jax.sharding.AxisType.Explicit,))
    b, t, d = 4 * n, 3, 2
    key = jax.random.key(0)
    clean = jax.random.normal(key, (b, t, d))
    masked = clean * 1.5
    with jax.set_mesh(mesh):
        spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch", None, None))
        sharded_clean = jax.device_put(clean, spec)
        sharded_masked = jax.device_put(masked, spec)
        got = float(
            jax.jit(lambda m, c: relative_squared_error(m, c, normalization=PER_POSITION))(
                sharded_masked, sharded_clean
            )
        )
    unsharded = float(relative_squared_error(masked, clean, normalization=PER_POSITION))
    assert got == pytest.approx(unsharded, rel=1e-5)
