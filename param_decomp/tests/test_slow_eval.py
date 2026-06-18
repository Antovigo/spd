"""CPU tests for the JAX-native slow (plot-type) eval pass.

Pins the reduction semantics against hand-rolled numpy (component activation density and
mean-CI per component are exact under micro-batching), the `pre_sigmoid`-vs-`lower`
distinction, the `n_batches_accum` cap on the histogram sample, and that the renderer
emits valid PNGs under the exact torch `slow_eval/figures/*` keys. Also covers the in-loop
slow tier (SPEC S28/S29): the `slow_every` / `slow_on_first_step` cadence and the rank-0
background `SlowEvalRenderer` logging figures on the live `_step` axis.
"""

import sys
import types
from typing import Any

import jax
import numpy as np
import pytest

from param_decomp.ci_fn import CIArch, init_ci_fn, lower_leaky_hard_sigmoid
from param_decomp.llama8b import (
    llama_decomposed_lm,
    llama_site_specs,
    mlp_family_site_cs,
)
from param_decomp.run import SlowEvalRenderer, slow_eval_due
from param_decomp.slow_eval import (
    accumulate_site_reductions,
    make_slow_eval_step,
    render_slow_eval_figures,
)
from param_decomp.tests.test_llama8b import (
    _tiny_cfg,
    _tiny_target,
)


def _tiny_setup(threshold: float):
    cfg = _tiny_cfg()
    tgt = _tiny_target(cfg, 4, jax.random.PRNGKey(0))
    C = 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, C))
    lm = llama_decomposed_lm(cfg, sites)
    ci_fn = init_ci_fn(CIArch(16, 1, 2, 32), lm.sites, jax.random.PRNGKey(2))
    step = make_slow_eval_step(lm, threshold)
    return cfg, lm, tgt, ci_fn, step, C


def test_reductions_match_hand_rolled_per_component():
    cfg, lm, tgt, ci_fn, step, C = _tiny_setup(threshold=0.0)
    b, t = 3, 16
    residual = jax.random.normal(jax.random.PRNGKey(4), (b, t, cfg.n_embd)) * 0.5

    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], n_batches_accum=None)

    site_inputs = lm.site_inputs(tgt, residual)
    lower = {s: lower_leaky_hard_sigmoid(ci_fn.site_logits(site_inputs)[s]) for s in lm.site_names}
    for site in lm.site_names:
        flat = np.asarray(lower[site]).reshape(-1, C).astype(np.float32)
        r = reductions[site]
        assert r.n_positions == b * t
        np.testing.assert_allclose(r.density_counts, (flat > 0.0).sum(0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(r.ci_sums, flat.sum(0), rtol=1e-4, atol=1e-4)


def test_density_threshold_caps_counts_at_n_positions():
    cfg, _, tgt, ci_fn, step, _ = _tiny_setup(threshold=-1.0)  # everything "alive"
    residual = jax.random.normal(jax.random.PRNGKey(7), (2, 16, cfg.n_embd)) * 0.5
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], n_batches_accum=None)
    for r in reductions.values():
        np.testing.assert_array_equal(r.density_counts, np.full_like(r.density_counts, 2 * 16))


def test_cross_batch_sum_accumulates_linearly():
    cfg, lm, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    res_a = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd)) * 0.5
    res_b = jax.random.normal(jax.random.PRNGKey(5), (2, 16, cfg.n_embd)) * 0.5

    one = accumulate_site_reductions(step, ci_fn, tgt, [res_a], None)
    two = accumulate_site_reductions(step, ci_fn, tgt, [res_a, res_b], None)
    other = accumulate_site_reductions(step, ci_fn, tgt, [res_b], None)
    for site in lm.site_names:
        assert two[site].n_positions == one[site].n_positions + other[site].n_positions
        np.testing.assert_allclose(
            two[site].ci_sums, one[site].ci_sums + other[site].ci_sums, rtol=1e-4, atol=1e-4
        )


def test_n_batches_accum_caps_histogram_sample_only():
    cfg, lm, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    batches = [
        jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(9), i), (2, 16, cfg.n_embd))
        for i in range(3)
    ]
    capped = accumulate_site_reductions(step, ci_fn, tgt, batches, n_batches_accum=1)
    full = accumulate_site_reductions(step, ci_fn, tgt, batches, n_batches_accum=None)
    for site in lm.site_names:
        # the cap only limits the histogram raw-value sample; counts/sums span all batches
        assert capped[site].n_positions == full[site].n_positions == 3 * 2 * 16
        assert capped[site].lower_sample.size == 2 * 16 * 8  # one batch
        assert full[site].lower_sample.size == 3 * 2 * 16 * 8


def test_pre_sigmoid_differs_from_lower():
    cfg, _, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)
    for r in reductions.values():
        # lower is clamped to [0, 1]; logits are unbounded — they cannot be identical
        assert r.lower_sample.min() >= 0.0 and r.lower_sample.max() <= 1.0
        assert not np.allclose(r.lower_sample, r.logits_sample)


def test_render_emits_torch_keyed_pngs():
    cfg, _, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)
    figures = render_slow_eval_figures(reductions)
    assert set(figures) == {
        "figures/causal_importance_values",
        "figures/causal_importance_values_pre_sigmoid",
        "figures/component_activation_density",
        "figures/ci_mean_per_component",
        "figures/ci_mean_per_component_log",
    }
    for png in figures.values():
        assert png[:4] == b"\x89PNG", "renderer must emit valid PNG bytes"


def test_finite_reductions():
    cfg, _, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)
    for r in reductions.values():
        assert np.all(np.isfinite(r.density_counts))
        assert np.all(np.isfinite(r.ci_sums))
        assert np.all(np.isfinite(r.lower_sample))
        assert np.all(np.isfinite(r.logits_sample))


def test_slow_eval_due_fires_on_cadence_and_first_step():
    # multiples of slow_every fire; non-multiples don't
    assert slow_eval_due(now_step=10000, every=1000, slow_every=10000, slow_on_first_step=False)
    assert not slow_eval_due(now_step=2000, every=1000, slow_every=10000, slow_on_first_step=False)
    assert slow_eval_due(now_step=20000, every=1000, slow_every=10000, slow_on_first_step=False)
    # slow_on_first_step additionally fires at the first eval step (now_step == every)
    assert slow_eval_due(now_step=1000, every=1000, slow_every=10000, slow_on_first_step=True)
    assert not slow_eval_due(now_step=1000, every=1000, slow_every=10000, slow_on_first_step=False)
    # the first eval step is the ONLY extra one slow_on_first_step adds
    assert not slow_eval_due(now_step=2000, every=1000, slow_every=10000, slow_on_first_step=True)


class _FakeWandb(types.ModuleType):
    """Minimal stand-in for the `wandb` module the background renderer imports."""

    class errors(types.ModuleType):  # noqa: N801 — mirrors the real `wandb.errors` submodule
        class CommError(Exception):
            pass

    def __init__(self):
        super().__init__("wandb")
        self.logged: list[tuple[dict[str, Any], int]] = []

    def Image(self, img: Any) -> Any:  # noqa: N802 — mirrors `wandb.Image`
        return img

    def log(self, payload: dict[str, Any], step: int) -> None:
        self.logged.append((payload, step))


def test_renderer_logs_figures_on_live_step_axis(monkeypatch: pytest.MonkeyPatch):
    cfg, _, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    renderer = SlowEvalRenderer(is_main=True)
    renderer.submit(reductions, now_step=4242)
    renderer.join()  # flush the background render

    assert len(fake.logged) == 1
    payload, logged_step = fake.logged[0]
    assert logged_step == 4242  # on the live `_step` axis at the eval step
    assert set(payload) == {
        "slow_eval/figures/causal_importance_values",
        "slow_eval/figures/causal_importance_values_pre_sigmoid",
        "slow_eval/figures/component_activation_density",
        "slow_eval/figures/ci_mean_per_component",
        "slow_eval/figures/ci_mean_per_component_log",
    }


def test_renderer_noop_off_main_rank(monkeypatch: pytest.MonkeyPatch):
    cfg, _, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))
    reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    renderer = SlowEvalRenderer(is_main=False)
    renderer.submit(reductions, now_step=4242)
    renderer.join()
    assert fake.logged == []  # non-main ranks do the collective pull but never render/log


def test_in_loop_slow_tier_fires_on_cadence_without_stalling(monkeypatch: pytest.MonkeyPatch):
    """Smoke: drive the in-loop slow-tier block (collective accumulate -> background
    render) over a sequence of eval steps and assert figures land ONLY on slow steps, on
    the live `_step` axis, and the main loop never blocks waiting on a render."""
    import time

    cfg, _, tgt, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd))

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    every, slow_every = 1000, 3000
    renderer = SlowEvalRenderer(is_main=True)
    t0 = time.time()
    for now_step in range(every, 10 * every + 1, every):  # 1000, 2000, ..., 10000
        if slow_eval_due(now_step, every, slow_every, slow_on_first_step=True):
            # the COLLECTIVE part (runs on every rank in the real loop)
            reductions = accumulate_site_reductions(step, ci_fn, tgt, [residual], None)
            renderer.submit(reductions, now_step)  # rank-0 background render
    main_loop_s = time.time() - t0
    renderer.join()  # flush

    logged_steps = sorted(s for _, s in fake.logged)
    # slow_on_first_step adds 1000; multiples of 3000 add 3000, 6000, 9000
    assert logged_steps == [1000, 3000, 6000, 9000]
    for payload, _ in fake.logged:
        assert all(k.startswith("slow_eval/figures/") for k in payload)
    # the dispatch loop itself must not block on rendering — accumulate + submit are quick
    assert main_loop_s < 30.0, main_loop_s
