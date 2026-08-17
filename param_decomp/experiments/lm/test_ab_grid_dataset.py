"""CPU tests for the (a,b)-grid snapshot eval (`ab_grid_dataset.py`).

Pins: the fused step's CI / normalized inner activations at the recorded positions with the
batch axis kept as the grid, and its pad-masked CI sums; the floor cut; chunk-count
invariance of which components are saved; the applet's `[comp, pos, op, a, b]` payload
layout (a pos/comp axis swap is the bug this catches); and the snapshot write + manifest.
"""

import base64
import json
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.ci_fn import ci_preactivations, lower_leaky_hard_sigmoid
from param_decomp.core.components import init_component_stacks
from param_decomp.core.model import prepare_compute_weights
from param_decomp.core.tests.test_slow_eval import _build_ci_fn
from param_decomp.experiments.lm.ab_grid_dataset import (
    APPLET_FILENAME,
    ABGridSnapshot,
    ab_grid_payload,
    collect_ab_grid_snapshot,
    encode_ci_u8,
    make_ab_grid_step,
    saved_indices,
    write_ab_grid_snapshot,
)
from param_decomp.experiments.lm.ab_grid_operation import resolve_positions
from param_decomp.experiments.lm.arithmetic_probe import ArithmeticGrid
from param_decomp.targets.glu_transformer import glu_site_specs, mlp_family_site_cs
from param_decomp.targets.testing import capture_clean, tiny_glu_cfg, tiny_glu_decomposed_lm

N_A, N_B = 3, 4
T = 5
POSITIONS = (0, T - 1)
SITE = "layers.4.mlp.gate_proj"


@cache
def _tiny_setup():
    cfg = tiny_glu_cfg()
    C = 8
    sites = glu_site_specs(cfg, mlp_family_site_cs(4, 5, C))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))
    return cfg, model, ci_fn, C


@cache
def _grid_step():
    """One trace shared by every test: the step is a pure function of the (cached) model and
    the recorded positions, both fixed for this file. The row count is a traced arg, so a
    chunked pass reuses this same trace."""
    _, model, ci_fn, _ = _tiny_setup()
    return make_ab_grid_step(model, ci_fn.capture_keys, POSITIONS)


def _grid() -> ArithmeticGrid:
    return ArithmeticGrid(a_values=tuple(range(N_A)), b_values=tuple(range(N_B)), symbol="+")


def _components():
    return init_component_stacks(_tiny_setup()[1].sites, jax.random.PRNGKey(1))


def _tokens(n_rows: int) -> jax.Array:
    cfg = _tiny_setup()[0]
    return jax.random.randint(jax.random.PRNGKey(4), (n_rows, T), 0, cfg.vocab_size)


def test_grid_step_ci_inner_and_pad_masked_sums_match_hand_rolled():
    _cfg, model, ci_fn, C = _tiny_setup()
    vu = _components()
    n_prompts = N_A * N_B
    n_pad = n_prompts + 2  # two garbage tail rows, as the sharding pad would append
    tokens = _tokens(n_pad)
    ci_grids, inner_grids, ci_sums = _grid_step()(model, vu, ci_fn, tokens, jnp.asarray(n_prompts))

    preactivations = ci_preactivations(
        ci_fn, capture_clean(model, tokens, ci_fn.capture_keys), remat=False
    )
    _clean, raw_activations = model.component_activation_forward(
        prepare_compute_weights(model, vu), tokens, capture_keys=ci_fn.capture_keys
    )
    for site in model.site_names:
        ci = np.asarray(ci_grids[site])
        assert ci.shape == (n_pad, len(POSITIONS), C)
        assert ci.min() >= 0.0 and ci.max() <= 1.0
        ci_expected = np.asarray(lower_leaky_hard_sigmoid(preactivations[site]), np.float32)
        np.testing.assert_allclose(ci, ci_expected[:, POSITIONS, :], rtol=1e-4, atol=1e-4)

        # inner is x@V divided by the component's own ‖V_c‖, at the recorded positions only
        v_norms = np.linalg.norm(np.asarray(vu.site(site).V, np.float32), axis=0)
        inner_expected = np.asarray(raw_activations[site], np.float32)[:, POSITIONS, :] / v_norms
        np.testing.assert_allclose(np.asarray(inner_grids[site]), inner_expected, rtol=1e-4)

        # the sums are over the REAL rows only — the garbage tail must not move a mean
        np.testing.assert_allclose(
            np.asarray(ci_sums[site]), ci[:n_prompts].sum(axis=0), rtol=1e-5, atol=1e-5
        )


def test_collect_snapshot_saves_only_components_above_the_floor():
    _cfg, model, ci_fn, C = _tiny_setup()
    vu = _components()
    n_prompts = N_A * N_B
    tokens = _tokens(n_prompts)
    everything = collect_ab_grid_snapshot(
        _grid_step(), model, vu, ci_fn, ((tokens, n_prompts),), n_prompts, mean_ci_floor=0.0
    )
    for site in model.site_names:
        mean_ci = everything.mean_ci[site]
        assert mean_ci.shape == (len(POSITIONS), C)
        assert everything.saved[site].tolist() == list(range(C))  # floor 0 saves everything
        assert everything.ci_columns[site].shape == (n_prompts, len(POSITIONS), C)
        assert everything.inner_columns[site].shape == (n_prompts, len(POSITIONS), C)

    floor = float(np.median([m.max() for m in everything.mean_ci.values()]))
    cut = collect_ab_grid_snapshot(
        _grid_step(), model, vu, ci_fn, ((tokens, n_prompts),), n_prompts, mean_ci_floor=floor
    )
    for site in model.site_names:
        expected = saved_indices(everything.mean_ci[site], floor)
        np.testing.assert_array_equal(cut.saved[site], expected)
        # the mean-CI vector is kept for EVERY component even when its grids are cut
        np.testing.assert_allclose(cut.mean_ci[site], everything.mean_ci[site], rtol=1e-6)
        np.testing.assert_allclose(
            cut.ci_columns[site], everything.ci_columns[site][:, :, expected], rtol=1e-6
        )
    assert any(cut.saved[site].size < C for site in model.site_names)


def test_chunking_the_grid_changes_nothing():
    """Chunking bounds the forward, not the result: the summed CI over chunks and the
    chunk-order column concatenation reproduce the single-forward pass up to float
    reassociation (SPEC D4). WHICH components are saved must match exactly — that is the
    part a chunking bug would break."""
    _cfg, model, ci_fn, _C = _tiny_setup()
    vu = _components()
    n_prompts = N_A * N_B
    tokens = _tokens(n_prompts)
    step = _grid_step()

    whole = collect_ab_grid_snapshot(
        step, model, vu, ci_fn, ((tokens, n_prompts),), n_prompts, mean_ci_floor=0.02
    )
    split = n_prompts // 2
    chunked = collect_ab_grid_snapshot(
        step,
        model,
        vu,
        ci_fn,
        ((tokens[:split], split), (tokens[split:], n_prompts - split)),
        n_prompts,
        mean_ci_floor=0.02,
    )
    for site in model.site_names:
        np.testing.assert_array_equal(chunked.saved[site], whole.saved[site])
        np.testing.assert_allclose(chunked.mean_ci[site], whole.mean_ci[site], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            chunked.ci_columns[site], whole.ci_columns[site], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            chunked.inner_columns[site], whole.inner_columns[site], rtol=1e-5, atol=1e-6
        )


def test_saved_indices_cuts_on_the_best_position():
    mean_ci = np.array([[0.9, 0.01, 0.0], [0.0, 0.2, 0.01]])  # (n_pos, C)
    assert saved_indices(mean_ci, 0.05).tolist() == [0, 1]  # component 1 clears at position 1
    assert saved_indices(mean_ci, 0.5).tolist() == [0]
    assert saved_indices(mean_ci, 0.0).tolist() == [0, 1, 2]


def test_encode_ci_u8():
    ci = np.array([0.0, 0.5, 1.0, 1.7, -0.2], dtype=np.float32)
    encoded = encode_ci_u8(ci)
    assert encoded.dtype == np.uint8
    assert encoded.tolist() == [0, 128, 255, 255, 0]


def _synthetic_snapshot(n_pos: int, saved: list[int], C: int) -> ABGridSnapshot:
    """Component `saved[1]` is on ONLY at the last position and only where a == b — the
    pattern a pos/comp axis swap in the payload moves to the wrong slice."""
    n_prompts = N_A * N_B
    ci = np.zeros((n_prompts, n_pos, len(saved)), np.float32)
    ci[:, :, 0] = 1.0
    for a in range(N_A):
        for b in range(N_B):
            ci[a * N_B + b, n_pos - 1, 1] = float(a == b)
    mean_ci = np.zeros((n_pos, C), np.float32)
    mean_ci[:, saved] = 1.0
    return ABGridSnapshot(
        mean_ci={SITE: mean_ci},
        saved={SITE: np.asarray(saved)},
        ci_columns={SITE: ci},
        inner_columns={SITE: ci * -2.0},
    )


def test_payload_is_comp_major_with_a_length_one_op_axis():
    n_pos, C = len(POSITIONS), 8
    snapshot = _synthetic_snapshot(n_pos, [0, 3], C)
    payload = ab_grid_payload(snapshot, _grid(), POSITIONS, T, step=1000, mean_ci_floor=0.05)

    assert payload["positions"] == list(POSITIONS)
    assert payload["ops"] == ["+"] and payload["ci_roles"] == ["output"]
    assert (payload["a_min"], payload["n_a"], payload["b_min"], payload["n_b"]) == (0, N_A, 0, N_B)
    module = payload["modules"][0]
    assert module["name"] == SITE and module["C"] == C and module["saved"] == [0, 3]

    ci = np.frombuffer(base64.b64decode(module["ci"]), np.uint8).reshape(2, n_pos, 1, N_A, N_B)
    assert (ci[0] == 255).all()  # component 0 on everywhere, at both positions
    assert (ci[1, 0] == 0).all()  # component 1 off at the first position
    for a in range(N_A):
        for b in range(N_B):
            assert ci[1, 1, 0, a, b] == (255 if a == b else 0)
    inner = np.frombuffer(base64.b64decode(module["inner"]), np.float16).reshape(
        2, n_pos, 1, N_A, N_B
    )
    assert inner[1, 1, 0, 1, 1] == -2.0
    mean_ci = np.frombuffer(base64.b64decode(module["mean_ci"]), np.float32)
    assert mean_ci.shape == (n_pos * C,)  # every component, row-major (pos, C)


def test_payload_omits_grids_for_a_module_with_nothing_saved():
    snapshot = ABGridSnapshot(
        mean_ci={SITE: np.zeros((1, 4), np.float32)},
        saved={SITE: np.asarray([], dtype=int)},
        ci_columns={SITE: np.zeros((N_A * N_B, 1, 0), np.float32)},
        inner_columns={SITE: np.zeros((N_A * N_B, 1, 0), np.float32)},
    )
    module = ab_grid_payload(snapshot, _grid(), (T - 1,), T, 1000, 0.05)["modules"][0]
    assert module["saved"] == [] and "ci" not in module and "inner" not in module
    assert "mean_ci" in module  # the cut stays visible


def test_write_snapshot_ships_the_applet_and_orders_the_manifest_by_step(tmp_path: Path):
    snapshot = _synthetic_snapshot(len(POSITIONS), [0, 3], 8)
    payload = ab_grid_payload(snapshot, _grid(), POSITIONS, T, step=1000, mean_ci_floor=0.05)
    write_ab_grid_snapshot(tmp_path, 1000, payload)
    write_ab_grid_snapshot(tmp_path, 200, payload | {"step": 200})

    out = tmp_path / "ab_grids"
    manifest = (out / "manifest.js").read_text()
    assert manifest == 'window.AB_GRIDS_MANIFEST = ["step_200.js", "step_1000.js"];\n'
    js = (out / "step_1000.js").read_text()
    assert js.startswith("window.registerABGrids(") and js.endswith(");")
    assert json.loads(js[len("window.registerABGrids(") : -2])["step"] == 1000
    applet = Path(__file__).parent / APPLET_FILENAME
    assert (out / "index.html").read_bytes() == applet.read_bytes()


def test_resolve_positions():
    assert resolve_positions(None, T) == (T - 1,)
    assert resolve_positions([0, -1], T) == (0, T - 1)
    with pytest.raises(AssertionError, match="out of range"):
        resolve_positions([T], T)
    with pytest.raises(AssertionError, match="duplicates"):
        resolve_positions([-1, T - 1], T)
