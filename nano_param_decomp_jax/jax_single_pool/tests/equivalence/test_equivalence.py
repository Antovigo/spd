"""Pytest entry for the cross-framework PD equivalence harness.

Two kinds of check:

  * **Numeric (cross-framework).** `test_jax_matches_torch_reference` runs the JAX side of
    the harness on the committed fixtures and asserts every loss term matches the committed
    `torch_reference.json` (produced by `torch_reference.py` in the torch env) to fp32
    tolerance. This is the watertight numeric verification: identical fixtures into both
    frameworks, the torch values from the REAL reference functions
    (`faithfulness_loss` / `importance_minimality_loss` / `recon_loss_kl` /
    `get_ppgd_mask_infos` / `LinearComponents.forward`), compared at ~1e-4.

  * **Structural.** `test_structure_*` pin the design facts that aren't a single number:
    the stochastic recon does ONE forward PER CHUNK (12 for the production 20..31 / 3-site
    chunks), recon is KL (not MSE), and the PPGD source carries the trailing weight-delta
    channel.

Regenerate the cross-framework golden (only needed if the math or fixtures change):

    # JAX env:
    python jax_single_pool/tests/equivalence/gen_fixtures.py
    # torch (param-decomp) env:
    python jax_single_pool/tests/equivalence/torch_reference.py
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import jax_single_pool.llama8b_step as step_mod
from jax_single_pool.tests.equivalence.jax_equivalence import compute_jax_terms

HERE = Path(__file__).resolve().parent
RTOL = 2e-4
ATOL = 1e-5


@pytest.mark.parametrize("term", ["faith", "imp", "stoch", "ppgd"])
def test_jax_matches_torch_reference(term: str) -> None:
    ref_path = HERE / "torch_reference.json"
    assert ref_path.exists(), "run torch_reference.py (torch env) to produce the golden first"
    ref = json.loads(ref_path.read_text())
    jaxv = compute_jax_terms(dict(np.load(HERE / "fixtures.npz")))
    jv, tv = jaxv[term], ref[term]
    assert abs(jv - tv) <= ATOL + RTOL * abs(tv), (
        f"{term}: jax {jv:.8e} vs torch {tv:.8e} (rel {abs(jv - tv) / (abs(tv) + 1e-30):.2e})"
    )


def test_structure_stoch_is_per_chunk() -> None:
    """`_stochastic_recon` runs ONE forward per chunk (== n_layers), matching the torch
    chunkwise pool (sites_per_chunk=3 → 12 chunks for layers 20..31), not one fused
    forward over all sites."""
    src = inspect.getsource(step_mod._stochastic_recon)
    assert "for chunk_idx in range(n_layers)" in src, "stoch must loop one forward per chunk"
    assert "/ n_layers" in src, "stoch must average over the n_layers per-chunk forwards"


def test_structure_recon_is_kl_not_mse() -> None:
    """Recon is KL on logits (`recon_loss_kl`), not MSE."""
    src = inspect.getsource(step_mod._kl_per_position)
    assert "log_softmax" in src and "log_p - log_q" in src, "recon must be KL"
    assert "** 2" not in src and "**2" not in src, "recon must not be MSE"


def test_structure_ppgd_has_delta_channel() -> None:
    """PPGD source carries a trailing weight-delta channel; masks interpolate
    `ci + (1-ci)*source[:-1]` and use `source[-1]` as the delta mask."""
    src = inspect.getsource(step_mod._ppgd_masks_and_deltas)
    assert "[..., :-1]" in src and "[..., -1:]" in src, "ppgd source needs the delta channel"
    assert "ci[k] + (1.0 - ci[k]) * comp_src" in src, "ppgd must interpolate mask=ci+(1-ci)*src"
