"""The finished-run loader delegates component activations to the target."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.experiments.lm import load_run
from param_decomp.targets.glu_transformer import KIND_ORDER, glu_site_specs, site_name
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_chunkwise_ci_fn,
    tiny_glu_decomposed_lm,
)


def test_open_jax_run_uses_target_owned_component_activations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The post-training forward returns one component-activation tensor per site."""
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, tuple(SiteC(site_name(2, kind), 2) for kind in KIND_ORDER))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    decomposition = SimpleNamespace(
        components=init_component_stacks(sites, jax.random.PRNGKey(1)),
        ci_fn=tiny_glu_chunkwise_ci_fn(model, jax.random.PRNGKey(2), n_blocks=1),
    )
    deliverable = SimpleNamespace(target=SimpleNamespace(), ci_fn=object())

    monkeypatch.setattr(load_run, "load_deliverable", lambda *_args: deliverable)
    monkeypatch.setattr(load_run, "hsdp_mesh", lambda: object())
    monkeypatch.setattr(load_run, "build_target", lambda *_args: (model, cfg.vocab_size))

    def restore(*_args: Any) -> tuple[Any, int]:
        return decomposition, 7

    monkeypatch.setattr(load_run, "_restore_decomposition", restore)

    run = load_run.open_jax_run(tmp_path / "p-canonical-capture", data_root=tmp_path)
    result = run.forward(jnp.arange(4, dtype=jnp.int32)[None, :])

    assert run.step == 7
    assert result.component_activations_by_site.keys() == dict.fromkeys(model.site_names).keys()
    assert all(value.shape == (1, 4, 2) for value in result.component_activations_by_site.values())
