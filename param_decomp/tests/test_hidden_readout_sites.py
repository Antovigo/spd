"""Readout sites: measuring the hidden objective somewhere other than a decomposed matrix."""

from typing import override

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.component_model import ComponentModel, OutputWithCache
from param_decomp.components import LinearComponents
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import ComponentsMaskInfo, make_mask_infos
from param_decomp.metrics.hidden_acts import (
    clean_site_outputs,
    select_sites,
    site_squared_errors,
)
from param_decomp_lab.batch_and_loss_fns import run_batch_passthrough

READOUTS = {"resid_post_attn": "post_attn_norm", "resid_post_mlp": "out_norm"}


class ResidualBlockModel(nn.Module):
    """One transformer-shaped block: two residual writes with an observable stream between."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.attn = nn.Linear(d, d, bias=False)
        self.post_attn_norm = nn.Identity()
        self.mlp = nn.Linear(d, d, bias=False)
        self.out_norm = nn.Identity()

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.post_attn_norm(x + self.attn(x))
        return self.out_norm(h + self.mlp(h))


def _model(readouts: dict[str, str] = READOUTS) -> ComponentModel:
    torch.manual_seed(0)
    target = ResidualBlockModel(d=4)
    target.requires_grad_(False)
    return ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        decomposition_targets=[
            DecompositionTarget(module_path="attn", C=2),
            DecompositionTarget(module_path="mlp", C=2),
        ],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        sigmoid_type="leaky_hard",
        hidden_readout_sites=readouts,
    )


def _target(model: ComponentModel) -> ResidualBlockModel:
    target = model.target_model
    assert isinstance(target, ResidualBlockModel)
    return target


def _silenced_mask_infos(model: ComponentModel) -> dict[str, ComponentsMaskInfo]:
    """Masks over components whose output is identically zero (`U` zeroed)."""
    for path in model.target_module_paths:
        components = model.components[path]
        assert isinstance(components, LinearComponents)
        with torch.no_grad():
            components.U.zero_()
    return make_mask_infos(
        {path: torch.ones(model.module_to_c[path]) for path in model.target_module_paths}
    )


def test_readouts_join_the_measurable_sites() -> None:
    model = _model()
    assert model.measurement_sites == ["attn", "mlp", "resid_post_attn", "resid_post_mlp"]
    assert select_sites(model.measurement_sites, ["resid_*"]) == [
        "resid_post_attn",
        "resid_post_mlp",
    ]


def test_clean_pass_caches_the_stream_itself() -> None:
    model = _model()
    x = torch.randn(3, 4)
    out: OutputWithCache = model(x, cache_type="input")

    target = _target(model)
    resid_post_attn = x + target.attn(x)
    torch.testing.assert_close(out.cache["resid_post_attn"], resid_post_attn)
    torch.testing.assert_close(
        out.cache["resid_post_mlp"], resid_post_attn + target.mlp(resid_post_attn)
    )
    # The decomposed sites' own pre-weight acts are untouched by the extra hooks.
    torch.testing.assert_close(out.cache["attn"], x)


def test_masked_stream_is_measured_against_the_clean_stream() -> None:
    """With both components silenced, every residual write vanishes — an exact target."""
    model = _model()
    x = torch.randn(3, 4)
    clean_cache = model(x, cache_type="input").cache
    mask_infos = _silenced_mask_infos(model)

    outputs = model.site_outputs(x, mask_infos)
    assert set(outputs) == {"attn", "mlp", "resid_post_attn", "resid_post_mlp"}
    torch.testing.assert_close(outputs["resid_post_attn"], x)
    torch.testing.assert_close(outputs["resid_post_mlp"], x)

    sites = ["resid_post_attn", "resid_post_mlp"]
    targets = clean_site_outputs(model, clean_cache, sites)
    for site in sites:
        torch.testing.assert_close(targets[site], clean_cache[site])

    errors = site_squared_errors(outputs, targets, mask_infos)
    for site in sites:
        sq_err, sq_target = errors[site]
        torch.testing.assert_close(sq_err, (x - clean_cache[site]).pow(2).sum())
        torch.testing.assert_close(sq_target, clean_cache[site].float().pow(2).sum())


def test_readout_error_covers_every_position() -> None:
    """A readout has no routing mask: attention spreads error to unrouted positions."""
    model = _model()
    x = torch.randn(3, 4)
    clean_cache = model(x, cache_type="input").cache
    mask_infos = _silenced_mask_infos(model)
    routing = torch.tensor([True, False, False])
    for info in mask_infos.values():
        info.routing_mask = routing

    outputs = model.site_outputs(x, mask_infos)
    targets = clean_site_outputs(model, clean_cache, ["attn", "resid_post_mlp"])
    errors = site_squared_errors(outputs, targets, mask_infos)

    # The decomposed site sees only the one routed position...
    _, sq_target_routed = errors["attn"]
    torch.testing.assert_close(sq_target_routed, targets["attn"][routing].float().pow(2).sum())
    # ...while the readout weighs the whole stream.
    _, sq_target_all = errors["resid_post_mlp"]
    torch.testing.assert_close(sq_target_all, clean_cache["resid_post_mlp"].float().pow(2).sum())


def test_readout_config_is_validated() -> None:
    with pytest.raises(AssertionError, match="collide"):
        _model({"attn": "post_attn_norm"})
    with pytest.raises(AssertionError, match="same module"):
        _model({"a": "post_attn_norm", "b": "post_attn_norm"})
    with pytest.raises(AssertionError, match="already cached"):
        _model({"a": "attn"})
    with pytest.raises(AttributeError):
        _model({"a": "nonexistent_module"})
