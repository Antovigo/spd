"""The fused hidden-acts auxiliary matches the standalone per-site MSE computation."""

import torch

from param_decomp.masks import AllLayersRouter, calc_stochastic_component_mask_info
from param_decomp.metrics.stochastic_hidden_acts_recon import calc_hidden_acts_mse
from param_decomp.metrics.stochastic_recon_subset import _stochastic_recon_subset_loss_update
from param_decomp.tests.metrics.fixtures import make_one_layer_component_model
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse


def test_fused_aux_matches_standalone_hidden_acts_mse():
    torch.manual_seed(0)
    fc_weight = torch.randn(3, 4)
    model = make_one_layer_component_model(weight=fc_weight, C=2)
    batch = torch.randn(5, 4)
    target_out = model(batch)
    ci = {"fc": torch.rand(5, 2)}

    x = model(batch, cache_type="input").cache["fc"]
    targets = {"fc": torch.nn.functional.linear(x, model.target_weight("fc"))}

    torch.manual_seed(1)
    _, _, sum_sq_err, n_elems = _stochastic_recon_subset_loss_update(
        model=model,
        sampling="continuous",
        n_mask_samples=1,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=None,
        router=AllLayersRouter(),
        reconstruction_loss=recon_loss_mse,
        hidden_acts_targets=targets,
    )

    torch.manual_seed(1)
    mask_infos = calc_stochastic_component_mask_info(
        causal_importances=ci,
        component_mask_sampling="continuous",
        weight_deltas=None,
        router=AllLayersRouter(),
    )
    clean_site_outputs = model(batch, cache_type="output").cache
    per_module, _ = calc_hidden_acts_mse(
        model=model, batch=batch, mask_infos=mask_infos, target_acts=clean_site_outputs
    )
    ref_mse, ref_n = per_module["fc"]

    assert n_elems == ref_n
    assert torch.allclose(sum_sq_err, ref_mse, atol=1e-5), (sum_sq_err, ref_mse)
