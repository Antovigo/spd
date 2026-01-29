from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import (
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LossMetricConfigType,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SamplingType,
    StochasticHiddenActsReconLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    UnmaskedReconLossConfig,
)
from spd.metrics import (
    ci_masked_recon_layerwise_loss,
    ci_masked_recon_loss,
    ci_masked_recon_subset_loss,
    faithfulness_loss,
    importance_minimality_loss,
    pgd_recon_layerwise_loss,
    pgd_recon_loss,
    pgd_recon_subset_loss,
    stochastic_hidden_acts_recon_loss,
    stochastic_recon_layerwise_loss,
    stochastic_recon_loss,
    stochastic_recon_subset_loss,
    unmasked_recon_loss,
)
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.general_utils import get_linear_annealed_value


def compute_total_loss(
    loss_metric_configs: list[LossMetricConfigType],
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    ci: CIOutputs,
    target_out: Tensor,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    current_frac_of_training: float,
    sampling: SamplingType,
    use_delta_component: bool,
    n_mask_samples: int,
    output_loss_type: Literal["mse", "kl", "mem"],
    force_delta_mask_one: bool = False,
) -> tuple[Float[Tensor, ""], dict[str, float], dict[str, float]]:
    """Compute weighted total loss and per-term raw values using new loss primitives.

    Returns (total, terms_dict, scheduled_params). terms_dict contains raw per-term values
    (no coeffs) and a weighted total. scheduled_params contains scheduled values for logging.
    """
    total = torch.tensor(0.0, device=batch.device)
    terms: dict[str, float] = {}
    scheduled_params: dict[str, float] = {}

    for cfg in loss_metric_configs:
        assert cfg.coeff is not None, "All loss metric configs must have a coeff"
        coeff = cfg.coeff
        match cfg:
            case FaithfulnessLossConfig():
                loss = faithfulness_loss(weight_deltas=weight_deltas)
            case ImportanceMinimalityLossConfig():
                if cfg.coeff_anneal_final_value is not None and cfg.coeff_anneal_start_frac < 1.0:
                    coeff = get_linear_annealed_value(
                        current_frac_of_training=current_frac_of_training,
                        initial_value=cfg.coeff,
                        anneal_start_frac=cfg.coeff_anneal_start_frac,
                        anneal_final_value=cfg.coeff_anneal_final_value,
                        anneal_end_frac=cfg.coeff_anneal_end_frac,
                    )
                    scheduled_params["scheduled/ImportanceMinimalityLoss_coeff"] = coeff

                if cfg.p_anneal_final_p is not None and cfg.p_anneal_start_frac < 1.0:
                    pnorm = get_linear_annealed_value(
                        current_frac_of_training=current_frac_of_training,
                        initial_value=cfg.pnorm,
                        anneal_start_frac=cfg.p_anneal_start_frac,
                        anneal_final_value=cfg.p_anneal_final_p,
                        anneal_end_frac=cfg.p_anneal_end_frac,
                    )
                    scheduled_params["scheduled/ImportanceMinimalityLoss_pnorm"] = pnorm

                loss = importance_minimality_loss(
                    ci_upper_leaky=ci.upper_leaky,
                    current_frac_of_training=current_frac_of_training,
                    pnorm=cfg.pnorm,
                    beta=cfg.beta,
                    eps=cfg.eps,
                    p_anneal_start_frac=cfg.p_anneal_start_frac,
                    p_anneal_final_p=cfg.p_anneal_final_p,
                    p_anneal_end_frac=cfg.p_anneal_end_frac,
                )
            case UnmaskedReconLossConfig():
                loss = unmasked_recon_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                )
            case CIMaskedReconSubsetLossConfig():
                loss = ci_masked_recon_subset_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    routing=cfg.routing,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case CIMaskedReconLayerwiseLossConfig():
                loss = ci_masked_recon_layerwise_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case CIMaskedReconLossConfig():
                loss = ci_masked_recon_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case StochasticReconLayerwiseLossConfig():
                loss = stochastic_recon_layerwise_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case StochasticReconLossConfig():
                loss = stochastic_recon_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case StochasticReconSubsetLossConfig():
                loss = stochastic_recon_subset_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    routing=cfg.routing,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case PGDReconLossConfig():
                loss = pgd_recon_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    pgd_config=cfg,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case PGDReconSubsetLossConfig():
                loss = pgd_recon_subset_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    pgd_config=cfg,
                    routing=cfg.routing,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case PGDReconLayerwiseLossConfig():
                loss = pgd_recon_layerwise_loss(
                    model=model,
                    output_loss_type=output_loss_type,
                    batch=batch,
                    target_out=target_out,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                    pgd_config=cfg,
                    force_delta_mask_one=force_delta_mask_one,
                )
            case StochasticHiddenActsReconLossConfig():
                loss = stochastic_hidden_acts_recon_loss(
                    model=model,
                    sampling=sampling,
                    n_mask_samples=n_mask_samples,
                    batch=batch,
                    pre_weight_acts=pre_weight_acts,
                    ci=ci.lower_leaky,
                    weight_deltas=weight_deltas if use_delta_component else None,
                )

        terms[f"loss/{cfg.classname}"] = loss.item()

        total = total + coeff * loss

    terms["loss/total"] = total.item()

    return total, terms, scheduled_params
