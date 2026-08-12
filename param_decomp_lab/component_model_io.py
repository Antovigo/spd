"""Lab-side `ComponentModel` helpers for postprocessing, the app, and harvest.

Rebuilds a `ComponentModel` from a saved checkpoint, and reads per-component activations
from cached pre-weight acts.
"""

from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_fns import (
    CiConfig,
    GlobalCiConfig,
    LayerwiseCiConfig,
)
from param_decomp.component_model import ComponentModel
from param_decomp.configs import PDConfig
from param_decomp.decomposition_targets import (
    DecompositionTargetConfig,
    insert_identity_operations_,
    resolve_decomposition_targets,
)


def _validate_checkpoint_ci_config_compatibility(
    state_dict: dict[str, Tensor], ci_config: CiConfig
) -> None:
    """Assert the checkpoint's CI weight keys match the layerwise/global mode in `ci_config`."""
    has_layerwise_ci_fns = any(k.startswith("ci_fn._ci_fns") for k in state_dict)
    has_global_ci_fn = any(k.startswith("ci_fn._global_ci_fn") for k in state_dict)

    match ci_config:
        case LayerwiseCiConfig():
            assert has_layerwise_ci_fns, (
                f"Config specifies layerwise CI but checkpoint has no ci_fn._ci_fns keys "
                f"(has ci_fn._global_ci_fn: {has_global_ci_fn})"
            )
        case GlobalCiConfig():
            assert has_global_ci_fn, (
                f"Config specifies global CI but checkpoint has no ci_fn._global_ci_fn keys "
                f"(has ci_fn._ci_fns: {has_layerwise_ci_fns})"
            )


def _validate_checkpoint_trunk_sharing(state_dict: dict[str, Tensor], shared_trunk: bool) -> None:
    """Assert the checkpoint's two CI nets agree with `shared_trunk` about their trunk.

    A shared-trunk state dict stores the one trunk under both nets' names, so the keys
    alone can't distinguish the two topologies — but the *values* can. Loading an
    independent-net checkpoint under `dual_hidden_ci_shared_trunk` would otherwise
    succeed silently, with the hidden net's trunk landing on the shared parameters and
    the output net quietly scoring off the wrong representation.
    """
    if not shared_trunk:
        return
    trunk_keys = [
        k for k in state_dict if k.startswith("ci_fn._global_ci_fn.") and "_output_head" not in k
    ]
    assert trunk_keys, "shared trunk configured but the checkpoint has no global CI trunk"
    mismatched = [
        k
        for k in trunk_keys
        if not torch.equal(state_dict[k], state_dict[k.replace("ci_fn.", "ci_fn_hidden.", 1)])
    ]
    assert not mismatched, (
        "dual_hidden_ci_shared_trunk is set but this checkpoint's two CI nets have "
        f"different trunk weights ({len(mismatched)}/{len(trunk_keys)} tensors differ) — "
        "it was trained with independent trunks, so loading it here would silently "
        "overwrite the output net's trunk with the hidden net's"
    )


def load_component_model(
    pd_config: PDConfig,
    checkpoint_path: Path,
    target_model: nn.Module,
    run_batch: RunBatch,
) -> ComponentModel:
    """Rebuild a `ComponentModel` from a saved PD checkpoint plus a caller-supplied target.

    The caller owns target loading (HF, in-repo pretrain, custom); everything else
    needed to reconstruct the model comes from `pd_config`.
    """
    target_model.eval()
    target_model.requires_grad_(False)

    identity_targets = pd_config.identity_decomposition_targets
    if identity_targets is not None:
        insert_identity_operations_(target_model, identity_decomposition_targets=identity_targets)

    all_targets = list(pd_config.decomposition_targets)
    if identity_targets is not None:
        for target in identity_targets:
            all_targets.append(
                DecompositionTargetConfig(
                    module_pattern=f"{target.module_pattern}.pre_identity", C=target.C
                )
            )
    resolved_targets = resolve_decomposition_targets(target_model, all_targets)

    comp_model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        decomposition_targets=resolved_targets,
        ci_config=pd_config.ci_config,
        sigmoid_type=pd_config.sigmoid_type,
        dual_hidden_ci=pd_config.dual_hidden_ci,
        dual_hidden_ci_shared_trunk=pd_config.dual_hidden_ci_shared_trunk,
        hidden_ci_floor=pd_config.hidden_ci_floor,
        output_ci_cap=pd_config.output_ci_cap,
        hidden_readout_sites=pd_config.hidden_readout_sites,
    )

    comp_model_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_ci_config_compatibility(comp_model_weights, pd_config.ci_config)
    _validate_checkpoint_trunk_sharing(comp_model_weights, pd_config.dual_hidden_ci_shared_trunk)
    comp_model.load_state_dict(comp_model_weights)

    if pd_config.tied_weights is not None:
        for src_name, tgt_name in pd_config.tied_weights:
            tgt = comp_model.components[tgt_name]
            src = comp_model.components[src_name]
            assert tgt is not None and src is not None, (
                f"Cannot tie weights between {src_name} and {tgt_name} - one or both are None"
            )
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    return comp_model


def get_all_component_acts(
    model: ComponentModel,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "..."]],
) -> dict[str, Float[Tensor, "... C"]]:
    """Per-component activations `V^T @ x` for every decomposed layer.

    Layers in `pre_weight_acts` with no matching entry in `model.components` are skipped
    silently.
    """
    return {
        layer: model.components[layer].get_component_acts(acts)
        for layer, acts in pre_weight_acts.items()
        if layer in model.components
    }
