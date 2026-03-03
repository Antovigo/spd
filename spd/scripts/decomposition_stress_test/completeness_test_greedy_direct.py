"""Test whether active components of an SPD decomposition fully capture model behavior.

Uses greedy coordinate descent with random restarts to optimize binary masks directly,
avoiding the binarization gap of PGD-based optimization.

Given an input, identifies which components are active (CI > threshold), then adversarially
searches for component masks that maximize divergence between model outputs with and without
the "delta component" (everything NOT captured by the active components). A high completeness
loss means the active components are insufficient.

Usage:
    # Toy model (TMS/ResidMLP) — input is a feature dimension index:
    python -m spd.scripts.decomposition_stress_test.completeness_test_greedy_direct <model_path> 3

    # Language model — input is a prompt string:
    python -m spd.scripts.decomposition_stress_test.completeness_test_greedy_direct <model_path> "Once upon a time"
"""

import argparse
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.experiments.resid_mlp.models import ResidMLP
from spd.experiments.tms.models import TMSModel
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.utils.general_utils import calc_kl_divergence_lm, calc_sum_recon_loss_lm


def detect_model_type(output_loss_type: Literal["mse", "kl"]) -> Literal["toy", "lm"]:
    match output_loss_type:
        case "mse":
            return "toy"
        case "kl":
            return "lm"


def build_input_tensor(
    model: ComponentModel,
    input_str: str,
    model_type: Literal["toy", "lm"],
    tokenizer_name: str | None,
    device: torch.device,
) -> tuple[Tensor, PreTrainedTokenizerBase | None]:
    match model_type:
        case "toy":
            dim_idx = int(input_str)
            assert isinstance(model.target_model, ResidMLP | TMSModel)
            n_features = model.target_model.config.n_features
            assert 0 <= dim_idx < n_features, f"dim_idx {dim_idx} out of range [0, {n_features})"
            input_tensor = torch.zeros(1, n_features, device=device)
            input_tensor[0, dim_idx] = 0.75
            return input_tensor, None
        case "lm":
            assert tokenizer_name is not None, "tokenizer_name required for LM models"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            assert isinstance(tokenizer, PreTrainedTokenizerBase)
            tokens = tokenizer.encode(input_str, return_tensors="pt")
            assert isinstance(tokens, Tensor)
            return tokens.to(device), tokenizer


def compute_ci(model: ComponentModel, input_tensor: Tensor) -> dict[str, Float[Tensor, "... C"]]:
    output_with_cache = model(input_tensor, cache_type="input")
    ci = model.calc_causal_importances(
        pre_weight_acts=output_with_cache.cache,
        sampling="continuous",
    ).upper_leaky
    return ci


def get_active_masks(
    ci: dict[str, Float[Tensor, "... C"]], ci_thr: float
) -> dict[str, Float[Tensor, "... C"]]:
    return {module: (ci_vals > ci_thr).float() for module, ci_vals in ci.items()}


def eval_loss(
    model: ComponentModel,
    input_tensor: Tensor,
    sources: dict[str, Tensor],
    active: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    loss_type: Literal["mse", "kl"],
) -> tuple[Float[Tensor, ""], Tensor, Tensor]:
    mask_no_delta = {m: active[m] * sources[m] for m in active}
    mask_with_delta = {m: active[m] * sources[m] + (1 - active[m]) for m in active}

    wdam: dict[str, WeightDeltaAndMask] = {
        m: (weight_deltas[m], torch.ones(active[m].shape[:-1], device=active[m].device))
        for m in active
    }

    out_no_delta = model(input_tensor, mask_infos=make_mask_infos(mask_no_delta))
    out_with_delta = model(
        input_tensor, mask_infos=make_mask_infos(mask_with_delta, weight_deltas_and_masks=wdam)
    )

    n_examples = out_no_delta.shape[:-1].numel() if loss_type == "kl" else out_no_delta.numel()
    loss = (
        calc_sum_recon_loss_lm(pred=out_no_delta, target=out_with_delta, loss_type=loss_type)
        / n_examples
    )
    return loss, out_no_delta, out_with_delta


def _per_element_loss_direct(
    out_no_delta: Tensor,
    out_with_delta: Tensor,
    loss_type: Literal["mse", "kl"],
    n_batch: int,
) -> Float[Tensor, " n_batch"]:
    """Compute per-batch-element completeness loss (direct variant)."""
    match loss_type:
        case "mse":
            return ((out_no_delta - out_with_delta) ** 2).flatten(1).mean(1)
        case "kl":
            kl_per_pos = calc_kl_divergence_lm(
                pred=out_no_delta, target=out_with_delta, reduce=False
            )
            return kl_per_pos.reshape(n_batch, -1).mean(1)


def run_greedy(
    model: ComponentModel,
    input_tensor: Tensor,
    active: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    loss_type: Literal["mse", "kl"],
    n_restarts: int,
) -> tuple[dict[str, Tensor], Float[Tensor, ""], Tensor, Tensor]:
    """Run greedy coordinate descent with random restarts to maximize completeness loss.

    Each sweep batches all single-bit flips into one forward pass instead of evaluating
    each flip individually.

    Returns (best_sources, best_loss, out_no_delta, out_with_delta).
    """
    modules = list(active.keys())
    # Only iterate over coordinates where the active mask is non-zero
    coords: list[tuple[str, int]] = []
    for m in modules:
        flat_active = active[m].reshape(-1)
        for i in range(flat_active.numel()):
            if flat_active[i] > 0:
                coords.append((m, i))

    n_coords = len(coords)
    print(f"  {n_coords} active coordinates")

    best_loss_val = float("-inf")
    best_final_loss: Float[Tensor, ""] | None = None
    best_sources: dict[str, Tensor] | None = None
    best_out_no_delta: Tensor | None = None
    best_out_with_delta: Tensor | None = None

    for restart in range(n_restarts):
        sources: dict[str, Tensor] = {
            m: torch.randint(0, 2, active[m].shape, device=active[m].device).float()
            for m in modules
        }

        with torch.no_grad():
            current_loss, _, _ = eval_loss(
                model, input_tensor, sources, active, weight_deltas, loss_type
            )

        n_sweeps = 0
        while n_coords > 0:
            n_sweeps += 1

            # Batch all single-bit flips: create n_coords copies with one bit flipped each
            batched_sources = {
                m: sources[m].expand(n_coords, *sources[m].shape[1:]).clone()
                for m in modules
            }
            for i, (m, flat_idx) in enumerate(coords):
                batched_sources[m][i].reshape(-1)[flat_idx] = (
                    1.0 - batched_sources[m][i].reshape(-1)[flat_idx]
                )

            batched_active = {
                m: active[m].expand(n_coords, *active[m].shape[1:]) for m in modules
            }
            mask_no_delta = {m: batched_active[m] * batched_sources[m] for m in modules}
            mask_with_delta = {
                m: batched_active[m] * batched_sources[m] + (1 - batched_active[m])
                for m in modules
            }
            wdam: dict[str, WeightDeltaAndMask] = {
                m: (
                    weight_deltas[m],
                    torch.ones(batched_active[m].shape[:-1], device=active[m].device),
                )
                for m in modules
            }
            batched_input = input_tensor.expand(n_coords, *input_tensor.shape[1:])

            with torch.no_grad():
                out_no = model(batched_input, mask_infos=make_mask_infos(mask_no_delta))
                out_with = model(
                    batched_input,
                    mask_infos=make_mask_infos(mask_with_delta, weight_deltas_and_masks=wdam),
                )

            per_coord_loss = _per_element_loss_direct(out_no, out_with, loss_type, n_coords)

            # Apply all improving flips
            applied_flips: list[tuple[str, int]] = []
            for i, (m, flat_idx) in enumerate(coords):
                if per_coord_loss[i] > current_loss:
                    sources[m].reshape(-1)[flat_idx] = 1.0 - sources[m].reshape(-1)[flat_idx]
                    applied_flips.append((m, flat_idx))

            if not applied_flips:
                break

            # Re-evaluate actual loss after all flips applied together
            with torch.no_grad():
                new_loss, _, _ = eval_loss(
                    model, input_tensor, sources, active, weight_deltas, loss_type
                )

            if new_loss > current_loss:
                current_loss = new_loss
            else:
                for m, flat_idx in applied_flips:
                    sources[m].reshape(-1)[flat_idx] = 1.0 - sources[m].reshape(-1)[flat_idx]
                break

        print(
            f"  Restart {restart + 1}/{n_restarts}:"
            f" loss = {current_loss.item():.6f} ({n_sweeps} sweeps)"
        )

        if current_loss.item() > best_loss_val:
            best_loss_val = current_loss.item()
            best_sources = {m: sources[m].clone() for m in modules}
            with torch.no_grad():
                best_final_loss, best_out_no_delta, best_out_with_delta = eval_loss(
                    model, input_tensor, sources, active, weight_deltas, loss_type
                )

    assert best_final_loss is not None
    assert best_sources is not None
    assert best_out_no_delta is not None
    assert best_out_with_delta is not None
    return best_sources, best_final_loss, best_out_no_delta, best_out_with_delta


def print_results(
    active: dict[str, Float[Tensor, "... C"]],
    binary_sources: dict[str, Tensor],
    final_loss: Float[Tensor, ""],
    model_type: Literal["toy", "lm"],
    ci_thr: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Completeness Test Results (CI threshold = {ci_thr})")
    print(f"{'=' * 60}")
    print(f"Final completeness loss (binary masks): {final_loss.item():.6f}")

    for module in active:
        active_mask = active[module]
        binary = binary_sources[module]

        print(f"\nModule: {module}")

        match model_type:
            case "toy":
                active_indices = active_mask[0].nonzero(as_tuple=True)[0].tolist()
                unmasked = ((active_mask[0] * binary[0]) > 0.5).nonzero(as_tuple=True)[0].tolist()
                print(f"  Active components: {active_indices}, unmasked: {unmasked}")
            case "lm":
                for pos in range(active_mask.shape[1]):
                    active_indices = active_mask[0, pos].nonzero(as_tuple=True)[0].tolist()
                    unmasked = (
                        ((active_mask[0, pos] * binary[0, pos]) > 0.5)
                        .nonzero(as_tuple=True)[0]
                        .tolist()
                    )
                    print(
                        f"  Position {pos}: active components: {active_indices},"
                        f" unmasked: {unmasked}"
                    )


def print_divergences(
    original: Tensor,
    out_no_delta: Tensor,
    out_with_delta: Tensor,
    loss_type: Literal["mse", "kl"],
) -> None:
    print(f"\n{'=' * 60}")
    print("Divergence from Original Model Output")
    print(f"{'=' * 60}")

    match loss_type:
        case "mse":
            mse_no_delta = ((original - out_no_delta) ** 2).mean().item()
            mse_with_delta = ((original - out_with_delta) ** 2).mean().item()
            print(f"  MSE(original, no_delta)   = {mse_no_delta:.6f}")
            print(f"  MSE(original, with_delta) = {mse_with_delta:.6f}")
        case "kl":
            kl_no_delta = calc_kl_divergence_lm(
                pred=out_no_delta, target=original, reduce=True
            ).item()
            kl_with_delta = calc_kl_divergence_lm(
                pred=out_with_delta, target=original, reduce=True
            ).item()
            print(f"  KL(original, no_delta)    = {kl_no_delta:.6f}")
            print(f"  KL(original, with_delta)  = {kl_with_delta:.6f}")


def print_top_outputs(
    original: Tensor,
    out_no_delta: Tensor,
    out_with_delta: Tensor,
    model_type: Literal["toy", "lm"],
    tokenizer: PreTrainedTokenizerBase | None,
    k: int = 10,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Top-{k} Output Dimensions/Tokens")
    print(f"{'=' * 60}")

    match model_type:
        case "toy":
            _, indices = torch.topk(original[0].abs(), k=min(k, original.shape[-1]))
            print(f"  {'Dim':<6} {'Original':>10} {'No-Delta':>10} {'With-Delta':>12}")
            print(f"  {'-' * 42}")
            for idx in indices:
                i = int(idx)
                print(
                    f"  {i:<6} {original[0, i].item():>10.4f}"
                    f" {out_no_delta[0, i].item():>10.4f}"
                    f" {out_with_delta[0, i].item():>12.4f}"
                )
        case "lm":
            assert tokenizer is not None
            probs_orig = torch.softmax(original[0, -1], dim=-1)
            probs_no = torch.softmax(out_no_delta[0, -1], dim=-1)
            probs_with = torch.softmax(out_with_delta[0, -1], dim=-1)

            _, indices = torch.topk(probs_orig, k=min(k, probs_orig.shape[-1]))
            print(f"  {'Token':<20} {'Original':>10} {'No-Delta':>10} {'With-Delta':>12}")
            print(f"  {'-' * 56}")
            for idx in indices:
                i = int(idx)
                token_str = repr(tokenizer.decode([i]))
                print(
                    f"  {token_str:<20} {probs_orig[i].item():>10.4f}"
                    f" {probs_no[i].item():>10.4f}"
                    f" {probs_with[i].item():>12.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="SPD Completeness Test (Greedy)")
    parser.add_argument("model_path", help="Path to decomposed model (wandb or local)")
    parser.add_argument("input", help="Prompt string (LM) or dimension index (toy model)")
    parser.add_argument(
        "--ci-thr", type=float, default=0.01, help="CI threshold for active components"
    )
    parser.add_argument("--n-restarts", type=int, default=10, help="Number of random restarts")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top output dims to show")
    parser.add_argument("--device", default="cpu", help="Device to run on")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load model and config
    print(f"Loading model from {args.model_path}...")
    run_info = SPDRunInfo.from_path(args.model_path)
    config = run_info.config
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    model_type = detect_model_type(config.output_loss_type)
    print(f"Model type: {model_type} (output_loss_type={config.output_loss_type})")

    # 2. Construct input tensor
    input_tensor, tokenizer = build_input_tensor(
        model, args.input, model_type, config.tokenizer_name, device
    )

    # 3. Compute CI values and identify active components
    ci = compute_ci(model, input_tensor)
    active = get_active_masks(ci, args.ci_thr)
    n_active = sum(int(mask.any(dim=-1).sum()) for mask in active.values())
    print(f"Computing causal importances... {n_active} active components")

    # 4. Compute weight deltas
    weight_deltas = model.calc_weight_deltas()

    # 5. Compute original model output (no masks = target model behavior)
    with torch.no_grad():
        original_output = model(input_tensor)

    # 6. Greedy coordinate descent optimization
    print(f"\nRunning greedy coordinate descent ({args.n_restarts} restarts)...")
    binary_sources, final_loss, out_no_delta, out_with_delta = run_greedy(
        model=model,
        input_tensor=input_tensor,
        active=active,
        weight_deltas=weight_deltas,
        loss_type=config.output_loss_type,
        n_restarts=args.n_restarts,
    )

    # 7. Print results
    print_results(active, binary_sources, final_loss, model_type, args.ci_thr)
    print_divergences(original_output, out_no_delta, out_with_delta, config.output_loss_type)
    print_top_outputs(
        original_output, out_no_delta, out_with_delta, model_type, tokenizer, args.top_n
    )


if __name__ == "__main__":
    main()
