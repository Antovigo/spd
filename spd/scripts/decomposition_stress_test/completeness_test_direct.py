"""Test whether active components of an SPD decomposition fully capture model behavior.

Given an input, identifies which components are active (CI > threshold), then adversarially
searches for component masks that maximize divergence between model outputs with and without
the "delta component" (everything NOT captured by the active components). A high completeness
loss means the active components are insufficient.

Usage:
    # Toy model (TMS/ResidMLP) — input is a feature dimension index:
    python -m spd.scripts.decomposition_stress_test.completeness_test_direct <model_path> 3

    # Language model — input is a prompt string:
    python -m spd.scripts.decomposition_stress_test.completeness_test_direct <model_path> "Once upon a time"
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


def run_pgd(
    model: ComponentModel,
    input_tensor: Tensor,
    active: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    loss_type: Literal["mse", "kl"],
    n_steps: int,
    step_size: float,
) -> tuple[dict[str, Tensor], Float[Tensor, ""], Tensor, Tensor]:
    """Run PGD to maximize completeness loss.

    Returns (optimized_sources, final_loss, out_no_delta, out_with_delta).
    """
    # Initialize sources randomly
    sources: dict[str, Tensor] = {}
    for module, active_mask in active.items():
        sources[module] = torch.rand_like(active_mask).requires_grad_(True)

    for step in range(n_steps):
        # Construct masks for both forward passes
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
        sum_loss = calc_sum_recon_loss_lm(
            pred=out_no_delta, target=out_with_delta, loss_type=loss_type
        )
        loss = sum_loss / n_examples

        grads = torch.autograd.grad(loss, list(sources.values()))
        with torch.no_grad():
            for (m, src), grad in zip(sources.items(), grads, strict=True):
                sources[m] = (src + step_size * grad.sign()).clamp(0.0, 1.0).requires_grad_(True)

        if step % 20 == 0 or step == n_steps - 1:
            print(f"  PGD step {step:4d}/{n_steps}: completeness loss = {loss.item():.6f}")

    # Final evaluation with binary masks
    with torch.no_grad():
        binary_sources = {m: (sources[m] > 0.5).float() for m in sources}
        mask_no_delta = {m: active[m] * binary_sources[m] for m in active}
        mask_with_delta = {m: active[m] * binary_sources[m] + (1 - active[m]) for m in active}

        wdam = {
            m: (weight_deltas[m], torch.ones(active[m].shape[:-1], device=active[m].device))
            for m in active
        }

        out_no_delta = model(input_tensor, mask_infos=make_mask_infos(mask_no_delta))
        out_with_delta = model(
            input_tensor, mask_infos=make_mask_infos(mask_with_delta, weight_deltas_and_masks=wdam)
        )

        n_examples = out_no_delta.shape[:-1].numel() if loss_type == "kl" else out_no_delta.numel()
        final_loss = (
            calc_sum_recon_loss_lm(pred=out_no_delta, target=out_with_delta, loss_type=loss_type)
            / n_examples
        )

    return binary_sources, final_loss, out_no_delta, out_with_delta


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
    parser = argparse.ArgumentParser(description="SPD Completeness Test")
    parser.add_argument("model_path", help="Path to decomposed model (wandb or local)")
    parser.add_argument("input", help="Prompt string (LM) or dimension index (toy model)")
    parser.add_argument(
        "--ci-thr", type=float, default=0.01, help="CI threshold for active components"
    )
    parser.add_argument("--n-steps", type=int, default=100, help="Number of PGD steps")
    parser.add_argument("--step-size", type=float, default=0.01, help="PGD step size")
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

    # 5. Compute weight deltas
    weight_deltas = model.calc_weight_deltas()

    # 6. Compute original model output (no masks = target model behavior)
    with torch.no_grad():
        original_output = model(input_tensor)

    # 7. PGD adversarial optimization
    print(f"\nRunning PGD ({args.n_steps} steps, step_size={args.step_size})...")
    binary_sources, final_loss, out_no_delta, out_with_delta = run_pgd(
        model=model,
        input_tensor=input_tensor,
        active=active,
        weight_deltas=weight_deltas,
        loss_type=config.output_loss_type,
        n_steps=args.n_steps,
        step_size=args.step_size,
    )

    # 8. Print results
    print_results(active, binary_sources, final_loss, model_type, args.ci_thr)
    print_divergences(original_output, out_no_delta, out_with_delta, config.output_loss_type)
    print_top_outputs(
        original_output, out_no_delta, out_with_delta, model_type, tokenizer, args.top_n
    )


if __name__ == "__main__":
    main()
