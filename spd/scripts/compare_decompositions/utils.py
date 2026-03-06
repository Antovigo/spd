"""Utilities for comparing attention components across SPD decompositions.

Loads two decomposition runs, computes CI on a prompt, identifies active components,
splits component weights by attention head, and computes pairwise cosine similarity.
"""

import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.figure import Figure
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from spd.models.component_model import CIOutputs, ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents


@dataclass
class DecompositionInfo:
    model: ComponentModel
    run_info: SPDRunInfo
    label: str


@dataclass
class ProjComparisonResult:
    layer_idx: int
    proj_type: str  # "q_proj", "k_proj", "v_proj", "o_proj"
    label_a: str
    label_b: str
    active_indices_a: list[int]
    active_indices_b: list[int]
    # sim_tensor[i, j] is (n_split, n_split) cosine sim between
    # head h1 of comp active_indices_a[i] vs head h2 of comp active_indices_b[j]
    sim_tensor: Float[Tensor, "Na Nb n_split_a n_split_b"]


def label_from_path(path: str) -> str:
    """Derive a label from a wandb/local path via SPDRunInfo.checkpoint_path."""
    run_info = SPDRunInfo.from_path(path)
    return run_info.checkpoint_path.parent.name


def load_decomposition(
    wandb_path: str, label: str | None = None, device: str = "cuda"
) -> DecompositionInfo:
    run_info = SPDRunInfo.from_path(wandb_path)
    if label is None:
        label = run_info.checkpoint_path.parent.name
    model = ComponentModel.from_run_info(run_info)
    model.to(device).eval()
    return DecompositionInfo(model=model, run_info=run_info, label=label)


def get_tokenizer(decomp: DecompositionInfo) -> PreTrainedTokenizerBase:
    tokenizer_name = decomp.run_info.config.tokenizer_name
    assert tokenizer_name is not None, "Config has no tokenizer_name"
    return AutoTokenizer.from_pretrained(tokenizer_name)


@torch.no_grad()
def compute_ci(
    model: ComponentModel,
    tokens: Tensor,
    sampling: str = "deterministic",
) -> dict[str, Float[Tensor, "... C"]]:
    out = model(tokens, cache_type="input")
    ci: CIOutputs = model.calc_causal_importances(
        pre_weight_acts=out.cache,
        sampling=sampling,
        detach_inputs=True,
    )
    return ci.lower_leaky


def get_active_component_indices(
    ci_dict: dict[str, Float[Tensor, "... C"]],
    threshold: float,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for module_path, ci in ci_dict.items():
        # Max CI across all non-component dims (batch, seq, etc.)
        max_ci = ci.amax(dim=tuple(range(ci.ndim - 1)))
        active_mask = max_ci > threshold
        result[module_path] = sorted(active_mask.nonzero(as_tuple=True)[0].tolist())
    return result


_ATTN_PATTERN = re.compile(r"h\.(\d+)\.attn\.(q_proj|k_proj|v_proj|o_proj)")


def get_attn_info(model: ComponentModel) -> tuple[int, int, int]:
    """Returns (n_heads, n_kv_heads, d_head) from the target model's first attention module."""
    for name, module in model.target_model.named_modules():
        if hasattr(module, "n_head") and hasattr(module, "n_key_value_heads") and hasattr(module, "head_dim"):
            return module.n_head, module.n_key_value_heads, module.head_dim
    raise ValueError("Could not find attention module with n_head/n_key_value_heads/head_dim")


def parse_attn_module_path(path: str) -> tuple[int, str] | None:
    m = _ATTN_PATTERN.search(path)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def get_component_weight(
    model: ComponentModel, module_path: str, idx: int
) -> Float[Tensor, "d_in d_out"]:
    comp = model.components[module_path]
    assert isinstance(comp, LinearComponents)
    # V[:, idx:idx+1] @ U[idx:idx+1, :] -> (d_in, d_out)
    return comp.V[:, idx : idx + 1] @ comp.U[idx : idx + 1, :]


def split_weight_by_head(
    weight: Float[Tensor, "d_in d_out"],
    n_split_heads: int,
    proj_type: str,
) -> Float[Tensor, "n_heads flat_dim"]:
    """Split a component weight matrix by attention head, returning flattened per-head vectors."""
    d_in, d_out = weight.shape
    if proj_type == "o_proj":
        # O proj: d_in = n_heads * d_head, split input dim
        d_head = d_in // n_split_heads
        # (n_heads, d_head, d_out)
        reshaped = weight.reshape(n_split_heads, d_head, d_out)
        return reshaped.reshape(n_split_heads, -1)
    else:
        # Q/K/V proj: d_out = n_split_heads * d_head, split output dim
        d_head = d_out // n_split_heads
        # (d_in, n_heads, d_head) -> (n_heads, d_in, d_head)
        reshaped = weight.reshape(d_in, n_split_heads, d_head).permute(1, 0, 2)
        return reshaped.reshape(n_split_heads, -1)


def compute_pairwise_cosine_sim(
    weights_a: Float[Tensor, "Na flat"],
    weights_b: Float[Tensor, "Nb flat"],
) -> Float[Tensor, "Na Nb"]:
    a_norm = F.normalize(weights_a, dim=-1)
    b_norm = F.normalize(weights_b, dim=-1)
    return a_norm @ b_norm.T


def compare_attention_heads(
    decomp_a: DecompositionInfo,
    decomp_b: DecompositionInfo,
    active_a: dict[str, list[int]],
    active_b: dict[str, list[int]],
) -> list[ProjComparisonResult]:
    n_heads, n_kv_heads, _d_head = get_attn_info(decomp_a.model)

    shared_modules = [p for p in active_a if p in active_b and parse_attn_module_path(p) is not None]

    results: list[ProjComparisonResult] = []
    for module_path in sorted(shared_modules):
        parsed = parse_attn_module_path(module_path)
        assert parsed is not None
        layer_idx, proj_type = parsed

        indices_a = active_a[module_path]
        indices_b = active_b[module_path]
        if not indices_a or not indices_b:
            continue

        n_split = n_heads if proj_type in ("q_proj", "o_proj") else n_kv_heads

        # Get per-component weights split by head: (N_active, n_split, flat_dim)
        stacked_a = torch.stack([
            split_weight_by_head(
                get_component_weight(decomp_a.model, module_path, idx), n_split, proj_type
            )
            for idx in indices_a
        ])
        stacked_b = torch.stack([
            split_weight_by_head(
                get_component_weight(decomp_b.model, module_path, idx), n_split, proj_type
            )
            for idx in indices_b
        ])

        # For each pair (i, j), compute (n_split, n_split) cosine sim
        na, nb = len(indices_a), len(indices_b)
        sim_tensor = torch.zeros(na, nb, n_split, n_split)
        for i in range(na):
            for j in range(nb):
                sim_tensor[i, j] = compute_pairwise_cosine_sim(
                    stacked_a[i], stacked_b[j]
                )

        results.append(
            ProjComparisonResult(
                layer_idx=layer_idx,
                proj_type=proj_type,
                label_a=decomp_a.label,
                label_b=decomp_b.label,
                active_indices_a=indices_a,
                active_indices_b=indices_b,
                sim_tensor=sim_tensor,
            )
        )

    return results


def plot_head_comparisons(result: ProjComparisonResult) -> Figure:
    """Plot a grid of heatmaps for a single ProjComparisonResult.

    Grid: rows = components from run A, cols = components from run B.
    Each subplot: (n_split, n_split) heatmap with head indices on both axes.
    """
    na = len(result.active_indices_a)
    nb = len(result.active_indices_b)
    n_split = result.sim_tensor.shape[2]

    fig, axes = plt.subplots(
        na, nb, figsize=(3 * nb + 1, 3 * na + 1), squeeze=False
    )

    im = None
    for i in range(na):
        for j in range(nb):
            ax = axes[i][j]
            sim_np = result.sim_tensor[i, j].cpu().numpy()
            im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(n_split))
            ax.set_xticklabels(range(n_split), fontsize=6)
            ax.set_yticks(range(n_split))
            ax.set_yticklabels(range(n_split), fontsize=6)

            if i == 0:
                ax.set_title(f"{result.label_b} C{result.active_indices_b[j]}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{result.label_a} C{result.active_indices_a[i]}", fontsize=8)
            if i == na - 1:
                ax.set_xlabel("Head", fontsize=7)

    fig.suptitle(f"Layer {result.layer_idx} {result.proj_type}", fontsize=14)
    fig.tight_layout()
    assert im is not None
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Cosine Similarity")
    return fig


def plot_comparison_summary(results: list[ProjComparisonResult]) -> Figure:
    """Bar chart of max |cos sim| and mean best-match per (layer, proj)."""
    labels: list[str] = []
    mean_best_matches: list[float] = []

    for r in results:
        # Collapse head dims: take max |cos sim| across heads for each (comp_a, comp_b) pair
        # sim_tensor: (Na, Nb, n_split, n_split)
        per_pair_max = r.sim_tensor.abs().amax(dim=(-2, -1))  # (Na, Nb)

        # Best match for each component in A (max over B), then average
        best_a = per_pair_max.max(dim=1).values.mean().item()
        best_b = per_pair_max.max(dim=0).values.mean().item()
        mean_best = (best_a + best_b) / 2

        labels.append(f"L{r.layer_idx}.{r.proj_type}")
        mean_best_matches.append(mean_best)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 4))
    x = np.arange(len(labels))
    ax.bar(x, mean_best_matches, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8, ha="right")
    ax.set_ylabel("Mean Best-Match |cos sim|")
    ax.set_title("Component Similarity Summary")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
