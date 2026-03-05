"""Match components between two SPD decomposition models.

Reports per-component similarity metrics in a TSV file, enabling comparison of
decompositions from different seeds/hyperparameters to see if they discover the
same subcomponents.

Usage:
    python3 spd/scripts/match_components.py \
        "wandb:goodfire/spd/runs/run_id_1" \
        "wandb:goodfire/spd/runs/run_id_2"
"""

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from tqdm import tqdm

from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo


@dataclass
class MatchedComponentRow:
    matrix: str
    index_1: int
    index_2: int
    rank_one_cosine_sim: float
    rank_one_pvalue: float
    rank_one_sim: float
    u_cosine_sim: float
    u_pvalue: float
    v_cosine_sim: float
    v_pvalue: float
    weight_norm_1: float
    weight_norm_2: float


COLUMNS = [
    "matrix",
    "index_1",
    "index_2",
    "rank_one_cosine_sim",
    "rank_one_pvalue",
    "rank_one_sim",
    "u_cosine_sim",
    "u_pvalue",
    "v_cosine_sim",
    "v_pvalue",
    "weight_norm_1",
    "weight_norm_2",
]


def cosine_sim_pvalue(cos_sim: float, dim: int) -> float:
    """Two-tailed p-value: P(|random_cos_sim| >= |cos_sim|) using normal approximation.

    For random unit vectors in R^d, cosine similarity ~ Normal(0, 1/d) for large d.
    """
    return float(2 * norm.sf(abs(cos_sim) * math.sqrt(dim)))


def _common_prefix(names: list[str]) -> str:
    """Find the longest common prefix ending at a '.' boundary."""
    if len(names) <= 1:
        return ""
    prefix = os.path.commonprefix(names)
    # Trim to last '.' boundary so we don't cut mid-word
    dot_idx = prefix.rfind(".")
    if dot_idx == -1:
        return ""
    return prefix[: dot_idx + 1]


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def match_components(
    model_1_path: str,
    model_2_path: str,
    output_path: str,
) -> None:
    """Match components between two SPD models and write per-component metrics to TSV.

    Args:
        model_1_path: Path to first model (wandb: or local path)
        model_2_path: Path to second model (wandb: or local path)
        output_path: Path for output TSV file
    """
    logger.info(f"Loading model 1 from: {model_1_path}")
    run_info_1 = SPDRunInfo.from_path(model_1_path)
    model_1 = ComponentModel.from_run_info(run_info_1)
    model_1.eval()

    logger.info(f"Loading model 2 from: {model_2_path}")
    run_info_2 = SPDRunInfo.from_path(model_2_path)
    model_2 = ComponentModel.from_run_info(run_info_2)
    model_2.eval()

    shared_layers = [name for name in model_1.components if name in model_2.components]
    assert shared_layers, "No shared layers between models"
    logger.info(f"Shared layers: {shared_layers}")

    prefix = _common_prefix(shared_layers)
    if prefix:
        logger.info(f"Stripping common prefix: {prefix!r}")

    rows: list[MatchedComponentRow] = []

    for layer_name in tqdm(shared_layers, desc="Matching layers"):
        comp_1 = model_1.components[layer_name]
        comp_2 = model_2.components[layer_name]

        u_1 = comp_1.U.detach()  # (C1, d_out)
        v_1 = comp_1.V.detach()  # (d_in, C1)
        u_2 = comp_2.U.detach()  # (C2, d_out)
        v_2 = comp_2.V.detach()  # (d_in, C2)

        c_1, d_out = u_1.shape
        d_in, c_2 = v_2.shape
        assert v_1.shape == (d_in, c_1)
        assert u_2.shape == (c_2, d_out)

        c_max = max(c_1, c_2)

        # Pad with zeros if component counts differ
        if c_max > c_1:
            u_1 = F.pad(u_1, (0, 0, 0, c_max - c_1))  # (c_max, d_out)
            v_1 = F.pad(v_1, (0, c_max - c_1))  # (d_in, c_max)
        if c_max > c_2:
            u_2 = F.pad(u_2, (0, 0, 0, c_max - c_2))
            v_2 = F.pad(v_2, (0, c_max - c_2))

        # Factored computation: avoid materializing (C, d_in, d_out) tensors.
        # ||v1_i ⊗ u1_i||_F = ||v1_i|| * ||u1_i||, and
        # <v1_i ⊗ u1_i, v2_j ⊗ u2_j>_F = (v1_i · v2_j) * (u1_i · u2_j)
        u_norms_1_sq = (u_1**2).sum(dim=1)  # (c_max,)
        u_norms_2_sq = (u_2**2).sum(dim=1)  # (c_max,)
        v_norms_1_sq = (v_1**2).sum(dim=0)  # (c_max,)
        v_norms_2_sq = (v_2**2).sum(dim=0)  # (c_max,)

        # Rank-one norms: ||v_i ⊗ u_i||_F² = ||v_i||² * ||u_i||²
        ro_norms_1_sq = v_norms_1_sq * u_norms_1_sq  # (c_max,)
        ro_norms_2_sq = v_norms_2_sq * u_norms_2_sq  # (c_max,)

        # Cross dot products via factored form
        u_dots = u_1 @ u_2.T  # (c_max, c_max)
        v_dots = v_1.T @ v_2  # (c_max, c_max)
        ro_dots = u_dots * v_dots  # (c_max, c_max)

        # Frobenius distance: ||A - B||_F² = ||A||² + ||B||² - 2<A, B>
        dist_sq = ro_norms_1_sq.unsqueeze(1) + ro_norms_2_sq.unsqueeze(0) - 2 * ro_dots
        frob_dist_matrix = dist_sq.clamp(min=0).sqrt()  # (c_max, c_max)
        cost_matrix = frob_dist_matrix.cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Rank-one cosine similarity via factored form
        ro_norms_1 = ro_norms_1_sq.clamp(min=1e-12).sqrt()  # (c_max,)
        ro_norms_2 = ro_norms_2_sq.clamp(min=1e-12).sqrt()  # (c_max,)
        sim_matrix = ro_dots / (ro_norms_1.unsqueeze(1) * ro_norms_2.unsqueeze(0))

        short_name = layer_name.removeprefix(prefix)

        for i, j in zip(row_indices, col_indices, strict=True):
            raw_rank_one_sim = sim_matrix[i, j].item()

            # Sign canonicalization: if raw sim is negative, flip model 2's U and V signs
            abs_rank_one_sim = abs(raw_rank_one_sim)
            rank_one_dim = d_in * d_out
            sign = -1.0 if raw_rank_one_sim < 0 else 1.0
            u2_j = u_2[j] * sign
            v2_j = v_2[:, j] * sign

            # Magnitude-aware similarity: 1 - ||A - B||_F / (||A||_F + ||B||_F)
            norm_sum = ro_norms_1[i].item() + ro_norms_2[j].item()
            rank_one_sim = 1.0 - frob_dist_matrix[i, j].item() / norm_sum if norm_sum > 0 else 0.0

            u1_i = u_1[i]
            v1_i = v_1[:, i]

            u_cos_sim = F.cosine_similarity(u1_i.unsqueeze(0), u2_j.unsqueeze(0)).item()
            v_cos_sim = F.cosine_similarity(v1_i.unsqueeze(0), v2_j.unsqueeze(0)).item()

            weight_norm_1 = u1_i.norm().item() * v1_i.norm().item()
            weight_norm_2 = u_2[j].norm().item() * v_2[:, j].norm().item()  # original (unsigned)

            rows.append(
                MatchedComponentRow(
                    matrix=short_name,
                    index_1=i,
                    index_2=j,
                    rank_one_cosine_sim=abs_rank_one_sim,
                    rank_one_pvalue=cosine_sim_pvalue(abs_rank_one_sim, rank_one_dim),
                    rank_one_sim=rank_one_sim,
                    u_cosine_sim=u_cos_sim,
                    u_pvalue=cosine_sim_pvalue(u_cos_sim, d_out),
                    v_cosine_sim=v_cos_sim,
                    v_pvalue=cosine_sim_pvalue(v_cos_sim, d_in),
                    weight_norm_1=weight_norm_1,
                    weight_norm_2=weight_norm_2,
                )
            )

    # Sort by layer name, then by weight_norm_1 descending within each layer
    rows.sort(key=lambda r: (r.matrix, -r.weight_norm_1))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.matrix,
                    row.index_1,
                    row.index_2,
                    f"{row.rank_one_cosine_sim:.6f}",
                    f"{row.rank_one_pvalue:.6e}",
                    f"{row.rank_one_sim:.6f}",
                    f"{row.u_cosine_sim:.6f}",
                    f"{row.u_pvalue:.6e}",
                    f"{row.v_cosine_sim:.6f}",
                    f"{row.v_pvalue:.6e}",
                    f"{row.weight_norm_1:.6f}",
                    f"{row.weight_norm_2:.6f}",
                ]
            )

    logger.info(f"Wrote {len(rows)} matched component pairs to {output}")

    # Summary
    if rows:
        similarities = [r.rank_one_sim for r in rows]
        logger.info(
            f"Rank-one similarity: mean={sum(similarities) / len(similarities):.4f}, "
            f"min={min(similarities):.4f}, max={max(similarities):.4f}"
        )


if __name__ == "__main__":
    fire.Fire(match_components)
