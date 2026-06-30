"""Validate the InterpBench case-19 conversion + synthetic-data pipeline.

Two checks (roadmap Objective 1):
  1. The `nn.Linear` converted model computes the same function as the original
     HookedTransformer weights — compared against a folded-parameter reference forward
     (the HookedTransformer math read straight off the state dict).
  2. Generated inputs, run through the converted model and decoded, recover the
     ground-truth task labels — confirming the vendored vocab, encoding map, and
     tokenization are correct.

Marked slow: downloads the case from the HuggingFace Hub. Run with
`pytest param_decomp_lab/experiments/lm/interpbench/ --runslow`.
"""

import pickle
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import einsum
from huggingface_hub import hf_hub_download
from torch import Tensor

from param_decomp_lab.experiments.lm.interpbench.data import (
    CASE_SPECS,
    encode_tokens,
    sample_content,
)
from param_decomp_lab.experiments.lm.interpbench.model import (
    INTERPBENCH_HF_REPO,
    InterpBenchTransformer,
)

CASE_ID = "19"
Case19 = tuple[InterpBenchTransformer, dict[str, Tensor], dict[str, Any]]


def _hooked_reference(state: dict[str, Tensor], cfg: dict[str, Any], input_ids: Tensor) -> Tensor:
    """HookedTransformer forward straight off the folded state dict (no LayerNorm).

    Independent code path from `InterpBenchTransformer` (folded `[n_heads, d_model,
    d_head]` params vs unfolded `nn.Linear`s) so agreement validates the conversion.
    """
    nl, dh = cfg["n_layers"], cfg["d_head"]
    attn_scale = dh**0.5 if cfg["use_attn_scale"] else 1.0
    causal = cfg["attention_dir"] == "causal"
    t = input_ids.shape[1]
    x = state["embed.W_E"][input_ids] + state["pos_embed.W_pos"][:t]
    for layer in range(nl):
        p = f"blocks.{layer}.attn."
        q = einsum(x, state[p + "W_Q"], "b t m, h m d -> b h t d") + state[p + "b_Q"][:, None, :]
        k = einsum(x, state[p + "W_K"], "b t m, h m d -> b h t d") + state[p + "b_K"][:, None, :]
        v = einsum(x, state[p + "W_V"], "b t m, h m d -> b h t d") + state[p + "b_V"][:, None, :]
        scores = einsum(q, k, "b h q d, b h k d -> b h q k") / attn_scale
        if causal:
            future = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        z = einsum(scores.softmax(-1), v, "b h q k, b h k d -> b h q d")
        x = x + einsum(z, state[p + "W_O"], "b h q d, h d m -> b q m") + state[p + "b_O"]
        m = f"blocks.{layer}.mlp."
        x = (
            x
            + F.gelu(x @ state[m + "W_in"] + state[m + "b_in"]) @ state[m + "W_out"]
            + state[m + "b_out"]
        )
    return x @ state["unembed.W_U"] + state["unembed.b_U"]


@pytest.fixture(scope="module")
def case19() -> Case19:
    with open(hf_hub_download(INTERPBENCH_HF_REPO, f"{CASE_ID}/ll_model_cfg.pkl"), "rb") as f:
        cfg = pickle.load(f)
    state = torch.load(
        hf_hub_download(INTERPBENCH_HF_REPO, f"{CASE_ID}/ll_model.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model = InterpBenchTransformer.from_hf(CASE_ID)
    return model, state, cfg


@pytest.mark.slow
def test_converted_model_matches_hooked_reference(case19: Case19):
    """Converted nn.Linear forward == folded HookedTransformer reference (exact in fp64)."""
    model, state, cfg = case19
    model64 = InterpBenchTransformer.from_hf(CASE_ID).double()
    state64 = {k: v.double() for k, v in state.items()}
    ids = torch.randint(0, cfg["d_vocab"], (16, cfg["n_ctx"]))

    ref = _hooked_reference(state64, cfg, ids)
    got = model64(ids)
    assert torch.allclose(ref, got, atol=1e-9), f"max diff {(ref - got).abs().max().item():.2e}"
    # fp32 model agrees on argmax (decisions identical despite reduction-order noise)
    assert (model(ids).argmax(-1) == ref.argmax(-1)).all()


@pytest.mark.slow
def test_data_pipeline_recovers_task_labels(case19: Case19):
    """Generated + decoded inputs match the ground-truth task on interior positions.

    Position 1 (BOS-adjacent) is excluded: the trained model is intrinsically imperfect
    at the first content token; all other content positions are exact.
    """
    model, _, _ = case19
    spec = CASE_SPECS[CASE_ID]
    rng = np.random.default_rng(0)
    contents = [sample_content(spec, rng) for _ in range(256)]
    ids = torch.tensor([encode_tokens(c, spec) for c in contents])
    pred = model(ids).argmax(-1)

    total = correct = 0
    for row, content in enumerate(contents):
        labels = spec.task(content)  # per content position; None where dropped
        for j, label in enumerate(labels):
            pos = j + 1  # +1 for BOS at position 0
            if pos == 1 or label is None:  # skip BOS-adjacent + dropped (ambiguous) positions
                continue
            total += 1
            correct += spec.output_id_to_token[int(pred[row, pos])] == label
    accuracy = correct / total
    assert accuracy >= 0.99, f"interior-position task accuracy {accuracy:.4f} (n={total})"
