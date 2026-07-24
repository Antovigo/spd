import base64
import json
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from param_decomp_lab.eval_metrics.ab_grid_dataset import (
    ABGridDataset,
    ABGridDatasetConfig,
    PromptGrid,
    encode_ci_u8,
)
from param_decomp_lab.run_artifacts import RunDirArtifact


def make_prompts(ops: list[str], n: int) -> list[str]:
    return [f"{a}{op}{b}=" for op in ops for a in range(1, n + 1) for b in range(1, n + 1)]


class TestPromptGrid:
    def test_full_grid_round_trip(self):
        prompts = make_prompts(["+", "-"], 3)
        grid = PromptGrid.from_prompts(prompts)
        assert grid.ops == ["+", "-"]
        assert (grid.n_a, grid.n_b, grid.a_min, grid.b_min) == (3, 3, 1, 1)

        values = np.arange(len(prompts), dtype=np.float32)
        scattered = grid.scatter(values, fill=np.nan)
        assert scattered.shape == (2, 3, 3)
        assert scattered[0, 0, 0] == 0.0  # "1+1="
        assert scattered[0, 0, 1] == 1.0  # "1+2="
        assert scattered[1, 2, 2] == len(prompts) - 1  # "3-3="
        assert not np.isnan(scattered).any()

    def test_sparse_pool_fills(self):
        grid = PromptGrid.from_prompts(["2+5=", "4+9="])
        assert (grid.a_min, grid.b_min, grid.n_a, grid.n_b) == (2, 5, 3, 5)
        scattered = grid.scatter(np.array([1.0, 2.0], dtype=np.float32), fill=np.nan)
        assert scattered[0, 0, 0] == 1.0
        assert scattered[0, 2, 4] == 2.0
        assert np.isnan(scattered).sum() == 15 - 2

    def test_trailing_dims(self):
        prompts = make_prompts(["+"], 2)
        grid = PromptGrid.from_prompts(prompts)
        values = np.stack([np.arange(4), np.arange(4) * 10], axis=1).astype(np.float32)
        scattered = grid.scatter(values, fill=0.0)
        assert scattered.shape == (1, 2, 2, 2)
        assert scattered[0, 1, 0, 1] == 20.0  # "2+1=", second channel

    def test_rejects_malformed_prompt(self):
        with pytest.raises(AssertionError, match="does not match"):
            PromptGrid.from_prompts(["1+1=", "hello"])

    def test_rejects_duplicates(self):
        with pytest.raises(AssertionError, match="duplicate"):
            PromptGrid.from_prompts(["1+1=", "1+1="])


def test_encode_ci_u8():
    ci = np.array([0.0, 0.5, 1.0, 1.7, -0.2], dtype=np.float32)
    out = encode_ci_u8(ci)
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 128, 255, 255, 0]


class TestRunDirArtifact:
    def test_write_and_manifest(self, tmp_path: Path):
        artifact = RunDirArtifact(
            dir="ab_grids",
            files={"step_1000.js": b"a", "index.html": b"<html>"},
            manifest_var="AB_GRIDS_MANIFEST",
        )
        artifact.write(tmp_path)
        assert (tmp_path / "ab_grids" / "step_1000.js").read_bytes() == b"a"
        manifest = (tmp_path / "ab_grids" / "manifest.js").read_text()
        assert manifest == 'window.AB_GRIDS_MANIFEST = ["step_1000.js"];\n'

        RunDirArtifact(
            dir="ab_grids", files={"step_200.js": b"b"}, manifest_var="AB_GRIDS_MANIFEST"
        ).write(tmp_path)
        manifest = (tmp_path / "ab_grids" / "manifest.js").read_text()
        assert manifest == 'window.AB_GRIDS_MANIFEST = ["step_200.js", "step_1000.js"];\n'

    def test_b64_payload_layout(self):
        arr = np.array([[1.5, -2.0]], dtype=np.float16)
        decoded = np.frombuffer(base64.b64decode(base64.b64encode(arr.tobytes())), dtype=np.float16)
        assert decoded.tolist() == [1.5, -2.0]


class _FakeTokenizer:
    def encode(self, s: str) -> list[int]:
        return [ord(c) for c in s]


D_IN, C, SEQ = 6, 4, 4


class _StubModel:
    """Single-digit `a<op>b=` prompts; cache channel 0 carries each position's char code."""

    def __init__(self):
        self.components = {"blk.mlp": types.SimpleNamespace(V=torch.eye(D_IN)[:, :C].clone())}
        self.module_to_c = {"blk.mlp": C}

    def __call__(self, chunk: torch.Tensor, cache_type: str) -> Any:
        x = torch.zeros(chunk.shape[0], SEQ, D_IN)
        x[:, :, 0] = chunk.float()
        return types.SimpleNamespace(
            output=torch.zeros(chunk.shape[0], SEQ, 7), cache={"blk.mlp": x}
        )

    def calc_causal_importances(
        self, pre_weight_acts: dict[str, torch.Tensor], detach_inputs: bool, sampling: str
    ) -> Any:
        del detach_inputs, sampling
        x = pre_weight_acts["blk.mlp"]
        ci = torch.zeros(x.shape[0], SEQ, C)
        ci[:, :, 0] = 1.0
        ci[:, -1, 1] = (x[:, 0, 0] == x[:, 2, 0]).float()  # a == b, answer position only
        return types.SimpleNamespace(lower_leaky={"blk.mlp": ci})


def make_bound_metric(tmp_path: Path, positions: list[int] | None) -> ABGridDataset:
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("\n".join(make_prompts(["+", "-"], 3)) + "\n")
    cfg = ABGridDatasetConfig(
        type="ABGridDataset",
        prompts_file=str(prompts_file),
        tokenizer_name="fake",
        forward_batch_size=4,
        mean_ci_floor=0.05,
        positions=positions,
    )
    with patch("param_decomp_lab.eval_metrics.ab_grid_dataset.AutoTokenizer") as auto_tok:
        auto_tok.from_pretrained.return_value = _FakeTokenizer()
        metric = ABGridDataset(cfg)
    metric.bind(model=cast(Any, _StubModel()), device="cpu")
    return metric


def decode_payload(result: dict[str, Any]) -> dict[str, Any]:
    js = result["ab_grids"].files["step_1000.js"].decode()
    return json.loads(js[len("window.registerABGrids(") : -2])


class TestABGridDatasetCompute:
    def test_multi_position_comp_major_layout(self, tmp_path: Path):
        """The diagonal component is CI=1 only at the answer position — a pos/comp axis
        swap in the payload layout moves it to the wrong slice."""
        metric = make_bound_metric(tmp_path, positions=[0, -1])
        metric.update(cast(Any, types.SimpleNamespace(step=1000)))
        payload = decode_payload(cast("dict[str, Any]", metric.compute()))

        assert payload["positions"] == [0, 3]
        m = payload["modules"][0]
        assert m["saved"] == [0, 1]
        n_pos, n_ops = 2, 2
        ci = np.frombuffer(base64.b64decode(m["ci"]), np.uint8).reshape(2, n_pos, n_ops, 3, 3)
        assert (ci[0] == 255).all()  # comp 0 on everywhere at both positions
        assert (ci[1, 0] == 0).all()  # comp 1 off at position 0
        for op in range(n_ops):
            for a in range(3):
                for b in range(3):
                    assert ci[1, 1, op, a, b] == (255 if a == b else 0)

    def test_answer_position_default_and_artifact(self, tmp_path: Path):
        metric = make_bound_metric(tmp_path, positions=None)
        metric.update(cast(Any, types.SimpleNamespace(step=1000)))
        result = cast("dict[str, Any]", metric.compute())
        payload = decode_payload(result)
        assert payload["positions"] == [3]
        assert result["ab_grids_saved_components"] == 2
        assert "index.html" in result["ab_grids"].files

    def test_out_of_range_position_rejected(self, tmp_path: Path):
        with pytest.raises(AssertionError, match="out of range"):
            make_bound_metric(tmp_path, positions=[SEQ])
