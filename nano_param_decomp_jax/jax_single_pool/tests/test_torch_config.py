"""The shared-config route must reproduce the hand-written native config exactly.

`configs/llama8b_l18_b128_cmp32.yaml` (native) and
`configs/torch/llama8b_l18_b128_cmp32_1pool.yaml` (torch `LMExperimentConfig`) describe
the same run by construction — the torch-vs-JAX comparison pair. Converting the torch
yaml through `load_torch_wrapper` must therefore yield the native `ExperimentConfig`
field-for-field (only `run_name` differs).
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from jax_single_pool.config import load_config
from jax_single_pool.torch_config import convert_torch_lm_config, load_torch_wrapper

CONFIGS = Path(__file__).parent.parent / "configs"


def test_torch_wrapper_reproduces_native_config():
    converted, torch_yaml_path, torch_raw = load_torch_wrapper(
        CONFIGS / "llama8b_l18_b128_cmp32_from_torch.yaml"
    )
    assert torch_yaml_path == CONFIGS / "torch" / "llama8b_l18_b128_cmp32_1pool.yaml"
    assert torch_raw["pd"]["batch_size"] == 128

    native = load_config(CONFIGS / "llama8b_l18_b128_cmp32.yaml")
    assert converted.run_name == "jax-l18-b128-cmp32-from-torch"
    assert dataclasses.replace(converted, run_name=native.run_name) == native


def _reference_torch_cfg():
    from param_decomp_config.lm import LMExperimentConfig

    raw = yaml.safe_load((CONFIGS / "torch" / "llama8b_l18_b128_cmp32_1pool.yaml").read_text())
    return LMExperimentConfig(**raw), raw


def test_eval_block_maps_and_defers_offline_metrics(capsys: pytest.CaptureFixture[str]):
    torch_cfg, raw = _reference_torch_cfg()
    raw["eval"] = {
        "batch_size": 128,
        "every": 1000,
        "n_steps": 1,
        "slow_every": 10000,
        "metrics": [
            {"type": "CEandKLLosses", "rounding_threshold": 0.0},
            {"type": "CI_L0", "groups": None, "ci_alive_threshold": 0.0},
            {
                "type": "PGDReconLoss",
                "coeff": None,
                "init": "random",
                "mask_scope": "shared_across_batch",
                "n_steps": 20,
                "step_size": 0.1,
            },
            {"type": "CIHistograms", "n_batches_accum": 1},
            {"type": "ComponentActivationDensity", "ci_alive_threshold": 0.0},
        ],
    }
    torch_cfg = type(torch_cfg)(**raw)
    cfg = convert_torch_lm_config(
        torch_cfg, run_name="t", out_dir=Path("/tmp"), remat_recon_forwards=True
    )
    assert cfg.eval is not None
    assert (cfg.eval.batch_size, cfg.eval.every, cfg.eval.n_steps) == (128, 1000, 1)
    assert cfg.eval.rounding_threshold == 0.0 and cfg.eval.ci_alive_threshold == 0.0
    assert cfg.eval.pgd is not None and (cfg.eval.pgd.n_steps, cfg.eval.pgd.step_size) == (20, 0.1)
    assert "deferred to the offline path" in capsys.readouterr().out


def test_unsupported_settings_refuse():
    torch_cfg, raw = _reference_torch_cfg()

    binomial = dict(raw, pd=dict(raw["pd"], sampling="binomial"))
    with pytest.raises(AssertionError):
        convert_torch_lm_config(
            type(torch_cfg)(**binomial), run_name="t", out_dir=Path("/tmp"),
            remat_recon_forwards=True,
        )  # fmt: skip

    extra_loss = dict(
        raw,
        pd=dict(
            raw["pd"],
            loss_metrics=raw["pd"]["loss_metrics"] + [{"type": "UnmaskedReconLoss", "coeff": 1.0}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported loss metric"):
        convert_torch_lm_config(
            type(torch_cfg)(**extra_loss), run_name="t", out_dir=Path("/tmp"),
            remat_recon_forwards=True,
        )  # fmt: skip

    non_mlp_target = dict(
        raw,
        pd=dict(
            raw["pd"],
            decomposition_targets=[{"module_pattern": "layers.18.self_attn.q_proj", "C": 512}],
        ),
    )
    with pytest.raises(AssertionError, match="unsupported decomposition target"):
        convert_torch_lm_config(
            type(torch_cfg)(**non_mlp_target), run_name="t", out_dir=Path("/tmp"),
            remat_recon_forwards=True,
        )  # fmt: skip


def test_c49k_yaml_converts_with_documented_divergences(capsys: pytest.CaptureFixture[str]):
    """The C49k/200k yaml (raw-HF target spec, fp32 weights_dtype, `model.`-prefixed
    site patterns) must convert, printing the fp32-frozen divergence note."""
    converted, _torch_path, _raw = load_torch_wrapper(
        CONFIGS / "llama8b_l18_C49k_200k_from_torch.yaml"
    )
    printed = capsys.readouterr().out
    assert "fp32 frozen target" in printed
    assert converted.target.C == 49152
    assert converted.steps == 200000
    assert converted.data.global_batch == 512 and converted.data.seq_len == 2048
    assert converted.vu_optimizer.lr == 5e-05 and converted.ci_optimizer.lr == 5e-05
    assert converted.eval is not None and converted.eval.pgd is not None
    assert converted.wandb is not None and converted.wandb.entity is None


def test_load_run_dir_config_handles_wrapper_runs(tmp_path: Path):
    """The exporter reads run dirs via `load_run_dir_config`; wrapper runs pin the
    wrapper as config.yaml + the torch yaml as torch_config.yaml (run.py's
    `_pin_config_copy`), and the rebuilt config must equal the launch-time conversion."""
    wrapper = CONFIGS / "llama8b_l18_C49k_200k_from_torch.yaml"
    expected, torch_yaml_path, _ = load_torch_wrapper(wrapper)
    (tmp_path / "config.yaml").write_text(wrapper.read_text())
    (tmp_path / "torch_config.yaml").write_text(torch_yaml_path.read_text())
    from jax_single_pool.torch_config import load_run_dir_config

    assert load_run_dir_config(tmp_path) == expected

    native = CONFIGS / "llama8b_l18_b128_cmp32.yaml"
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    (native_dir / "config.yaml").write_text(native.read_text())
    from jax_single_pool.config import load_config

    assert load_run_dir_config(native_dir) == load_config(native)
