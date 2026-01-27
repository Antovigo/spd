"""Tests for template variable resolution in output_dir_name."""

import pytest

from spd.utils.run_utils import (
    PathSegment,
    parse_path_segments,
    resolve_config_path,
    resolve_template_string,
)


class TestParsePathSegments:
    def test_simple_field(self) -> None:
        segments = parse_path_segments("seed")
        assert segments == [PathSegment(field="seed")]

    def test_nested_field(self) -> None:
        segments = parse_path_segments("task_config.feature_probability")
        assert segments == [
            PathSegment(field="task_config"),
            PathSegment(field="feature_probability"),
        ]

    def test_list_lookup(self) -> None:
        segments = parse_path_segments("loss_metric_configs[ImportanceMinimalityLoss].coeff")
        assert segments == [
            PathSegment(field="loss_metric_configs", list_key="ImportanceMinimalityLoss"),
            PathSegment(field="coeff"),
        ]

    def test_deeply_nested(self) -> None:
        segments = parse_path_segments("a.b.c.d")
        assert segments == [
            PathSegment(field="a"),
            PathSegment(field="b"),
            PathSegment(field="c"),
            PathSegment(field="d"),
        ]


class TestResolveConfigPath:
    def test_simple_field(self) -> None:
        config = {"seed": 42}
        assert resolve_config_path("seed", config) == 42

    def test_nested_field(self) -> None:
        config = {"task_config": {"feature_probability": 0.05}}
        assert resolve_config_path("task_config.feature_probability", config) == 0.05

    def test_list_lookup(self) -> None:
        config = {
            "loss_metric_configs": [
                {"classname": "FaithfulnessLoss", "coeff": 1.0},
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.003},
            ]
        }
        result = resolve_config_path("loss_metric_configs[ImportanceMinimalityLoss].coeff", config)
        assert result == 0.003

    def test_invalid_field(self) -> None:
        config = {"seed": 42}
        with pytest.raises(AssertionError, match="Field 'nonexistent' not found"):
            resolve_config_path("nonexistent", config)

    def test_invalid_list_key(self) -> None:
        config = {
            "loss_metric_configs": [
                {"classname": "FaithfulnessLoss", "coeff": 1.0},
            ]
        }
        with pytest.raises(AssertionError, match="No item with classname='NonExistent'"):
            resolve_config_path("loss_metric_configs[NonExistent].coeff", config)

    def test_not_a_list(self) -> None:
        config = {"loss_metric_configs": {"not": "a list"}}
        with pytest.raises(AssertionError, match="Expected list"):
            resolve_config_path("loss_metric_configs[Something].coeff", config)

    def test_not_a_dict_in_path(self) -> None:
        config = {"seed": 42}
        with pytest.raises(AssertionError, match="Expected dict"):
            resolve_config_path("seed.nested", config)


class TestResolveTemplateString:
    def test_simple_replacement(self) -> None:
        config = {"seed": 42}
        result = resolve_template_string("Seed {seed}", config)
        assert result == "Seed 42"

    def test_nested_replacement(self) -> None:
        config = {"task_config": {"feature_probability": 0.05}}
        result = resolve_template_string("FP {task_config.feature_probability}", config)
        assert result == "FP 0.05"

    def test_list_lookup_replacement(self) -> None:
        config = {
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.003},
            ]
        }
        result = resolve_template_string(
            "Coeff {loss_metric_configs[ImportanceMinimalityLoss].coeff}", config
        )
        assert result == "Coeff 0.003"

    def test_multiple_placeholders(self) -> None:
        config = {
            "seed": 42,
            "loss_metric_configs": [
                {"classname": "ImportanceMinimalityLoss", "coeff": 0.003},
            ],
        }
        result = resolve_template_string(
            "seed_{seed}_coeff_{loss_metric_configs[ImportanceMinimalityLoss].coeff}", config
        )
        assert result == "seed_42_coeff_0.003"

    def test_no_placeholders(self) -> None:
        config = {"seed": 42}
        result = resolve_template_string("static_name", config)
        assert result == "static_name"

    def test_dict_value_raises(self) -> None:
        config = {"task_config": {"nested": {"value": 1}}}
        with pytest.raises(AssertionError, match="resolved to dict"):
            resolve_template_string("{task_config.nested}", config)

    def test_list_value_raises(self) -> None:
        config = {"items": [1, 2, 3]}
        with pytest.raises(AssertionError, match="resolved to list"):
            resolve_template_string("{items}", config)


class TestConfigResolveOutputDirName:
    def test_none_returns_none(self) -> None:
        from spd.configs import (
            Config,
            ModulePatternInfoConfig,
            ScheduleConfig,
            TMSTaskConfig,
        )

        config = Config(
            output_dir_name=None,
            seed=42,
            n_mask_samples=10,
            module_info=[ModulePatternInfoConfig(module_pattern="*", C=5)],
            output_loss_type="mse",
            lr_schedule=ScheduleConfig(start_val=0.001),
            steps=100,
            batch_size=32,
            train_log_freq=10,
            eval_freq=50,
            slow_eval_freq=50,
            eval_batch_size=32,
            n_eval_steps=10,
            n_examples_until_dead=1000,
            pretrained_model_class="torch.nn.Linear",
            task_config=TMSTaskConfig(feature_probability=0.1),
        )
        assert config.resolve_output_dir_name() is None

    def test_template_resolved(self) -> None:
        from spd.configs import (
            Config,
            ModulePatternInfoConfig,
            ScheduleConfig,
            TMSTaskConfig,
        )

        config = Config(
            output_dir_name="seed_{seed}",
            seed=123,
            n_mask_samples=10,
            module_info=[ModulePatternInfoConfig(module_pattern="*", C=5)],
            output_loss_type="mse",
            lr_schedule=ScheduleConfig(start_val=0.001),
            steps=100,
            batch_size=32,
            train_log_freq=10,
            eval_freq=50,
            slow_eval_freq=50,
            eval_batch_size=32,
            n_eval_steps=10,
            n_examples_until_dead=1000,
            pretrained_model_class="torch.nn.Linear",
            task_config=TMSTaskConfig(feature_probability=0.1),
        )
        assert config.resolve_output_dir_name() == "seed_123"

    def test_nested_template_resolved(self) -> None:
        from spd.configs import (
            Config,
            ModulePatternInfoConfig,
            ScheduleConfig,
            TMSTaskConfig,
        )

        config = Config(
            output_dir_name="fp_{task_config.feature_probability}",
            seed=0,
            n_mask_samples=10,
            module_info=[ModulePatternInfoConfig(module_pattern="*", C=5)],
            output_loss_type="mse",
            lr_schedule=ScheduleConfig(start_val=0.001),
            steps=100,
            batch_size=32,
            train_log_freq=10,
            eval_freq=50,
            slow_eval_freq=50,
            eval_batch_size=32,
            n_eval_steps=10,
            n_examples_until_dead=1000,
            pretrained_model_class="torch.nn.Linear",
            task_config=TMSTaskConfig(feature_probability=0.05),
        )
        assert config.resolve_output_dir_name() == "fp_0.05"
