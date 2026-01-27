import pytest

from spd.utils.general_utils import get_linear_annealed_value


class TestGetLinearAnnealedValue:
    def test_no_annealing_when_final_value_none(self) -> None:
        result = get_linear_annealed_value(
            current_frac_of_training=0.5,
            initial_value=2.0,
            anneal_start_frac=0.0,
            anneal_final_value=None,
            anneal_end_frac=1.0,
        )
        assert result == 2.0

    def test_no_annealing_when_start_frac_is_one(self) -> None:
        result = get_linear_annealed_value(
            current_frac_of_training=0.5,
            initial_value=2.0,
            anneal_start_frac=1.0,
            anneal_final_value=1.0,
            anneal_end_frac=1.0,
        )
        assert result == 2.0

    def test_before_annealing_starts(self) -> None:
        result = get_linear_annealed_value(
            current_frac_of_training=0.3,
            initial_value=2.0,
            anneal_start_frac=0.5,
            anneal_final_value=1.0,
            anneal_end_frac=1.0,
        )
        assert result == 2.0

    def test_after_annealing_ends(self) -> None:
        result = get_linear_annealed_value(
            current_frac_of_training=0.9,
            initial_value=2.0,
            anneal_start_frac=0.0,
            anneal_final_value=1.0,
            anneal_end_frac=0.5,
        )
        assert result == 1.0

    def test_during_annealing_midpoint(self) -> None:
        # At 50% through annealing (0.25 between 0.0 and 0.5)
        result = get_linear_annealed_value(
            current_frac_of_training=0.25,
            initial_value=2.0,
            anneal_start_frac=0.0,
            anneal_final_value=1.0,
            anneal_end_frac=0.5,
        )
        # p should be: 2.0 + (1.0 - 2.0) * 0.5 = 1.5
        assert result == 1.5

    def test_during_annealing_quarter(self) -> None:
        # At 25% through annealing
        result = get_linear_annealed_value(
            current_frac_of_training=0.125,
            initial_value=2.0,
            anneal_start_frac=0.0,
            anneal_final_value=1.0,
            anneal_end_frac=0.5,
        )
        # progress = 0.125 / 0.5 = 0.25
        # value = 2.0 + (1.0 - 2.0) * 0.25 = 1.75
        assert result == 1.75

    def test_annealing_upward(self) -> None:
        # Test annealing from lower to higher value
        result = get_linear_annealed_value(
            current_frac_of_training=0.5,
            initial_value=0.001,
            anneal_start_frac=0.0,
            anneal_final_value=0.01,
            anneal_end_frac=1.0,
        )
        # progress = 0.5, value = 0.001 + (0.01 - 0.001) * 0.5 = 0.0055
        assert result == pytest.approx(0.0055)

    def test_exact_at_start_frac(self) -> None:
        result = get_linear_annealed_value(
            current_frac_of_training=0.5,
            initial_value=2.0,
            anneal_start_frac=0.5,
            anneal_final_value=1.0,
            anneal_end_frac=1.0,
        )
        # At exactly the start, should return initial value
        assert result == 2.0

    def test_exact_at_end_frac(self) -> None:
        result = get_linear_annealed_value(
            current_frac_of_training=0.5,
            initial_value=2.0,
            anneal_start_frac=0.0,
            anneal_final_value=1.0,
            anneal_end_frac=0.5,
        )
        # At exactly the end, should return final value
        assert result == 1.0

    def test_assertion_end_before_start(self) -> None:
        with pytest.raises(AssertionError):
            get_linear_annealed_value(
                current_frac_of_training=0.5,
                initial_value=2.0,
                anneal_start_frac=0.8,
                anneal_final_value=1.0,
                anneal_end_frac=0.3,
            )
