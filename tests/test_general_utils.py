"""Test p-annealing functionality."""

from spd.utils.general_utils import get_linear_annealed_value


class TestLinearAnnealedValue:
    """Test get_linear_annealed_value function."""

    def test_no_annealing_cases(self):
        """Test all edge cases where annealing should be a no-op."""
        initial_value = 2.0

        # Case 1: final_value is None
        for frac in [0.0, 0.5, 1.0]:
            value = get_linear_annealed_value(
                current_frac_of_training=frac,
                initial_value=initial_value,
                anneal_start_frac=0.5,
                anneal_final_value=None,
                anneal_end_frac=1.0,
            )
            assert value == initial_value

        # Case 2: start_frac >= 1.0
        for frac in [0.0, 0.5, 1.0]:
            value = get_linear_annealed_value(
                current_frac_of_training=frac,
                initial_value=initial_value,
                anneal_start_frac=1.0,
                anneal_final_value=0.9,
                anneal_end_frac=1.0,
            )
            assert value == initial_value

    def test_annealing_scenarios(self):
        """Test various annealing scenarios."""
        initial_value = 2.0
        final_value = 0.5

        test_cases = [
            # (start_frac, end_frac, test_points)
            # start=0, end=1: anneal throughout
            (0.0, 1.0, [(0.0, initial_value), (0.5, 1.25), (1.0, final_value)]),
            # start=0.25, end=1: skip first quarter
            (0.25, 1.0, [(0.0, initial_value), (0.25, initial_value), (0.625, 1.25), (1.0, final_value)]),
            # start=0.25, end=0.75: anneal in middle
            (
                0.25,
                0.75,
                [(0.0, initial_value), (0.25, initial_value), (0.5, 1.25), (0.75, final_value), (1.0, final_value)],
            ),
            # start=0, end=0.75: anneal then plateau
            (0.0, 0.75, [(0.0, initial_value), (0.375, 1.25), (0.75, final_value), (1.0, final_value)]),
        ]

        for start_frac, end_frac, test_points in test_cases:
            for frac, expected in test_points:
                value = get_linear_annealed_value(
                    current_frac_of_training=frac,
                    initial_value=initial_value,
                    anneal_start_frac=start_frac,
                    anneal_final_value=final_value,
                    anneal_end_frac=end_frac,
                )
                assert abs(value - expected) < 1e-6, (
                    f"start={start_frac}, end={end_frac}, frac={frac}: expected {expected}, got {value}"
                )
