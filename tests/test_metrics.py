from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.evaluation.metrics import (
    mean_square_error,
    error_by_time,
)


class MetricsTests(unittest.TestCase):
    def test_identical_arrays_have_zero_error(self):
        states = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        self.assertEqual(mean_square_error(states, states), 0.0)
        np.testing.assert_array_equal(
            error_by_time(states, states),
            np.zeros(2),
        )

    def test_known_errors(self):
        truth = np.zeros((2, 2))
        prediction = np.array([
            [1.0, 3.0],
            [2.0, 4.0],
        ])

        self.assertAlmostEqual(
            mean_square_error(truth, prediction),
            7.5,
        )
        np.testing.assert_allclose(
            error_by_time(truth, prediction),
            [5.0, 10.0],
        )

    def test_mismatched_shapes_are_rejected(self):
        truth = np.zeros((3, 2))
        prediction = np.zeros((3, 1))

        for metric in (mean_square_error, error_by_time):
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(ValueError):
                    metric(truth, prediction)