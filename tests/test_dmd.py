from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.models.dmd import (
    fit_matrix,
    predict_one_step,
    rollout,
)


class DMDTests(unittest.TestCase):
    def test_fit_recovers_known_linear_system(self):
        A_true = np.array([
            [0.9, 0.1],
            [-0.2, 0.8],
        ])

        rng = np.random.default_rng(0)
        X = rng.normal(size=(2, 100))
        Y = A_true @ X

        A_learned = fit_matrix(X, Y)

        np.testing.assert_allclose(
            A_learned,
            A_true,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_one_step_predictions(self):
        A = np.array([
            [0.5, 0.0],
            [0.0, 2.0],
        ])
        states = np.array([
            [4.0, 8.0],
            [1.0, 3.0],
        ])

        predictions = predict_one_step(A, states)

        expected = np.array([
            [2.0, 4.0],
            [2.0, 6.0],
        ])
        np.testing.assert_allclose(predictions, expected)

    def test_rollout_includes_initial_state(self):
        A = np.array([
            [0.5, 0.0],
            [0.0, 2.0],
        ])
        initial_state = np.array([4.0, 1.0])

        predictions = rollout(A, initial_state, num_steps=4)

        expected = np.array([
            [4.0, 2.0, 1.0, 0.5],
            [1.0, 2.0, 4.0, 8.0],
        ])

        self.assertEqual(predictions.shape, (2, 4))
        np.testing.assert_allclose(predictions, expected)