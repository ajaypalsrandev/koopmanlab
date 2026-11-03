"""Checks for the fixed EDMD dictionary and linear evolution of its features."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.models import edmd


class EDMDTests(unittest.TestCase):
    def test_lift_states_has_expected_feature_order(self):
        states = np.array([[0.0, np.pi / 2], [2.0, -1.0]])
        expected = np.array([
            [0.0, np.pi / 2],
            [2.0, -1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])

        np.testing.assert_allclose(edmd.lift_states(states), expected, atol=1e-14)

    def test_lift_states_rejects_invalid_shapes(self):
        for shape in [(), (2,), (1, 3), (3, 2), (2, 1, 1)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    edmd.lift_states(np.zeros(shape))

    def test_fit_recovers_identity_for_stationary_data(self):
        rng = np.random.default_rng(0)
        states = rng.uniform(-2.0, 2.0, size=(2, 100))
        self.assertEqual(np.linalg.matrix_rank(edmd.lift_states(states)), 5)

        K = edmd.fit_matrix(states, states)

        np.testing.assert_allclose(K, np.eye(5), atol=1e-12)

    def test_fit_recovers_known_feature_dynamics(self):
        # Constant angle and halving velocity are exactly closed in this dictionary.
        rng = np.random.default_rng(1)
        X = rng.uniform(-2.0, 2.0, size=(2, 100))
        Y = X.copy()
        Y[1] *= 0.5

        K = edmd.fit_matrix(X, Y)

        np.testing.assert_allclose(K, np.diag([1., 0.5, 1., 1., 1.]), atol=1e-12)
        np.testing.assert_allclose(
            edmd.rollout(K, np.array([0.7, 4.0]), 4),
            [[0.7, 0.7, 0.7, 0.7], [4.0, 2.0, 1.0, 0.5]],
            atol=1e-12,
        )

    def test_fit_rejects_invalid_training_data(self):
        for X, Y in [
            (np.zeros(2), np.zeros(2)),
            (np.zeros((2, 3)), np.zeros(2)),
            (np.zeros((2, 3)), np.zeros((2, 4))),
            (np.zeros((3, 4)), np.zeros((3, 4))),
            (np.zeros((2, 0)), np.zeros((2, 0))),
        ]:
            with self.subTest(X_shape=X.shape, Y_shape=Y.shape):
                with self.assertRaises(ValueError):
                    edmd.fit_matrix(X, Y)

    def test_identity_one_step_preserves_states(self):
        states = np.array([[0.0, 0.7], [2.0, -1.0]])
        np.testing.assert_allclose(edmd.predict_one_step(np.eye(5), states), states)

    def test_one_step_uses_nonlinear_and_constant_features(self):
        K = np.zeros((5, 5))
        K[0, 2] = 2.0  # Predicted angle = 2 sin(theta) + 3.
        K[0, 4] = 3.0
        K[1, 1] = 0.5  # Predicted velocity = omega / 2 + cos(theta).
        K[1, 3] = 1.0
        states = np.array([[0.0, np.pi / 2], [2.0, -2.0]])

        np.testing.assert_allclose(
            edmd.predict_one_step(K, states), [[3.0, 5.0], [2.0, -1.0]],
            atol=1e-14,
        )

    def test_prediction_and_rollout_reject_invalid_matrices(self):
        for shape in [(2, 2), (5, 4), (4, 5), (5,), (5, 5, 1)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    edmd.predict_one_step(np.zeros(shape), np.zeros((2, 1)))
                with self.assertRaises(ValueError):
                    edmd.rollout(np.zeros(shape), np.zeros(2), 3)

    def test_one_step_rejects_invalid_states(self):
        with self.assertRaises(ValueError):
            edmd.predict_one_step(np.eye(5), np.zeros(2))

    def test_identity_rollout_preserves_initial_state(self):
        predicted = edmd.rollout(np.eye(5), np.array([0.0, 2.0]), 4)
        np.testing.assert_allclose(predicted, [[0., 0., 0., 0.], [2., 2., 2., 2.]])

    def test_rollout_advances_and_includes_initial_state(self):
        K = np.diag([0.5, 2.0, 1.0, 1.0, 1.0])
        predicted = edmd.rollout(K, np.array([4.0, 1.0]), 4)
        np.testing.assert_allclose(predicted, [[4., 2., 1., 0.5], [1., 2., 4., 8.]])

    def test_rollout_evolves_features_without_relifting(self):
        # Freeze the sine feature, then add it to the angle each step.
        # Recomputing sine from the predicted angle would change the third state.
        K = np.eye(5)
        K[0, 2] = 1.0
        predicted = edmd.rollout(K, np.array([np.pi / 2, 2.0]), 3)
        np.testing.assert_allclose(
            predicted,
            [[np.pi / 2, np.pi / 2 + 1, np.pi / 2 + 2], [2., 2., 2.]],
        )

    def test_single_step_returns_only_initial_state(self):
        predicted = edmd.rollout(np.eye(5), np.array([0.3, -0.8]), np.int64(1))
        np.testing.assert_allclose(predicted, [[0.3], [-0.8]])

    def test_rollout_rejects_invalid_initial_state_shapes(self):
        for shape in [(2, 1), (1, 2), (3,), ()]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    edmd.rollout(np.eye(5), np.zeros(shape), 3)

    def test_rollout_rejects_invalid_step_counts(self):
        for steps in [0, -1, 1.5, "3", None]:
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    edmd.rollout(np.eye(5), np.zeros(2), steps)


if __name__ == "__main__":
    unittest.main()
