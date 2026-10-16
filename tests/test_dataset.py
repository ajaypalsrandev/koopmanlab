from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.data.generate_dataset import (
    create_transition_matrices,
    split_trajectories,
)


class DatasetTests(unittest.TestCase):
    def test_transitions_do_not_cross_trajectory_boundaries(self):
        trajectories = np.array([
            [[0., 10.], [1., 11.], [2., 12.]],
            [[100., 110.], [101., 111.], [102., 112.]],
        ])

        X, Y = create_transition_matrices(trajectories)

        expected_X = np.array([
            [0., 1., 100., 101.],
            [10., 11., 110., 111.],
        ])
        expected_Y = np.array([
            [1., 2., 101., 102.],
            [11., 12., 111., 112.],
        ])

        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(Y, expected_Y)

    def test_split_preserves_complete_trajectories(self):
        trajectories = np.arange(20 * 3 * 2).reshape(20, 3, 2)

        train, val, test = split_trajectories(trajectories)

        self.assertEqual(train.shape, (14, 3, 2))
        self.assertEqual(val.shape, (3, 3, 2))
        self.assertEqual(test.shape, (3, 3, 2))

        np.testing.assert_array_equal(train, trajectories[:14])
        np.testing.assert_array_equal(val, trajectories[14:17])
        np.testing.assert_array_equal(test, trajectories[17:])