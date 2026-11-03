"""Check evaluator alignment and saved artifacts using hand-computed errors."""
from pathlib import Path
import sys
import json
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_edmd import evaluate_model, save_model_results
from data_driven_nonlinear_dynamics.models import dmd, edmd


class EvaluatorTests(unittest.TestCase):
    def test_identity_model_errors_exclude_initial_state(self):
        trajectories = np.array([[[0., 0.], [1., 2.], [3., 6.]]])
        for model, matrix in [(dmd, np.eye(2)), (edmd, np.eye(5))]:
            with self.subTest(model=model.__name__):
                result = evaluate_model(model, matrix, trajectories)
                self.assertEqual(result['one_step_mse'], 6.25)
                self.assertEqual(result['rollout_mse'], 12.5)
                np.testing.assert_allclose(result['rollout_mse_by_time'], [2.5, 22.5])
                np.testing.assert_allclose(result['rollout_mse_by_state'], [5., 20.])
                self.assertEqual(result['one_step_predictions'].shape, (1, 2, 2))
                self.assertEqual(result['rollout_states'].shape, (1, 3, 2))

    def test_exact_linear_trajectory_has_zero_error(self):
        trajectory = np.array([[[1., 4.], [1., 2.], [1., 1.]]])
        for model, matrix in [(dmd, np.diag([1., .5])), (edmd, np.diag([1., .5, 1., 1., 1.]))]:
            with self.subTest(model=model.__name__):
                result = evaluate_model(model, matrix, trajectory)
                self.assertEqual(result['rollout_mse'], 0.)
                self.assertEqual(result['one_step_mse'], 0.)

    def test_saved_predictions_and_metrics_match_evaluation(self):
        truth = np.array([[[0., 0.], [1., 2.], [3., 6.]]])
        result = evaluate_model(edmd, np.eye(5), truth)
        times = np.array([0., .02, .04])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_model_results('EDMD', np.eye(5), result, times, {'seed': 42}, {'val': 1}, directory)
            saved = json.loads((directory / 'metrics.json').read_text())
            self.assertEqual(saved['model'], 'EDMD')
            self.assertEqual(saved['evaluated_split'], 'validation')
            self.assertEqual(saved['metrics']['rollout_mse_mean'], 12.5)
            with np.load(directory / 'predictions.npz') as archive:
                np.testing.assert_array_equal(archive['true_states'], truth)
                np.testing.assert_array_equal(archive['rollout_states'], result['rollout_states'])
                np.testing.assert_array_equal(archive['times'], times)

    def test_empty_or_single_time_trajectories_are_rejected(self):
        for shape in [(0, 3, 2), (1, 1, 2), (3, 2)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    evaluate_model(dmd, np.eye(2), np.zeros(shape))


if __name__ == '__main__':
    unittest.main()
