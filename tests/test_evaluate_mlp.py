"""Check physical-unit scoring, rollout alignment, and checkpoint evaluation."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_mlp import evaluate_mlp_model, run_evaluation
from data_driven_nonlinear_dynamics.models.mlp import MLP


class MLPEvaluationTests(unittest.TestCase):
    def test_identity_predictions_and_scoring_alignment(self):
        truth = np.array([[[0., 0.], [1., 2.], [3., 6.]]])
        result = evaluate_mlp_model(
            torch.nn.Identity(), torch.tensor([1., 2.]),
            torch.tensor([2., 4.]), truth,
        )
        self.assertAlmostEqual(result['one_step_mse'], 6.25)
        self.assertAlmostEqual(result['rollout_mse'], 12.5)
        np.testing.assert_allclose(result['rollout_mse_by_time'], [2.5, 22.5])
        self.assertEqual(result['one_step_predictions'].shape, (1, 2, 2))
        self.assertEqual(result['rollout_states'].shape, (1, 3, 2))

    def test_zero_normalized_prediction_is_denormalized(self):
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        truth = np.zeros((1, 3, 2))
        result = evaluate_mlp_model(model, torch.tensor([2., 4.]), torch.tensor([3., 5.]), truth)
        np.testing.assert_allclose(result['one_step_predictions'], [[[2., 4.], [2., 4.]]])
        self.assertAlmostEqual(result['one_step_mse'], 10.)
        self.assertAlmostEqual(result['rollout_mse'], 10.)

    def test_checkpoint_to_saved_comparison(self):
        config = dict(num_trajectories=20, seed=42, model_seed=7,
                      theta_range=(-.5, .5), omega_range=(-.5, .5),
                      final_time=.1, time_step=.02, hidden_dim=4, epochs=1)
        checkpoint = {
            'model_state_dict': MLP(4).state_dict(), 'config': config,
            'mean': torch.zeros(2), 'std': torch.ones(2),
            'history': [(1., .8)], 'best_epoch': 1, 'best_val_loss': .8,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            output = Path(tmp) / 'evaluation'
            torch.save(checkpoint, path)
            results = run_evaluation(path, output)
            comparison = json.loads((output / 'comparison.json').read_text())
            self.assertEqual(comparison['split_sizes'], dict(train=14, val=3, test=3))
            self.assertEqual(comparison['evaluated_split'], 'validation')
            self.assertEqual(set(comparison['metrics']), {'DMD', 'EDMD', 'MLP'})
            for name in results:
                np.testing.assert_array_equal(results[name]['true_states'], results['MLP']['true_states'])
            with np.load(output / 'mlp' / 'predictions.npz') as saved:
                np.testing.assert_array_equal(saved['rollout_states'], results['MLP']['rollout_states'])
            for filename in ['representative.png', 'worst.png', 'error_by_time.png', 'training_history.png']:
                self.assertTrue((output / filename).is_file())
            del checkpoint['config']['seed']
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'dataset seed'):
                run_evaluation(path, output)


if __name__ == '__main__':
    unittest.main()
