"""Protect seed aggregation against mixing runs or using trajectory deviations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from summarize_mlp import aggregate, summarize, METRICS, MODELS
from mlp_experiment import REGIMES


class SummaryTests(unittest.TestCase):
    def test_sample_standard_deviation(self):
        result = aggregate([1., 2., 3.])
        self.assertEqual(result['mean'], 2.)
        self.assertEqual(result['sample_std_across_seeds'], 1.)
        with self.assertRaises(ValueError):
            aggregate([1., float('nan')])

    def make_runs(self, root):
        for regime, ranges in REGIMES.items():
            for seed, value in [(42, 1.), (43, 2.), (44, 3.)]:
                folder = root / regime / f'dataset_seed_{seed}' / 'model_seed_42/epochs_100/checkpoint_test'
                folder.mkdir(parents=True)
                data = dict(evaluated_split='validation',
                            config=dict(seed=seed, model_seed=42, epochs=100, **ranges),
                            split_sizes=dict(train=140, val=30, test=30),
                            checkpoint_sha256='test', best_epoch=25,
                            metrics={m: {**{k: value for k in METRICS},
                                         'rollout_mse_std_across_trajectories': 999.}
                                     for m in MODELS})
                (folder / 'comparison.json').write_text(json.dumps(data))

    def test_complete_runs_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_runs(root)
            summary = summarize(root, [42, 43, 44], 42, 100)
            score = summary['regimes']['small_angle']['models']['MLP']['rollout_mse_mean']
            self.assertEqual(score['mean'], 2.)
            self.assertEqual(score['sample_std_across_seeds'], 1.)
            folder = root / 'small_angle/dataset_seed_42/model_seed_42/epochs_100'
            duplicate = folder / 'checkpoint_duplicate'
            duplicate.mkdir()
            (duplicate / 'comparison.json').write_text((folder / 'checkpoint_test/comparison.json').read_text())
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                summarize(root, [42, 43, 44], 42, 100)

    def test_missing_runs_and_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                summarize(root, [42, 43, 44], 42, 100)
            self.make_runs(root)
            path = root / 'small_angle/dataset_seed_43/model_seed_42/epochs_100/checkpoint_test/comparison.json'
            data = json.loads(path.read_text())
            data['config']['learning_rate'] = .5
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                summarize(root, [42, 43, 44], 42, 100)


if __name__ == '__main__':
    unittest.main()
