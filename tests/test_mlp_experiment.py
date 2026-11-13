"""Keep experiment selection consistent between the two MLP commands."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from mlp_experiment import (add_experiment_arguments, experiment_config,
                            experiment_path, checkpoint_path, evaluation_path)


class MLPExperimentTests(unittest.TestCase):
    def test_default_and_explicit_selection(self):
        parser = argparse.ArgumentParser()
        add_experiment_arguments(parser)
        config = experiment_config(parser.parse_args([]))
        self.assertEqual(experiment_path(config), Path('small_angle/dataset_seed_42/model_seed_42/epochs_100'))
        config = experiment_config(parser.parse_args(['--regime', 'broad_motion', '--seed', '43', '--model-seed', '7', '--epochs', '5']))
        expected = Path('broad_motion/dataset_seed_43/model_seed_7/epochs_5')
        self.assertEqual(config['theta_range'], (-2.5, 2.5))
        self.assertEqual(checkpoint_path(ROOT, config), ROOT / 'results/mlp' / expected / 'best_model.pt')
        self.assertEqual(evaluation_path(ROOT, config, '1234567890abcdef'), ROOT / 'docs/benchmarks/mlp' / expected / 'checkpoint_1234567890ab')
        config['theta_range'] = list(config['theta_range'])
        self.assertEqual(experiment_path(config), expected)

    def test_custom_ranges_are_not_mislabeled(self):
        config = dict(theta_range=[-1, 1], omega_range=[-2, 2], seed=42, model_seed=42, epochs=100)
        self.assertEqual(experiment_path(config).parts[0], 'custom')


if __name__ == '__main__':
    unittest.main()
