"""Shared experiment names and paths for MLP training and evaluation."""
from pathlib import Path

REGIMES = {
    'small_angle': {'theta_range': (-0.5, 0.5), 'omega_range': (-0.5, 0.5)},
    'broad_motion': {'theta_range': (-2.5, 2.5), 'omega_range': (-2.0, 2.0)},
}


def add_experiment_arguments(parser):
    parser.add_argument('--regime', choices=REGIMES, default='small_angle')
    parser.add_argument('--seed', type=int, default=42, help='Dataset seed')
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100, help='Training budget, not selected epoch')


def experiment_config(args):
    if args.epochs <= 0:
        raise ValueError('epochs must be positive')
    return dict(num_trajectories=200, seed=args.seed, model_seed=args.model_seed,
                epochs=args.epochs, final_time=10.0, time_step=0.02,
                batch_size=32, learning_rate=0.001, hidden_dim=32,
                **REGIMES[args.regime])


def experiment_path(config):
    regime = 'custom'
    for name, ranges in REGIMES.items():
        if all(tuple(config[key]) == tuple(value) for key, value in ranges.items()):
            regime = name
            break
    return (Path(regime) / f"dataset_seed_{config['seed']}"
            / f"model_seed_{config['model_seed']}" / f"epochs_{config['epochs']}")


def checkpoint_path(root, config):
    return root / 'results' / 'mlp' / experiment_path(config) / 'best_model.pt'


def evaluation_path(root, config, digest):
    return (root / 'docs' / 'benchmarks' / 'mlp' / experiment_path(config)
            / f'checkpoint_{digest[:12]}')
