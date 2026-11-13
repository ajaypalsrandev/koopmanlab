"""Summarize matched DMD/EDMD/MLP validation runs across dataset seeds."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from mlp_experiment import REGIMES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('DMD', 'EDMD', 'MLP')
METRICS = {
    'one_step_mse_mean': 'One-step state MSE',
    'rollout_mse_mean': 'Rollout state MSE',
    'rollout_angle_mse': 'Rollout angle MSE (rad²)',
    'rollout_angular_velocity_mse': 'Rollout velocity MSE ((rad/s)²)',
    'persistence_one_step_mse': 'Persistence one-step state MSE',
}


def aggregate(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('Need at least two finite seed-level scores')
    return {'mean': mean(values), 'sample_std_across_seeds': stdev(values),
            'values_by_seed_order': values}


def summarize(root, seeds, model_seed, epochs):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Provide at least two distinct dataset seeds')
    summary = {'dataset_seeds': seeds, 'model_seed': model_seed,
               'training_epochs': epochs, 'evaluated_split': 'validation',
               'standard_deviation': 'sample standard deviation across seed-level scores (ddof=1)',
               'regimes': {}}
    for regime, ranges in REGIMES.items():
        rows, sources = [], []
        for seed in seeds:
            folder = root / regime / f'dataset_seed_{seed}' / f'model_seed_{model_seed}' / f'epochs_{epochs}'
            matches = sorted(folder.glob('checkpoint_*/comparison.json'))
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one comparison for {folder}; found {len(matches)}. Resolve missing or duplicate runs before summarizing.')
            path = matches[0]
            data = json.loads(path.read_text())
            config = data['config']
            if (data['evaluated_split'] != 'validation' or config['seed'] != seed
                    or config['model_seed'] != model_seed or config['epochs'] != epochs
                    or any(tuple(config[k]) != tuple(v) for k, v in ranges.items())):
                raise ValueError(f'Folder and experiment metadata disagree: {path}')
            if rows:
                expected = {k: v for k, v in rows[0]['config'].items() if k != 'seed'}
                actual = {k: v for k, v in config.items() if k != 'seed'}
                if actual != expected or data['split_sizes'] != rows[0]['split_sizes']:
                    raise ValueError(f'Experiment settings differ across seeds: {path}')
            rows.append(data)
            sources.append({'dataset_seed': seed, 'comparison_file': str(path.relative_to(root)),
                            'checkpoint_sha256': data['checkpoint_sha256'],
                            'best_mlp_epoch': data['best_epoch']})
        models = {
            model: {key: aggregate([float(row['metrics'][model][key]) for row in rows])
                    for key in METRICS}
            for model in MODELS
        }
        rankings = {
            key: [{ 'dataset_seed': seed,
                    'models_best_to_worst': sorted(MODELS, key=lambda m: row['metrics'][m][key])}
                  for seed, row in zip(seeds, rows)]
            for key in ('one_step_mse_mean', 'rollout_mse_mean')
        }
        summary['regimes'][regime] = {
            'shared_config': {k: v for k, v in rows[0]['config'].items() if k != 'seed'},
            'split_sizes': rows[0]['split_sizes'], 'sources': sources,
            'models': models, 'rankings_by_seed': rankings,
        }
    return summary


def render_markdown(summary):
    seeds = ', '.join(map(str, summary['dataset_seeds']))
    lines = [
        '# DMD, EDMD, and MLP validation summary', '',
        f"Dataset seeds: **{seeds}**. MLP model seed: **{summary['model_seed']}**. "
        f"MLP training budget: **{summary['training_epochs']} epochs**.", '',
        'Each table entry is the mean ± sample standard deviation of the seed-level '
        'scores (ddof=1). Each run averages over its validation trajectories. '
        'These are not confidence intervals or within-run trajectory standard deviations.', '',
        'The MLP uses the checkpoint selected by one-step validation loss within each '
        'training run. Dataset seed varies; model initialization seed is held fixed. '
        'All three models use identical training and validation data within each run. '
        'Runs with other training budgets are excluded.', '',
        'Aggregate state MSE combines angle and angular-velocity squared errors without '
        'normalization; separate state errors retain their respective units. '
        'Persistence is a one-step reference, not a rollout baseline.', '',
    ]
    for regime, result in summary['regimes'].items():
        lines += [f"## {regime.replace('_', ' ').capitalize()}", '',
                  '| Metric | DMD | EDMD | MLP |', '|---|---:|---:|---:|']
        for key, label in METRICS.items():
            cells = []
            for model in MODELS:
                stats = result['models'][model][key]
                cells.append(f"{stats['mean']:.4g} ± {stats['sample_std_across_seeds']:.4g}")
            lines.append('| ' + ' | '.join([label] + cells) + ' |')
        lines += ['', '### Individual rollout scores', '',
                  '| Dataset seed | DMD | EDMD | MLP | Selected MLP epoch |',
                  '|---|---:|---:|---:|---:|']
        for i, source in enumerate(result['sources']):
            scores = [f"{result['models'][m]['rollout_mse_mean']['values_by_seed_order'][i]:.6g}" for m in MODELS]
            lines.append('| ' + ' | '.join([str(source['dataset_seed'])] + scores + [str(source['best_mlp_epoch'])]) + ' |')
        lines += ['', 'Rankings below run from lowest to highest error:', '']
        for key, label in [('one_step_mse_mean', 'One-step'), ('rollout_mse_mean', 'Rollout')]:
            for row in result['rankings_by_seed'][key]:
                lines.append(f"- {label}, seed {row['dataset_seed']}: " + ' → '.join(row['models_best_to_worst']) + '.')
        lines += ['', 'Source runs:', '']
        for source in result['sources']:
            lines.append(f"- [Seed {source['dataset_seed']}]({source['comparison_file']})")
        lines.append('')
    lines += ['## Reproduce this summary', '', 'From the repository root:', '',
              '```bash', f"python scripts/summarize_mlp.py --seeds {' '.join(map(str, summary['dataset_seeds']))} --model-seed {summary['model_seed']} --epochs {summary['training_epochs']}", '```', '',
              'The script reads existing comparison files; it does not retrain or '
              'evaluate any models. It rejects missing runs, duplicate checkpoints '
              'for a selected experiment, and inconsistent settings rather than '
              'silently averaging incompatible experiments.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'docs/benchmarks/mlp')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    summary = summarize(args.root, args.seeds, args.model_seed, args.epochs)
    suffix = f"model_seed_{args.model_seed}_epochs_{args.epochs}_seeds_" + '_'.join(map(str, args.seeds))
    path = args.root / f'summary_{suffix}'
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.md').write_text(render_markdown(summary))
    print(f'Saved {path.with_suffix(".md")}')
    print(f'Saved {path.with_suffix(".json")}')


if __name__ == '__main__':
    main()
