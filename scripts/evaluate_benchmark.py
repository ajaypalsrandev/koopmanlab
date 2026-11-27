"""Compare five frozen model configurations on a shared pendulum split."""
import argparse
import hashlib

import numpy as np

from evaluate_edmd import evaluate_model, write_json
from evaluate_koopman import evaluate_koopman_model
from evaluate_mlp import PROJECT_ROOT, evaluate_mlp_model
from mlp_experiment import (
    add_experiment_arguments, checkpoint_path, experiment_config, experiment_path,
)
from data_driven_nonlinear_dynamics.data.generate_dataset import (
    create_trajectories, create_transition_matrices, split_trajectories,
)
from data_driven_nonlinear_dynamics.models import dmd, edmd, koopman, mlp


def run_benchmark(config, regime, split):
    dataset_keys = ("num_trajectories", "seed", "theta_range", "omega_range", "final_time", "time_step")
    _, trajectories = create_trajectories(**{key: config[key] for key in dataset_keys})
    train, validation, test = split_trajectories(trajectories)
    selected = {"validation": validation, "test": test}[split]
    print(f"{regime}, dataset seed {config['seed']}, {split} {selected.shape}")

    inputs, targets = create_transition_matrices(train)
    evaluations = {}
    for name, module in (("DMD", dmd), ("EDMD", edmd)):
        matrix = module.fit_matrix(inputs, targets)
        evaluations[name] = evaluate_model(module, matrix, selected)

    mlp_path = checkpoint_path(PROJECT_ROOT, config)
    model, mean, std = mlp.load_checkpoint(mlp_path)
    evaluations["MLP"] = evaluate_mlp_model(model, mean, std, selected)
    paths = {"MLP": mlp_path}
    original = checkpoint_path(PROJECT_ROOT, config, model="koopman")
    for name, path, horizon in (
        ("Original Koopman", original, 0),
        ("10-step Koopman", original.parent / "rollout_horizon_10" / original.name, 10),
    ):
        model, checkpoint = koopman.load_checkpoint(path)
        if checkpoint["config"].get("rollout_horizon", 0) != horizon:
            raise ValueError(f"Unexpected training horizon in {path}")
        evaluations[name] = evaluate_koopman_model(model, checkpoint["mean"], checkpoint["std"], selected)
        paths[name] = path

    metrics = {}
    print(f"{'Model':18} {'One-step MSE':>14} {'Rollout MSE':>14}")
    for name, result in evaluations.items():
        np.testing.assert_array_equal(result["true_states"], selected)
        metrics[name] = {key: result[key] for key in ("one_step_mse", "rollout_mse")}
        print(f"{name:18} {result['one_step_mse']:14.7g} {result['rollout_mse']:14.7g}")

    comparison = {
        "regime": regime, "evaluated_split": split, "config": config,
        "split_sizes": dict(train=len(train), validation=len(validation), test=len(test)),
        "scored_shape": list(selected.shape),
        "checkpoints": {name: {"path": str(path.relative_to(PROJECT_ROOT)),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in paths.items()},
        "metrics": metrics,
    }
    directory = PROJECT_ROOT / "docs" / "benchmarks" / "final" / split / experiment_path(config)
    write_json(directory / "comparison.json", comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    run_benchmark(experiment_config(args), args.regime, args.split)


if __name__ == "__main__":
    main()
