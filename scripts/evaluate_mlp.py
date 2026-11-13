"""Evaluate a saved MLP against DMD and EDMD on the same validation data."""

from mlp_experiment import (
    add_experiment_arguments, experiment_config, checkpoint_path as default_checkpoint_path,
    evaluation_path,
)
import argparse
import hashlib
from pathlib import Path
import sys
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.data.generate_dataset import create_trajectories, create_transition_matrices, split_trajectories
from data_driven_nonlinear_dynamics.models import dmd, edmd, mlp
from evaluate_edmd import (
    evaluate_model, plot_comparison, save_model_results, save_figure, write_json,
)
import matplotlib.pyplot as plt

def evaluate_mlp_model(model, mean, std, trajectories):
    """Evaluate the MLP model on validation trajectories."""

    if trajectories.ndim != 3 or trajectories.shape[0] == 0 or trajectories.shape[1] < 2 or trajectories.shape[2] != 2:
        raise ValueError("trajectories must have shape (N, T, 2), with N > 0 and T >= 2")
    truth = trajectories[:, 1:, :]
    one_step = np.stack([
        mlp.predict_one_step(model, trajectory[:-1].T, mean, std).T
        for trajectory in trajectories
    ])
    rollouts = np.stack([
        mlp.rollout(model, trajectory[0], len(trajectory), mean, std).T
        for trajectory in trajectories
    ])
    if not all(np.isfinite(a).all() for a in (trajectories, one_step, rollouts)):
        raise ValueError("Non-finite states or predictions; inspect model stability")
    one_error = (truth - one_step) ** 2
    roll_error = (truth - rollouts[:, 1:]) ** 2
    return {
        "true_states": trajectories,
        "one_step_predictions": one_step,
        "rollout_states": rollouts,
        "one_step_mse": float(one_error.mean()),
        "rollout_mse": float(roll_error.mean()),
        "one_step_mse_per_trajectory": one_error.mean(axis=(1, 2)),
        "rollout_mse_per_trajectory": roll_error.mean(axis=(1, 2)),
        "one_step_mse_by_time": one_error.mean(axis=(0, 2)),
        "rollout_mse_by_time": roll_error.mean(axis=(0, 2)),
        "one_step_mse_by_state": one_error.mean(axis=(0, 1)),
        "rollout_mse_by_state": roll_error.mean(axis=(0, 1)),
        "persistence_one_step_mse": float(np.mean((truth - trajectories[:, :-1]) ** 2)),
    }

    


def save_mlp_results(evaluation, checkpoint, checkpoint_path, times, split_sizes, directory):
    """Save physical-unit scores separately from normalized training losses."""
    errors = evaluation["rollout_mse_per_trajectory"]
    metrics = {
        "one_step_mse_mean": evaluation["one_step_mse"],
        "rollout_mse_mean": evaluation["rollout_mse"],
        "rollout_mse_median": float(np.median(errors)),
        "rollout_mse_max": float(errors.max()),
        "rollout_mse_std_across_trajectories": float(errors.std()),
        "rollout_angle_mse": float(evaluation["rollout_mse_by_state"][0]),
        "rollout_angular_velocity_mse": float(evaluation["rollout_mse_by_state"][1]),
        "persistence_one_step_mse": evaluation["persistence_one_step_mse"],
    }
    summary = {
        "model": "MLP", "evaluated_split": "validation",
        "config": checkpoint["config"], "split_sizes": split_sizes,
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "best_epoch": checkpoint["best_epoch"],
        "best_normalized_validation_loss": checkpoint["best_val_loss"],
        "training_history": checkpoint["history"],
        "normalization": {key: checkpoint[key].tolist() for key in ("mean", "std")},
        "metrics": metrics,
        "prediction_times": times[1:].tolist(),
    }
    for key in (
        "one_step_mse_per_trajectory", "rollout_mse_per_trajectory",
        "one_step_mse_by_time", "rollout_mse_by_time",
        "one_step_mse_by_state", "rollout_mse_by_state",
    ):
        summary[key] = evaluation[key].tolist()
    write_json(directory / "metrics.json", summary)
    np.savez_compressed(
        directory / "predictions.npz", times=times,
        true_states=evaluation["true_states"],
        one_step_predictions=evaluation["one_step_predictions"],
        rollout_states=evaluation["rollout_states"],
    )
    return metrics


def plot_training_history(checkpoint, path):
    history = np.asarray(checkpoint["history"])
    epochs = np.arange(1, len(history) + 1)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, history[:, 0], label="Training")
    axis.plot(epochs, history[:, 1], label="Validation")
    axis.axvline(checkpoint["best_epoch"], color="gray", linestyle="--", label="Selected epoch")
    axis.set(xlabel="Epoch", ylabel="Normalized state MSE", title="MLP training history")
    axis.legend()
    axis.grid(alpha=0.2)
    save_figure(figure, path)


def run_evaluation(checkpoint_path, output_directory=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if "seed" not in config:
        raise ValueError("Checkpoint has no dataset seed. Regenerate it with seed recorded in the training config.")
    model, mean, std = mlp.load_checkpoint(checkpoint_path)
    dataset_keys = ("num_trajectories", "seed", "theta_range", "omega_range", "final_time", "time_step")
    dataset_config = {key: config[key] for key in dataset_keys}
    times, trajectories = create_trajectories(**dataset_config)
    train, val, test = split_trajectories(trajectories)
    split_sizes = dict(train=len(train), val=len(val), test=len(test))
    X, Y = create_transition_matrices(train)
    matrices = {"DMD": dmd.fit_matrix(X, Y), "EDMD": edmd.fit_matrix(X, Y)}
    evaluations = {
        "DMD": evaluate_model(dmd, matrices["DMD"], val),
        "EDMD": evaluate_model(edmd, matrices["EDMD"], val),
        "MLP": evaluate_mlp_model(model, mean, std, val),
    }
    # The content hash distinguishes checkpoints even when their filenames match.
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if output_directory is None:
        output_directory = evaluation_path(PROJECT_ROOT, config, digest)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name in ("DMD", "EDMD"):
        metrics[name] = save_model_results(
            name, matrices[name], evaluations[name], times, dataset_config,
            split_sizes, output_directory / name.lower(),
        )
    metrics["MLP"] = save_mlp_results(
        evaluations["MLP"], checkpoint, checkpoint_path, times,
        split_sizes, output_directory / "mlp",
    )
    indices = plot_comparison(
        times, evaluations, output_directory,
        f"Dataset seed {config['seed']} / MLP model seed {config['model_seed']}",
    )
    plot_training_history(checkpoint, output_directory / "training_history.png")
    write_json(output_directory / "comparison.json", {
        "evaluated_split": "validation", "config": config, "split_sizes": split_sizes,
        "checkpoint_sha256": digest, "best_epoch": checkpoint["best_epoch"],
        "selection_basis": "EDMD rollout MSE; all three models use the same selected trajectories",
        "selected_validation_indices": indices, "metrics": metrics,
    })
    print("Validation comparison (physical-unit state MSE):")
    for name, scores in metrics.items():
        print(f"{name:5s} one-step={scores['one_step_mse_mean']:.6g}  rollout={scores['rollout_mse_mean']:.6g}")
    print(f"Saved comparison: {output_directory}")
    return evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Explicit checkpoint overrides experiment selection; saved config controls evaluation")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    selected_checkpoint = args.checkpoint or default_checkpoint_path(
        PROJECT_ROOT, experiment_config(args)
    )
    run_evaluation(selected_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
