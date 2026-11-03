"""Compare DMD and EDMD on shared validation trajectories and save artifacts."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.data.generate_dataset import (
    create_trajectories, create_transition_matrices, split_trajectories,
)
from data_driven_nonlinear_dynamics.models import dmd, edmd


def evaluate_model(model, matrix, trajectories):
    """Return predictions and errors; the supplied initial state is not scored."""
    if trajectories.ndim != 3 or trajectories.shape[0] == 0 or trajectories.shape[1] < 2 or trajectories.shape[2] != 2:
        raise ValueError("trajectories must have shape (N, T, 2), with N > 0 and T >= 2")
    truth = trajectories[:, 1:, :]
    one_step = np.stack([
        model.predict_one_step(matrix, trajectory[:-1].T).T
        for trajectory in trajectories
    ])
    rollouts = np.stack([
        model.rollout(matrix, trajectory[0], len(trajectory)).T
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


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_model_results(name, matrix, evaluation, times, config, split_sizes, directory):
    """JSON is tracked; compressed full predictions are local, ignored artifacts."""
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
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(matrix)))),
    }
    summary = {
        "model": name, "evaluated_split": "validation", "config": config,
        "split_sizes": split_sizes, "metrics": metrics,
        "features": ["theta", "omega"] if name == "DMD" else ["theta", "omega", "sin(theta)", "cos(theta)", "1"],
        "learned_matrix": matrix.tolist(),
        "one_step_mse_by_state": evaluation["one_step_mse_by_state"].tolist(),
        "rollout_mse_per_trajectory": errors.tolist(),
        "one_step_mse_per_trajectory": evaluation["one_step_mse_per_trajectory"].tolist(),
        "prediction_times": times[1:].tolist(),
        "one_step_mse_by_time": evaluation["one_step_mse_by_time"].tolist(),
        "rollout_mse_by_time": evaluation["rollout_mse_by_time"].tolist(),
    }
    write_json(directory / "metrics.json", summary)
    np.savez_compressed(
        directory / "predictions.npz", times=times,
        true_states=evaluation["true_states"],
        one_step_predictions=evaluation["one_step_predictions"],
        rollout_states=evaluation["rollout_states"],
    )
    return metrics


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_comparison(times, evaluations, directory, title):
    # Use EDMD's selected cases for BOTH models so the initial conditions match.
    errors = evaluations["EDMD"]["rollout_mse_per_trajectory"]
    indices = {
        "representative": int(np.argmin(np.abs(errors - np.median(errors)))),
        "worst": int(np.argmax(errors)),
    }
    for selection, index in indices.items():
        truth = evaluations["EDMD"]["true_states"][index]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        figure.suptitle(f"{title} | EDMD {selection} case | validation trajectory {index}")
        axes[0, 0].plot(times, truth[:, 0], color="black", label="True")
        axes[0, 1].plot(times, truth[:, 1], color="black", label="True")
        axes[1, 0].plot(truth[:, 0], truth[:, 1], color="black", label="True")
        for name, result in evaluations.items():
            predicted = result["rollout_states"][index]
            axes[0, 0].plot(times, predicted[:, 0], label=name)
            axes[0, 1].plot(times, predicted[:, 1], label=name)
            axes[1, 0].plot(predicted[:, 0], predicted[:, 1], label=name)
            error = np.mean((truth[1:] - predicted[1:]) ** 2, axis=1)
            axes[1, 1].plot(times[1:], error, label=name)
        axes[0, 0].set(xlabel="Time (s)", ylabel="Angle (rad)", title="Recursive angle prediction")
        axes[0, 1].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s)", title="Recursive velocity prediction")
        axes[1, 0].set(xlabel="Angle (rad)", ylabel="Angular velocity (rad/s)", title="Phase portrait")
        axes[1, 1].set(xlabel="Time (s)", ylabel="State MSE", title="Error for this trajectory")
        for axis in axes.flat:
            axis.legend()
            axis.grid(alpha=0.2)
        save_figure(figure, directory / f"{selection}.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    figure.suptitle(f"{title} | mean across all validation trajectories")
    for axis, key, label in zip(axes, ["one_step_mse_by_time", "rollout_mse_by_time"], ["One-step", "Recursive rollout"]):
        for name, result in evaluations.items():
            axis.plot(times[1:], result[key], label=name)
        axis.set(title=label, xlabel="Time (s)", ylabel="State MSE")
        axis.legend()
        axis.grid(alpha=0.2)
    save_figure(figure, directory / "error_by_time.png")
    return indices


def run_benchmarks(seeds, output_directory):
    regimes = {
        "small_angle": {"theta_range": (-0.5, 0.5), "omega_range": (-0.5, 0.5)},
        "broad_motion": {"theta_range": (-2.5, 2.5), "omega_range": (-2.0, 2.0)},
    }
    aggregate = {"seeds": seeds, "evaluated_split": "validation", "regimes": {}}
    for regime_name, ranges in regimes.items():
        runs = {"DMD": [], "EDMD": []}
        for seed in seeds:
            config = dict(num_trajectories=200, seed=seed, final_time=10.0, time_step=0.02, **ranges)
            times, trajectories = create_trajectories(**config)
            train, validation, test = split_trajectories(trajectories)
            split_sizes = dict(train=len(train), val=len(validation), test=len(test))
            X, Y = create_transition_matrices(train)
            directory = output_directory / regime_name / f"seed_{seed}"
            evaluations = {}
            for name, model in [("DMD", dmd), ("EDMD", edmd)]:
                matrix = model.fit_matrix(X, Y)
                result = evaluate_model(model, matrix, validation)
                evaluations[name] = result
                metrics = save_model_results(name, matrix, result, times, config, split_sizes, directory / name.lower())
                runs[name].append(metrics)
                print(f"{regime_name} seed={seed} {name}: one-step={result['one_step_mse']:.6g}, rollout={result['rollout_mse']:.6g}", flush=True)
            indices = plot_comparison(times, evaluations, directory, f"{regime_name} / seed {seed}")
            write_json(directory / "comparison.json", {
                "evaluated_split": "validation", "config": config,
                "selection_basis": "EDMD rollout MSE; both models plotted on the same trajectory",
                "selected_validation_indices": indices,
                "metrics": {name: runs[name][-1] for name in runs},
            })
        aggregate["regimes"][regime_name] = {}
        for name, rows in runs.items():
            aggregate["regimes"][regime_name][name] = {
                key: {
                    "mean": float(np.mean([row[key] for row in rows])),
                    "sample_std_across_seeds": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else None,
                }
                for key in ("one_step_mse_mean", "rollout_mse_mean", "rollout_angle_mse", "rollout_angular_velocity_mse")
            }
    write_json(output_directory / "summary.json", aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs" / "benchmarks" / "edmd")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    run_benchmarks(args.seeds, args.output_dir)


if __name__ == "__main__":
    main()
