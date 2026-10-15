from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.evaluation.metrics import error_by_time
import data_driven_nonlinear_dynamics.models.dmd as dmd_model
import train_dmd as dmd_trainer

# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1
# $env:PYTHONPATH = "$PWD\src"
# python scripts\train_dmd.py

# source .venv/bin/activate
# export PYTHONPATH="$PWD/src"

def main() -> None:
    # Train the DMD model and get the learned matrix A and dataset
    A1, dataset1 = dmd_trainer.train_dmd_model(200, 42, (-0.5, 0.5), (-0.5, 0.5), 10.0, 0.02)
    A2, dataset2 = dmd_trainer.train_dmd_model(200, 42, (-2.5, 2.5), (-2.0, 2.0), 10.0, 0.02)
    times1 = dataset1["times"]
    times2 = dataset2["times"]


    # Extract validation data
    evaluation1 = evaluate_split(A1, dataset1["val"])
    evaluation2 = evaluate_split(A2, dataset2["val"])

    for name, evaluation in [
        ("Small-angle", evaluation1),
        ("Broad-motion", evaluation2),
    ]:
        one_step_errors = evaluation["one_step_mse_per_trajectory"]
        rollout_errors = evaluation["rollout_mse_per_trajectory"]

        print(f"\n{name}")
        print("Trajectories evaluated:", len(rollout_errors))
        print("Mean one-step MSE:", one_step_errors.mean())
        print("Mean rollout MSE:", rollout_errors.mean())
        print("Median rollout MSE:", np.median(rollout_errors))
        print("Worst rollout MSE:", rollout_errors.max())
        print("Rollout MSE [angle, velocity]:",
            evaluation["rollout_mse_by_state"])
        print("Persistence one-step MSE:",
            evaluation["persistence_one_step_mse"])

    plot_trajectory_comparison(
        times1,
        evaluation1,
        "dmd_evaluation_small_angle.png",
    )
    plot_trajectory_comparison(
        times2,
        evaluation2,
        "dmd_evaluation_broad_motion.png",
    )
    
    print("Learned DMD matrix A:")
    print(A1)
    print("\nDataset shapes:")
    print("Training:", dataset1["train"].shape)
    print("Validation:", dataset1["val"].shape)
    print("Test:", dataset1["test"].shape)

    eigenvalues = np.linalg.eigvals(A1)
    print("\nEigenvalues of the learned DMD matrix A:")
    print(eigenvalues)
    print("\nEigenvalues magnitudes:")
    print(np.abs(eigenvalues))

    print("\nLearned DMD matrix A2:")
    print(A2)
    print("\nDataset shapes:")
    print("Training:", dataset2["train"].shape)
    print("Validation:", dataset2["val"].shape)
    print("Test:", dataset2["test"].shape)

    eigenvalues2 = np.linalg.eigvals(A2)
    print("\nEigenvalues of the learned DMD matrix A2:")
    print(eigenvalues2)
    print("\nEigenvalues magnitudes:")
    print(np.abs(eigenvalues2))


def evaluate_split(A, trajectories):
    truth = trajectories[:, 1:, :]
    current = trajectories[:, :-1, :]

    one_step = current @ A.T

    rollouts = np.stack([dmd_model.rollout(A, traj[0], len(traj)).T for traj in trajectories])
    predicted_rollouts = rollouts[:, 1:, :]

    one_step_squared_error = (truth - one_step) ** 2
    rollout_squared_error = (truth - predicted_rollouts) ** 2

    return {
        # Full arrays for selecting and plotting individual trajectories.
        "true_states": trajectories,
        "rollout_states": rollouts,
        "one_step_predictions": one_step,

        # One score per trajectory.
        "one_step_mse_per_trajectory":
            one_step_squared_error.mean(axis=(1, 2)),
        "rollout_mse_per_trajectory":
            rollout_squared_error.mean(axis=(1, 2)),

        # One score per prediction time, averaged over all trajectories.
        "one_step_mse_by_time":
            one_step_squared_error.mean(axis=(0, 2)),
        "rollout_mse_by_time":
            rollout_squared_error.mean(axis=(0, 2)),

        # Separate scores for angle and angular velocity.
        "one_step_mse_by_state":
            one_step_squared_error.mean(axis=(0, 1)),
        "rollout_mse_by_state":
            rollout_squared_error.mean(axis=(0, 1)),

        # Baseline: predict that the next state equals the current state.
        "persistence_one_step_mse":
            float(np.mean((truth - current) ** 2)),
    }

def plot_trajectory_comparison(times, evaluation, output_filename):
        errors = evaluation["rollout_mse_per_trajectory"]
        index  = int(np.argmin(np.abs(errors - np.median(errors))))
    
        true_states = evaluation["true_states"][index]
        rollout_states = evaluation["rollout_states"][index]
        one_step_predictions = evaluation["one_step_predictions"][index]
    
        # plot the true vs predicted states for the first validation trajectory for both rollout and one-step prediction
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        angle_ax = axes[0, 0]
        angle_ax.plot(times, true_states[:, 0], label="True")
        angle_ax.plot(times, rollout_states[:, 0], label="Rollout")
        angle_ax.plot(
            times[1:],
            one_step_predictions[:, 0],
        label="One-step prediction",
        )
        angle_ax.set(
            title="Representative validation trajectory",
            xlabel="Time (s)",
            ylabel="Angle (rad)",
        )
        angle_ax.legend()
    
        velocity_ax = axes[0, 1]
        velocity_ax.plot(times, true_states[:, 1], label="True", color="tab:blue")
        velocity_ax.plot(times, rollout_states[:, 1], label="Rollout", color="tab:orange")
        velocity_ax.plot(times[1:], one_step_predictions[:, 1], label="One-step Prediction", color="tab:green")
        velocity_ax.set(title="True vs Predicted Angular Velocities", xlabel="Time Step", ylabel="State")
        velocity_ax.legend()
    
        phase_ax = axes[1, 0]
        phase_ax.plot(true_states[:, 0], true_states[:, 1], label="True", color="tab:blue")
        phase_ax.plot(rollout_states[:, 0], rollout_states[:, 1], label="Rollout", color="tab:orange")
        phase_ax.plot(one_step_predictions[:, 0], one_step_predictions[:, 1], label="One-step Prediction", color="tab:green")
        phase_ax.set(title="Phase Portrait", xlabel="Angle", ylabel="Angular Velocity")
        phase_ax.legend()
    
        error_ax = axes[1, 1]
        error_ax.plot(times[1:], evaluation["rollout_mse_by_time"], label="Rollout", color="tab:orange")
        error_ax.plot(times[1:], evaluation["one_step_mse_by_time"], label="One-step Prediction", color="tab:green")
        error_ax.set(title="Mean error across all validation trajectories", xlabel="Time (s)", ylabel="Mean state squared error")
        error_ax.legend()
    
        results_directory = PROJECT_ROOT / "results"
        results_directory.mkdir(parents=True, exist_ok=True)
        output_path = results_directory / output_filename
        plt.tight_layout()
        figure.savefig(output_path, dpi=180)
        plt.close(figure)

if __name__ == "__main__":
    main()
