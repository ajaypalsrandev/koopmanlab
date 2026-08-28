from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from koopmanlab.evaluation.metrics import error_by_time
import koopmanlab.models.dmd as dmd_model
import train_dmd as dmd_trainer

# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1
# $env:PYTHONPATH = "$PWD\src"
# python scripts\train_dmd.py
def main() -> None:
    # Train the DMD model and get the learned matrix A and dataset
    A, dataset = dmd_trainer.train_dmd_model(200, 42, (-0.5, 0.5), (-0.5, 0.5), 50.0, 0.02)
    times = dataset["times"]

    # Extract validation data
    val_traj = dataset["val"][0]  # Use the first validation trajectory
    initial_state = val_traj[0]  # Initial state for rollout

    num_steps = val_traj.shape[0]  # Number of steps to rollout
    rollout_states = dmd_model.rollout(A, initial_state, num_steps).T
    one_step_predictions = dmd_model.predict_one_step(A, val_traj[:-1].T).T  # Predict next states
    rollout_error = error_by_time(val_traj[1:], rollout_states[1:])
    one_step_error = error_by_time(val_traj[1:], one_step_predictions)

    # plot the true vs predicted states for the first validation trajectory for both rollout and one-step prediction
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    angle_ax = axes[0, 0]
    angle_ax.plot(times, val_traj[:, 0], label="True", color="tab:blue")
    angle_ax.plot(times, rollout_states[:, 0], label="Rollout", color="tab:orange")
    angle_ax.plot(times[1:], one_step_predictions[:, 0], label="One-step Prediction", color="tab:green")
    angle_ax.set(title="True vs Predicted Angles", xlabel="Time Step", ylabel="State")
    angle_ax.legend()

    velocity_ax = axes[0, 1]
    velocity_ax.plot(times, val_traj[:, 1], label="True", color="tab:blue")
    velocity_ax.plot(times, rollout_states[:, 1], label="Rollout", color="tab:orange")
    velocity_ax.plot(times[1:], one_step_predictions[:, 1], label="One-step Prediction", color="tab:green")
    velocity_ax.set(title="True vs Predicted Angular Velocities", xlabel="Time Step", ylabel="State")
    velocity_ax.legend()

    phase_ax = axes[1, 0]
    phase_ax.plot(val_traj[:, 0], val_traj[:, 1], label="True", color="tab:blue")
    phase_ax.plot(rollout_states[:, 0], rollout_states[:, 1], label="Rollout", color="tab:orange")
    phase_ax.plot(one_step_predictions[:, 0], one_step_predictions[:, 1], label="One-step Prediction", color="tab:green")
    phase_ax.set(title="Phase Portrait", xlabel="Angle", ylabel="Angular Velocity")
    phase_ax.legend()

    error_ax = axes[1, 1]
    error_ax.plot(times[1:], rollout_error, label="Rollout", color="tab:orange")
    error_ax.plot(times[1:], one_step_error, label="One-step Prediction", color="tab:green")
    error_ax.set(title="Prediction Error vs Time", xlabel="Time Step", ylabel="Error")
    error_ax.legend()

    results_directory = PROJECT_ROOT / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    output_path = results_directory / "dmd_evaluation.png"
    plt.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    print("Learned DMD matrix A:")
    print(A)
    print("\nDataset shapes:")
    print("Training:", dataset["train"].shape)
    print("Validation:", dataset["val"].shape)
    print("Test:", dataset["test"].shape)

    eigenvalues = np.linalg.eigvals(A)
    print("\nEigenvalues of the learned DMD matrix A:")
    print(eigenvalues)
    print("\nEigenvalues magnitudes:")
    print(np.abs(eigenvalues))
    
if __name__ == "__main__":
    main()