from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from koopmanlab.data.generate_dataset import (
    create_trajectories,
    create_transition_matrices,
    split_trajectories,
)
from koopmanlab.models.dmd import fit_matrix

def train_dmd_model(
    num_trajectories: int = 100,
    seed: int = 42, 
    theta_range: tuple[float, float] = (-0.5, 0.5),
    omega_range: tuple[float, float] = (-0.5, 0.5),
    final_time: float = 10.0,
    time_step: float = 0.02,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Generate trajectories
    times, trajectories = create_trajectories(
        num_trajectories=num_trajectories,
        seed=seed,
        theta_range=theta_range,
        omega_range=omega_range,
        final_time=final_time,
        time_step=time_step,
    )

    # Split trajectories into train, validation, and test sets
    train, val, test = split_trajectories(trajectories)

    # Create transition matrices for each set
    X_train, Y_train = create_transition_matrices(train)
    X_val, Y_val = create_transition_matrices(val)
    X_test, Y_test = create_transition_matrices(test)

    # Fit DMD model on training data
    A = fit_matrix(X_train, Y_train)

    dataset = {
        "times" : times,
        "train": train,
        "val": val,
        "test": test,
    }
    return A, dataset

def main() -> None:
    A, dataset = train_dmd_model()

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