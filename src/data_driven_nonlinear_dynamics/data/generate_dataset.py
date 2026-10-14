from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.systems.pendulum import (  # noqa: E402
    PendulumParameters,
    mechanical_energy,
    simulate_pendulum,
)

def create_trajectories(
        num_trajectories: int = 10,
        seed: int = 42,
        theta_range: tuple[float, float] = (-np.pi, np.pi),
        omega_range: tuple[float, float] = (-2.0, 2.0),
        final_time: float = 10.0,
        time_step: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Create trajectories for the pendulum system."""

    rng = np.random.default_rng(seed)
    parameters = PendulumParameters(damping=0.1)
    trajectory_list = []
    times = None
    for _ in range(num_trajectories):
        initial_theta = rng.uniform(*theta_range)
        initial_omega = rng.uniform(*omega_range)
        times, states = simulate_pendulum(
            initial_state=(initial_theta, initial_omega),
            parameters=parameters,
            final_time=final_time,
            time_step=time_step,
        )
        trajectory_list.append(states)

    trajectories = np.stack(trajectory_list, axis=0)  # Shape: (num_trajectories, num_time_steps, 2)
    return times, np.array(trajectories)

def create_transition_matrices(
    trajectories: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Create transition matrices X and Y from trajectories."""
    current_states = trajectories[:, :-1, :]  # States at time t
    next_states = trajectories[:, 1:, :]   # States at time t+1

    num_features = current_states.shape[2]

    X = current_states.reshape(-1, num_features).T  # Shape: (num_features, num_samples)
    Y = next_states.reshape(-1, num_features).T  # Shape: (num_features, num_samples)
    return X, Y

def split_trajectories(
    trajectories: np.ndarray,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_trajectories = trajectories.shape[0]
    train_end = int(num_trajectories * train_fraction)
    val_end = train_end + int(num_trajectories * val_fraction)

    train = trajectories[:train_end]
    val = trajectories[train_end:val_end]
    test = trajectories[val_end:]

    return train, val, test

        