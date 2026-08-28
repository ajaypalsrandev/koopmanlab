import numpy as np


def fit_matrix(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Fit a matrix A such that Y ≈ AX.

    Args:
        X: Current states with shape (n_features, n_samples).
        Y: Corresponding next states with shape
            (n_features, n_samples).

    Returns:
        Fitted DMD matrix with shape
        (n_features, n_features).
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2D matrices")

    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have the same shape, got "
            f"{X.shape} and {Y.shape}"
        )

    if X.shape[1] == 0:
        raise ValueError("X and Y must contain at least one sample")

    A = Y @ np.linalg.pinv(X)

    return A


def predict_one_step(
    A: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Predict the next state given the current state and DMD matrix.

    Args:
        A: DMD matrix with shape (n_features, n_features).
        states: Current states with shape (n_features, n_samples).
    Returns:
        Predicted next states with shape (n_features, n_samples).
    """
    if A.ndim != 2 or states.ndim != 2:
        raise ValueError("A and states must both be 2D matrices")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")

    if A.shape[1] != states.shape[0]:
        raise ValueError(
            f"A and states must have compatible shapes, got "
            f"{A.shape} and {states.shape}"
        )

    next_states = A @ states

    return next_states

def rollout(
        A: np.ndarray,
        initial_state: np.ndarray,
        num_steps: int,
) -> np.ndarray:
    """Rollout the DMD model for a given number of steps.

    Args:
        A: DMD matrix with shape (n_features, n_features).
        initial_state: Initial state with shape (n_features,).
        num_steps: Number of steps to rollout.

    Returns:
        Rollout states with shape (n_features, num_steps).

    """
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")

    if initial_state.ndim != 1:
        raise ValueError("initial_state must be a 1D array")

    if A.shape[1] != initial_state.shape[0]:
        raise ValueError(
            f"A and initial_state must have compatible shapes, got "
            f"{A.shape} and {initial_state.shape}"
        )

    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")

    states = np.zeros((A.shape[0], num_steps))
    states[:, 0] = initial_state

    for i in range(1, num_steps):
        states[:, i] = A @ states[:, i - 1]

    return states

