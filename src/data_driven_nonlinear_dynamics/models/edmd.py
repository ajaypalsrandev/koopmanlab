import numpy as np

def lift_states(states: np.ndarray) -> np.ndarray:
    """converts states of shape (2, N) into features of shape (5, N) by adding adding sine and cosine of the angle, as well as a constatnt"""

    if states.ndim != 2 or states.shape[0] != 2:
        raise ValueError("states must be a 2D matrix with shape (2, N)")
    
    theta = states[0, :]

    return np.vstack((states, np.sin(theta), np.cos(theta), np.ones_like(theta)))


def fit_matrix(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Fit a matrix K such that lifted_states(Y) ≈ K * lifted_states(X).

    Args:
        X: Current states with shape (2, N).
        Y: Corresponding next states with shape (2, N).

    Returns:
        Fitted EDMD matrix with shape
        (5, 5).
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

    Z_current = lift_states(X)
    Z_next = lift_states(Y)

    K = Z_next @ np.linalg.pinv(Z_current)
    return K

def predict_one_step(
    K: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Predict the next state given the current state and EDMD matrix.

    Args:
        K: EDMD matrix with shape (5, 5).
        states: Current states with shape (2, N).

    Returns:
        Predicted next states with shape (2, N).
    
    """

    if K.shape != (5, 5):
        raise ValueError(f"K must be a 5x5 matrix, got shape {K.shape}")

    predicted_features = K @ lift_states(states)
    return predicted_features[:2, :]  # Return only the original state dimensions

def rollout(
    K: np.ndarray,
    initial_state: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Roll out the dynamics for a given number of steps.

    Args:
        K: EDMD matrix with shape (5, 5).
        initial_state: Initial state with shape (2,).
        n_steps: Number of steps to roll out including the initial state.

    Returns:
        Rolled out states with shape (2, n_steps).
    """
    if K.shape != (5, 5):
        raise ValueError(f"K must be a 5x5 matrix, got shape {K.shape}")

    if initial_state.shape != (2,):
        raise ValueError(
            f"initial_state must have shape (2,), got {initial_state.shape}"
        )

    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError(f"n_steps must be a positive integer")

    initial_features = lift_states(initial_state[:, None])[:, 0]  # Convert to shape (5,)
    rolled_out_features = np.zeros((5, n_steps))
    rolled_out_features[:, 0] = initial_features

    for i in range(1, n_steps):
        rolled_out_features[:, i] = K @ rolled_out_features[:, i - 1]

    return rolled_out_features[:2, :]  # Return only the original state dimensions


