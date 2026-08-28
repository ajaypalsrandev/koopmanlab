import numpy as np

def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean squared error between true and predicted values.

    Args:
        y_true: True values with shape (n_samples, n_features).
        y_pred: Predicted values with shape (n_samples, n_features).
    Returns:
        Mean squared error as a float.
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must both be 2D matrices")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )

    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def error_by_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute the mean squared error at each time step.

    Args:
        y_true: True values with shape (n_samples, n_features).
        y_pred: Predicted values with shape (n_samples, n_features).
    Returns:
        Mean squared error at each time step with shape (n_samples,).
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must both be 2D matrices")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )

    mse_by_time = np.mean((y_true - y_pred) ** 2, axis=1)
    return mse_by_time