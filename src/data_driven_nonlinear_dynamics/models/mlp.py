import torch
from torch import nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    
    model = MLP(hidden_dim=checkpoint["config"]["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["std"]

def predict_one_step(model, states, mean, std):
    """Predict physical states; input and output have shape (2, N)."""
    states = np.asarray(states)

    if states.ndim != 2 or states.shape[0] != 2:
        raise ValueError("states must have shape (2, N)")

    # Existing project models use columns; PyTorch uses rows.
    inputs = torch.as_tensor(states.T, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        normalized_inputs = (inputs - mean) / std
        normalized_predictions = model(normalized_inputs)
        predictions = normalized_predictions * std + mean

    return predictions.cpu().numpy().T

def rollout(model, initial_state, n_steps, mean, std):
    """Return shape (2, n_steps), including the initial state."""
    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (2,):
        raise ValueError("initial_state must have shape (2,)")
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")

    states = np.empty((2, n_steps))
    states[:, 0] = initial_state

    for step in range(1, n_steps):
        current = states[:, step - 1:step]

        states[:, step] = predict_one_step(
            model, current, mean, std
        )[:, 0]

    return states