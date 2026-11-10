
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.models.mlp import (MLP, load_checkpoint, predict_one_step)
import torch
import copy
import numpy as np
from data_driven_nonlinear_dynamics.data.generate_dataset import create_trajectories, create_transition_matrices, split_trajectories
from torch.utils.data import DataLoader, TensorDataset

def train_mlp_model(
        num_trajectories: int = 100,
        seed: int = 42,
        theta_range: tuple[float, float] = (-0.5, 0.5),
        omega_range: tuple[float, float] = (-0.5, 0.5),
        final_time: float = 10.0,
        time_step: float = 0.02,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        hidden_dim: int = 32,
        model_seed: int = 42,
):
    
    times, trajectories = create_trajectories(
        num_trajectories=num_trajectories,
        seed=seed,
        theta_range=theta_range,
        omega_range=omega_range,
        final_time=final_time,
        time_step=time_step,
    )

    train, validation, test = split_trajectories(trajectories)
    X_train, Y_train = create_transition_matrices(train)
    X_val, Y_val = create_transition_matrices(validation)


    X_train = torch.tensor(X_train.T, dtype=torch.float32)
    Y_train = torch.tensor(Y_train.T, dtype=torch.float32)
    X_val = torch.tensor(X_val.T, dtype=torch.float32)
    Y_val = torch.tensor(Y_val.T, dtype=torch.float32)

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0, correction=0).clamp(min=1e-8)
    X_train_norm = (X_train - mean) / std
    Y_train_norm = (Y_train - mean) / std

    X_val_norm = (X_val - mean) / std
    Y_val_norm = (Y_val - mean) / std

    train_dataset = TensorDataset(X_train_norm, Y_train_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(model_seed)
    model = MLP(hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()


    best_val_loss = float("inf")
    best_weights = None

    history = []

    best_epoch = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)

        train_loss = total_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_norm)
            val_loss = loss_fn(val_predictions, Y_val_norm).item()

        history.append((train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
            f"Epoch {epoch + 1}: "
            f"train={train_loss:.6f}, "
            f"validation={val_loss:.6f}"
        )

    model.load_state_dict(best_weights)
    model.eval()

    return {
        "model": model,
        "mean": mean,
        "std": std,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }

def main() -> None:
    config = {
        "num_trajectories": 200,
        "seed": 42,
        "model_seed": 42,
        "theta_range": (-0.5, 0.5),
        "omega_range": (-0.5, 0.5),
        "final_time": 10.0,
        "time_step": 0.02,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_dim": 32,
    }

    result = train_mlp_model(**config)

    checkpoint = {
        "model_state_dict": result["model"].state_dict(),
        "mean": result["mean"],
        "std": result["std"],
        "config": config,
        "history": result["history"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
    }

    output_directory = PROJECT_ROOT / "results" / "mlp" / "small_angle" / "seed_42"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "best_model.pt"
    torch.save(checkpoint, output_path)

    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best validation loss: {result['best_val_loss']:.6g}")
    print(f"Saved checkpoint to: {output_path}")

    sample = np.array([
        [0.2, -0.3],
        [0.1, 0.0],
    ])

    before = predict_one_step(
        result["model"],
        sample,
        result["mean"],
        result["std"],
    )

    loaded_model, loaded_mean, loaded_std = load_checkpoint(
        output_path
    )

    after = predict_one_step(
        loaded_model,
        sample,
        loaded_mean,
        loaded_std,
    )

    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
    print("Checkpoint reload reproduces predictions.")


if __name__ == "__main__":
    main()
