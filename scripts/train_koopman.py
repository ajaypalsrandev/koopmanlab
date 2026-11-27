import argparse
from mlp_experiment import add_experiment_arguments, experiment_config, checkpoint_path
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from data_driven_nonlinear_dynamics.models.koopman import KoopmanAutoencoder, load_checkpoint
from data_driven_nonlinear_dynamics.data.generate_dataset import create_trajectories, split_trajectories, create_transition_matrices
import torch
from torch.utils.data import DataLoader, TensorDataset

def loss_calc_helper(model, norm_states, norm_next_states):
    """Compute the loss for a batch of states."""

    encoded_current = model.encode(norm_states)
    recon_current = model.decode(encoded_current)
    encoded_next = model.encode(norm_next_states)
    latent_prediction = model.advance_latent(encoded_current)
    predicted_next = model.decode(latent_prediction)

    reconstruction_loss = ((recon_current - norm_states) ** 2).mean()
    prediction_loss = ((predicted_next - norm_next_states) ** 2).mean()
    latent_loss = ((latent_prediction - encoded_next) ** 2).mean()

    total_loss = reconstruction_loss + prediction_loss + latent_loss

    return total_loss, reconstruction_loss, prediction_loss, latent_loss 

def trajectory_windows(states, horizon):
    """Slide along time within each trajectory, then combine the windows."""
    if not 1 <= horizon < states.size(1):
        raise ValueError("rollout horizon must be shorter than a trajectory and positive")
    windows = states.unfold(1, horizon + 1, 1).transpose(-1, -2)
    return windows.flatten(0, 1)


def rollout_loss(model, windows):
    """Forecast every target from a single encoding of the initial state."""
    latent = model.encode(windows[:, 0])
    forecasts = []
    for _ in range(windows.size(1) - 1):
        latent = model.advance_latent(latent)
        forecasts.append(latent)
    predictions = model.decode(torch.stack(forecasts, dim=1))
    return (predictions - windows[:, 1:]).square().mean()


def train_koopman_model(num_trajectories: int = 100,
        seed: int = 42,
        theta_range: tuple[float, float] = (-0.5, 0.5),
        omega_range: tuple[float, float] = (-0.5, 0.5),
        final_time: float = 10.0,
        time_step: float = 0.02,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        hidden_dim: int = 32,
        latent_dim: int = 4,
        model_seed: int = 42,
        rollout_horizon: int = 0
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

    if rollout_horizon < 0:
        raise ValueError("rollout horizon must be non-negative")
    if rollout_horizon:
        states = (torch.as_tensor(train, dtype=torch.float32) - mean) / std
        train_dataset = trajectory_windows(states, rollout_horizon)
    else:
        train_dataset = TensorDataset(X_train_norm, Y_train_norm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(model_seed)
    model = KoopmanAutoencoder(hidden_dim=hidden_dim, latent_dim=latent_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_weights = None

    history = []
    best_epoch = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if rollout_horizon:
                batch_states, batch_next_states = batch[:, 0], batch[:, 1]
            else:
                batch_states, batch_next_states = batch
            loss, _, _, _ = loss_calc_helper(model, batch_states, batch_next_states)
            if rollout_horizon:
                loss = loss + rollout_loss(model, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_states.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            _, recon, prediction, latent = loss_calc_helper(model, X_val_norm, Y_val_norm)
            val_loss = prediction.item()

        print(f"Epoch {epoch + 1}/{epochs}, train total={train_loss:.4f}, "
              f"val reconstruction={recon.item():.4f}, val prediction={val_loss:.4f}, "
              f"val latent={latent.item():.4f}")

        history.append({"train_total": train_loss, "val_prediction": val_loss,
                        "val_reconstruction": recon.item(), "val_latent": latent.item()})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch + 1

    if best_weights is None:
        raise ValueError("Training produced no best epoch. Check the epoch count and validation loss.")
    model.load_state_dict(best_weights)
    return {"model": model, "mean": mean, "std": std,
            "history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}

    
def main():
    parser = argparse.ArgumentParser(description="Train the pendulum Koopman autoencoder.")
    
    add_experiment_arguments(parser)
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=0,
        help="Number of future steps in the rollout loss. Use 0 for one-step training."
    )

    args = parser.parse_args()

    if args.rollout_horizon < 0:
        parser.error("rollout_horizon must be non-negative")

    config = experiment_config(args)
    config["latent_dim"] = 4
    config["rollout_horizon"] = args.rollout_horizon

    result = train_koopman_model(**config)
    model = result.pop("model")
    path = checkpoint_path(PROJECT_ROOT, config, model="koopman")
    if args.rollout_horizon > 0:
        path = path.parent / f"rollout_horizon_{args.rollout_horizon}" / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(result, config=config, model_state_dict=model.state_dict()), path)

    restored, checkpoint = load_checkpoint(path)
    state = torch.tensor([[0.2, -0.1]])
    with torch.no_grad():
        expected = model((state - result["mean"]) / result["std"]) * result["std"] + result["mean"]
        mean, std = checkpoint["mean"], checkpoint["std"]
        actual = restored((state - mean) / std) * std + mean
    torch.testing.assert_close(actual, expected)
    print(f"Verified checkpoint from epoch {result['best_epoch']}: {path}")


if __name__ == "__main__":
    main()
