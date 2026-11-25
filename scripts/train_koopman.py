import argparse
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from data_driven_nonlinear_dynamics.models.koopman import KoopmanAutoencoder
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
    model = KoopmanAutoencoder(hidden_dim=hidden_dim, latent_dim=latent_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_weights = None

    best_epoch = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_states, batch_next_states in train_loader:
            optimizer.zero_grad()
            loss, _, _, _ = loss_calc_helper(model, batch_states, batch_next_states)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch + 1

    if best_weights is None:
        raise ValueError("Training produced no best epoch. Check the epoch count and validation loss.")
    model.load_state_dict(best_weights)
    return {"model": model, "mean": mean, "std": std,
            "best_epoch": best_epoch, "best_val_loss": best_val_loss}

    
def main():
    parser = argparse.ArgumentParser(description="Train the pendulum Koopman autoencoder.")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    train_koopman_model(epochs=args.epochs)

if __name__ == "__main__":
    main()
