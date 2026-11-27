import torch
from torch import nn
import numpy as np

class KoopmanAutoencoder(nn.Module):
    def __init__(self, hidden_dim: int = 32, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.latent = nn.Linear(latent_dim, latent_dim, bias=False)  # Linear layer for latent space dynamics

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def advance_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.latent(latent)

    def reconstruct(self, states: torch.Tensor) -> torch.Tensor:
        latent = self.encode(states)
        reconstructed = self.decode(latent)
        return reconstructed

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        latent = self.encode(states)
        latent_next = self.advance_latent(latent)
        reconstructed = self.decode(latent_next)
        return reconstructed


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = KoopmanAutoencoder(config["hidden_dim"], config["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
