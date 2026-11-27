"""Score a trained Koopman autoencoder on the pendulum validation trajectories."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from evaluate_mlp import PROJECT_ROOT, save_mlp_results, score_predictions
from mlp_experiment import add_experiment_arguments, checkpoint_path, evaluation_path, experiment_config
from data_driven_nonlinear_dynamics.data.generate_dataset import create_trajectories, split_trajectories
from data_driven_nonlinear_dynamics.models.koopman import load_checkpoint


@torch.no_grad()
def evaluate_koopman_model(model, mean, std, trajectories):
    states = torch.as_tensor(trajectories, dtype=torch.float32)
    normalized = (states - mean) / std
    one_step = model(normalized[:, :-1]) * std + mean

    rollouts = np.empty_like(trajectories)
    rollouts[:, 0] = trajectories[:, 0]
    latent = model.encode(normalized[:, 0])
    for step in range(1, states.shape[1]):
        latent = model.advance_latent(latent)
        rollouts[:, step] = (model.decode(latent) * std + mean).numpy()
    return score_predictions(trajectories, one_step.numpy(), rollouts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    path = args.checkpoint or checkpoint_path(PROJECT_ROOT, experiment_config(args), model="koopman")
    model, checkpoint = load_checkpoint(path)
    config = checkpoint["config"]
    keys = ("num_trajectories", "seed", "theta_range", "omega_range", "final_time", "time_step")
    times, trajectories = create_trajectories(**{key: config[key] for key in keys})
    train, val, test = split_trajectories(trajectories)
    scores = evaluate_koopman_model(model, checkpoint["mean"], checkpoint["std"], val)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    directory = evaluation_path(PROJECT_ROOT, config, digest, model="koopman")
    directory.mkdir(parents=True, exist_ok=True)
    metrics = save_mlp_results(scores, checkpoint, path, times,
                              dict(train=len(train), val=len(val), test=len(test)),
                              directory, model_name="Koopman")
    print(f"Validation one-step MSE={metrics['one_step_mse_mean']:.6g}, "
          f"rollout MSE={metrics['rollout_mse_mean']:.6g}")
    print(f"Saved evaluation: {directory}")


if __name__ == "__main__":
    main()
