from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_koopman import KoopmanAutoencoder, loss_calc_helper, rollout_loss, trajectory_windows


class KoopmanTrainingTests(unittest.TestCase):
    def test_windows_include_last_target_and_stay_inside_trajectory(self):
        states = torch.arange(16).reshape(2, 4, 2)
        windows = trajectory_windows(states, horizon=2)
        expected = torch.stack([states[0, :3], states[0, 1:], states[1, :3], states[1, 1:]])
        torch.testing.assert_close(windows, expected)

    def test_rollout_aligns_with_one_step_and_advances_same_latent(self):
        model = KoopmanAutoencoder()
        windows = torch.randn(3, 3, 2)
        _, _, prediction_loss, _ = loss_calc_helper(model, windows[:, 0], windows[:, 1])
        torch.testing.assert_close(rollout_loss(model, windows[:, :2]), prediction_loss)
        z1 = model.advance_latent(model.encode(windows[:, 0]))
        second_error = (model.decode(model.advance_latent(z1)) - windows[:, 2]).square().mean()
        loss = rollout_loss(model, windows)
        expected = (prediction_loss + second_error) / 2
        torch.testing.assert_close(loss, expected)
        params = tuple(model.parameters())
        for actual_grad, expected_grad in zip(torch.autograd.grad(loss, params),
                                              torch.autograd.grad(expected, params)):
            torch.testing.assert_close(actual_grad, expected_grad)
