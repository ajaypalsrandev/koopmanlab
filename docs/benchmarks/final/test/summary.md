# Held-out pendulum benchmark

For each regime and dataset seed, all five models scored the same 30 test trajectories. DMD and EDMD used the 140 training trajectories for fitting. The neural models used model seed 42, a budget of 100 epochs, and their saved validation-selected weights and normalization. Each trajectory has 500 forecast steps at 0.02 s per step. Both metrics average squared errors over angle and angular velocity in their original units. Rollout MSE averages all forecast steps, excluding the given initial state.

Values are the mean ± sample standard deviation of the three dataset-seed scores.

## Small angle

| Model | One-step MSE | 10-second rollout MSE |
|---|---:|---:|
| DMD | 3.5436e-08 ± 5.18e-09 | 0.00035664 ± 5.86e-05 |
| EDMD | 1.2133e-10 ± 1.77e-11 | 0.00034848 ± 5.59e-05 |
| MLP | 5.4286e-08 ± 1.14e-08 | 0.0012984 ± 0.00113 |
| Original Koopman | 4.5512e-07 ± 1.3e-07 | 0.00045613 ± 5.6e-05 |
| 10-step Koopman | 7.2025e-07 ± 1.73e-07 | 0.00050506 ± 0.000201 |

[Seed 42](small_angle/dataset_seed_42/model_seed_42/epochs_100/comparison.json) · [Seed 43](small_angle/dataset_seed_43/model_seed_42/epochs_100/comparison.json) · [Seed 44](small_angle/dataset_seed_44/model_seed_42/epochs_100/comparison.json)

## Broad motion

| Model | One-step MSE | 10-second rollout MSE |
|---|---:|---:|
| DMD | 0.00035082 ± 7.01e-05 | 2.4492 ± 0.393 |
| EDMD | 6.7318e-07 ± 8.31e-08 | 2.2103 ± 0.279 |
| MLP | 9.1295e-07 ± 2.28e-07 | 0.29692 ± 0.297 |
| Original Koopman | 9.8214e-06 ± 2.99e-06 | 2.4163 ± 0.293 |
| 10-step Koopman | 1.5271e-05 ± 1.16e-06 | 2.5163 ± 0.355 |

[Seed 42](broad_motion/dataset_seed_42/model_seed_42/epochs_100/comparison.json) · [Seed 43](broad_motion/dataset_seed_43/model_seed_42/epochs_100/comparison.json) · [Seed 44](broad_motion/dataset_seed_44/model_seed_42/epochs_100/comparison.json)

