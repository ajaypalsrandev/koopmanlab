# DMD, EDMD, and MLP validation summary

Dataset seeds: **42, 43, 44**. MLP model seed: **42**. MLP training budget: **100 epochs**.

Each table entry is the mean ± sample standard deviation of the seed-level scores (ddof=1). Each run averages over its validation trajectories. These are not confidence intervals or within-run trajectory standard deviations.

The MLP uses the checkpoint selected by one-step validation loss within each training run. Dataset seed varies; model initialization seed is held fixed. All three models use identical training and validation data within each run. Runs with other training budgets are excluded.

Aggregate state MSE combines angle and angular-velocity squared errors without normalization; separate state errors retain their respective units. Persistence is a one-step reference, not a rollout baseline.

## Small angle

| Metric | DMD | EDMD | MLP |
|---|---:|---:|---:|
| One-step state MSE | 3.807e-08 ± 1.567e-09 | 1.352e-10 ± 5.104e-12 | 5.911e-08 ± 2.169e-08 |
| Rollout state MSE | 0.000348 ± 1.702e-05 | 0.000316 ± 1.262e-05 | 0.001468 ± 0.001217 |
| Rollout angle MSE (rad²) | 6.647e-05 ± 3.431e-06 | 6.272e-05 ± 2.57e-06 | 0.0002654 ± 0.0002201 |
| Rollout velocity MSE ((rad/s)²) | 0.0006295 ± 3.061e-05 | 0.0005693 ± 2.268e-05 | 0.002671 ± 0.002215 |
| Persistence one-step state MSE | 0.0006262 ± 5.574e-05 | 0.0006262 ± 5.574e-05 | 0.0006262 ± 5.574e-05 |

### Individual rollout scores

| Dataset seed | DMD | EDMD | MLP | Selected MLP epoch |
|---|---:|---:|---:|---:|
| 42 | 0.00034207 | 0.000317497 | 0.00101417 | 25 |
| 43 | 0.000367183 | 0.000327817 | 0.000543089 | 32 |
| 44 | 0.000334729 | 0.000302705 | 0.00284735 | 56 |

Rankings below run from lowest to highest error:

- One-step, seed 42: EDMD → DMD → MLP.
- One-step, seed 43: EDMD → MLP → DMD.
- One-step, seed 44: EDMD → DMD → MLP.
- Rollout, seed 42: EDMD → DMD → MLP.
- Rollout, seed 43: EDMD → DMD → MLP.
- Rollout, seed 44: EDMD → DMD → MLP.

Source runs:

- [Seed 42](small_angle/dataset_seed_42/model_seed_42/epochs_100/checkpoint_6a860141d134/comparison.json)
- [Seed 43](small_angle/dataset_seed_43/model_seed_42/epochs_100/checkpoint_1bd90d8510fe/comparison.json)
- [Seed 44](small_angle/dataset_seed_44/model_seed_42/epochs_100/checkpoint_9b02d33f93b9/comparison.json)

## Broad motion

| Metric | DMD | EDMD | MLP |
|---|---:|---:|---:|
| One-step state MSE | 0.0003978 ± 2.097e-05 | 7.623e-07 ± 3.196e-08 | 7.838e-07 ± 3.13e-07 |
| Rollout state MSE | 2.689 ± 0.1302 | 2.331 ± 0.2158 | 0.3993 ± 0.4741 |
| Rollout angle MSE (rad²) | 0.6558 ± 0.03708 | 0.5791 ± 0.05071 | 0.1005 ± 0.1219 |
| Rollout velocity MSE ((rad/s)²) | 4.721 ± 0.2234 | 4.084 ± 0.381 | 0.6981 ± 0.8262 |
| Persistence one-step state MSE | 0.008736 ± 0.0009384 | 0.008736 ± 0.0009384 | 0.008736 ± 0.0009384 |

### Individual rollout scores

| Dataset seed | DMD | EDMD | MLP | Selected MLP epoch |
|---|---:|---:|---:|---:|
| 42 | 2.64486 | 2.29147 | 0.206605 | 41 |
| 43 | 2.83506 | 2.56447 | 0.939398 | 68 |
| 44 | 2.58586 | 2.1384 | 0.0519566 | 35 |

Rankings below run from lowest to highest error:

- One-step, seed 42: EDMD → MLP → DMD.
- One-step, seed 43: MLP → EDMD → DMD.
- One-step, seed 44: MLP → EDMD → DMD.
- Rollout, seed 42: MLP → EDMD → DMD.
- Rollout, seed 43: MLP → EDMD → DMD.
- Rollout, seed 44: MLP → EDMD → DMD.

Source runs:

- [Seed 42](broad_motion/dataset_seed_42/model_seed_42/epochs_100/checkpoint_85e2da7c72f1/comparison.json)
- [Seed 43](broad_motion/dataset_seed_43/model_seed_42/epochs_100/checkpoint_318b7ad80631/comparison.json)
- [Seed 44](broad_motion/dataset_seed_44/model_seed_42/epochs_100/checkpoint_7908c0d3cb6e/comparison.json)

## Reproduce this summary

From the repository root:

```bash
python scripts/summarize_mlp.py --seeds 42 43 44 --model-seed 42 --epochs 100
```

The script reads existing comparison files; it does not retrain or evaluate any models. It rejects missing runs, duplicate checkpoints for a selected experiment, and inconsistent settings rather than silently averaging incompatible experiments.
