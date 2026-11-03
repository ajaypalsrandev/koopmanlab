# Running the DMD and EDMD comparison

Run from the repository root with the project environment activated:

```bash
python scripts/evaluate_edmd.py
```

This runs seeds 42, 43, and 44 for both small-angle and broad-motion pendulum
initial conditions. For a single run, use `--seeds 42`. You can choose a separate
output location with `--output-dir /path/to/output`.

For every regime and seed, the script generates 200 trajectories once, splits
whole trajectories into 140 training, 30 validation, and 30 test trajectories,
and fits both models on the same training snapshot matrices. Only validation
trajectories are scored. The test split is reserved for later comparisons.

EDMD uses the fixed dictionary `[theta, omega, sin(theta), cos(theta), 1]`.
Its rollout lifts the initial state once and evolves all five features linearly.
Neither model is normalized or regularized in this comparison.

## Outputs

Results go to `docs/benchmarks/edmd/` by default:

```text
summary.json
small_angle/seed_42/
    comparison.json
    representative.png
    worst.png
    error_by_time.png
    dmd/
        metrics.json
        predictions.npz
    edmd/
        metrics.json
        predictions.npz
```

The same layout is repeated for each selected regime and seed. Repeating a run
replaces that run's artifacts. `summary.json` describes only the seeds requested
in the current invocation; older seed directories may still be present.

Each model's JSON includes configuration, split sizes, fitted matrix, spectral
radius, per-trajectory scores, and error curves. The compressed archives contain
`times`, `true_states`, `one_step_predictions`, and `rollout_states`. They are
local generated artifacts excluded by the repository's existing `*.npz` rule;
rerun the script to recreate them after cloning.

The evaluator also returns these full arrays in memory. True states and rollouts
have shape `(trajectories, times, 2)`. One-step predictions have shape
`(trajectories, times - 1, 2)`. The supplied initial state is excluded from scoring.

## Reading the figures and summaries

Representative and worst cases are selected using EDMD's rollout MSE. Both
models are plotted on that SAME validation trajectory, rather than choosing a
different example for each model. The selection indices and rule are recorded in
`comparison.json`. The error panel in each example figure refers to that example;
`error_by_time.png` averages across all validation trajectories and separates
one-step from recursive errors.

`summary.json` reports the mean and sample standard deviation of seed-level
scores (`ddof=1`). With one seed, standard deviation is `null`. This variation
across sampled datasets is different from variation across trajectories within
one run. Each model's metrics file records the latter with `ddof=0`.

State MSE averages angle and angular-velocity squared errors without rescaling,
so it combines different units. Separate angle and velocity rollout errors are
also reported. A small one-step error does not guarantee an accurate recursive
rollout, and EDMD is not assumed to outperform DMD.
