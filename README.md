# Data Driven Nonlinear Dynamics

Data Driven Nonlinear Dynamics is a scientific machine-learning
project focused on learning approximately linear latent dynamics for nonlinear
systems. The first benchmark is a damped pendulum, which is simple enough to
analyze physically but still rich enough to expose the limitations of linear
models.

The project is organized around a practical research workflow:

- model the nonlinear dynamics with a first-order state-space system,
- generate trajectories from randomized initial conditions,
- fit a linear baseline using DMD,
- compare one-step and multi-step predictive performance,
- build toward richer Koopman and neural baselines.

## Current project status

The codebase is already beyond the initial simulator-only milestone. The current
repository includes:

- a pendulum ODE simulator and physical validation tests,
- randomized trajectory generation for a dataset of pendulum rollouts,
- a train/validation/test split over entire trajectories,
- DMD fitting with the least-squares snapshot matrix formula,
- evaluation scripts for one-step and rollout error,
- plotting utilities for trajectories, phase portraits, and predictive error.

In other words, the project is currently at the DMD-baseline stage, with the
pendulum used as the benchmark system for evaluating linear forecasting methods.

## Physical model

We represent the pendulum state as

```text
x(t) = [theta(t), omega(t)]
```

with dynamics

```text
dtheta/dt = omega
domega/dt = -(g/L) sin(theta) - b omega
```

This is implemented in the system model and numerically integrated with SciPy.

## Repository layout

```text
.
├── README.md
├── LEARNING_PLAN.md
├── requirements.txt
├── results/
├── scripts/
│   ├── run_pendulum.py
│   ├── train_dmd.py
│   └── evaluate_dmd.py
├── src/
│   └── data_driven_nonlinear_dynamics/
│       ├── data/
│       │   └── generate_dataset.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── models/
│       │   └── dmd.py
│       └── systems/
│           └── pendulum.py
├── tests/
│   └── test_pendulum.py
└── .gitignore
```

## Key modules

### Pendulum simulation

The core dynamics live in `src/data_driven_nonlinear_dynamics/systems/pendulum.py`.

- `PendulumParameters` defines the physical settings.
- `pendulum_rhs` returns the first-order state derivative.
- `simulate_pendulum` integrates the system with `solve_ivp`.
- `mechanical_energy` computes total energy for checking physical behavior.

### Dataset construction

The trajectory generation workflow is implemented in
`src/data_driven_nonlinear_dynamics/data/generate_dataset.py`.

- `create_trajectories` samples random initial conditions and simulates rollouts.
- `create_transition_matrices` builds the DMD snapshot matrices `X` and `Y`.
- `split_trajectories` divides data into train, validation, and test sets.

### DMD model

The linear dynamics baseline is implemented in `src/data_driven_nonlinear_dynamics/models/dmd.py`.

- `fit_matrix(X, Y)` learns the matrix by solving the least-squares problem
  `Y ≈ A X` using the pseudoinverse.
- `predict_one_step(A, states)` computes a single-step prediction.
- `rollout(A, initial_state, num_steps)` recursively advances the model.

### Evaluation

The metrics used for comparing true and predicted trajectories are in
`src/data_driven_nonlinear_dynamics/evaluation/metrics.py`.

- `mean_square_error` computes a global scalar error.
- `error_by_time` measures the MSE per time step, which is useful for plotting
  error growth over a rollout horizon.

## Current workflows

### 1. Run the pendulum overview

```bash
python scripts/run_pendulum.py
```

This saves plots to `results/` and prints a short numerical summary, including
initial and final energy values.

### 2. Train the DMD baseline

```bash
python scripts/train_dmd.py
```

This script:

- generates randomized pendulum trajectories,
- splits them into train/validation/test sets,
- fits a DMD matrix `A`,
- prints the matrix and its eigenvalues.

### 3. Evaluate the DMD model

```bash
python scripts/evaluate_dmd.py
```

This script fits a DMD matrix on training data and evaluates validation trajectories, rolls the model forward,
compares true versus predicted trajectories, and saves a summary figure to
`results/dmd_evaluation.png`.

### 4. Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests currently validate the physical assumptions behind the pendulum model,
including equilibrium behavior, shape correctness, and energy loss under damping.

## What the project is aiming to learn

The central research question is:

> Can we learn an approximately linear latent representation for a nonlinear
> dynamical system, and how well does that representation predict the system over
> time?

The pendulum is the first benchmark because it is nonlinear, low-dimensional, and
easy to understand physically. Linear baselines like DMD help establish a clear
reference point before introducing richer latent-variable models such as neural
or Koopman-based approaches.

## Research roadmap

The current codebase is aligned with the following roadmap:

- Milestone 1: pendulum simulation and physical validation — complete
- Milestone 2: trajectory dataset and train/validation/test split — complete
- Milestone 3: DMD baseline and predictive evaluation — complete
- Milestone 4: one-step MLP baseline — next
- Milestone 5: Koopman encoder, latent linear dynamics, and decoder — next
- Milestone 6: multistep training and long-horizon evaluation — planned
- Milestone 7: noise, distribution-shift, and latent-dimension experiments — planned
- Milestone 8: Duffing oscillator, final report, and demonstration — planned

The cart-pole control extension should come only after the pendulum and Koopman
research stack are working reliably.

## Practical notes

- The repository is intentionally designed as a learning project: small files,
  explicit steps, and easy-to-inspect scripts.
- The DMD model is a good baseline, but it is not the final research target.
- For physically meaningful experiments, compare one-step predictions and long
  recursive rollouts separately; they do not always agree.

## Getting started

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run the project workflow described above.
