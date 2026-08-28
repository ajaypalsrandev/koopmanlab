# KoopmanLab

Beginner-friendly scientific machine-learning project for learning approximately
linear latent representations of nonlinear dynamical systems.

# Current milestone

Milestone 1 is intentionally limited to a damped pendulum. Before building any
machine-learning model, we will:

1. Express the second-order pendulum equation as a first-order state-space model.
2. Numerically integrate that model with SciPy.
3. Plot angle, angular velocity, phase space, and mechanical energy.
4. Verify basic physical expectations with automated tests.

The state is

```text
x(t) = [theta(t), omega(t)]
```

and it evolves according to

```text
dtheta/dt = omega
domega/dt = -(g/L) sin(theta) - b omega
```

# Set up

Open a terminal in this directory. On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the simulation:

```bash
python scripts/run_pendulum.py
```

This creates `results/pendulum_overview.png` and prints a short numerical
summary.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

# What to inspect first

Open `src/koopmanlab/systems/pendulum.py` and trace the following path:

```text
initial state -> pendulum_rhs -> solve_ivp -> trajectory arrays -> plots
```

Then change one parameter at a time:

- Set `damping=0.0`. Energy should remain nearly constant.
- Set `damping=0.5`. Oscillations should decay faster.
- Set `theta0=0.1`. Motion should resemble a linear oscillator.
- Set `theta0=3.0`. Nonlinear behavior should be much more apparent.

# Project roadmap

- Milestone 1: pendulum simulator and physical validation
- Milestone 2: trajectory dataset and train/validation/test split
- Milestone 3: persistence and DMD baselines
- Milestone 4: PyTorch one-step MLP baseline
- Milestone 5: Koopman encoder, linear latent dynamics, and decoder
- Milestone 6: multistep training and long-horizon evaluation
- Milestone 7: noise, distribution-shift, and latent-dimension experiments
- Milestone 8: Duffing oscillator, final report, and demonstration

The cart-pole control extension should only begin after these milestones work.
