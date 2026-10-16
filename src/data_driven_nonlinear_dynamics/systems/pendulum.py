"""Simulation utilities for a damped nonlinear pendulum.

The physical equation is

    theta_ddot + damping * theta_dot + (g / length) * sin(theta) = 0.

We rewrite it as two first-order equations because numerical ODE solvers work
with a state vector and its first derivative.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PendulumParameters:
    """Physical parameters for the pendulum model."""

    gravity: float = 9.81
    length: float = 1.0
    damping: float = 0.1
    mass: float = 1.0

    def __post_init__(self) -> None:
        if self.gravity <= 0:
            raise ValueError("gravity must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.damping < 0:
            raise ValueError("damping cannot be negative")


def pendulum_rhs(
    _time: float,
    state: np.ndarray,
    parameters: PendulumParameters,
) -> np.ndarray:
    """Return the time derivative [theta_dot, omega_dot]."""

    theta, omega = state
    theta_dot = omega
    omega_dot = (
        -(parameters.gravity / parameters.length) * np.sin(theta)
        - parameters.damping * omega
    )
    return np.array([theta_dot, omega_dot], dtype=float)


def simulate_pendulum(
    initial_state: tuple[float, float] = (1.0, 0.0),
    parameters: Optional[PendulumParameters] = None,
    final_time: float = 10.0,
    time_step: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a trajectory and return `(times, states)`.

    `states[i]` contains `[theta, omega]` at `times[i]`, so the state array has
    shape `(number_of_times, 2)`.
    """

    if final_time <= 0:
        raise ValueError("final_time must be positive")
    if time_step <= 0:
        raise ValueError("time_step must be positive")

    parameters = parameters or PendulumParameters()
    times = np.arange(0.0, final_time + 0.5 * time_step, time_step)
    times = times[times <= final_time]

    solution = solve_ivp(
        fun=lambda t, x: pendulum_rhs(t, x, parameters),
        t_span=(0.0, final_time),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    return solution.t, solution.y.T


def mechanical_energy(
    states: np.ndarray,
    parameters: Optional[PendulumParameters] = None,
) -> np.ndarray:
    """Return kinetic plus gravitational potential energy for every state."""

    parameters = parameters or PendulumParameters()
    states = np.asarray(states, dtype=float)
    theta = states[..., 0]
    omega = states[..., 1]

    kinetic = 0.5 * parameters.mass * parameters.length**2 * omega**2
    potential = (
        parameters.mass
        * parameters.gravity
        * parameters.length
        * (1.0 - np.cos(theta))
    )
    return kinetic + potential

