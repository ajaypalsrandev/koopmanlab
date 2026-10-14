"""Run the pendulum milestone and save its overview figure."""

from pathlib import Path
import sys
import numpy as np

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_driven_nonlinear_dynamics.systems.pendulum import (  # noqa: E402
    PendulumParameters,
    mechanical_energy,
    simulate_pendulum,
)


def main() -> None:
    parameters = PendulumParameters(damping=0.1)
    times, states = simulate_pendulum(
        initial_state=(1.5, 0.0),
        parameters=parameters,
        final_time=12.0,
        time_step=0.02,
    )

    theta = states[:, 0]
    omega = states[:, 1]
    omega_dot = (
        -(parameters.gravity / parameters.length) * np.sin(theta)
        - parameters.damping * omega
    )
    energy = mechanical_energy(states, parameters)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(times, theta)
    axes[0, 0].set(title="Angle over time", xlabel="Time (s)", ylabel="Angle (rad)")

    axes[0, 1].plot(times, omega, color="tab:orange")
    axes[0, 1].set(
        title="Angular velocity over time",
        xlabel="Time (s)",
        ylabel="Angular velocity (rad/s)",
    )

    axes[1, 0].plot(theta, omega, color="tab:green")
    axes[1, 0].scatter(theta[0], omega[0], color="black", label="Initial state")
    axes[1, 0].set(
        title="Phase portrait",
        xlabel="Angle (rad)",
        ylabel="Angular velocity (rad/s)",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(times, energy, color="tab:red")
    axes[1, 1].set(title="Mechanical energy", xlabel="Time (s)", ylabel="Energy (J)")

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    figure.suptitle("Damped nonlinear pendulum")
    figure.tight_layout()

    results_directory = PROJECT_ROOT / "results"
    results_directory.mkdir(exist_ok=True)
    output_path = results_directory / "pendulum_overview.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    figure2, axes2 = plt.subplots(1, 1, figsize=(6, 5))
    axes2.plot(times, omega_dot, color="tab:green")
    axes2.set(
        title="Angular acceleration vs. angle",
        xlabel="Angle (rad)",
        ylabel="Angular acceleration (rad/s²)",
    )
    axes2.grid(alpha=0.25)
    output_path2 = results_directory / "pendulum_acceleration.png"
    figure2.savefig(output_path2, dpi=180)
    plt.close(figure2)

    print(f"States shape: {states.shape}")
    print(f"Initial energy: {energy[0]:.6f} J")
    print(f"Final energy:   {energy[-1]:.6f} J")
    print(f"Saved figure:   {output_path}")


if __name__ == "__main__":
    main()

