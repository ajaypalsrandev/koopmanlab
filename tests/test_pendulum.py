"""Basic physical and software checks for the pendulum simulator."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from koopmanlab.systems.pendulum import (  # noqa: E402
    PendulumParameters,
    mechanical_energy,
    pendulum_rhs,
    simulate_pendulum,
)


class PendulumTests(unittest.TestCase):
    def test_rhs_at_downward_equilibrium_is_zero(self) -> None:
        derivative = pendulum_rhs(
            0.0,
            np.array([0.0, 0.0]),
            PendulumParameters(),
        )
        self.assertTrue(np.allclose(derivative, np.zeros(2)))

    def test_simulation_has_expected_shape_and_initial_state(self) -> None:
        times, states = simulate_pendulum(
            initial_state=(0.5, -0.2),
            final_time=1.0,
            time_step=0.1,
        )
        self.assertEqual(states.shape, (len(times), 2))
        self.assertTrue(np.allclose(states[0], np.array([0.5, -0.2])))

    def test_undamped_energy_is_nearly_conserved(self) -> None:
        parameters = PendulumParameters(damping=0.0)
        _, states = simulate_pendulum(
            initial_state=(1.0, 0.3),
            parameters=parameters,
            final_time=10.0,
            time_step=0.02,
        )
        energy = mechanical_energy(states, parameters)
        relative_drift = np.max(np.abs(energy - energy[0])) / energy[0]
        self.assertLess(relative_drift, 1e-6)

    def test_damped_system_loses_energy(self) -> None:
        parameters = PendulumParameters(damping=0.2)
        _, states = simulate_pendulum(
            initial_state=(1.0, 0.0),
            parameters=parameters,
            final_time=10.0,
            time_step=0.02,
        )
        energy = mechanical_energy(states, parameters)
        self.assertLess(energy[-1], energy[0])


if __name__ == "__main__":
    unittest.main()
