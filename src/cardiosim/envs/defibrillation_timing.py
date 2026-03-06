"""DefibrillationTiming-v0: Optimal defibrillation shock timing and energy.

The agent decides when to deliver defibrillation shocks and at what energy
level to terminate ventricular fibrillation, minimizing energy delivery
and time in fibrillation.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cardiosim.models.aliev_panfilov import AlievPanfilovModel


DIFFICULTY_TIERS = {
    "easy": {"fib_severity": 0.3, "defibrillation_success_base": 0.7, "noise_std": 0.01},
    "medium": {"fib_severity": 0.5, "defibrillation_success_base": 0.5, "noise_std": 0.02},
    "hard": {"fib_severity": 0.8, "defibrillation_success_base": 0.3, "noise_std": 0.03},
}


class DefibrillationTimingEnv(gym.Env):
    """Gymnasium environment for defibrillation timing optimization.

    Observation space (Box, 6 dimensions):
        - membrane_potential: Current Aliev-Panfilov u variable
        - recovery_variable: Current Aliev-Panfilov v variable
        - fibrillation_index: Measure of rhythm irregularity (0-1)
        - time_in_fibrillation: Seconds in fibrillation
        - shocks_delivered: Number of shocks delivered so far
        - energy_delivered: Cumulative energy delivered (J)

    Action space (Box, 2 dimensions):
        - shock_decision: Whether to shock (>0.5 = shock)
        - shock_energy: Energy level (0 to 360 J, biphasic equivalent)

    Reward:
        - Large positive for successful defibrillation
        - Negative proportional to time in fibrillation
        - Negative for each shock (minimize unnecessary shocks)
        - Negative proportional to energy delivered
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty: str = "medium",
        max_steps: int = 100,
    ) -> None:
        super().__init__()

        tier = DIFFICULTY_TIERS.get(difficulty, DIFFICULTY_TIERS["medium"])
        self.difficulty = difficulty
        self.fib_severity = tier["fib_severity"]
        self.defibrillation_success_base = tier["defibrillation_success_base"]
        self.max_steps = max_steps

        self.cell_model = AlievPanfilovModel(noise_std=tier["noise_std"])

        # Observation: [u, v, fib_index, time_in_fib, shocks, energy]
        self.observation_space = spaces.Box(
            low=np.array([-0.5, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 25.0, 1.0, 300.0, 20.0, 5000.0], dtype=np.float32),
        )

        # Action: [shock_decision, shock_energy_joules]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 360.0], dtype=np.float32),
        )

        self.steps = 0
        self.in_fibrillation = True
        self.fibrillation_time = 0.0
        self.shocks_delivered = 0
        self.total_energy = 0.0
        self._voltage_history: list[float] = []
        self._rng: np.random.Generator | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self.cell_model.reset(self._rng)
        # Initialize in fibrillation state - chaotic dynamics
        self.cell_model.u = self._rng.uniform(0.3, 0.9)
        self.cell_model.v = self._rng.uniform(0.0, 5.0)

        self.steps = 0
        self.in_fibrillation = True
        self.fibrillation_time = 0.0
        self.shocks_delivered = 0
        self.total_energy = 0.0
        self._voltage_history = []

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        shock_decision = float(action[0])
        shock_energy = float(np.clip(action[1], 0.0, 360.0))

        deliver_shock = shock_decision > 0.5

        # Simulate cardiac dynamics
        if self.in_fibrillation:
            # Chaotic stimulus during fibrillation
            I_ext = self._rng.uniform(0.0, 0.3) * self.fib_severity
        else:
            I_ext = 0.0

        for _ in range(20):
            self.cell_model.step(I_ext)
            self._voltage_history.append(self.cell_model.u)

        # Handle defibrillation attempt
        if deliver_shock and self.in_fibrillation:
            self.shocks_delivered += 1
            self.total_energy += shock_energy

            # Success probability depends on energy and timing
            # Higher energy = higher success, but diminishing returns
            energy_factor = 1.0 - np.exp(-shock_energy / 150.0)
            # Later shocks have slightly lower success (tissue damage)
            fatigue_factor = 0.95 ** self.shocks_delivered
            success_prob = self.defibrillation_success_base * energy_factor * fatigue_factor

            if self._rng.random() < success_prob:
                self.in_fibrillation = False
                # Reset to near-resting state
                self.cell_model.u = 0.05
                self.cell_model.v = 0.01

        elif deliver_shock and not self.in_fibrillation:
            # Unnecessary shock - penalized
            self.shocks_delivered += 1
            self.total_energy += shock_energy

        # Update fibrillation time
        if self.in_fibrillation:
            self.fibrillation_time += 1.0

        # Compute reward
        reward = 0.0

        if not self.in_fibrillation:
            reward += 100.0  # Large bonus for successful defibrillation
        else:
            reward -= 0.1  # Small penalty per step in fibrillation

        if deliver_shock:
            reward -= 2.0  # Significant shock penalty (prefer fewer shocks)
            reward -= 0.005 * shock_energy  # Energy cost

        # Bonus for quick defibrillation with few shocks
        if not self.in_fibrillation and self.shocks_delivered <= 3:
            reward += 50.0

        self.steps += 1
        terminated = not self.in_fibrillation  # Success = episode ends
        truncated = self.steps >= self.max_steps

        # Large penalty for failing to defibrillate
        if truncated and self.in_fibrillation:
            reward -= 50.0

        obs = self._get_obs()
        info = {
            "in_fibrillation": self.in_fibrillation,
            "fibrillation_time": self.fibrillation_time,
            "shocks_delivered": self.shocks_delivered,
            "total_energy": self.total_energy,
            "defibrillated": not self.in_fibrillation,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        fib_index = self._compute_fibrillation_index()
        obs = np.array([
            self.cell_model.u,
            self.cell_model.v,
            fib_index,
            self.fibrillation_time,
            float(self.shocks_delivered),
            self.total_energy,
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _compute_fibrillation_index(self) -> float:
        """Compute a measure of rhythm irregularity from voltage history."""
        if len(self._voltage_history) < 10:
            return 1.0 if self.in_fibrillation else 0.0

        recent = self._voltage_history[-50:]
        self._voltage_history = self._voltage_history[-50:]
        # High variance = fibrillation, low variance = normal rhythm
        variance = np.var(recent)
        # Normalize to [0, 1]
        return float(min(1.0, variance / 0.1))
