"""PacingControl-v0: Optimize pacemaker parameters to maintain target heart rate.

The agent controls pacemaker timing parameters to restore and maintain
normal sinus rhythm in a simulated heart with conduction abnormalities.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cardiosim.models.conduction import CardiacConductionModel
from cardiosim.models.fitzhugh_nagumo import FitzHughNagumoModel


DIFFICULTY_TIERS = {
    "easy": {"conduction_block_prob": 0.1, "sa_rate_range": (55, 65), "target_hr": 72.0},
    "medium": {"conduction_block_prob": 0.3, "sa_rate_range": (40, 55), "target_hr": 72.0},
    "hard": {"conduction_block_prob": 0.5, "sa_rate_range": (30, 45), "target_hr": 72.0},
}


class PacingControlEnv(gym.Env):
    """Gymnasium environment for cardiac pacemaker control.

    Observation space (Box, 6 dimensions):
        - membrane_voltage: Current FHN voltage variable
        - recovery_variable: Current FHN recovery variable
        - heart_rate: Estimated instantaneous heart rate (bpm)
        - time_since_beat: Time since last ventricular beat (ms)
        - hr_error: Difference from target heart rate (bpm)
        - hrv: Heart rate variability (ms)

    Action space (Box, 2 dimensions):
        - pacing_rate: Pacing rate adjustment (-20 to +20 bpm)
        - pulse_amplitude: Pacing pulse strength (0 to 1)

    Reward:
        - Proximity to target heart rate (negative squared error)
        - Penalty for energy use (pulse amplitude)
        - Bonus for stable rhythm
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty: str = "medium",
        target_hr: float | None = None,
        max_steps: int = 500,
        dt: float = 1.0,
    ) -> None:
        super().__init__()

        tier = DIFFICULTY_TIERS.get(difficulty, DIFFICULTY_TIERS["medium"])
        self.target_hr = target_hr if target_hr is not None else tier["target_hr"]
        self.max_steps = max_steps
        self.difficulty = difficulty

        self.conduction = CardiacConductionModel(
            sa_rate=60.0,
            conduction_block_prob=tier["conduction_block_prob"],
            dt=dt,
        )
        self.cell_model = FitzHughNagumoModel(dt=0.05)

        self.sa_rate_range = tier["sa_rate_range"]

        # Observation: [voltage, recovery, heart_rate, time_since_beat, hr_error, hrv]
        self.observation_space = spaces.Box(
            low=np.array([-3.0, -3.0, 0.0, 0.0, -100.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 3.0, 250.0, 5000.0, 100.0, 500.0], dtype=np.float32),
        )

        # Action: [pacing_rate_adjust, pulse_amplitude]
        self.action_space = spaces.Box(
            low=np.array([-20.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 1.0], dtype=np.float32),
        )

        self.steps = 0
        self._rng: np.random.Generator | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Set pathological SA rate based on difficulty
        low, high = self.sa_rate_range
        pathological_rate = self._rng.uniform(low, high)
        self.conduction.reset(self._rng)
        self.conduction.set_sa_rate(pathological_rate)
        self.cell_model.reset(self._rng)

        self.steps = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        pacing_rate_adj = float(action[0])
        pulse_amplitude = float(np.clip(action[1], 0.0, 1.0))

        # Apply pacing stimulus if amplitude > threshold
        pacing_stimulus = pulse_amplitude if pulse_amplitude > 0.1 else 0.0

        # Simulate multiple conduction steps per RL step
        for _ in range(10):
            cond_result = self.conduction.step(pacing_stimulus)
            # Drive FHN model with conduction events
            I_ext = 0.5 if cond_result["ventricular_fired"] else 0.0
            self.cell_model.step(I_ext)

        # Adjust effective pacing rate
        current_hr = self.conduction.get_heart_rate()
        new_rate = current_hr + pacing_rate_adj * 0.1
        self.conduction.set_sa_rate(max(30.0, min(200.0, new_rate)))

        # Compute reward
        hr_error = abs(current_hr - self.target_hr)
        hr_reward = -((hr_error / 20.0) ** 2)  # Scaled squared error
        energy_penalty = -0.1 * pulse_amplitude
        stability_bonus = 0.0
        hrv = self.conduction.get_heart_rate_variability()
        if hrv < 50.0 and hr_error < 10.0:
            stability_bonus = 0.3

        reward = hr_reward + energy_penalty + stability_bonus

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {
            "heart_rate": current_hr,
            "hr_error": hr_error,
            "hrv": hrv,
            "pulse_amplitude": pulse_amplitude,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        hr = self.conduction.get_heart_rate()
        hrv = self.conduction.get_heart_rate_variability()
        time_since = self.conduction.time - self.conduction.last_ventricular

        return np.array([
            self.cell_model.v,
            self.cell_model.w,
            hr,
            min(time_since, 5000.0),
            hr - self.target_hr,
            hrv,
        ], dtype=np.float32)
