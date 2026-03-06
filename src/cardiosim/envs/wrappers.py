"""SB3-compatible wrappers for CardioSim environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to approximately [-1, 1] range for SB3."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_space = env.observation_space
        self._low = obs_space.low
        self._high = obs_space.high
        self._range = self._high - self._low
        self._range[self._range == 0] = 1.0  # Avoid division by zero

        self.observation_space = gym.spaces.Box(
            low=-np.ones_like(self._low),
            high=np.ones_like(self._high),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return 2.0 * (obs - self._low) / self._range - 1.0


class ClipAction(gym.ActionWrapper):
    """Clip actions to the valid range."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_space.low, self.action_space.high)
