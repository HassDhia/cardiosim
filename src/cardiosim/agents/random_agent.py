"""Random baseline agent for all CardioSim environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class RandomAgent:
    """Agent that takes uniformly random actions."""

    def __init__(self, env: gym.Env, seed: int | None = None) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a random action."""
        low = self.env.action_space.low
        high = self.env.action_space.high
        return self.rng.uniform(low, high).astype(np.float32)

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Evaluate random agent performance over n episodes."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": n_episodes,
        }
