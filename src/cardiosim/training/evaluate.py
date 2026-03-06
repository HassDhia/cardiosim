"""Evaluation utilities for CardioSim agents."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def evaluate_agent(
    env: gym.Env,
    predict_fn,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> dict:
    """Evaluate an agent on an environment.

    Args:
        env: Gymnasium environment.
        predict_fn: Callable that takes obs and returns action.
        n_episodes: Number of evaluation episodes.
        deterministic: Whether predictions are deterministic.

    Returns:
        Dict with evaluation metrics.
    """
    episode_rewards = []
    episode_lengths = []
    episode_infos = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        last_info = {}

        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            last_info = info
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_infos.append(last_info)

    rewards_arr = np.array(episode_rewards)
    lengths_arr = np.array(episode_lengths)

    return {
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "min_reward": float(np.min(rewards_arr)),
        "max_reward": float(np.max(rewards_arr)),
        "mean_length": float(np.mean(lengths_arr)),
        "n_episodes": n_episodes,
    }


def compute_reward_ratio(ppo_reward: float, baseline_reward: float) -> tuple[float, str]:
    """Compute PPO vs baseline reward ratio with edge case handling.

    Returns:
        Tuple of (ratio, note).
    """
    if abs(baseline_reward) < 0.01:
        return (ppo_reward - baseline_reward, "absolute_difference_near_zero_baseline")
    if baseline_reward < 0:
        improvement = abs(ppo_reward - baseline_reward) / abs(baseline_reward)
        note = "negative_baseline_abs_improvement"
        return (improvement, note)
    return (ppo_reward / baseline_reward, "standard_ratio")
