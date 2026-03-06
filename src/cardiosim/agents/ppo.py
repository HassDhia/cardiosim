"""PPO agent training for CardioSim environments.

Uses Stable-Baselines3 PPO with environment-specific hyperparameters.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def get_env_configs() -> dict:
    """Return PPO training configurations per environment."""
    return {
        "cardiosim/PacingControl-v0": {
            "total_timesteps": 100_000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
        "cardiosim/AntiarrhythmicDosing-v0": {
            "total_timesteps": 100_000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
        "cardiosim/DefibrillationTiming-v0": {
            "total_timesteps": 100_000,
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
    }


def train_single_env(env_id: str, output_dir: str = "results") -> dict:
    """Train PPO on a single environment and return results."""
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("stable-baselines3 and torch required. Install with: pip install cardiosim[train]")
        sys.exit(1)

    import cardiosim  # noqa: F401 - registers envs

    configs = get_env_configs()
    config = configs.get(env_id, configs["cardiosim/PacingControl-v0"])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/training_logs", exist_ok=True)

    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=config["n_steps"] * 2,
        verbose=0,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=eval_callback,
    )

    # Save model
    env_slug = env_id.split("/")[-1].lower()
    model.save(f"{output_dir}/models/{env_slug}_ppo")

    # Evaluate
    rewards = []
    for _ in range(20):
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)

    env.close()
    eval_env.close()

    return {
        "env_id": env_id,
        "ppo_mean_reward": float(np.mean(rewards)),
        "ppo_std_reward": float(np.std(rewards)),
        "total_timesteps": config["total_timesteps"],
        "n_eval_episodes": 20,
    }


def train_all(output_dir: str = "results") -> dict:
    """Train PPO on all environments and save results."""
    results = {}
    for env_id in get_env_configs():
        print(f"Training {env_id}...")
        result = train_single_env(env_id, output_dir)
        env_slug = env_id.split("/")[-1]
        results[env_slug] = result
        print(f"  Mean reward: {result['ppo_mean_reward']:.2f}")

    # Save results
    results_path = Path(output_dir) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


def main() -> None:
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train CardioSim PPO agents")
    parser.add_argument("--env", type=str, default=None, help="Specific env ID to train")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    if args.env:
        result = train_single_env(args.env, args.output)
        print(json.dumps(result, indent=2))
    else:
        train_all(args.output)


if __name__ == "__main__":
    main()
