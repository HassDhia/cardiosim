"""Train PPO agents on all CardioSim environments and save results."""

import json
import os

import gymnasium as gym
import numpy as np

import cardiosim  # noqa: F401
from stable_baselines3 import PPO
from cardiosim.agents.random_agent import RandomAgent
from cardiosim.agents.heuristic import HeuristicPacingAgent, HeuristicDosingAgent
from cardiosim.training.evaluate import evaluate_agent, compute_reward_ratio


def main():
    results = {}

    configs = [
        ("PacingControl-v0", "cardiosim/PacingControl-v0", 200000,
         HeuristicPacingAgent(), {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                                  "ent_coef": 0.01}),
        ("AntiarrhythmicDosing-v0", "cardiosim/AntiarrhythmicDosing-v0", 300000,
         HeuristicDosingAgent(), {"learning_rate": 5e-4, "n_steps": 1024, "batch_size": 64,
                                  "ent_coef": 0.02}),
        ("DefibrillationTiming-v0", "cardiosim/DefibrillationTiming-v0", 500000,
         None, {"learning_rate": 3e-4, "n_steps": 256, "batch_size": 64,
                "ent_coef": 0.1, "difficulty": "easy"}),
    ]

    for env_name, env_id, steps, heuristic, hp in configs:
        print(f"Training {env_name}...")
        env_kwargs = {}
        if "difficulty" in hp:
            env_kwargs["difficulty"] = hp.pop("difficulty")
        env = gym.make(env_id, **env_kwargs)
        eval_env = gym.make(env_id, **env_kwargs)

        random_agent = RandomAgent(env, seed=42)
        random_results = random_agent.evaluate(n_episodes=20)
        print(f"  Random: {random_results['mean_reward']:.2f}")

        heuristic_results = None
        if heuristic is not None:
            heuristic_results = evaluate_agent(eval_env, heuristic.predict, n_episodes=20)
            print(f"  Heuristic: {heuristic_results['mean_reward']:.2f}")

        model = PPO(
            "MlpPolicy", env,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            n_epochs=10, gamma=0.99, verbose=0,
            ent_coef=hp.get("ent_coef", 0.01),
        )
        model.learn(total_timesteps=steps)

        # Use stochastic eval for envs where deterministic collapse is a risk
        use_deterministic = "Defibrillation" not in env_name
        ppo_rewards = []
        for _ in range(20):
            obs, _ = eval_env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=use_deterministic)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            ppo_rewards.append(total_reward)

        ppo_mean = float(np.mean(ppo_rewards))
        ppo_std = float(np.std(ppo_rewards))
        print(f"  PPO: {ppo_mean:.2f} +/- {ppo_std:.2f}")

        random_mean = random_results["mean_reward"]
        ratio, ratio_note = compute_reward_ratio(ppo_mean, random_mean)
        print(f"  Ratio vs random: {ratio:.2f} ({ratio_note})")

        results[env_name] = {
            "env_id": env_id,
            "random_mean_reward": random_results["mean_reward"],
            "random_std_reward": random_results["std_reward"],
            "heuristic_mean_reward": heuristic_results["mean_reward"] if heuristic_results else None,
            "heuristic_std_reward": heuristic_results["std_reward"] if heuristic_results else None,
            "ppo_mean_reward": ppo_mean,
            "ppo_std_reward": ppo_std,
            "ppo_vs_random_ratio": ratio,
            "ppo_vs_random_ratio_note": ratio_note,
            "total_timesteps": steps,
            "n_eval_episodes": 20,
        }

        env.close()
        eval_env.close()

    os.makedirs("results", exist_ok=True)
    with open("results/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/training_results.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
