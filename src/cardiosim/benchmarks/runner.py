"""Benchmark runner for CardioSim environments."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym

import cardiosim  # noqa: F401
from cardiosim.agents.random_agent import RandomAgent
from cardiosim.agents.heuristic import HeuristicPacingAgent, HeuristicDosingAgent
from cardiosim.benchmarks.environments import BENCHMARK_CONFIGS
from cardiosim.training.evaluate import evaluate_agent


def run_benchmarks(n_episodes: int = 20, output_dir: str = "results") -> dict:
    """Run baseline benchmarks on all environments."""
    results = {}

    for env_name, config in BENCHMARK_CONFIGS.items():
        env_id = config["env_id"]
        print(f"\nBenchmarking {env_name}...")

        for tier_name, tier_config in config["tiers"].items():
            env = gym.make(env_id, **tier_config)

            # Random baseline
            random_agent = RandomAgent(env, seed=42)
            random_results = random_agent.evaluate(n_episodes=n_episodes)

            # Heuristic baseline
            if "Pacing" in env_name:
                heuristic = HeuristicPacingAgent()
                heuristic_results = evaluate_agent(
                    env, heuristic.predict, n_episodes=n_episodes
                )
            elif "Dosing" in env_name:
                heuristic = HeuristicDosingAgent()
                heuristic_results = evaluate_agent(
                    env, heuristic.predict, n_episodes=n_episodes
                )
            else:
                heuristic_results = None

            key = f"{env_name}_{tier_name}"
            results[key] = {
                "env_id": env_id,
                "tier": tier_name,
                "random": random_results,
                "heuristic": heuristic_results,
            }

            print(f"  {tier_name}: random={random_results['mean_reward']:.2f}")
            if heuristic_results:
                print(f"  {tier_name}: heuristic={heuristic_results['mean_reward']:.2f}")

            env.close()

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run CardioSim benchmarks")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    run_benchmarks(n_episodes=args.episodes, output_dir=args.output)


if __name__ == "__main__":
    main()
