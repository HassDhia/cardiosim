"""Tests for baseline agents."""

import gymnasium as gym
import numpy as np

import cardiosim  # noqa: F401
from cardiosim.agents.random_agent import RandomAgent
from cardiosim.agents.heuristic import HeuristicPacingAgent, HeuristicDosingAgent


class TestRandomAgent:
    def test_predict_pacing(self):
        env = gym.make("cardiosim/PacingControl-v0")
        agent = RandomAgent(env, seed=42)
        obs, _ = env.reset(seed=42)
        action = agent.predict(obs)
        assert action.shape == (2,)
        env.close()

    def test_predict_dosing(self):
        env = gym.make("cardiosim/AntiarrhythmicDosing-v0")
        agent = RandomAgent(env, seed=42)
        obs, _ = env.reset(seed=42)
        action = agent.predict(obs)
        assert action.shape == (1,)
        env.close()

    def test_predict_defib(self):
        env = gym.make("cardiosim/DefibrillationTiming-v0")
        agent = RandomAgent(env, seed=42)
        obs, _ = env.reset(seed=42)
        action = agent.predict(obs)
        assert action.shape == (2,)
        env.close()

    def test_evaluate(self):
        env = gym.make("cardiosim/PacingControl-v0")
        agent = RandomAgent(env, seed=42)
        results = agent.evaluate(n_episodes=3)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert results["n_episodes"] == 3
        env.close()


class TestHeuristicPacingAgent:
    def test_predict(self):
        agent = HeuristicPacingAgent(target_hr=72.0)
        obs = np.array([0.0, 0.0, 60.0, 100.0, -12.0, 10.0], dtype=np.float32)
        action = agent.predict(obs)
        assert action.shape == (2,)
        # HR below target, should increase pacing
        assert action[0] > 0  # Positive adjustment

    def test_predict_above_target(self):
        agent = HeuristicPacingAgent(target_hr=72.0)
        obs = np.array([0.0, 0.0, 90.0, 100.0, 18.0, 10.0], dtype=np.float32)
        action = agent.predict(obs)
        assert action[0] < 0  # Negative adjustment

    def test_full_episode(self):
        env = gym.make("cardiosim/PacingControl-v0")
        agent = HeuristicPacingAgent()
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        for _ in range(100):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        env.close()


class TestHeuristicDosingAgent:
    def test_predict_subtherapeutic(self):
        agent = HeuristicDosingAgent()
        obs = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0], dtype=np.float32)
        action = agent.predict(obs)
        assert action[0] > 20  # Should give loading dose

    def test_predict_therapeutic(self):
        agent = HeuristicDosingAgent()
        obs = np.array([0.0, 0.0, 3.0, 0.0, 0.5, 0.5], dtype=np.float32)
        action = agent.predict(obs)
        assert action[0] < 20  # Maintenance dose

    def test_predict_near_toxic(self):
        agent = HeuristicDosingAgent()
        obs = np.array([0.0, 0.0, 4.5, 0.0, 0.8, 0.8], dtype=np.float32)
        action = agent.predict(obs)
        assert action[0] == 0.0  # Should withhold

    def test_full_episode(self):
        env = gym.make("cardiosim/AntiarrhythmicDosing-v0")
        agent = HeuristicDosingAgent()
        obs, _ = env.reset(seed=42)
        for _ in range(100):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
