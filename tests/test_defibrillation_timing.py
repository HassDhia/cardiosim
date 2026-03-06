"""Tests for DefibrillationTiming-v0 environment."""

import gymnasium as gym
import numpy as np

import cardiosim  # noqa: F401


class TestDefibrillationTimingEnv:
    def setup_method(self):
        self.env = gym.make("cardiosim/DefibrillationTiming-v0")

    def teardown_method(self):
        self.env.close()

    def test_make_env(self):
        assert self.env is not None

    def test_observation_space_shape(self):
        assert self.env.observation_space.shape == (6,)

    def test_action_space_shape(self):
        assert self.env.action_space.shape == (2,)

    def test_reset_returns_obs(self):
        obs, info = self.env.reset(seed=42)
        assert obs.shape == (6,)
        assert isinstance(info, dict)

    def test_obs_in_bounds(self):
        obs, _ = self.env.reset(seed=42)
        assert self.env.observation_space.contains(obs)

    def test_step_returns_tuple(self):
        self.env.reset(seed=42)
        action = np.array([0.0, 0.0], dtype=np.float32)
        result = self.env.step(action)
        assert len(result) == 5

    def test_no_shock_action(self):
        self.env.reset(seed=42)
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert info["shocks_delivered"] == 0

    def test_shock_action(self):
        self.env.reset(seed=42)
        action = np.array([1.0, 200.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert info["shocks_delivered"] == 1
        assert info["total_energy"] > 0

    def test_episode_terminates(self):
        self.env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 150:
            action = self.env.action_space.sample()
            _, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            steps += 1
        assert done

    def test_defibrillation_possible(self):
        """Verify that defibrillation can succeed."""
        successes = 0
        for seed in range(20):
            self.env.reset(seed=seed)
            for _ in range(50):
                action = np.array([1.0, 300.0], dtype=np.float32)
                _, _, terminated, _, info = self.env.step(action)
                if terminated and info.get("defibrillated"):
                    successes += 1
                    break
        assert successes > 0

    def test_info_contains_fields(self):
        self.env.reset(seed=42)
        action = self.env.action_space.sample()
        _, _, _, _, info = self.env.step(action)
        assert "in_fibrillation" in info
        assert "shocks_delivered" in info
        assert "total_energy" in info

    def test_seed_reproducibility(self):
        obs1, _ = self.env.reset(seed=123)
        obs2, _ = self.env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


class TestDefibrillationDifficulties:
    def test_easy(self):
        env = gym.make("cardiosim/DefibrillationTiming-v0", difficulty="easy")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (6,)
        env.close()

    def test_hard(self):
        env = gym.make("cardiosim/DefibrillationTiming-v0", difficulty="hard")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (6,)
        env.close()
