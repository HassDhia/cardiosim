"""Tests for PacingControl-v0 environment."""

import gymnasium as gym
import numpy as np

import cardiosim  # noqa: F401


class TestPacingControlEnv:
    def setup_method(self):
        self.env = gym.make("cardiosim/PacingControl-v0")

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
        action = self.env.action_space.sample()
        result = self.env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self):
        self.env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 600:
            action = self.env.action_space.sample()
            _, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            steps += 1
        assert done

    def test_info_contains_heart_rate(self):
        self.env.reset(seed=42)
        action = self.env.action_space.sample()
        _, _, _, _, info = self.env.step(action)
        assert "heart_rate" in info
        assert "hr_error" in info

    def test_seed_reproducibility(self):
        obs1, _ = self.env.reset(seed=123)
        obs2, _ = self.env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


class TestPacingControlDifficulties:
    def test_easy_difficulty(self):
        env = gym.make("cardiosim/PacingControl-v0", difficulty="easy")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (6,)
        env.close()

    def test_hard_difficulty(self):
        env = gym.make("cardiosim/PacingControl-v0", difficulty="hard")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (6,)
        env.close()

    def test_custom_target_hr(self):
        env = gym.make("cardiosim/PacingControl-v0", target_hr=80.0)
        obs, _ = env.reset(seed=42)
        env.close()
