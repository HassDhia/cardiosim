"""Tests for environment wrappers."""

import gymnasium as gym
import numpy as np

import cardiosim  # noqa: F401
from cardiosim.envs.wrappers import NormalizeObservation, ClipAction


class TestNormalizeObservation:
    def test_normalize_pacing(self):
        env = gym.make("cardiosim/PacingControl-v0")
        wrapped = NormalizeObservation(env)
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == (6,)
        assert np.all(obs >= -2.0)  # Allow small overshoot
        assert np.all(obs <= 2.0)
        wrapped.close()

    def test_normalize_dosing(self):
        env = gym.make("cardiosim/AntiarrhythmicDosing-v0")
        wrapped = NormalizeObservation(env)
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == (6,)
        wrapped.close()


class TestClipAction:
    def test_clip_pacing(self):
        env = gym.make("cardiosim/PacingControl-v0")
        wrapped = ClipAction(env)
        wrapped.reset(seed=42)
        # Test with out-of-bounds action
        action = np.array([100.0, 5.0], dtype=np.float32)
        obs, _, _, _, _ = wrapped.step(action)
        assert obs.shape == (6,)
        wrapped.close()
