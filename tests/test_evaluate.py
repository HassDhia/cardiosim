"""Tests for evaluation utilities."""

import gymnasium as gym
import pytest

import cardiosim  # noqa: F401
from cardiosim.training.evaluate import evaluate_agent, compute_reward_ratio


class TestEvaluateAgent:
    def test_evaluate_random(self):
        env = gym.make("cardiosim/PacingControl-v0")
        results = evaluate_agent(
            env,
            predict_fn=lambda obs: env.action_space.sample(),
            n_episodes=3,
        )
        assert "mean_reward" in results
        assert "std_reward" in results
        assert results["n_episodes"] == 3
        env.close()


class TestComputeRewardRatio:
    def test_standard_ratio(self):
        ratio, note = compute_reward_ratio(10.0, 5.0)
        assert ratio == pytest.approx(2.0)
        assert note == "standard_ratio"

    def test_negative_baseline(self):
        ratio, note = compute_reward_ratio(5.0, -10.0)
        assert ratio > 0
        assert note == "negative_baseline_abs_improvement"

    def test_near_zero_baseline(self):
        ratio, note = compute_reward_ratio(5.0, 0.001)
        assert note == "absolute_difference_near_zero_baseline"

    def test_equal_rewards(self):
        ratio, note = compute_reward_ratio(5.0, 5.0)
        assert ratio == pytest.approx(1.0)
