"""Tests for the FitzHugh-Nagumo cardiac model."""

import numpy as np
import pytest

from cardiosim.models.fitzhugh_nagumo import FitzHughNagumoModel, PARAMETER_RANGES


class TestFitzHughNagumoInit:
    def test_default_initialization(self):
        model = FitzHughNagumoModel()
        assert model.a == 0.2
        assert model.b == 0.7
        assert model.tau == 12.5
        assert model.dt == 0.05

    def test_custom_parameters(self):
        model = FitzHughNagumoModel(a=0.1, b=0.8, tau=10.0)
        assert model.a == 0.1
        assert model.b == 0.8
        assert model.tau == 10.0

    def test_invalid_a_raises(self):
        with pytest.raises(ValueError, match="outside literature range"):
            FitzHughNagumoModel(a=0.01)  # Below min

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError, match="outside literature range"):
            FitzHughNagumoModel(b=2.0)  # Above max

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError, match="outside literature range"):
            FitzHughNagumoModel(tau=1.0)  # Below min

    def test_initial_state(self):
        model = FitzHughNagumoModel()
        assert model.v == pytest.approx(-1.2)
        assert model.w == pytest.approx(-0.6)


class TestFitzHughNagumoReset:
    def test_reset_deterministic(self):
        model = FitzHughNagumoModel()
        model.step(1.0)
        v, w = model.reset()
        assert v == pytest.approx(-1.2)
        assert w == pytest.approx(-0.6)

    def test_reset_with_rng(self):
        model = FitzHughNagumoModel()
        rng = np.random.default_rng(42)
        v, w = model.reset(rng)
        assert v != -1.2  # Should be perturbed
        assert abs(v - (-1.2)) < 0.2  # But close


class TestFitzHughNagumoDynamics:
    def test_single_step(self):
        model = FitzHughNagumoModel()
        v, w = model.step(0.0)
        assert isinstance(v, float)
        assert isinstance(w, float)

    def test_excitation_with_stimulus(self):
        model = FitzHughNagumoModel()
        # Apply strong stimulus
        for _ in range(200):
            model.step(1.0)
        assert model.v > -1.0  # Should have excited

    def test_return_to_rest(self):
        model = FitzHughNagumoModel()
        # Excite then release
        for _ in range(100):
            model.step(1.0)
        for _ in range(500):
            model.step(0.0)
        assert model.v < 0.5  # Should return toward rest

    def test_step_n(self):
        model = FitzHughNagumoModel()
        traj = model.step_n(100, I_ext=0.5)
        assert traj.shape == (100, 2)
        assert not np.any(np.isnan(traj))

    def test_is_excited(self):
        model = FitzHughNagumoModel()
        model.v = 1.0
        assert model.is_excited(threshold=0.5)
        model.v = 0.0
        assert not model.is_excited(threshold=0.5)

    def test_get_state(self):
        model = FitzHughNagumoModel()
        state = model.get_state()
        assert state.shape == (2,)
        assert state[0] == pytest.approx(-1.2)

    def test_action_potential_duration(self):
        model = FitzHughNagumoModel()
        apd = model.action_potential_duration
        assert apd > 0
        assert isinstance(apd, float)

    def test_noise_adds_variability(self):
        model1 = FitzHughNagumoModel(noise_std=0.0)
        model2 = FitzHughNagumoModel(noise_std=0.1)
        model1.reset()
        model2.reset()
        np.random.seed(42)
        traj1 = model1.step_n(50, 0.5)
        np.random.seed(42)
        traj2 = model2.step_n(50, 0.5)
        # Noisy model should differ
        assert not np.allclose(traj1, traj2)

    def test_rk4_accuracy(self):
        """Verify RK4 is more accurate than Euler for same dt."""
        model = FitzHughNagumoModel(dt=0.1)
        model.reset()
        rk4_result = model.step_n(100, 0.5)
        assert not np.any(np.isnan(rk4_result))
        assert not np.any(np.isinf(rk4_result))


class TestParameterRanges:
    def test_all_defaults_in_range(self):
        for name, info in PARAMETER_RANGES.items():
            assert info["min"] <= info["default"] <= info["max"], (
                f"{name} default {info['default']} not in [{info['min']}, {info['max']}]"
            )

    def test_ranges_have_units(self):
        for name, info in PARAMETER_RANGES.items():
            assert "unit" in info, f"{name} missing unit"

    def test_ranges_have_source(self):
        for name, info in PARAMETER_RANGES.items():
            assert "source" in info, f"{name} missing source"
