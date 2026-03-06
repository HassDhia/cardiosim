"""Tests for the Aliev-Panfilov cardiac model."""

import numpy as np
import pytest

from cardiosim.models.aliev_panfilov import AlievPanfilovModel, PARAMETER_RANGES


class TestAlievPanfilovInit:
    def test_default_initialization(self):
        model = AlievPanfilovModel()
        assert model.k == 8.0
        assert model.a_param == 0.15
        assert model.epsilon_0 == 0.002

    def test_custom_parameters(self):
        model = AlievPanfilovModel(k=10.0, a_param=0.1)
        assert model.k == 10.0
        assert model.a_param == 0.1

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="outside literature range"):
            AlievPanfilovModel(k=1.0)

    def test_invalid_a_param_raises(self):
        with pytest.raises(ValueError, match="outside literature range"):
            AlievPanfilovModel(a_param=0.5)

    def test_initial_state(self):
        model = AlievPanfilovModel()
        assert model.u == pytest.approx(0.0)
        assert model.v == pytest.approx(0.0)


class TestAlievPanfilovDynamics:
    def test_single_step(self):
        model = AlievPanfilovModel()
        u, v = model.step(0.0)
        assert isinstance(u, (float, np.floating))
        assert isinstance(v, (float, np.floating))

    def test_excitation_with_stimulus(self):
        model = AlievPanfilovModel()
        for _ in range(100):
            model.step(0.5)
        assert model.u > 0.1

    def test_return_to_rest(self):
        model = AlievPanfilovModel()
        for _ in range(50):
            model.step(0.5)
        for _ in range(500):
            model.step(0.0)
        assert model.u < 0.5

    def test_step_n(self):
        model = AlievPanfilovModel()
        traj = model.step_n(100, I_ext=0.3)
        assert traj.shape == (100, 2)
        assert not np.any(np.isnan(traj))

    def test_is_excited(self):
        model = AlievPanfilovModel()
        model.u = 0.5
        assert model.is_excited(threshold=0.3)
        model.u = 0.1
        assert not model.is_excited(threshold=0.3)

    def test_get_state(self):
        model = AlievPanfilovModel()
        state = model.get_state()
        assert state.shape == (2,)

    def test_clamping(self):
        model = AlievPanfilovModel()
        model.u = 100.0
        model.step(0.0)
        assert model.u <= 1.5  # Should be clamped

    def test_reset_with_rng(self):
        model = AlievPanfilovModel()
        rng = np.random.default_rng(42)
        u, v = model.reset(rng)
        assert isinstance(u, float)
        assert isinstance(v, float)

    def test_noise_variability(self):
        model = AlievPanfilovModel(noise_std=0.05)
        model.reset()
        np.random.seed(42)
        traj = model.step_n(50, 0.3)
        assert not np.all(np.diff(traj[:, 0]) == 0)


class TestAlievPanfilovParameterRanges:
    def test_all_defaults_in_range(self):
        for name, info in PARAMETER_RANGES.items():
            assert info["min"] <= info["default"] <= info["max"]

    def test_ranges_have_units(self):
        for name, info in PARAMETER_RANGES.items():
            assert "unit" in info

    def test_ranges_have_source(self):
        for name, info in PARAMETER_RANGES.items():
            assert "source" in info

    def test_epsilon_computation(self):
        model = AlievPanfilovModel()
        eps = model._epsilon(0.5, 1.0)
        assert eps > 0
        assert isinstance(eps, float)
