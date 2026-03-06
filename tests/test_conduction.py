"""Tests for the cardiac conduction system model."""

import numpy as np
import pytest

from cardiosim.models.conduction import CardiacConductionModel, PARAMETER_RANGES


class TestConductionInit:
    def test_default_initialization(self):
        model = CardiacConductionModel()
        assert model.sa_rate == 72.0
        assert model.av_delay == 160.0
        assert model.refractory_period == 300.0

    def test_sa_interval_computed(self):
        model = CardiacConductionModel(sa_rate=60.0)
        assert model.sa_interval == pytest.approx(1000.0)


class TestConductionDynamics:
    def test_single_step(self):
        model = CardiacConductionModel()
        result = model.step()
        assert isinstance(result, dict)
        assert "heart_rate" in result
        assert "sa_fired" in result

    def test_sa_node_fires(self):
        model = CardiacConductionModel(sa_rate=60.0)
        # Run for more than one cycle
        fired = False
        for _ in range(1200):
            result = model.step()
            if result["sa_fired"]:
                fired = True
                break
        assert fired

    def test_pacing_stimulus(self):
        model = CardiacConductionModel(sa_rate=40.0)
        # Run with external pacing
        for _ in range(500):
            model.step(pacing_stimulus=1.0)
        hr = model.get_heart_rate()
        assert hr > 0

    def test_heart_rate_bradycardia(self):
        model = CardiacConductionModel(sa_rate=50.0)
        for _ in range(2000):
            model.step()
        assert model.is_bradycardia()

    def test_heart_rate_tachycardia(self):
        model = CardiacConductionModel(sa_rate=110.0)
        for _ in range(2000):
            model.step()
        assert model.is_tachycardia()

    def test_normal_sinus_rhythm(self):
        model = CardiacConductionModel(sa_rate=72.0)
        for _ in range(2000):
            model.step()
        assert model.is_normal_sinus()

    def test_set_sa_rate(self):
        model = CardiacConductionModel()
        model.set_sa_rate(90.0)
        assert model.sa_rate == 90.0
        assert model.sa_interval == pytest.approx(60000.0 / 90.0)

    def test_set_sa_rate_clamps(self):
        model = CardiacConductionModel()
        model.set_sa_rate(300.0)
        assert model.sa_rate == 200.0
        model.set_sa_rate(10.0)
        assert model.sa_rate == 30.0

    def test_hrv(self):
        model = CardiacConductionModel(sa_rate=72.0)
        for _ in range(5000):
            model.step()
        hrv = model.get_heart_rate_variability()
        assert hrv >= 0.0

    def test_conduction_block(self):
        model = CardiacConductionModel(conduction_block_prob=0.5)
        model.reset(np.random.default_rng(42))
        sa_fires = 0
        vent_fires = 0
        for _ in range(5000):
            result = model.step()
            if result["sa_fired"]:
                sa_fires += 1
            if result["ventricular_fired"]:
                vent_fires += 1
        # With 50% block, vent fires should be fewer than SA fires
        assert vent_fires <= sa_fires

    def test_reset(self):
        model = CardiacConductionModel()
        for _ in range(1000):
            model.step()
        model.reset()
        assert model.time == 0.0
        assert len(model.heart_rate_history) == 0


class TestConductionParameterRanges:
    def test_all_defaults_in_range(self):
        for name, info in PARAMETER_RANGES.items():
            assert info["min"] <= info["default"] <= info["max"]

    def test_ranges_have_units(self):
        for name, info in PARAMETER_RANGES.items():
            assert "unit" in info
