"""Tests for the pharmacokinetics model."""

import pytest

from cardiosim.models.pharmacokinetics import SingleCompartmentPKModel, PARAMETER_RANGES


class TestPKModelInit:
    def test_default_initialization(self):
        model = SingleCompartmentPKModel()
        assert model.vd == 2.0
        assert model.ke == 0.1
        assert model.ka == 1.0

    def test_initial_concentration_zero(self):
        model = SingleCompartmentPKModel()
        assert model.concentration == 0.0


class TestPKModelDynamics:
    def test_dose_increases_concentration(self):
        model = SingleCompartmentPKModel()
        for _ in range(50):
            model.step(dose_mg=10.0)
        assert model.concentration > 0.0

    def test_elimination_reduces_concentration(self):
        model = SingleCompartmentPKModel()
        for _ in range(20):
            model.step(dose_mg=50.0)
        peak = model.concentration
        for _ in range(100):
            model.step(dose_mg=0.0)
        assert model.concentration < peak

    def test_concentration_stays_nonnegative(self):
        model = SingleCompartmentPKModel()
        model.step(0.0)
        assert model.concentration >= 0.0

    def test_reset(self):
        model = SingleCompartmentPKModel()
        model.step(50.0)
        model.reset()
        assert model.concentration == 0.0
        assert model.gut_amount == 0.0

    def test_therapeutic_window(self):
        model = SingleCompartmentPKModel()
        model.concentration = 3.0
        assert model.is_therapeutic()
        assert not model.is_toxic()
        assert not model.is_subtherapeutic()

    def test_toxic_detection(self):
        model = SingleCompartmentPKModel()
        model.concentration = 10.0
        assert model.is_toxic()

    def test_subtherapeutic_detection(self):
        model = SingleCompartmentPKModel()
        model.concentration = 0.5
        assert model.is_subtherapeutic()

    def test_therapeutic_fraction(self):
        model = SingleCompartmentPKModel()
        model.concentration = 0.0
        assert model.get_therapeutic_fraction() == 0.0
        model.concentration = 3.25  # Middle of window
        frac = model.get_therapeutic_fraction()
        assert 0.0 < frac < 1.0

    def test_efficacy_sigmoid(self):
        model = SingleCompartmentPKModel()
        model.concentration = 0.0
        assert model.get_efficacy() == pytest.approx(0.0)
        model.concentration = 10.0
        assert model.get_efficacy() > 0.5

    def test_efficacy_monotonic(self):
        model = SingleCompartmentPKModel()
        effs = []
        for c in [0.0, 1.0, 3.0, 5.0, 10.0]:
            model.concentration = c
            effs.append(model.get_efficacy())
        for i in range(1, len(effs)):
            assert effs[i] >= effs[i - 1]

    def test_administer_dose(self):
        model = SingleCompartmentPKModel()
        model.administer_dose(100.0)
        assert model.gut_amount == 100.0
        assert model.total_dose_given == 100.0

    def test_multiple_doses_accumulate(self):
        model = SingleCompartmentPKModel()
        model.administer_dose(50.0)
        model.administer_dose(50.0)
        assert model.total_dose_given == 100.0


class TestPKParameterRanges:
    def test_all_defaults_in_range(self):
        for name, info in PARAMETER_RANGES.items():
            assert info["min"] <= info["default"] <= info["max"]

    def test_therapeutic_below_toxic(self):
        assert PARAMETER_RANGES["therapeutic_max"]["default"] < PARAMETER_RANGES["toxic_threshold"]["default"]
