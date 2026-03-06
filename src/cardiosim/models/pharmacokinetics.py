"""Single-compartment pharmacokinetic model for antiarrhythmic drugs.

Models drug absorption, distribution, and elimination using a one-compartment
model with first-order kinetics. Used by the AntiarrhythmicDosing environment
to simulate drug concentration dynamics.

References:
    Derendorf, H., & Meibohm, B. (1999). Modeling of pharmacokinetic/
    pharmacodynamic (PK/PD) relationships. Pharmaceutical Research, 16(2),
    176-185.

    Nattel, S., & Singh, B. N. (1999). Evolution, mechanisms, and
    classification of antiarrhythmic drugs. American Journal of Cardiology,
    84(9), 11-19.
"""

from __future__ import annotations


# SIMPLIFICATION: Using a single-compartment PK model instead of a
# multi-compartment (e.g., 2- or 3-compartment) model. Clinical PK
# uses multi-compartment models for accurate tissue distribution.
# Single-compartment is acceptable for benchmark RL environments.

PARAMETER_RANGES = {
    "volume_of_distribution": {"min": 0.5, "max": 10.0, "unit": "L/kg",
                               "source": "Derendorf & Meibohm, 1999", "default": 2.0},
    "elimination_rate": {"min": 0.01, "max": 0.5, "unit": "1/h",
                         "source": "Nattel & Singh, 1999", "default": 0.1},
    "absorption_rate": {"min": 0.1, "max": 5.0, "unit": "1/h",
                        "source": "Derendorf & Meibohm, 1999", "default": 1.0},
    "therapeutic_min": {"min": 0.5, "max": 3.0, "unit": "mg/L",
                        "source": "Nattel & Singh, 1999", "default": 1.5},
    "therapeutic_max": {"min": 3.0, "max": 10.0, "unit": "mg/L",
                        "source": "Nattel & Singh, 1999", "default": 5.0},
    "toxic_threshold": {"min": 5.0, "max": 15.0, "unit": "mg/L",
                        "source": "Nattel & Singh, 1999", "default": 8.0},
}


class SingleCompartmentPKModel:
    """Single-compartment pharmacokinetic model.

    State variable:
        C: plasma drug concentration (mg/L)

    Equations:
        dC/dt = (dose * ka) / Vd - ke * C

    where:
        ka = absorption rate constant
        ke = elimination rate constant
        Vd = volume of distribution
    """

    def __init__(
        self,
        volume_of_distribution: float = 2.0,
        elimination_rate: float = 0.1,
        absorption_rate: float = 1.0,
        therapeutic_min: float = 1.5,
        therapeutic_max: float = 5.0,
        toxic_threshold: float = 8.0,
        dt: float = 0.1,
        body_weight: float = 70.0,
    ) -> None:
        self.vd = volume_of_distribution
        self.ke = elimination_rate
        self.ka = absorption_rate
        self.therapeutic_min = therapeutic_min
        self.therapeutic_max = therapeutic_max
        self.toxic_threshold = toxic_threshold
        self.dt = dt
        self.body_weight = body_weight

        self.concentration = 0.0
        self.gut_amount = 0.0
        self.total_dose_given = 0.0

    def reset(self) -> float:
        """Reset drug concentration to zero."""
        self.concentration = 0.0
        self.gut_amount = 0.0
        self.total_dose_given = 0.0
        return self.concentration

    def administer_dose(self, dose_mg: float) -> None:
        """Add a dose to the gut compartment for absorption."""
        self.gut_amount += dose_mg
        self.total_dose_given += dose_mg

    def step(self, dose_mg: float = 0.0) -> float:
        """Advance one timestep, optionally administering a dose.

        Args:
            dose_mg: Drug dose in mg (0 if no dose this step).

        Returns:
            Current plasma concentration.
        """
        if dose_mg > 0:
            self.administer_dose(dose_mg)

        # Absorption from gut to plasma (clamped to prevent negative gut)
        absorbed = min(self.ka * self.gut_amount * self.dt, self.gut_amount)
        self.gut_amount -= absorbed

        # Volume-adjusted concentration change
        total_vd = self.vd * self.body_weight
        dC_absorption = absorbed / total_vd
        dC_elimination = -self.ke * self.concentration * self.dt

        self.concentration += dC_absorption + dC_elimination
        self.concentration = max(0.0, min(self.concentration, 1000.0))

        return self.concentration

    def is_therapeutic(self) -> bool:
        """Check if concentration is within therapeutic window."""
        return self.therapeutic_min <= self.concentration <= self.therapeutic_max

    def is_toxic(self) -> bool:
        """Check if concentration exceeds toxic threshold."""
        return self.concentration >= self.toxic_threshold

    def is_subtherapeutic(self) -> bool:
        """Check if concentration is below therapeutic minimum."""
        return self.concentration < self.therapeutic_min

    def get_therapeutic_fraction(self) -> float:
        """Return position within therapeutic window as fraction [0, 1].

        0.0 = at or below minimum, 1.0 = at or above maximum.
        Values > 1.0 indicate approaching toxicity.
        """
        if self.concentration <= self.therapeutic_min:
            return 0.0
        window = self.therapeutic_max - self.therapeutic_min
        return min(2.0, (self.concentration - self.therapeutic_min) / window)

    def get_efficacy(self) -> float:
        """Compute drug efficacy using sigmoid Emax model.

        Returns a value between 0 and 1 representing the fraction
        of maximum pharmacodynamic effect.
        """
        # SIMPLIFICATION: Using simple sigmoid Hill equation instead of
        # full population PK/PD model with inter-individual variability.
        ec50 = (self.therapeutic_min + self.therapeutic_max) / 2.0
        hill_coeff = 2.0
        c = min(self.concentration, 1e6)  # Clamp to prevent overflow
        if c <= 0.0:
            return 0.0
        ratio = c / ec50
        # Numerically stable Hill equation: 1 / (1 + (ec50/c)^n)
        return ratio ** hill_coeff / (1.0 + ratio ** hill_coeff)
