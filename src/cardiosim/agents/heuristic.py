"""Heuristic baseline agents for CardioSim environments.

These implement simple rule-based policies that domain experts would
recognize as reasonable clinical heuristics. They serve as meaningful
baselines beyond random performance.
"""

from __future__ import annotations

import numpy as np


class HeuristicPacingAgent:
    """Simple rule-based pacemaker controller.

    Strategy: If heart rate is below target, increase pacing rate.
    If above target, decrease. Pulse amplitude proportional to error.
    """

    def __init__(self, target_hr: float = 72.0) -> None:
        self.target_hr = target_hr

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs[4] = hr_error (current_hr - target_hr)."""
        hr_error = obs[4]

        # Proportional control
        pacing_adjust = -hr_error * 0.5  # Negative feedback
        pacing_adjust = np.clip(pacing_adjust, -20.0, 20.0)

        # Pulse amplitude proportional to error magnitude
        amplitude = min(1.0, abs(hr_error) / 30.0)

        return np.array([pacing_adjust, amplitude], dtype=np.float32)


class HeuristicDosingAgent:
    """Simple rule-based antiarrhythmic dosing controller.

    Strategy: If concentration is subtherapeutic, give a dose.
    If in therapeutic range, give maintenance dose.
    If approaching toxicity, withhold dose.
    """

    def __init__(
        self,
        loading_dose: float = 50.0,
        maintenance_dose: float = 10.0,
        therapeutic_threshold: float = 1.5,
        toxic_threshold: float = 5.0,
    ) -> None:
        self.loading_dose = loading_dose
        self.maintenance_dose = maintenance_dose
        self.therapeutic_threshold = therapeutic_threshold
        self.toxic_threshold = toxic_threshold

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs[2] = drug_concentration, obs[3] = arrhythmia_indicator."""
        concentration = obs[2]
        arrhythmia = obs[3]

        if concentration < self.therapeutic_threshold * 0.5:
            # Well below therapeutic - loading dose
            dose = self.loading_dose
        elif concentration < self.therapeutic_threshold:
            # Below therapeutic - moderate dose
            dose = self.maintenance_dose * 2.0
        elif concentration > self.toxic_threshold * 0.8:
            # Approaching toxicity - withhold
            dose = 0.0
        elif arrhythmia > 0.5:
            # In range but arrhythmia active - slight increase
            dose = self.maintenance_dose * 1.5
        else:
            # In therapeutic range, maintain
            dose = self.maintenance_dose

        return np.array([dose], dtype=np.float32)
