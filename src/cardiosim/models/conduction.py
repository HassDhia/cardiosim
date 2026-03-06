"""Cardiac conduction system model.

Models the timing and propagation of electrical signals through the
heart's conduction system: SA node, AV node, His-Purkinje system.
Used by the PacingControl environment.

References:
    Malmivuo, J., & Plonsey, R. (1995). Bioelectromagnetism: Principles
    and Applications. Oxford University Press.

    Kusumoto, F. M. (2020). ECG Interpretation: From Pathophysiology to
    Clinical Application. Springer.
"""

from __future__ import annotations

import numpy as np


# SIMPLIFICATION: Using a timing-based conduction model instead of
# spatially resolved cable equation. Clinical cardiac conduction uses
# 3D spatial propagation through tissue. This lumped-parameter model
# captures SA-AV-His timing without spatial detail.

PARAMETER_RANGES = {
    "sa_rate": {"min": 40.0, "max": 120.0, "unit": "bpm",
                "source": "Malmivuo & Plonsey, 1995", "default": 72.0},
    "av_delay": {"min": 120.0, "max": 300.0, "unit": "ms",
                 "source": "Kusumoto, 2020", "default": 160.0},
    "his_purkinje_delay": {"min": 30.0, "max": 80.0, "unit": "ms",
                           "source": "Kusumoto, 2020", "default": 50.0},
    "refractory_period": {"min": 200.0, "max": 400.0, "unit": "ms",
                          "source": "Malmivuo & Plonsey, 1995", "default": 300.0},
}


class CardiacConductionModel:
    """Lumped-parameter cardiac conduction system model.

    Models the electrical activation sequence:
    SA node -> AV node -> His bundle -> Purkinje fibers -> ventricular activation

    Tracks timing of each node's firing and implements refractory periods.
    """

    def __init__(
        self,
        sa_rate: float = 72.0,
        av_delay: float = 160.0,
        his_purkinje_delay: float = 50.0,
        refractory_period: float = 300.0,
        dt: float = 1.0,
        conduction_block_prob: float = 0.0,
    ) -> None:
        self.sa_rate = sa_rate
        self.av_delay = av_delay
        self.his_purkinje_delay = his_purkinje_delay
        self.refractory_period = refractory_period
        self.dt = dt
        self.conduction_block_prob = conduction_block_prob

        self.sa_interval = 60000.0 / sa_rate  # ms between SA beats
        self.time = 0.0
        self.last_sa_fire = -self.sa_interval
        self.last_av_fire = -1000.0
        self.last_ventricular = -1000.0
        self.last_pacing_stimulus = -1000.0

        self._rng = np.random.default_rng()
        self.heart_rate_history: list[float] = []
        self.rr_intervals: list[float] = []

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset conduction system to initial state."""
        if rng is not None:
            self._rng = rng
        self.time = 0.0
        self.last_sa_fire = -self.sa_interval
        self.last_av_fire = -1000.0
        self.last_ventricular = -1000.0
        self.last_pacing_stimulus = -1000.0
        self.heart_rate_history = []
        self.rr_intervals = []

    def set_sa_rate(self, bpm: float) -> None:
        """Update the intrinsic SA node rate."""
        self.sa_rate = np.clip(bpm, 30.0, 200.0)
        self.sa_interval = 60000.0 / self.sa_rate

    def step(self, pacing_stimulus: float = 0.0) -> dict:
        """Advance one timestep.

        Args:
            pacing_stimulus: External pacing current (>0 triggers pacing pulse).

        Returns:
            Dict with keys: sa_fired, av_fired, ventricular_fired, heart_rate,
            time_since_last_beat.
        """
        self.time += self.dt

        sa_fired = False
        av_fired = False
        ventricular_fired = False

        # SA node intrinsic firing
        time_since_sa = self.time - self.last_sa_fire
        if time_since_sa >= self.sa_interval:
            sa_fired = True
            self.last_sa_fire = self.time

        # External pacing can also trigger if myocardium is not refractory
        if pacing_stimulus > 0.0:
            time_since_vent = self.time - self.last_ventricular
            if time_since_vent > self.refractory_period:
                self.last_pacing_stimulus = self.time
                sa_fired = True  # Pacing captures the rhythm

        # AV conduction (with possible block)
        if sa_fired:
            if self._rng.random() > self.conduction_block_prob:
                time_since_av = self.time - self.last_av_fire
                if time_since_av > self.refractory_period:
                    av_fired = True
                    self.last_av_fire = self.time

        # Ventricular activation after His-Purkinje delay
        if av_fired:
            ventricular_fired = True
            self.last_ventricular = self.time

            # Record RR interval
            if len(self.rr_intervals) > 0 or self.time > self.sa_interval * 2:
                rr = time_since_sa if time_since_sa < 3000 else self.sa_interval
                self.rr_intervals.append(rr)

        # Compute instantaneous heart rate
        if len(self.rr_intervals) > 0:
            recent_rr = self.rr_intervals[-min(5, len(self.rr_intervals)):]
            avg_rr = np.mean(recent_rr)
            heart_rate = 60000.0 / avg_rr if avg_rr > 0 else 0.0
        else:
            heart_rate = self.sa_rate

        self.heart_rate_history.append(heart_rate)

        return {
            "sa_fired": sa_fired,
            "av_fired": av_fired,
            "ventricular_fired": ventricular_fired,
            "heart_rate": heart_rate,
            "time_since_last_beat": self.time - self.last_ventricular,
        }

    def get_heart_rate(self) -> float:
        """Return current heart rate estimate in bpm."""
        if self.heart_rate_history:
            return self.heart_rate_history[-1]
        return self.sa_rate

    def get_heart_rate_variability(self) -> float:
        """Return HRV as standard deviation of recent RR intervals."""
        if len(self.rr_intervals) < 3:
            return 0.0
        recent = self.rr_intervals[-min(10, len(self.rr_intervals)):]
        return float(np.std(recent))

    def is_bradycardia(self) -> bool:
        """Heart rate < 60 bpm."""
        return self.get_heart_rate() < 60.0

    def is_tachycardia(self) -> bool:
        """Heart rate > 100 bpm."""
        return self.get_heart_rate() > 100.0

    def is_normal_sinus(self) -> bool:
        """Heart rate between 60-100 bpm."""
        hr = self.get_heart_rate()
        return 60.0 <= hr <= 100.0
