"""FitzHugh-Nagumo cardiac action potential model.

The FitzHugh-Nagumo (FHN) model is a two-variable simplification of the
Hodgkin-Huxley model. It captures the essential dynamics of excitable media
using a fast voltage-like variable (v) and a slow recovery variable (w).

References:
    FitzHugh, R. (1961). Impulses and physiological states in theoretical
    models of nerve membrane. Biophysical Journal, 1(6), 445-466.

    Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse
    transmission line simulating nerve axon. Proceedings of the IRE, 50(10),
    2061-2070.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# SIMPLIFICATION: Using the 2-variable FitzHugh-Nagumo model instead of
# the full Hodgkin-Huxley 4-variable model. FHN captures qualitative
# excitation/recovery dynamics but not ionic current detail.

PARAMETER_RANGES = {
    "a": {"min": 0.05, "max": 0.5, "unit": "dimensionless",
           "source": "FitzHugh, 1961", "default": 0.2},
    "b": {"min": 0.1, "max": 1.5, "unit": "dimensionless",
           "source": "FitzHugh, 1961", "default": 0.7},
    "tau": {"min": 5.0, "max": 25.0, "unit": "ms",
            "source": "FitzHugh, 1961", "default": 12.5},
    "I_ext_range": {"min": 0.0, "max": 2.0, "unit": "dimensionless",
                    "source": "FitzHugh, 1961", "default": 0.5},
}


class FitzHughNagumoModel:
    """FitzHugh-Nagumo two-variable excitable membrane model.

    State variables:
        v: membrane voltage analogue (fast, excitatory)
        w: recovery variable (slow, inhibitory)

    Equations:
        dv/dt = v - v^3/3 - w + I_ext
        dw/dt = (v + a - b*w) / tau
    """

    def __init__(
        self,
        a: float = 0.2,
        b: float = 0.7,
        tau: float = 12.5,
        dt: float = 0.05,
        noise_std: float = 0.0,
    ) -> None:
        self._validate_param("a", a)
        self._validate_param("b", b)
        self._validate_param("tau", tau)

        self.a = a
        self.b = b
        self.tau = tau
        self.dt = dt
        self.noise_std = noise_std

        self.v = -1.2
        self.w = -0.6

    @staticmethod
    def _validate_param(name: str, value: float) -> None:
        info = PARAMETER_RANGES[name]
        if not info["min"] <= value <= info["max"]:
            raise ValueError(
                f"Parameter '{name}' = {value} outside literature range "
                f"[{info['min']}, {info['max']}] ({info['source']})"
            )

    def reset(self, rng: np.random.Generator | None = None) -> tuple[float, float]:
        """Reset to resting state with optional small perturbation."""
        if rng is not None:
            self.v = -1.2 + rng.normal(0, 0.05)
            self.w = -0.6 + rng.normal(0, 0.05)
        else:
            self.v = -1.2
            self.w = -0.6
        return self.v, self.w

    def _derivatives(self, v: float, w: float, I_ext: float) -> tuple[float, float]:
        """Compute dv/dt, dw/dt."""
        dvdt = v - (v**3) / 3.0 - w + I_ext
        dwdt = (v + self.a - self.b * w) / self.tau
        return dvdt, dwdt

    def step(self, I_ext: float = 0.0) -> tuple[float, float]:
        """Advance one timestep using 4th-order Runge-Kutta integration.

        Args:
            I_ext: External stimulus current.

        Returns:
            Tuple of (v, w) after integration.
        """
        v, w = self.v, self.w
        dt = self.dt

        k1v, k1w = self._derivatives(v, w, I_ext)
        k2v, k2w = self._derivatives(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, I_ext)
        k3v, k3w = self._derivatives(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, I_ext)
        k4v, k4w = self._derivatives(v + dt * k3v, w + dt * k3w, I_ext)

        self.v += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.w += (dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)

        if self.noise_std > 0:
            self.v += np.random.normal(0, self.noise_std)

        return self.v, self.w

    def step_n(self, n: int, I_ext: float = 0.0) -> NDArray[np.float64]:
        """Run n integration steps, returning trajectory as (n, 2) array."""
        trajectory = np.zeros((n, 2))
        for i in range(n):
            v, w = self.step(I_ext)
            trajectory[i] = [v, w]
        return trajectory

    def is_excited(self, threshold: float = 0.5) -> bool:
        """Check if the membrane is in an excited (depolarized) state."""
        return self.v > threshold

    def get_state(self) -> NDArray[np.float64]:
        """Return current state as numpy array."""
        return np.array([self.v, self.w], dtype=np.float64)

    @property
    def action_potential_duration(self) -> float:
        """Estimate APD from model parameters (approximate)."""
        # SIMPLIFICATION: Analytical APD estimate for FHN.
        # Clinical APD measurement uses 90% repolarization (APD90).
        return self.tau * np.log(1.0 + 1.0 / self.b)
