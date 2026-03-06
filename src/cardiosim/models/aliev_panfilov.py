"""Aliev-Panfilov cardiac tissue model.

The Aliev-Panfilov model extends FitzHugh-Nagumo with more realistic
cardiac action potential morphology. It uses a modified cubic nonlinearity
and multiplicative recovery coupling.

References:
    Aliev, R. R., & Panfilov, A. V. (1996). A simple two-variable model
    of cardiac excitation. Chaos, Solitons & Fractals, 7(3), 293-301.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# SIMPLIFICATION: Using the single-cell Aliev-Panfilov model without
# spatial diffusion. Full tissue simulation would require PDE solver
# with diffusion term D * nabla^2(u). Acceptable for RL benchmark.

PARAMETER_RANGES = {
    "k": {"min": 4.0, "max": 12.0, "unit": "dimensionless",
           "source": "Aliev & Panfilov, 1996", "default": 8.0},
    "a_param": {"min": 0.05, "max": 0.2, "unit": "dimensionless",
                "source": "Aliev & Panfilov, 1996", "default": 0.15},
    "epsilon_0": {"min": 0.001, "max": 0.01, "unit": "1/ms",
                  "source": "Aliev & Panfilov, 1996", "default": 0.002},
    "mu_1": {"min": 0.1, "max": 0.3, "unit": "dimensionless",
             "source": "Aliev & Panfilov, 1996", "default": 0.2},
    "mu_2": {"min": 0.2, "max": 0.4, "unit": "dimensionless",
             "source": "Aliev & Panfilov, 1996", "default": 0.3},
}


class AlievPanfilovModel:
    """Aliev-Panfilov two-variable cardiac model.

    State variables:
        u: normalized membrane potential (0 to 1, fast)
        v: recovery variable (slow)

    Equations:
        du/dt = -k * u * (u - a) * (u - 1) - u * v + I_ext
        dv/dt = epsilon(u, v) * (-v - k * u * (u - a - 1))

    where epsilon(u, v) = epsilon_0 + mu_1 * v / (u + mu_2)
    """

    def __init__(
        self,
        k: float = 8.0,
        a_param: float = 0.15,
        epsilon_0: float = 0.002,
        mu_1: float = 0.2,
        mu_2: float = 0.3,
        dt: float = 0.1,
        noise_std: float = 0.0,
    ) -> None:
        self._validate_params(k, a_param, epsilon_0, mu_1, mu_2)

        self.k = k
        self.a_param = a_param
        self.epsilon_0 = epsilon_0
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.dt = dt
        self.noise_std = noise_std

        self.u = 0.0
        self.v = 0.0
        self._rng: np.random.Generator = np.random.default_rng()

    @staticmethod
    def _validate_params(k: float, a: float, eps0: float, mu1: float, mu2: float) -> None:
        params = {"k": k, "a_param": a, "epsilon_0": eps0, "mu_1": mu1, "mu_2": mu2}
        for name, value in params.items():
            info = PARAMETER_RANGES[name]
            if not info["min"] <= value <= info["max"]:
                raise ValueError(
                    f"Parameter '{name}' = {value} outside literature range "
                    f"[{info['min']}, {info['max']}] ({info['source']})"
                )

    def reset(self, rng: np.random.Generator | None = None) -> tuple[float, float]:
        """Reset to resting state."""
        if rng is not None:
            self._rng = rng
            self.u = max(0.0, rng.normal(0.0, 0.01))
            self.v = max(0.0, rng.normal(0.0, 0.01))
        else:
            self.u = 0.0
            self.v = 0.0
        return self.u, self.v

    def _epsilon(self, u: float, v: float) -> float:
        """Compute the variable time-scale parameter."""
        return self.epsilon_0 + self.mu_1 * v / (u + self.mu_2)

    def _derivatives(self, u: float, v: float, I_ext: float) -> tuple[float, float]:
        """Compute du/dt, dv/dt."""
        k, a = self.k, self.a_param
        dudt = -k * u * (u - a) * (u - 1.0) - u * v + I_ext
        eps = self._epsilon(u, v)
        dvdt = eps * (-v - k * u * (u - a - 1.0))
        return dudt, dvdt

    def step(self, I_ext: float = 0.0) -> tuple[float, float]:
        """Advance one timestep using RK4 integration."""
        u, v = self.u, self.v
        dt = self.dt

        k1u, k1v = self._derivatives(u, v, I_ext)
        k2u, k2v = self._derivatives(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v, I_ext)
        k3u, k3v = self._derivatives(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v, I_ext)
        k4u, k4v = self._derivatives(u + dt * k3u, v + dt * k3v, I_ext)

        self.u += (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        self.v += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

        # Clamp to physiological range
        self.u = np.clip(self.u, -0.1, 1.5)
        self.v = np.clip(self.v, -0.1, 20.0)

        if self.noise_std > 0:
            self.u += self._rng.normal(0, self.noise_std)

        return self.u, self.v

    def step_n(self, n: int, I_ext: float = 0.0) -> NDArray[np.float64]:
        """Run n steps, returning (n, 2) trajectory."""
        trajectory = np.zeros((n, 2))
        for i in range(n):
            u, v = self.step(I_ext)
            trajectory[i] = [u, v]
        return trajectory

    def is_excited(self, threshold: float = 0.3) -> bool:
        """Check if membrane is in excited state."""
        return self.u > threshold

    def get_state(self) -> NDArray[np.float64]:
        """Return current state as numpy array."""
        return np.array([self.u, self.v], dtype=np.float64)
