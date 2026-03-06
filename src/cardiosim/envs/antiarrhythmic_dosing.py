"""AntiarrhythmicDosing-v0: Drug dosing to suppress cardiac arrhythmias.

The agent decides drug dosing to maintain therapeutic plasma concentration
while suppressing arrhythmias and avoiding toxicity.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cardiosim.models.fitzhugh_nagumo import FitzHughNagumoModel
from cardiosim.models.pharmacokinetics import SingleCompartmentPKModel


DIFFICULTY_TIERS = {
    "easy": {"arrhythmia_prob": 0.3, "noise_std": 0.01, "pk_variability": 0.0},
    "medium": {"arrhythmia_prob": 0.5, "noise_std": 0.03, "pk_variability": 0.2},
    "hard": {"arrhythmia_prob": 0.7, "noise_std": 0.05, "pk_variability": 0.4},
}


class AntiarrhythmicDosingEnv(gym.Env):
    """Gymnasium environment for antiarrhythmic drug dosing.

    Observation space (Box, 6 dimensions):
        - membrane_voltage: Current FHN voltage
        - recovery_variable: Current FHN recovery
        - drug_concentration: Plasma drug level (mg/L)
        - arrhythmia_indicator: 1.0 if arrhythmia detected, else 0.0
        - drug_efficacy: Current pharmacodynamic effect (0-1)
        - time_in_therapeutic: Fraction of episode in therapeutic range

    Action space (Box, 1 dimension):
        - dose: Drug dose to administer (0 to 100 mg)

    Reward:
        - Positive for maintaining therapeutic concentration
        - Negative for arrhythmia presence
        - Strong negative for toxicity
        - Small negative for each dose (minimize total drug exposure)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty: str = "medium",
        max_steps: int = 200,
    ) -> None:
        super().__init__()

        tier = DIFFICULTY_TIERS.get(difficulty, DIFFICULTY_TIERS["medium"])
        self.difficulty = difficulty
        self.arrhythmia_prob = tier["arrhythmia_prob"]
        self.max_steps = max_steps

        self.cell_model = FitzHughNagumoModel(noise_std=tier["noise_std"])
        self.pk_model = SingleCompartmentPKModel()

        # Apply PK variability
        if tier["pk_variability"] > 0:
            self._pk_variability = tier["pk_variability"]
        else:
            self._pk_variability = 0.0

        # Observation: [voltage, recovery, concentration, arrhythmia, efficacy, therapeutic_frac]
        self.observation_space = spaces.Box(
            low=np.array([-3.0, -3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 3.0, 20.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Action: [dose_mg]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([100.0], dtype=np.float32),
        )

        self.steps = 0
        self.therapeutic_steps = 0
        self.arrhythmia_active = False
        self._rng: np.random.Generator | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self.cell_model.reset(self._rng)
        self.pk_model.reset()

        # Randomize PK parameters slightly
        if self._pk_variability > 0:
            var = self._pk_variability
            self.pk_model.ke *= (1.0 + self._rng.uniform(-var, var))
            self.pk_model.vd *= (1.0 + self._rng.uniform(-var, var))

        self.steps = 0
        self.therapeutic_steps = 0
        self.arrhythmia_active = self._rng.random() < self.arrhythmia_prob

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        dose = float(np.clip(action[0], 0.0, 100.0))

        # Administer drug and advance PK model
        self.pk_model.step(dose)

        # Drug effect on arrhythmia: efficacy suppresses arrhythmia
        efficacy = self.pk_model.get_efficacy()
        suppression_prob = efficacy * 0.8  # Max 80% suppression per step

        # Update arrhythmia state
        if self.arrhythmia_active:
            if self._rng.random() < suppression_prob:
                self.arrhythmia_active = False
        else:
            # Arrhythmia can recur
            recurrence_base = self.arrhythmia_prob * 0.05
            recurrence_prob = recurrence_base * (1.0 - efficacy)
            if self._rng.random() < recurrence_prob:
                self.arrhythmia_active = True

        # Drive FHN model with arrhythmia-modulated stimulus
        I_ext = 0.8 if self.arrhythmia_active else 0.3
        for _ in range(20):
            self.cell_model.step(I_ext)

        # Compute reward
        reward = 0.0

        # Therapeutic window reward
        if self.pk_model.is_therapeutic():
            reward += 1.0
            self.therapeutic_steps += 1
        elif self.pk_model.is_subtherapeutic():
            reward -= 0.3
        elif self.pk_model.is_toxic():
            reward -= 3.0  # Strong toxicity penalty

        # Arrhythmia penalty
        if self.arrhythmia_active:
            reward -= 0.5

        # Drug exposure penalty (minimize total dose)
        reward -= 0.01 * dose / 100.0

        self.steps += 1
        terminated = self.pk_model.is_toxic() and self.pk_model.concentration > 12.0
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {
            "concentration": self.pk_model.concentration,
            "arrhythmia_active": self.arrhythmia_active,
            "efficacy": efficacy,
            "is_therapeutic": self.pk_model.is_therapeutic(),
            "is_toxic": self.pk_model.is_toxic(),
            "therapeutic_fraction": self.therapeutic_steps / max(1, self.steps),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        therapeutic_frac = self.therapeutic_steps / max(1, self.steps)
        conc = min(float(self.pk_model.concentration), 50.0)
        return np.array([
            self.cell_model.v,
            self.cell_model.w,
            conc,
            1.0 if self.arrhythmia_active else 0.0,
            self.pk_model.get_efficacy(),
            therapeutic_frac,
        ], dtype=np.float32)
