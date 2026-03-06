"""CardioSim Gymnasium environments."""

from gymnasium.envs.registration import register


def register_envs() -> None:
    """Register all CardioSim environments with Gymnasium."""
    register(
        id="cardiosim/PacingControl-v0",
        entry_point="cardiosim.envs.pacing_control:PacingControlEnv",
    )
    register(
        id="cardiosim/AntiarrhythmicDosing-v0",
        entry_point="cardiosim.envs.antiarrhythmic_dosing:AntiarrhythmicDosingEnv",
    )
    register(
        id="cardiosim/DefibrillationTiming-v0",
        entry_point="cardiosim.envs.defibrillation_timing:DefibrillationTimingEnv",
    )


__all__ = ["register_envs"]
