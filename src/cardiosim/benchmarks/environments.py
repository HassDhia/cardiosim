"""Benchmark environment configurations with difficulty tiers."""

from __future__ import annotations


BENCHMARK_CONFIGS = {
    "PacingControl": {
        "env_id": "cardiosim/PacingControl-v0",
        "tiers": {
            "easy": {"difficulty": "easy", "max_steps": 500},
            "medium": {"difficulty": "medium", "max_steps": 500},
            "hard": {"difficulty": "hard", "max_steps": 500},
        },
        "primary": True,
    },
    "AntiarrhythmicDosing": {
        "env_id": "cardiosim/AntiarrhythmicDosing-v0",
        "tiers": {
            "easy": {"difficulty": "easy", "max_steps": 200},
            "medium": {"difficulty": "medium", "max_steps": 200},
            "hard": {"difficulty": "hard", "max_steps": 200},
        },
        "primary": True,
    },
    "DefibrillationTiming": {
        "env_id": "cardiosim/DefibrillationTiming-v0",
        "tiers": {
            "easy": {"difficulty": "easy", "max_steps": 100},
            "medium": {"difficulty": "medium", "max_steps": 100},
            "hard": {"difficulty": "hard", "max_steps": 100},
        },
        "primary": True,
    },
}


def get_benchmark_env_ids() -> list[str]:
    """Return all benchmark environment IDs."""
    return [cfg["env_id"] for cfg in BENCHMARK_CONFIGS.values()]


def get_primary_env_ids() -> list[str]:
    """Return environment IDs marked as primary."""
    return [cfg["env_id"] for cfg in BENCHMARK_CONFIGS.values() if cfg.get("primary")]
