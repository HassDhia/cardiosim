"""Tests for benchmark infrastructure."""


from cardiosim.benchmarks.environments import (
    BENCHMARK_CONFIGS,
    get_benchmark_env_ids,
    get_primary_env_ids,
)


class TestBenchmarkConfigs:
    def test_three_environments(self):
        assert len(BENCHMARK_CONFIGS) == 3

    def test_all_have_tiers(self):
        for name, config in BENCHMARK_CONFIGS.items():
            assert "tiers" in config
            assert "easy" in config["tiers"]
            assert "medium" in config["tiers"]
            assert "hard" in config["tiers"]

    def test_all_have_env_id(self):
        for name, config in BENCHMARK_CONFIGS.items():
            assert config["env_id"].startswith("cardiosim/")

    def test_get_benchmark_env_ids(self):
        ids = get_benchmark_env_ids()
        assert len(ids) == 3
        assert all(id.startswith("cardiosim/") for id in ids)

    def test_get_primary_env_ids(self):
        ids = get_primary_env_ids()
        assert len(ids) == 3  # All are primary


class TestBenchmarkTierConfig:
    def test_tier_has_difficulty(self):
        for name, config in BENCHMARK_CONFIGS.items():
            for tier_name, tier in config["tiers"].items():
                assert "difficulty" in tier

    def test_tier_has_max_steps(self):
        for name, config in BENCHMARK_CONFIGS.items():
            for tier_name, tier in config["tiers"].items():
                assert "max_steps" in tier
                assert tier["max_steps"] > 0
