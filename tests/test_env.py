"""Tests for FleetRepositioningEnv."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from simulation.city import CityGraph
from simulation.demand import HistoricalDemandModel, SyntheticDemandModel
from simulation.env import FleetRepositioningEnv, Vehicle

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ZONES_DIR = DATA_DIR / "zones"
TRIPS_PATH = DATA_DIR / "processed" / "trips_manhattan.parquet"


@pytest.fixture
def city() -> CityGraph:
    return CityGraph(ZONES_DIR)


@pytest.fixture
def synthetic(city: CityGraph) -> SyntheticDemandModel:
    return SyntheticDemandModel.from_historical(TRIPS_PATH, city, seed=42)


@pytest.fixture
def env(city: CityGraph, synthetic: SyntheticDemandModel) -> FleetRepositioningEnv:
    return FleetRepositioningEnv(
        city=city,
        demand_model=synthetic,
        n_vehicles=50,
        step_minutes=5,
        episode_hours=2,
        max_wait_minutes=10.0,
    )


class TestReset:
    def test_observation_shape(self, env: FleetRepositioningEnv, city: CityGraph) -> None:
        obs, info = env.reset(seed=0)
        expected = city.n_zones * 3 + 4
        assert obs.shape == (expected,), f"Expected ({expected},), got {obs.shape}"

    def test_observation_in_bounds(self, env: FleetRepositioningEnv) -> None:
        obs, _ = env.reset(seed=0)
        assert obs.min() >= 0.0, f"Obs min {obs.min()} < 0"
        assert obs.max() <= 1.0, f"Obs max {obs.max()} > 1"
        assert obs.dtype == np.float32

    def test_info_keys(self, env: FleetRepositioningEnv) -> None:
        _, info = env.reset(seed=0)
        required = {"sim_time", "step", "n_idle", "n_occupied", "n_repositioning",
                     "pending_requests", "mean_wait_time_minutes", "fleet_utilization"}
        assert required.issubset(info.keys())

    def test_all_vehicles_idle_after_reset(self, env: FleetRepositioningEnv) -> None:
        _, info = env.reset(seed=0)
        assert info["n_idle"] == 50
        assert info["n_occupied"] == 0
        assert info["n_repositioning"] == 0

    def test_forced_date(self, env: FleetRepositioningEnv) -> None:
        _, info = env.reset(seed=0, options={"sim_date": datetime(2023, 1, 5)})
        assert "2023-01-05" in info["sim_time"]


class TestStep:
    def test_step_returns_valid_tuple(self, env: FleetRepositioningEnv) -> None:
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_in_bounds(self, env: FleetRepositioningEnv) -> None:
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_reward_in_range(self, env: FleetRepositioningEnv) -> None:
        env.reset(seed=0)
        for _ in range(5):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            assert -1.0 <= reward <= 1.0, f"Reward {reward} out of [-1, 1]"

    def test_ten_consecutive_steps(self, env: FleetRepositioningEnv) -> None:
        env.reset(seed=0)
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == env.observation_space.shape
            assert info["step"] == i + 1

    def test_zero_action_no_crash(self, env: FleetRepositioningEnv) -> None:
        """Zero action means no repositioning — env should still work."""
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape


class TestEpisode:
    def test_full_episode_terminates(
        self, city: CityGraph, synthetic: SyntheticDemandModel
    ) -> None:
        """A full episode should end with terminated=True."""
        env = FleetRepositioningEnv(
            city=city,
            demand_model=synthetic,
            n_vehicles=20,
            step_minutes=5,
            episode_hours=1,  # short episode: 12 steps
        )
        obs, _ = env.reset(seed=42)
        terminated = False
        steps = 0
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            assert steps <= 20, "Episode should terminate within expected steps"

        assert terminated
        assert steps == 12  # 1 hour / 5 min = 12 steps

    def test_trips_served_during_episode(
        self, city: CityGraph, synthetic: SyntheticDemandModel
    ) -> None:
        """With demand and vehicles, some trips should be served."""
        env = FleetRepositioningEnv(
            city=city,
            demand_model=synthetic,
            n_vehicles=100,
            step_minutes=5,
            episode_hours=1,
        )
        env.reset(seed=42)
        for _ in range(12):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, _, _, info = env.step(action)

        assert info["total_trips_requested"] > 0, "Should have some demand"
        assert info["total_trips_served"] > 0, "Some trips should be served"


class TestDemandImpact:
    def test_idle_vehicles_decrease_with_demand(
        self, city: CityGraph, synthetic: SyntheticDemandModel
    ) -> None:
        """During a busy period, idle vehicle count should decrease."""
        env = FleetRepositioningEnv(
            city=city,
            demand_model=synthetic,
            n_vehicles=50,
            step_minutes=5,
            episode_hours=2,
        )
        # Use noon (busy period) to ensure demand
        _, info_start = env.reset(
            seed=42, options={"sim_date": datetime(2023, 1, 3)}
        )
        initial_idle = info_start["n_idle"]

        # Run several steps with zero action (no repositioning)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        min_idle = initial_idle
        for _ in range(6):
            _, _, _, _, info = env.step(action)
            min_idle = min(min_idle, info["n_idle"])

        assert min_idle < initial_idle, (
            f"Idle count should decrease from {initial_idle} as vehicles serve demand"
        )


class TestSnapshot:
    def test_get_snapshot(self, env: FleetRepositioningEnv, city: CityGraph) -> None:
        env.reset(seed=0)
        env.step(env.action_space.sample())
        snap = env.get_snapshot()

        assert len(snap.vehicles) == 50
        assert len(snap.vehicle_counts_per_zone) == city.n_zones
        assert len(snap.demand_counts_per_zone) == city.n_zones
        assert len(snap.forecast_counts_per_zone) == city.n_zones
        assert isinstance(snap.pending_requests, int)
        assert isinstance(snap.metrics, dict)

    def test_snapshot_json_roundtrip(self, env: FleetRepositioningEnv) -> None:
        env.reset(seed=0)
        env.step(env.action_space.sample())
        snap = env.get_snapshot()
        json_str = snap.to_json()
        from simulation.snapshot import SimSnapshot
        snap2 = SimSnapshot.from_json(json_str)
        assert len(snap2.vehicles) == len(snap.vehicles)
        assert snap2.pending_requests == snap.pending_requests
