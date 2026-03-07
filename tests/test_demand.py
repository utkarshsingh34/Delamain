"""Tests for demand models."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from simulation.city import CityGraph
from simulation.demand import HistoricalDemandModel, RideRequest, SyntheticDemandModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ZONES_DIR = DATA_DIR / "zones"
TRIPS_PATH = DATA_DIR / "processed" / "trips_manhattan.parquet"


@pytest.fixture
def city() -> CityGraph:
    return CityGraph(ZONES_DIR)


@pytest.fixture
def historical(city: CityGraph) -> HistoricalDemandModel:
    return HistoricalDemandModel(TRIPS_PATH, city)


@pytest.fixture
def synthetic(city: CityGraph) -> SyntheticDemandModel:
    return SyntheticDemandModel.from_historical(TRIPS_PATH, city, seed=42)


# --- RideRequest ---


def test_ride_request_fields() -> None:
    req = RideRequest(
        request_id="abc123",
        timestamp=datetime(2023, 1, 3, 8, 0),
        pickup_zone_idx=5,
        dropoff_zone_idx=10,
        estimated_duration_s=600,
    )
    assert req.pickup_zone_idx == 5
    assert req.estimated_duration_s == 600


# --- HistoricalDemandModel ---


def test_historical_busy_hour(historical: HistoricalDemandModel, city: CityGraph) -> None:
    # Tuesday Jan 3, 2023 at 8:00 AM — should be a busy period
    requests = historical.get_requests(datetime(2023, 1, 3, 8, 0), duration_minutes=1)
    assert len(requests) > 10, f"Expected >10 requests at 8 AM Tuesday, got {len(requests)}"


def test_historical_quiet_hour(historical: HistoricalDemandModel) -> None:
    # 4 AM should have much fewer requests
    busy = historical.get_requests(datetime(2023, 1, 3, 8, 0), duration_minutes=1)
    quiet = historical.get_requests(datetime(2023, 1, 3, 4, 0), duration_minutes=1)
    assert len(quiet) < len(busy), "4 AM should have fewer requests than 8 AM"


def test_historical_empty_window(historical: HistoricalDemandModel) -> None:
    # Date outside the data range
    requests = historical.get_requests(datetime(2020, 1, 1, 12, 0), duration_minutes=1)
    assert len(requests) == 0


def test_historical_zone_indices_valid(
    historical: HistoricalDemandModel, city: CityGraph
) -> None:
    requests = historical.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=5)
    assert len(requests) > 0
    for req in requests:
        assert 0 <= req.pickup_zone_idx < city.n_zones, (
            f"pickup_zone_idx {req.pickup_zone_idx} out of range"
        )
        assert 0 <= req.dropoff_zone_idx < city.n_zones, (
            f"dropoff_zone_idx {req.dropoff_zone_idx} out of range"
        )


def test_historical_unique_request_ids(historical: HistoricalDemandModel) -> None:
    requests = historical.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=5)
    ids = [r.request_id for r in requests]
    assert len(ids) == len(set(ids)), "Request IDs should be unique"


def test_historical_duration_positive(historical: HistoricalDemandModel) -> None:
    requests = historical.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=5)
    for req in requests:
        assert req.estimated_duration_s > 0


def test_historical_available_dates(historical: HistoricalDemandModel) -> None:
    dates = historical.available_dates
    assert len(dates) > 50  # should span Jan + Jun
    # Check January dates present
    jan_dates = [d for d in dates if d.month == 1]
    assert len(jan_dates) >= 28  # at least most of January


# --- SyntheticDemandModel ---


def test_synthetic_from_historical_rates(synthetic: SyntheticDemandModel, city: CityGraph) -> None:
    rates = synthetic.hourly_rates
    assert rates.shape == (city.n_zones, 24)
    # Peak hours should have higher rates than overnight
    avg_8am = rates[:, 8].mean()
    avg_4am = rates[:, 4].mean()
    assert avg_8am > avg_4am, "8 AM should have higher rates than 4 AM"


def test_synthetic_generates_requests(synthetic: SyntheticDemandModel, city: CityGraph) -> None:
    requests = synthetic.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=5)
    assert len(requests) > 0, "Synthetic model should generate some requests"


def test_synthetic_zone_indices_valid(
    synthetic: SyntheticDemandModel, city: CityGraph
) -> None:
    requests = synthetic.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=5)
    for req in requests:
        assert 0 <= req.pickup_zone_idx < city.n_zones
        assert 0 <= req.dropoff_zone_idx < city.n_zones


def test_synthetic_timestamps_in_window(synthetic: SyntheticDemandModel) -> None:
    sim_time = datetime(2023, 1, 3, 12, 0)
    duration = 5
    requests = synthetic.get_requests(sim_time, duration_minutes=duration)
    for req in requests:
        assert sim_time <= req.timestamp < datetime(2023, 1, 3, 12, 5)


def test_synthetic_duration_reasonable(synthetic: SyntheticDemandModel) -> None:
    requests = synthetic.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=5)
    for req in requests:
        assert req.estimated_duration_s >= 60, "Duration should be at least 60s"


def test_synthetic_reproducible(city: CityGraph) -> None:
    m1 = SyntheticDemandModel.from_historical(TRIPS_PATH, city, seed=123)
    m2 = SyntheticDemandModel.from_historical(TRIPS_PATH, city, seed=123)
    r1 = m1.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=1)
    r2 = m2.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=1)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.pickup_zone_idx == b.pickup_zone_idx
        assert a.dropoff_zone_idx == b.dropoff_zone_idx


def test_synthetic_scales_with_duration(synthetic: SyntheticDemandModel) -> None:
    # Longer window should produce more requests on average
    short = []
    long = []
    for _ in range(10):
        short.append(len(synthetic.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=1)))
        long.append(len(synthetic.get_requests(datetime(2023, 1, 3, 12, 0), duration_minutes=10)))
    assert np.mean(long) > np.mean(short), "10-min window should produce more requests than 1-min"
