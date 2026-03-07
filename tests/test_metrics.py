"""Tests for MetricsTracker and EpisodeMetrics."""

from datetime import datetime

import pytest

from simulation.demand import RideRequest
from simulation.metrics import EpisodeMetrics, MetricsTracker


def _make_request(request_id: str = "r1") -> RideRequest:
    return RideRequest(
        request_id=request_id,
        timestamp=datetime(2023, 1, 3, 8, 0),
        pickup_zone_idx=5,
        dropoff_zone_idx=10,
        estimated_duration_s=600,
    )


class TestEpisodeMetrics:
    def test_empty(self) -> None:
        m = EpisodeMetrics.empty()
        assert m.total_trips_served == 0
        assert m.mean_wait_time_minutes == 0.0
        assert m.fleet_utilization == 0.0

    def test_to_dict_from_dict_roundtrip(self) -> None:
        m = EpisodeMetrics(
            mean_wait_time_minutes=3.5,
            median_wait_time_minutes=3.0,
            pct_served_within_5min=85.0,
            pct_served_total=95.0,
            fleet_utilization=0.65,
            deadhead_fraction=0.12,
            total_trips_served=100,
            total_trips_requested=105,
            total_repositioning_moves=20,
        )
        d = m.to_dict()
        m2 = EpisodeMetrics.from_dict(d)
        assert m2.total_trips_served == 100
        assert m2.total_trips_requested == 105
        assert abs(m2.mean_wait_time_minutes - 3.5) < 0.01


class TestMetricsTracker:
    def test_no_data(self) -> None:
        tracker = MetricsTracker()
        m = tracker.compute_episode_metrics()
        assert m.total_trips_requested == 0
        assert m.total_trips_served == 0
        assert m.mean_wait_time_minutes == 0.0
        assert m.pct_served_total == 0.0

    def test_mean_wait_time(self) -> None:
        tracker = MetricsTracker()
        r1 = _make_request("r1")
        r2 = _make_request("r2")
        r3 = _make_request("r3")
        tracker.record_request(r1)
        tracker.record_request(r2)
        tracker.record_request(r3)
        # Wait times: 120s, 180s, 300s -> 2, 3, 5 min -> mean = 10/3 min
        tracker.record_match(r1, 120.0)
        tracker.record_match(r2, 180.0)
        tracker.record_match(r3, 300.0)
        m = tracker.compute_episode_metrics()
        assert abs(m.mean_wait_time_minutes - (10.0 / 3.0)) < 0.01
        assert m.total_trips_served == 3
        assert m.total_trips_requested == 3

    def test_median_wait_time_odd(self) -> None:
        tracker = MetricsTracker()
        for i, wait_s in enumerate([60, 180, 600]):
            r = _make_request(f"r{i}")
            tracker.record_request(r)
            tracker.record_match(r, float(wait_s))
        m = tracker.compute_episode_metrics()
        # Sorted: 1, 3, 10 min -> median = 3
        assert abs(m.median_wait_time_minutes - 3.0) < 0.01

    def test_median_wait_time_even(self) -> None:
        tracker = MetricsTracker()
        for i, wait_s in enumerate([60, 120, 180, 240]):
            r = _make_request(f"r{i}")
            tracker.record_request(r)
            tracker.record_match(r, float(wait_s))
        m = tracker.compute_episode_metrics()
        # Sorted: 1, 2, 3, 4 min -> median = 2.5
        assert abs(m.median_wait_time_minutes - 2.5) < 0.01

    def test_pct_served_within_5min(self) -> None:
        tracker = MetricsTracker()
        for i, wait_s in enumerate([60, 180, 360]):
            r = _make_request(f"r{i}")
            tracker.record_request(r)
            tracker.record_match(r, float(wait_s))
        m = tracker.compute_episode_metrics()
        # 60s=1min, 180s=3min -> within 5min. 360s=6min -> outside.
        # 2/3 = 66.67%
        assert abs(m.pct_served_within_5min - 66.667) < 0.1

    def test_pct_served_total_with_expiry(self) -> None:
        tracker = MetricsTracker()
        r1, r2, r3 = _make_request("r1"), _make_request("r2"), _make_request("r3")
        tracker.record_request(r1)
        tracker.record_request(r2)
        tracker.record_request(r3)
        tracker.record_match(r1, 60.0)
        tracker.record_match(r2, 120.0)
        tracker.record_expiry(r3)
        m = tracker.compute_episode_metrics()
        # 2 served / 3 requested = 66.67%
        assert abs(m.pct_served_total - 66.667) < 0.1

    def test_fleet_utilization(self) -> None:
        tracker = MetricsTracker()
        # 3 steps: utilization = occupied / total
        tracker.step(n_idle=80, n_occupied=100, n_repositioning=20)  # 100/200 = 0.5
        tracker.step(n_idle=60, n_occupied=120, n_repositioning=20)  # 120/200 = 0.6
        tracker.step(n_idle=40, n_occupied=140, n_repositioning=20)  # 140/200 = 0.7
        m = tracker.compute_episode_metrics()
        # Mean: (0.5 + 0.6 + 0.7) / 3 = 0.6
        assert abs(m.fleet_utilization - 0.6) < 0.01

    def test_deadhead_fraction(self) -> None:
        tracker = MetricsTracker()
        tracker.record_trip(8.0)
        tracker.record_reposition(0, 1, 2.0)
        m = tracker.compute_episode_metrics()
        # deadhead = 2 / (8 + 2) = 0.2
        assert abs(m.deadhead_fraction - 0.2) < 0.01
        assert m.total_repositioning_moves == 1

    def test_deadhead_no_miles(self) -> None:
        tracker = MetricsTracker()
        m = tracker.compute_episode_metrics()
        assert m.deadhead_fraction == 0.0

    def test_reset(self) -> None:
        tracker = MetricsTracker()
        r = _make_request("r1")
        tracker.record_request(r)
        tracker.record_match(r, 60.0)
        tracker.step(100, 100, 0)
        tracker.record_trip(5.0)
        tracker.record_reposition(0, 1, 1.0)
        tracker.reset()
        m = tracker.compute_episode_metrics()
        assert m.total_trips_requested == 0
        assert m.total_trips_served == 0
