"""Tests for SimSnapshot and VehicleSnapshot serialization."""

from datetime import datetime

import pytest

from simulation.snapshot import SimSnapshot, VehicleSnapshot


class TestVehicleSnapshot:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        v = VehicleSnapshot(
            vehicle_id="v-001",
            zone_idx=5,
            state="idle",
            destination_zone_idx=None,
            eta_minutes=None,
        )
        d = v.to_dict()
        v2 = VehicleSnapshot.from_dict(d)
        assert v2.vehicle_id == "v-001"
        assert v2.zone_idx == 5
        assert v2.state == "idle"
        assert v2.destination_zone_idx is None
        assert v2.eta_minutes is None

    def test_moving_vehicle(self) -> None:
        v = VehicleSnapshot(
            vehicle_id="v-042",
            zone_idx=10,
            state="repositioning",
            destination_zone_idx=20,
            eta_minutes=3.5,
        )
        d = v.to_dict()
        v2 = VehicleSnapshot.from_dict(d)
        assert v2.destination_zone_idx == 20
        assert v2.eta_minutes == 3.5
        assert v2.state == "repositioning"


class TestSimSnapshot:
    def _make_snapshot(self) -> SimSnapshot:
        vehicles = [
            VehicleSnapshot("v-001", 5, "idle"),
            VehicleSnapshot("v-002", 10, "occupied", destination_zone_idx=15, eta_minutes=4.2),
            VehicleSnapshot("v-003", 3, "repositioning", destination_zone_idx=8, eta_minutes=2.0),
        ]
        return SimSnapshot(
            timestamp=datetime(2023, 1, 3, 8, 15, 30),
            sim_time=datetime(2023, 1, 3, 8, 15),
            vehicles=vehicles,
            pending_requests=12,
            metrics={"mean_wait_time_minutes": 3.5, "fleet_utilization": 0.65},
            vehicle_counts_per_zone=[2, 0, 1] + [0] * 64,
            demand_counts_per_zone=[5, 3, 8] + [0] * 64,
            forecast_counts_per_zone=[6.0, 4.0, 9.0] + [0.0] * 64,
        )

    def test_to_json_from_json_roundtrip(self) -> None:
        snap = self._make_snapshot()
        json_str = snap.to_json()
        snap2 = SimSnapshot.from_json(json_str)

        assert snap2.timestamp == snap.timestamp
        assert snap2.sim_time == snap.sim_time
        assert snap2.pending_requests == 12
        assert len(snap2.vehicles) == 3
        assert snap2.metrics["mean_wait_time_minutes"] == 3.5
        assert snap2.vehicle_counts_per_zone == snap.vehicle_counts_per_zone
        assert snap2.demand_counts_per_zone == snap.demand_counts_per_zone
        assert snap2.forecast_counts_per_zone == snap.forecast_counts_per_zone

    def test_vehicle_state_preserved(self) -> None:
        snap = self._make_snapshot()
        snap2 = SimSnapshot.from_json(snap.to_json())
        v_occupied = [v for v in snap2.vehicles if v.state == "occupied"][0]
        assert v_occupied.vehicle_id == "v-002"
        assert v_occupied.zone_idx == 10
        assert v_occupied.destination_zone_idx == 15
        assert v_occupied.eta_minutes == 4.2

    def test_to_dict_from_dict(self) -> None:
        snap = self._make_snapshot()
        d = snap.to_dict()
        snap2 = SimSnapshot.from_dict(d)
        assert snap2.pending_requests == snap.pending_requests
        assert len(snap2.vehicles) == len(snap.vehicles)

    def test_json_is_valid_string(self) -> None:
        import json
        snap = self._make_snapshot()
        json_str = snap.to_json()
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "vehicles" in parsed
        assert "sim_time" in parsed

    def test_empty_vehicles(self) -> None:
        snap = SimSnapshot(
            timestamp=datetime(2023, 1, 1, 0, 0),
            sim_time=datetime(2023, 1, 1, 0, 0),
            vehicles=[],
            pending_requests=0,
            metrics={},
            vehicle_counts_per_zone=[],
            demand_counts_per_zone=[],
            forecast_counts_per_zone=[],
        )
        snap2 = SimSnapshot.from_json(snap.to_json())
        assert snap2.vehicles == []
        assert snap2.pending_requests == 0
