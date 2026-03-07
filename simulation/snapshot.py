"""State serialization for simulation snapshots.

SimSnapshot and VehicleSnapshot dataclasses for
JSON-serializable state transport via Redis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class VehicleSnapshot:
    """Serializable state of a single vehicle.

    Attributes:
        vehicle_id: Unique vehicle identifier.
        zone_idx: Current zone as 0-based index.
        state: Vehicle state string ("idle", "en_route_pickup", "occupied", "repositioning").
        destination_zone_idx: Target zone index if moving, None if idle/occupied in place.
        eta_minutes: Estimated time of arrival in minutes, None if not moving.
    """

    vehicle_id: str
    zone_idx: int
    state: str
    destination_zone_idx: int | None = None
    eta_minutes: float | None = None

    def to_dict(self) -> dict:
        """Convert to a plain dictionary.

        Returns:
            Dictionary with all fields.
        """
        return {
            "vehicle_id": self.vehicle_id,
            "zone_idx": self.zone_idx,
            "state": self.state,
            "destination_zone_idx": self.destination_zone_idx,
            "eta_minutes": self.eta_minutes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VehicleSnapshot:
        """Reconstruct from a dictionary.

        Args:
            d: Dictionary with vehicle fields.

        Returns:
            VehicleSnapshot instance.
        """
        return cls(
            vehicle_id=d["vehicle_id"],
            zone_idx=d["zone_idx"],
            state=d["state"],
            destination_zone_idx=d.get("destination_zone_idx"),
            eta_minutes=d.get("eta_minutes"),
        )


@dataclass(slots=True)
class SimSnapshot:
    """Serializable snapshot of entire simulation state.

    Designed for Redis transport between simulation process, API server,
    and dashboard.

    Attributes:
        timestamp: Wall-clock time when snapshot was taken.
        sim_time: In-simulation datetime.
        vehicles: List of vehicle states.
        pending_requests: Number of unmatched ride requests.
        metrics: Episode metrics as a dictionary.
        vehicle_counts_per_zone: Vehicle count per zone (length n_zones).
        demand_counts_per_zone: Recent demand per zone (length n_zones).
        forecast_counts_per_zone: Predicted next-15min demand per zone (length n_zones).
    """

    timestamp: datetime
    sim_time: datetime
    vehicles: list[VehicleSnapshot]
    pending_requests: int
    metrics: dict
    vehicle_counts_per_zone: list[int]
    demand_counts_per_zone: list[int]
    forecast_counts_per_zone: list[float]

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, datetimes as ISO strings.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "sim_time": self.sim_time.isoformat(),
            "vehicles": [v.to_dict() for v in self.vehicles],
            "pending_requests": self.pending_requests,
            "metrics": self.metrics,
            "vehicle_counts_per_zone": self.vehicle_counts_per_zone,
            "demand_counts_per_zone": self.demand_counts_per_zone,
            "forecast_counts_per_zone": self.forecast_counts_per_zone,
        }

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            JSON string representation of the snapshot.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> SimSnapshot:
        """Reconstruct from a dictionary.

        Args:
            d: Dictionary with snapshot fields.

        Returns:
            SimSnapshot instance.
        """
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            sim_time=datetime.fromisoformat(d["sim_time"]),
            vehicles=[VehicleSnapshot.from_dict(v) for v in d["vehicles"]],
            pending_requests=d["pending_requests"],
            metrics=d["metrics"],
            vehicle_counts_per_zone=d["vehicle_counts_per_zone"],
            demand_counts_per_zone=d["demand_counts_per_zone"],
            forecast_counts_per_zone=d["forecast_counts_per_zone"],
        )

    @classmethod
    def from_json(cls, s: str) -> SimSnapshot:
        """Deserialize from JSON string.

        Args:
            s: JSON string produced by to_json().

        Returns:
            SimSnapshot instance.
        """
        return cls.from_dict(json.loads(s))
