"""Episode metrics tracking for fleet simulation.

Tracks wait times, utilization, deadhead fraction,
and per-step statistics across simulation episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simulation.demand import RideRequest


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Aggregate metrics for a simulation episode.

    Attributes:
        mean_wait_time_minutes: Mean wait time for served requests.
        median_wait_time_minutes: Median wait time for served requests.
        pct_served_within_5min: Percentage of requests matched within 5 minutes.
        pct_served_total: Percentage of all requests that were matched (not expired).
        fleet_utilization: Mean fraction of vehicles in OCCUPIED state.
        deadhead_fraction: Repositioning miles as fraction of total miles driven.
        total_trips_served: Number of requests successfully matched.
        total_trips_requested: Total incoming requests.
        total_repositioning_moves: Number of repositioning commands issued.
    """

    mean_wait_time_minutes: float
    median_wait_time_minutes: float
    pct_served_within_5min: float
    pct_served_total: float
    fleet_utilization: float
    deadhead_fraction: float
    total_trips_served: int
    total_trips_requested: int
    total_repositioning_moves: int

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization.

        Returns:
            Dictionary with all metric fields.
        """
        return {
            "mean_wait_time_minutes": round(self.mean_wait_time_minutes, 3),
            "median_wait_time_minutes": round(self.median_wait_time_minutes, 3),
            "pct_served_within_5min": round(self.pct_served_within_5min, 3),
            "pct_served_total": round(self.pct_served_total, 3),
            "fleet_utilization": round(self.fleet_utilization, 4),
            "deadhead_fraction": round(self.deadhead_fraction, 4),
            "total_trips_served": self.total_trips_served,
            "total_trips_requested": self.total_trips_requested,
            "total_repositioning_moves": self.total_repositioning_moves,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EpisodeMetrics:
        """Reconstruct from a dictionary.

        Args:
            d: Dictionary with metric fields.

        Returns:
            EpisodeMetrics instance.
        """
        return cls(**d)

    @classmethod
    def empty(cls) -> EpisodeMetrics:
        """Return zeroed-out metrics for initialization.

        Returns:
            EpisodeMetrics with all values at zero.
        """
        return cls(
            mean_wait_time_minutes=0.0,
            median_wait_time_minutes=0.0,
            pct_served_within_5min=0.0,
            pct_served_total=0.0,
            fleet_utilization=0.0,
            deadhead_fraction=0.0,
            total_trips_served=0,
            total_trips_requested=0,
            total_repositioning_moves=0,
        )


class MetricsTracker:
    """Accumulates per-step metrics across a simulation episode.

    Call record_* methods during simulation, then compute_episode_metrics()
    at the end to get aggregate results.
    """

    def __init__(self) -> None:
        self._wait_times_s: list[float] = []
        self._total_requested: int = 0
        self._total_expired: int = 0
        self._reposition_miles_km: float = 0.0
        self._trip_miles_km: float = 0.0
        self._total_repositioning_moves: int = 0

        # Per-step fleet state for utilization calculation
        self._step_idle: list[int] = []
        self._step_occupied: list[int] = []
        self._step_repositioning: list[int] = []

    def record_request(self, request: RideRequest) -> None:
        """Track an incoming ride request.

        Args:
            request: The incoming RideRequest.
        """
        self._total_requested += 1

    def record_match(self, request: RideRequest, wait_seconds: float) -> None:
        """Track a successful match between a request and a vehicle.

        Args:
            request: The matched RideRequest.
            wait_seconds: Time in seconds between request and vehicle arrival at pickup.
        """
        self._wait_times_s.append(wait_seconds)

    def record_expiry(self, request: RideRequest) -> None:
        """Track an expired (unmatched) request.

        Args:
            request: The expired RideRequest.
        """
        self._total_expired += 1

    def record_reposition(
        self, from_zone: int, to_zone: int, distance_km: float
    ) -> None:
        """Track a vehicle repositioning move.

        Args:
            from_zone: Origin zone index.
            to_zone: Destination zone index.
            distance_km: Distance of the repositioning move in km.
        """
        self._reposition_miles_km += distance_km
        self._total_repositioning_moves += 1

    def record_trip(self, distance_km: float) -> None:
        """Track miles driven serving a passenger trip.

        Args:
            distance_km: Trip distance in km.
        """
        self._trip_miles_km += distance_km

    def step(self, n_idle: int, n_occupied: int, n_repositioning: int) -> None:
        """Record fleet state at the current simulation step.

        Called once per simulation step to track vehicle utilization over time.

        Args:
            n_idle: Number of idle vehicles.
            n_occupied: Number of vehicles serving passengers.
            n_repositioning: Number of vehicles repositioning.
        """
        self._step_idle.append(n_idle)
        self._step_occupied.append(n_occupied)
        self._step_repositioning.append(n_repositioning)

    def compute_episode_metrics(self) -> EpisodeMetrics:
        """Compute aggregate metrics for the episode.

        Returns:
            EpisodeMetrics with all computed values.
        """
        total_served = len(self._wait_times_s)

        # Wait time stats
        if total_served > 0:
            sorted_waits = sorted(self._wait_times_s)
            wait_minutes = [w / 60.0 for w in sorted_waits]
            mean_wait = sum(wait_minutes) / len(wait_minutes)
            n = len(wait_minutes)
            if n % 2 == 1:
                median_wait = wait_minutes[n // 2]
            else:
                median_wait = (wait_minutes[n // 2 - 1] + wait_minutes[n // 2]) / 2.0
            served_within_5 = sum(1 for w in wait_minutes if w <= 5.0)
            pct_within_5 = served_within_5 / total_served * 100.0
        else:
            mean_wait = 0.0
            median_wait = 0.0
            pct_within_5 = 0.0

        # Service rate
        if self._total_requested > 0:
            pct_served = total_served / self._total_requested * 100.0
        else:
            pct_served = 0.0

        # Fleet utilization (mean fraction occupied across steps)
        if self._step_occupied:
            utilizations = []
            for occ, idle, repo in zip(
                self._step_occupied, self._step_idle, self._step_repositioning
            ):
                total = occ + idle + repo
                if total > 0:
                    utilizations.append(occ / total)
            fleet_util = sum(utilizations) / len(utilizations) if utilizations else 0.0
        else:
            fleet_util = 0.0

        # Deadhead fraction
        total_miles = self._trip_miles_km + self._reposition_miles_km
        if total_miles > 0:
            deadhead = self._reposition_miles_km / total_miles
        else:
            deadhead = 0.0

        return EpisodeMetrics(
            mean_wait_time_minutes=mean_wait,
            median_wait_time_minutes=median_wait,
            pct_served_within_5min=pct_within_5,
            pct_served_total=pct_served,
            fleet_utilization=fleet_util,
            deadhead_fraction=deadhead,
            total_trips_served=total_served,
            total_trips_requested=self._total_requested,
            total_repositioning_moves=self._total_repositioning_moves,
        )

    def reset(self) -> None:
        """Reset all accumulators for a new episode."""
        self._wait_times_s.clear()
        self._total_requested = 0
        self._total_expired = 0
        self._reposition_miles_km = 0.0
        self._trip_miles_km = 0.0
        self._total_repositioning_moves = 0
        self._step_idle.clear()
        self._step_occupied.clear()
        self._step_repositioning.clear()
