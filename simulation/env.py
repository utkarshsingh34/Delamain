"""Gymnasium environment for fleet repositioning.

FleetRepositioningEnv simulates a fleet of autonomous vehicles across
Manhattan taxi zones, where an RL agent decides how to reposition idle
vehicles to meet anticipated demand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import gymnasium
import numpy as np

from simulation.city import CityGraph
from simulation.demand import DemandModel, RideRequest
from simulation.metrics import MetricsTracker
from simulation.snapshot import SimSnapshot, VehicleSnapshot


@dataclass
class Vehicle:
    """Internal vehicle state for the simulation.

    Attributes:
        vehicle_id: Unique identifier.
        zone_idx: Current zone (0-based index).
        state: One of "idle", "en_route_pickup", "occupied", "repositioning".
        destination_zone_idx: Target zone if moving, None otherwise.
        eta_steps: Steps remaining until arrival, 0 if stationary.
        assigned_request: Request being served, None if not dispatched.
    """

    vehicle_id: str
    zone_idx: int
    state: str = "idle"
    destination_zone_idx: int | None = None
    eta_steps: int = 0
    assigned_request: RideRequest | None = None


class FleetRepositioningEnv(gymnasium.Env):
    """Gymnasium environment for fleet repositioning across Manhattan zones.

    The agent observes per-zone vehicle counts, demand forecasts, and
    supply-demand gaps, then outputs a target vehicle distribution.
    Vehicles reposition accordingly, new ride requests arrive, and
    idle vehicles are dispatched to the nearest pending request.

    Args:
        city: CityGraph with zone topology and travel times.
        demand_model: Model generating ride requests each step.
        n_vehicles: Total fleet size.
        step_minutes: Simulation minutes per environment step.
        episode_hours: Length of one episode in hours (e.g. 18 for 6AM-midnight).
        max_wait_minutes: Requests expire after this many minutes unmatched.
        start_hour: Hour of day when episodes begin.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        city: CityGraph,
        demand_model: DemandModel,
        n_vehicles: int = 200,
        step_minutes: int = 5,
        episode_hours: int = 18,
        max_wait_minutes: float = 10.0,
        start_hour: int = 6,
    ) -> None:
        super().__init__()

        self._city = city
        self._demand_model = demand_model
        self._n_vehicles = n_vehicles
        self._step_minutes = step_minutes
        self._episode_hours = episode_hours
        self._max_wait_minutes = max_wait_minutes
        self._start_hour = start_hour

        n = city.n_zones
        self._n_zones = n

        # Observation: vehicle_counts(n) + demand_forecast(n) + gap(n) + time(4)
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(n * 3 + 4,), dtype=np.float32
        )

        # Action: target distribution weights per zone
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(n,), dtype=np.float32
        )

        # Pre-compute travel time matrix in steps (rounded up)
        self._travel_steps = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    zid_from = city.index_to_zone_id(i)
                    zid_to = city.index_to_zone_id(j)
                    tt_min = city.travel_time_minutes(zid_from, zid_to)
                    self._travel_steps[i, j] = max(1, int(math.ceil(tt_min / step_minutes)))

        # State initialized in reset()
        self._vehicles: list[Vehicle] = []
        self._pending_requests: list[tuple[RideRequest, float]] = []  # (request, arrival_step)
        self._sim_time = datetime(2023, 1, 3, 6, 0)
        self._step_count = 0
        self._max_steps = int(episode_hours * 60 / step_minutes)
        self._metrics = MetricsTracker()

        # Rolling demand history for forecast approximation (last 3 steps per zone)
        self._demand_history: list[np.ndarray] = []

        # Reward components for info dict
        self._reward_alpha = 2.0
        self._reward_beta = 0.5

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to the start of a new episode.

        Picks a random date from the demand model (if historical) and
        distributes vehicles proportional to historical demand.

        Args:
            seed: Random seed.
            options: Optional config. Supports "sim_date" to force a date.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        self._step_count = 0
        self._pending_requests = []
        self._metrics = MetricsTracker()
        self._demand_history = []

        # Pick episode date
        options = options or {}
        if "sim_date" in options:
            sim_date = options["sim_date"]
        elif hasattr(self._demand_model, "available_dates"):
            dates = self._demand_model.available_dates  # type: ignore[attr-defined]
            idx = self.np_random.integers(0, len(dates))
            sim_date = dates[idx]
        else:
            sim_date = datetime(2023, 1, 3)

        if isinstance(sim_date, datetime):
            self._sim_time = sim_date.replace(
                hour=self._start_hour, minute=0, second=0, microsecond=0
            )
        else:
            self._sim_time = datetime(
                sim_date.year, sim_date.month, sim_date.day,
                self._start_hour, 0, 0,
            )

        # Distribute vehicles proportional to demand
        demand_weights = self._compute_initial_distribution()
        self._vehicles = []
        for i in range(self._n_vehicles):
            zone_idx = int(self.np_random.choice(self._n_zones, p=demand_weights))
            self._vehicles.append(
                Vehicle(vehicle_id=f"v-{i:04d}", zone_idx=zone_idx)
            )

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simulation step.

        Order of operations:
        1. Decode action -> repositioning moves (only idle vehicles)
        2. Execute repositioning commands
        3. Advance all moving vehicles (decrement eta, transition on arrival)
        4. Generate new demand
        5. Match idle vehicles to pending requests (nearest, FIFO)
        6. Expire old requests
        7. Compute reward and observation
        8. Check termination

        Args:
            action: Array of shape (n_zones,) with values in [-1, 1].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        # 1. Decode action to repositioning moves
        moves = self._decode_action(action)

        # 2. Execute repositioning
        self._execute_repositioning(moves)

        # 3. Advance moving vehicles
        self._advance_vehicles()

        # 4. Generate new demand
        new_requests = self._demand_model.get_requests(
            self._sim_time, duration_minutes=self._step_minutes
        )
        for req in new_requests:
            self._metrics.record_request(req)
            self._pending_requests.append((req, self._step_count))

        # Track demand per zone for this step
        step_demand = np.zeros(self._n_zones, dtype=np.float32)
        for req in new_requests:
            step_demand[req.pickup_zone_idx] += 1
        self._demand_history.append(step_demand)
        if len(self._demand_history) > 3:
            self._demand_history.pop(0)

        # 5. Match idle vehicles to pending requests
        self._match_requests()

        # 6. Expire old requests
        self._expire_requests()

        # 7. Record fleet state for metrics
        n_idle, n_occupied, n_repo = self._count_by_state()
        self._metrics.step(n_idle, n_occupied, n_repo)

        # Advance sim time
        self._sim_time += timedelta(minutes=self._step_minutes)
        self._step_count += 1

        # 8. Check termination
        terminated = self._step_count >= self._max_steps
        truncated = False

        reward = self._compute_reward()
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_action(self, action: np.ndarray) -> dict[int, int]:
        """Convert raw action to per-zone vehicle delta.

        Applies softmax to get target distribution weights, multiplies by
        idle vehicle count, then computes delta from current idle distribution.

        Args:
            action: Raw action array in [-1, 1] per zone.

        Returns:
            Dict mapping zone_idx -> net vehicle change (positive = add).
        """
        # Softmax for numerical stability
        action = np.clip(action, -10, 10)
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / exp_a.sum()

        idle_per_zone = self._idle_vehicles_per_zone()
        total_idle = idle_per_zone.sum()

        if total_idle == 0:
            return {}

        target = (weights * total_idle).astype(np.int32)

        # Ensure target sums to exactly total_idle by distributing remainder
        remainder = int(total_idle) - int(target.sum())
        if remainder > 0:
            top_zones = np.argsort(-weights)[:remainder]
            for z in top_zones:
                target[z] += 1
        elif remainder < 0:
            bot_zones = np.argsort(weights)[:abs(remainder)]
            for z in bot_zones:
                if target[z] > 0:
                    target[z] -= 1

        delta = target - idle_per_zone
        return {int(z): int(delta[z]) for z in range(self._n_zones) if delta[z] != 0}

    # ------------------------------------------------------------------
    # Repositioning execution
    # ------------------------------------------------------------------

    def _execute_repositioning(self, moves: dict[int, int]) -> None:
        """Execute repositioning commands.

        For zones with positive delta (need more vehicles), find the nearest
        zones with negative delta (excess vehicles) and move idle vehicles.

        Args:
            moves: Dict of zone_idx -> net vehicle change.
        """
        # Separate into zones needing vehicles and zones with excess
        need_zones = [(z, n) for z, n in moves.items() if n > 0]
        excess_zones = [(z, -n) for z, n in moves.items() if n < 0]

        if not need_zones or not excess_zones:
            return

        # Sort need zones by magnitude (fill biggest gaps first)
        need_zones.sort(key=lambda x: -x[1])

        # Build list of available idle vehicles in excess zones
        available: list[tuple[int, Vehicle]] = []  # (zone_idx, vehicle)
        for zone_idx, count in excess_zones:
            idle_in_zone = [
                v for v in self._vehicles
                if v.zone_idx == zone_idx and v.state == "idle"
            ]
            for v in idle_in_zone[:count]:
                available.append((zone_idx, v))

        # Assign vehicles to need zones (nearest first)
        for target_zone, n_needed in need_zones:
            if not available:
                break
            # Sort available by travel time to target
            available.sort(
                key=lambda x: self._travel_steps[x[0], target_zone]
            )
            assigned = 0
            remaining: list[tuple[int, Vehicle]] = []
            for from_zone, v in available:
                if assigned >= n_needed:
                    remaining.append((from_zone, v))
                    continue
                # Move this vehicle
                eta = int(self._travel_steps[from_zone, target_zone])
                v.state = "repositioning"
                v.destination_zone_idx = target_zone
                v.eta_steps = eta
                dist = self._city.distance_km(
                    self._city.index_to_zone_id(from_zone),
                    self._city.index_to_zone_id(target_zone),
                )
                self._metrics.record_reposition(from_zone, target_zone, dist)
                assigned += 1
            available = remaining

    # ------------------------------------------------------------------
    # Vehicle state advancement
    # ------------------------------------------------------------------

    def _advance_vehicles(self) -> None:
        """Advance all moving vehicles by one step.

        Decrements eta_steps for vehicles in transit. When eta reaches 0,
        vehicles arrive at their destination and transition state.
        """
        for v in self._vehicles:
            if v.state in ("repositioning", "en_route_pickup") and v.eta_steps > 0:
                v.eta_steps -= 1
                if v.eta_steps == 0:
                    if v.state == "repositioning":
                        # Arrived at reposition target
                        v.zone_idx = v.destination_zone_idx  # type: ignore[assignment]
                        v.destination_zone_idx = None
                        v.state = "idle"
                    elif v.state == "en_route_pickup":
                        # Arrived at pickup -> become occupied
                        v.zone_idx = v.destination_zone_idx  # type: ignore[assignment]
                        req = v.assigned_request
                        if req is not None:
                            # Set up trip to dropoff
                            v.state = "occupied"
                            v.destination_zone_idx = req.dropoff_zone_idx
                            trip_steps = max(
                                1,
                                int(math.ceil(req.estimated_duration_s / 60 / self._step_minutes))
                            )
                            v.eta_steps = trip_steps
                            trip_dist = self._city.distance_km(
                                self._city.index_to_zone_id(req.pickup_zone_idx),
                                self._city.index_to_zone_id(req.dropoff_zone_idx),
                            )
                            self._metrics.record_trip(trip_dist)
                        else:
                            v.state = "idle"
                            v.destination_zone_idx = None

            elif v.state == "occupied" and v.eta_steps > 0:
                v.eta_steps -= 1
                if v.eta_steps == 0:
                    # Trip complete, drop off passenger
                    v.zone_idx = v.destination_zone_idx  # type: ignore[assignment]
                    v.destination_zone_idx = None
                    v.assigned_request = None
                    v.state = "idle"

    # ------------------------------------------------------------------
    # Request matching
    # ------------------------------------------------------------------

    def _match_requests(self) -> None:
        """Match pending requests to nearest idle vehicles (FIFO order).

        Iterates through pending requests in arrival order and assigns
        the nearest idle vehicle to each.
        """
        # Sort pending by arrival step (FIFO)
        self._pending_requests.sort(key=lambda x: x[1])

        matched_indices: list[int] = []
        assigned_vehicle_ids: set[str] = set()

        for i, (req, arrival_step) in enumerate(self._pending_requests):
            # Find nearest idle vehicle to pickup zone
            best_vehicle: Vehicle | None = None
            best_travel = float("inf")

            for v in self._vehicles:
                if v.state != "idle" or v.vehicle_id in assigned_vehicle_ids:
                    continue
                travel = self._travel_steps[v.zone_idx, req.pickup_zone_idx]
                if travel < best_travel:
                    best_travel = travel
                    best_vehicle = v

            if best_vehicle is not None:
                # Dispatch vehicle
                wait_steps = self._step_count - arrival_step + best_travel
                wait_seconds = wait_steps * self._step_minutes * 60.0
                self._metrics.record_match(req, wait_seconds)

                best_vehicle.state = "en_route_pickup"
                best_vehicle.destination_zone_idx = req.pickup_zone_idx
                best_vehicle.eta_steps = int(self._travel_steps[
                    best_vehicle.zone_idx, req.pickup_zone_idx
                ])
                best_vehicle.assigned_request = req
                assigned_vehicle_ids.add(best_vehicle.vehicle_id)
                matched_indices.append(i)

        # Remove matched requests (iterate in reverse to preserve indices)
        for i in sorted(matched_indices, reverse=True):
            self._pending_requests.pop(i)

    # ------------------------------------------------------------------
    # Request expiry
    # ------------------------------------------------------------------

    def _expire_requests(self) -> None:
        """Remove pending requests older than max_wait_minutes."""
        max_wait_steps = self._max_wait_minutes / self._step_minutes
        expired_indices: list[int] = []

        for i, (req, arrival_step) in enumerate(self._pending_requests):
            if self._step_count - arrival_step >= max_wait_steps:
                self._metrics.record_expiry(req)
                expired_indices.append(i)

        for i in sorted(expired_indices, reverse=True):
            self._pending_requests.pop(i)

    # ------------------------------------------------------------------
    # Observation and reward
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation vector.

        Components:
        - vehicle_counts_per_zone / n_vehicles (n_zones)
        - demand_forecast_per_zone (rolling avg, normalized) (n_zones)
        - supply_demand_gap (normalized) (n_zones)
        - time features: sin(hour), cos(hour), sin(dow), cos(dow) (4)

        Returns:
            Float32 array of shape (n_zones * 3 + 4,).
        """
        n = self._n_zones

        # Vehicle counts per zone (normalized)
        v_counts = np.zeros(n, dtype=np.float32)
        for v in self._vehicles:
            v_counts[v.zone_idx] += 1
        v_norm = v_counts / max(self._n_vehicles, 1)

        # Demand forecast (rolling mean of last 3 steps)
        if self._demand_history:
            demand_forecast = np.mean(self._demand_history, axis=0).astype(np.float32)
        else:
            demand_forecast = np.zeros(n, dtype=np.float32)
        d_max = demand_forecast.max()
        d_norm = demand_forecast / max(d_max, 1.0)

        # Supply-demand gap (positive = oversupplied)
        gap = v_norm - d_norm
        # Normalize to [0, 1]
        gap_norm = (gap + 1.0) / 2.0

        # Time features
        hour = self._sim_time.hour + self._sim_time.minute / 60.0
        dow = self._sim_time.weekday()
        time_features = np.array([
            (math.sin(2 * math.pi * hour / 24) + 1) / 2,
            (math.cos(2 * math.pi * hour / 24) + 1) / 2,
            (math.sin(2 * math.pi * dow / 7) + 1) / 2,
            (math.cos(2 * math.pi * dow / 7) + 1) / 2,
        ], dtype=np.float32)

        obs = np.concatenate([v_norm, d_norm, gap_norm, time_features])
        return np.clip(obs, 0.0, 1.0)

    def _compute_reward(self) -> float:
        """Compute step reward.

        reward = -(mean_wait / max_wait) + alpha * utilization - beta * deadhead
        Clipped to [-1, 1].

        Returns:
            Scalar reward.
        """
        metrics = self._metrics.compute_episode_metrics()

        # Wait time component (lower is better)
        if metrics.mean_wait_time_minutes > 0:
            wait_penalty = metrics.mean_wait_time_minutes / self._max_wait_minutes
        else:
            wait_penalty = 0.0

        reward = (
            -wait_penalty
            + self._reward_alpha * metrics.fleet_utilization
            - self._reward_beta * metrics.deadhead_fraction
        )

        return float(np.clip(reward, -1.0, 1.0))

    def _get_info(self) -> dict[str, Any]:
        """Build info dict with episode metrics and state summary.

        Returns:
            Dict with metrics and fleet state counts.
        """
        metrics = self._metrics.compute_episode_metrics()
        n_idle, n_occupied, n_repo = self._count_by_state()
        return {
            "sim_time": self._sim_time.isoformat(),
            "step": self._step_count,
            "n_idle": n_idle,
            "n_occupied": n_occupied,
            "n_repositioning": n_repo,
            "n_en_route_pickup": sum(1 for v in self._vehicles if v.state == "en_route_pickup"),
            "pending_requests": len(self._pending_requests),
            "mean_wait_time_minutes": metrics.mean_wait_time_minutes,
            "fleet_utilization": metrics.fleet_utilization,
            "deadhead_fraction": metrics.deadhead_fraction,
            "total_trips_served": metrics.total_trips_served,
            "total_trips_requested": metrics.total_trips_requested,
            "pct_served_total": metrics.pct_served_total,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _idle_vehicles_per_zone(self) -> np.ndarray:
        """Count idle vehicles per zone.

        Returns:
            Int32 array of shape (n_zones,).
        """
        counts = np.zeros(self._n_zones, dtype=np.int32)
        for v in self._vehicles:
            if v.state == "idle":
                counts[v.zone_idx] += 1
        return counts

    def _count_by_state(self) -> tuple[int, int, int]:
        """Count vehicles by state category.

        Returns:
            Tuple of (n_idle, n_occupied, n_repositioning).
            en_route_pickup is counted with repositioning for utilization purposes.
        """
        n_idle = 0
        n_occupied = 0
        n_repo = 0
        for v in self._vehicles:
            if v.state == "idle":
                n_idle += 1
            elif v.state == "occupied":
                n_occupied += 1
            else:
                # en_route_pickup and repositioning
                n_repo += 1
        return n_idle, n_occupied, n_repo

    def _compute_initial_distribution(self) -> np.ndarray:
        """Compute initial vehicle distribution proportional to demand.

        Uses demand at the episode start hour to weight zones.
        Falls back to uniform if no demand data available.

        Returns:
            Probability array of shape (n_zones,) summing to 1.
        """
        # Get a sample of demand at start time
        requests = self._demand_model.get_requests(
            self._sim_time, duration_minutes=60
        )

        weights = np.ones(self._n_zones, dtype=np.float32)
        for req in requests:
            weights[req.pickup_zone_idx] += 1

        weights /= weights.sum()
        return weights

    def get_snapshot(self) -> SimSnapshot:
        """Create a SimSnapshot of the current state for API/Redis transport.

        Returns:
            SimSnapshot with current vehicle states, metrics, and zone data.
        """
        vehicles = [
            VehicleSnapshot(
                vehicle_id=v.vehicle_id,
                zone_idx=int(v.zone_idx),
                state=v.state,
                destination_zone_idx=(
                    int(v.destination_zone_idx)
                    if v.destination_zone_idx is not None else None
                ),
                eta_minutes=(
                    v.eta_steps * self._step_minutes if v.eta_steps > 0 else None
                ),
            )
            for v in self._vehicles
        ]

        # Vehicle counts per zone
        v_counts = [0] * self._n_zones
        for v in self._vehicles:
            v_counts[v.zone_idx] += 1

        # Demand counts (last step)
        if self._demand_history:
            demand = self._demand_history[-1].tolist()
        else:
            demand = [0] * self._n_zones

        # Forecast (rolling avg)
        if self._demand_history:
            forecast = np.mean(self._demand_history, axis=0).tolist()
        else:
            forecast = [0.0] * self._n_zones

        metrics = self._metrics.compute_episode_metrics()

        return SimSnapshot(
            timestamp=datetime.now(),
            sim_time=self._sim_time,
            vehicles=vehicles,
            pending_requests=len(self._pending_requests),
            metrics=metrics.to_dict(),
            vehicle_counts_per_zone=v_counts,
            demand_counts_per_zone=[int(d) for d in demand],
            forecast_counts_per_zone=[float(f) for f in forecast],
        )
