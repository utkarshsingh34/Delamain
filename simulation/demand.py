"""Demand models for ride request generation.

HistoricalDemandModel replays actual trips from parquet data.
SyntheticDemandModel generates requests via Poisson process.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from simulation.city import CityGraph


@dataclass(frozen=True, slots=True)
class RideRequest:
    """A single ride request in the simulation.

    Attributes:
        request_id: Unique identifier for this request.
        timestamp: When the request was made.
        pickup_zone_idx: Pickup zone as 0-based index.
        dropoff_zone_idx: Dropoff zone as 0-based index.
        estimated_duration_s: Estimated trip duration in seconds.
    """

    request_id: str
    timestamp: datetime
    pickup_zone_idx: int
    dropoff_zone_idx: int
    estimated_duration_s: int


class DemandModel(Protocol):
    """Protocol for demand models used by the simulation environment."""

    def get_requests(
        self, sim_time: datetime, duration_minutes: int = 1
    ) -> list[RideRequest]:
        """Return ride requests for a time window.

        Args:
            sim_time: Start of the query window.
            duration_minutes: Length of the window in minutes.

        Returns:
            List of RideRequest objects within [sim_time, sim_time + duration).
        """
        ...


class HistoricalDemandModel:
    """Replays actual trips from parquet at simulation speed.

    Loads all processed Manhattan trips, indexes them by pickup datetime,
    and returns matching trips for any queried time window.

    Args:
        trips_path: Path to trips_manhattan.parquet.
        city: CityGraph for zone ID to index conversion.
    """

    def __init__(self, trips_path: Path, city: CityGraph) -> None:
        self._city = city

        df = pl.read_parquet(trips_path)

        # Convert zone IDs to 0-based indices, dropping trips with unmapped zones
        valid_ids = set(city.zone_ids)
        df = df.filter(
            pl.col("PULocationID").is_in(valid_ids)
            & pl.col("DOLocationID").is_in(valid_ids)
        )

        # Sort by pickup time for binary search
        df = df.sort("tpep_pickup_datetime")

        # Extract arrays for fast access
        self._timestamps = df["tpep_pickup_datetime"].to_numpy()
        self._pu_zones = df["PULocationID"].to_numpy()
        self._do_zones = df["DOLocationID"].to_numpy()
        self._durations = df["trip_duration_seconds"].to_numpy()

        # Build zone ID -> index lookup array
        self._id_to_idx = np.vectorize(city.zone_id_to_index)

        # Pre-compute all zone indices
        self._pu_idx = np.array(
            [city.zone_id_to_index(int(z)) for z in self._pu_zones]
        )
        self._do_idx = np.array(
            [city.zone_id_to_index(int(z)) for z in self._do_zones]
        )

        # Available dates for random episode selection
        dates = df["tpep_pickup_datetime"].dt.date().unique().sort()
        self._available_dates: list[datetime] = [
            datetime(d.year, d.month, d.day) for d in dates.to_list()
        ]

    @property
    def available_dates(self) -> list[datetime]:
        """List of dates that have trip data available."""
        return list(self._available_dates)

    def get_requests(
        self, sim_time: datetime, duration_minutes: int = 1
    ) -> list[RideRequest]:
        """Return all historical trips with pickup in [sim_time, sim_time + duration).

        Args:
            sim_time: Start of the query window (naive datetime).
            duration_minutes: Length of the window in minutes.

        Returns:
            List of RideRequest objects matching the time window.
        """
        t_start = np.datetime64(sim_time, "ns")
        t_end = np.datetime64(sim_time + timedelta(minutes=duration_minutes), "ns")

        # Binary search for start and end indices
        i_start = int(np.searchsorted(self._timestamps, t_start, side="left"))
        i_end = int(np.searchsorted(self._timestamps, t_end, side="left"))

        if i_start >= i_end:
            return []

        requests: list[RideRequest] = []
        for i in range(i_start, i_end):
            ts = self._timestamps[i]
            req = RideRequest(
                request_id=uuid.uuid4().hex[:12],
                timestamp=ts.astype("datetime64[us]").astype(datetime),
                pickup_zone_idx=int(self._pu_idx[i]),
                dropoff_zone_idx=int(self._do_idx[i]),
                estimated_duration_s=int(self._durations[i]),
            )
            requests.append(req)

        return requests


class SyntheticDemandModel:
    """Poisson process per zone for training variation.

    Generates synthetic ride requests by sampling from learned per-zone,
    per-hour Poisson rates, with dropoff zones sampled from historical
    distributions.

    Args:
        hourly_rates: Array of shape (n_zones, 24) with average trips per hour.
        dropoff_probs: Array of shape (n_zones, n_zones) with dropoff probability
            distributions. dropoff_probs[i, j] = P(dropoff=j | pickup=i).
        mean_duration_s: Array of shape (n_zones, n_zones) with mean trip duration
            in seconds for each (pickup, dropoff) pair.
        city: CityGraph for zone metadata.
        rng: Optional numpy random generator for reproducibility.
    """

    def __init__(
        self,
        hourly_rates: np.ndarray,
        dropoff_probs: np.ndarray,
        mean_duration_s: np.ndarray,
        city: CityGraph,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._rates = hourly_rates  # (n_zones, 24)
        self._dropoff_probs = dropoff_probs  # (n_zones, n_zones)
        self._mean_duration = mean_duration_s  # (n_zones, n_zones)
        self._city = city
        self._rng = rng or np.random.default_rng()

    @classmethod
    def from_historical(
        cls,
        trips_path: Path,
        city: CityGraph,
        seed: int | None = None,
    ) -> SyntheticDemandModel:
        """Learn Poisson rates and dropoff distributions from historical data.

        Computes per-zone, per-hour average trip rates and per-zone dropoff
        probability distributions from the processed trip data.

        Args:
            trips_path: Path to trips_manhattan.parquet.
            city: CityGraph for zone ID to index conversion.
            seed: Random seed for reproducibility.

        Returns:
            Configured SyntheticDemandModel instance.
        """
        df = pl.read_parquet(trips_path)
        n = city.n_zones
        valid_ids = set(city.zone_ids)

        df = df.filter(
            pl.col("PULocationID").is_in(valid_ids)
            & pl.col("DOLocationID").is_in(valid_ids)
        )

        n_days = df["tpep_pickup_datetime"].dt.date().n_unique()

        # Compute hourly rates: trips per zone per hour, averaged across days
        hourly_rates = np.zeros((n, 24), dtype=np.float32)
        rate_agg = (
            df.group_by("PULocationID", "pickup_hour")
            .agg(pl.len().alias("count"))
        )
        for row in rate_agg.iter_rows(named=True):
            idx = city.zone_id_to_index(row["PULocationID"])
            hour = row["pickup_hour"]
            hourly_rates[idx, hour] = row["count"] / n_days

        # Compute dropoff distributions per pickup zone
        dropoff_probs = np.zeros((n, n), dtype=np.float32)
        do_agg = (
            df.group_by("PULocationID", "DOLocationID")
            .agg(pl.len().alias("count"))
        )
        for row in do_agg.iter_rows(named=True):
            pu_idx = city.zone_id_to_index(row["PULocationID"])
            do_idx = city.zone_id_to_index(row["DOLocationID"])
            dropoff_probs[pu_idx, do_idx] = row["count"]

        # Normalize rows to probabilities (handle zones with zero pickups)
        row_sums = dropoff_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        dropoff_probs /= row_sums

        # Compute mean duration per OD pair
        mean_duration = np.full((n, n), 600.0, dtype=np.float32)  # default 10 min
        dur_agg = (
            df.group_by("PULocationID", "DOLocationID")
            .agg(pl.col("trip_duration_seconds").mean().alias("mean_dur"))
        )
        for row in dur_agg.iter_rows(named=True):
            pu_idx = city.zone_id_to_index(row["PULocationID"])
            do_idx = city.zone_id_to_index(row["DOLocationID"])
            mean_duration[pu_idx, do_idx] = row["mean_dur"]

        rng = np.random.default_rng(seed)
        return cls(hourly_rates, dropoff_probs, mean_duration, city, rng)

    @property
    def hourly_rates(self) -> np.ndarray:
        """Per-zone, per-hour average trip rates. Shape (n_zones, 24)."""
        return self._rates

    def get_requests(
        self, sim_time: datetime, duration_minutes: int = 1
    ) -> list[RideRequest]:
        """Generate synthetic requests from Poisson process.

        For each zone, samples the number of requests from
        Poisson(rate[zone][hour] * duration_minutes / 60), then assigns
        random dropoff zones weighted by historical distribution.

        Args:
            sim_time: Current simulation time.
            duration_minutes: Window length in minutes.

        Returns:
            List of generated RideRequest objects.
        """
        hour = sim_time.hour
        n = self._city.n_zones
        scale = duration_minutes / 60.0

        # Sample request counts per zone from Poisson
        rates = self._rates[:, hour] * scale
        counts = self._rng.poisson(rates)

        requests: list[RideRequest] = []
        for pu_idx in range(n):
            n_reqs = int(counts[pu_idx])
            if n_reqs == 0:
                continue

            # Sample dropoff zones from learned distribution
            probs = self._dropoff_probs[pu_idx]
            if probs.sum() < 1e-8:
                # No historical dropoff data — uniform
                do_indices = self._rng.integers(0, n, size=n_reqs)
            else:
                do_indices = self._rng.choice(n, size=n_reqs, p=probs)

            for j in range(n_reqs):
                do_idx = int(do_indices[j])
                # Random timestamp within the window
                offset_s = self._rng.uniform(0, duration_minutes * 60)
                ts = sim_time + timedelta(seconds=offset_s)

                # Duration from learned mean with some noise
                mean_dur = self._mean_duration[pu_idx, do_idx]
                dur = max(60, int(self._rng.normal(mean_dur, mean_dur * 0.2)))

                requests.append(
                    RideRequest(
                        request_id=uuid.uuid4().hex[:12],
                        timestamp=ts,
                        pickup_zone_idx=pu_idx,
                        dropoff_zone_idx=do_idx,
                        estimated_duration_s=dur,
                    )
                )

        return requests
