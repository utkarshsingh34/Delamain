"""Manhattan zone topology and travel metadata.

Provides CityGraph class for zone lookups, distances,
travel times, and bidirectional zone ID mapping.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class CityGraph:
    """Manhattan zone topology: centroids, distances, travel times, and ID mappings.

    NYC taxi zone IDs are non-contiguous integers (e.g., 4, 12, 13, 24, ..., 263).
    Internally the simulation uses 0-based indices. This class provides bidirectional
    mapping between the two representations.

    Args:
        zones_dir: Path to directory containing zone data files
            (zone_centroids.csv, zone_distances.npz, zone_travel_times.npz,
             zone_mapping.json, manhattan_zones.geojson).
    """

    def __init__(self, zones_dir: Path) -> None:
        zones_dir = Path(zones_dir)

        # Load zone mapping (zone_id <-> 0-based index)
        with open(zones_dir / "zone_mapping.json") as f:
            mapping = json.load(f)
        self._id_to_idx: dict[int, int] = {
            int(k): int(v) for k, v in mapping["zone_id_to_index"].items()
        }
        self._idx_to_id: dict[int, int] = {
            int(k): int(v) for k, v in mapping["index_to_zone_id"].items()
        }

        # Load centroids
        centroids_df = pd.read_csv(zones_dir / "zone_centroids.csv")
        self._centroids: dict[int, tuple[float, float]] = {}
        self._names: dict[int, str] = {}
        self._name_to_id: dict[str, int] = {}
        for _, row in centroids_df.iterrows():
            zid = int(row["LocationID"])
            self._centroids[zid] = (float(row["lat"]), float(row["lon"]))
            self._names[zid] = str(row["zone_name"])
            self._name_to_id[str(row["zone_name"]).lower()] = zid

        # Load distance matrix (n_zones x n_zones, float32)
        dist_data = np.load(zones_dir / "zone_distances.npz")
        self._distances: np.ndarray = dist_data["distances"]

        # Load travel time matrix (n_zones x n_zones, float32)
        tt_data = np.load(zones_dir / "zone_travel_times.npz")
        self._travel_times: np.ndarray = tt_data["travel_times"]

        # Sorted list of real zone IDs
        self._zone_ids: list[int] = sorted(self._id_to_idx.keys())

        # Store geojson path for later use by map rendering
        self._geojson_path = zones_dir / "manhattan_zones.geojson"

    @property
    def n_zones(self) -> int:
        """Number of Manhattan taxi zones."""
        return len(self._zone_ids)

    @property
    def zone_ids(self) -> list[int]:
        """Sorted list of real NYC taxi zone IDs."""
        return list(self._zone_ids)

    @property
    def zone_indices(self) -> list[int]:
        """List of 0-based zone indices [0, 1, ..., n_zones-1]."""
        return list(range(self.n_zones))

    @property
    def geojson_path(self) -> Path:
        """Path to manhattan_zones.geojson for map rendering."""
        return self._geojson_path

    def zone_name(self, zone_id: int) -> str:
        """Get human-readable zone name from a real NYC zone ID.

        Args:
            zone_id: Real NYC taxi zone ID (e.g., 230 for Times Square).

        Returns:
            Zone name string (e.g., "Times Sq/Theatre District").

        Raises:
            KeyError: If zone_id is not a valid Manhattan zone.
        """
        return self._names[zone_id]

    def zone_id_to_name(self, zone_id: int) -> str:
        """Alias for zone_name. Get zone name from real NYC zone ID.

        Args:
            zone_id: Real NYC taxi zone ID.

        Returns:
            Zone name string.
        """
        return self.zone_name(zone_id)

    def name_to_zone_id(self, name: str) -> int:
        """Look up zone ID by name (case-insensitive partial match).

        Tries exact match first, then substring match. Returns the first match
        found by alphabetical order of zone names if multiple partial matches exist.

        Args:
            name: Full or partial zone name (e.g., "Times Sq", "midtown").

        Returns:
            Real NYC taxi zone ID.

        Raises:
            KeyError: If no matching zone name is found.
        """
        name_lower = name.lower()

        # Exact match
        if name_lower in self._name_to_id:
            return self._name_to_id[name_lower]

        # Partial match
        matches = [
            (full_name, zid)
            for full_name, zid in self._name_to_id.items()
            if name_lower in full_name
        ]
        if matches:
            # Return first alphabetically for deterministic behavior
            matches.sort(key=lambda x: x[0])
            return matches[0][1]

        raise KeyError(f"No zone matching '{name}' found")

    def centroid(self, zone_id: int) -> tuple[float, float]:
        """Get (lat, lon) centroid for a zone.

        Args:
            zone_id: Real NYC taxi zone ID.

        Returns:
            Tuple of (latitude, longitude) in WGS84.

        Raises:
            KeyError: If zone_id is not a valid Manhattan zone.
        """
        return self._centroids[zone_id]

    def distance_km(self, from_zone: int, to_zone: int) -> float:
        """Haversine distance between two zone centroids in kilometers.

        Args:
            from_zone: Origin zone ID (real NYC zone ID).
            to_zone: Destination zone ID (real NYC zone ID).

        Returns:
            Distance in kilometers.
        """
        i = self._id_to_idx[from_zone]
        j = self._id_to_idx[to_zone]
        return float(self._distances[i, j])

    def travel_time_minutes(self, from_zone: int, to_zone: int) -> float:
        """Estimated travel time between two zones in minutes.

        Uses 24 kph for Manhattan grid streets, 40 kph for highway-adjacent zones.

        Args:
            from_zone: Origin zone ID (real NYC zone ID).
            to_zone: Destination zone ID (real NYC zone ID).

        Returns:
            Travel time in minutes.
        """
        i = self._id_to_idx[from_zone]
        j = self._id_to_idx[to_zone]
        return float(self._travel_times[i, j])

    def neighbors(self, zone_id: int, radius_km: float = 2.0) -> list[int]:
        """Get all zones within a given radius of a zone.

        Args:
            zone_id: Center zone ID (real NYC zone ID).
            radius_km: Search radius in kilometers. Default 2.0 km.

        Returns:
            List of real NYC zone IDs within the radius (excluding the zone itself),
            sorted by distance (nearest first).
        """
        idx = self._id_to_idx[zone_id]
        dists = self._distances[idx]
        neighbor_indices = np.where((dists > 0) & (dists <= radius_km))[0]
        # Sort by distance
        sorted_indices = neighbor_indices[np.argsort(dists[neighbor_indices])]
        return [self._idx_to_id[int(i)] for i in sorted_indices]

    def zone_id_to_index(self, zone_id: int) -> int:
        """Convert a real NYC taxi zone ID to a 0-based index.

        Args:
            zone_id: Real NYC taxi zone ID (e.g., 230).

        Returns:
            0-based index in [0, n_zones).

        Raises:
            KeyError: If zone_id is not a valid Manhattan zone.
        """
        return self._id_to_idx[zone_id]

    def index_to_zone_id(self, index: int) -> int:
        """Convert a 0-based index to a real NYC taxi zone ID.

        Args:
            index: 0-based zone index in [0, n_zones).

        Returns:
            Real NYC taxi zone ID.

        Raises:
            KeyError: If index is out of range.
        """
        return self._idx_to_id[index]

    def __repr__(self) -> str:
        return f"CityGraph(n_zones={self.n_zones})"
