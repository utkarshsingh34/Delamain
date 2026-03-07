"""Clean and filter NYC TLC trip data to Manhattan zones.

Reads raw parquet files, filters to Manhattan, computes derived columns,
removes outliers, and outputs processed parquet files. Also processes
taxi zone shapefiles into GeoJSON, centroids, distance/travel-time matrices.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ZONES_DIR = DATA_DIR / "zones"

# Highway-adjacent zones with faster average speeds (FDR, West Side Hwy, etc.)
HIGHWAY_ZONE_IDS = {
    202,  # Roosevelt Island
    261,  # Yorkville East (FDR-adjacent)
    262,  # Yorkville West
    120,  # Highbridge Park
    127,  # Inwood
    128,  # Inwood Hill Park
    153,  # Marble Hill
    194,  # Randalls Island
    103,  # Governor's Island/Ellis Island/Liberty Island
}

MANHATTAN_SPEED_KPH = 24.0   # ~15 mph city grid
HIGHWAY_SPEED_KPH = 40.0     # ~25 mph FDR/highway zones


def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in kilometers."""
    R = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def process_zones() -> None:
    """Process taxi zone shapefile into GeoJSON, centroids, distances, and travel times."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ZONES_DIR.mkdir(parents=True, exist_ok=True)

    # Load shapefile (EPSG:2263 — NAD83 / NY Long Island feet)
    print("Loading taxi zone shapefile...")
    gdf = gpd.read_file(RAW_DIR / "taxi_zones")
    print(f"  Raw zones: {len(gdf)} features, CRS: {gdf.crs}")

    # Filter to Manhattan
    manhattan = gdf[gdf["borough"] == "Manhattan"].copy()
    print(f"  Manhattan zones before dissolve: {len(manhattan)} features")

    # Dissolve multi-polygon zones (LocationID 103 has 3 separate polygons)
    manhattan = manhattan.dissolve(by="LocationID", as_index=False, aggfunc="first")
    manhattan = manhattan.sort_values("LocationID").reset_index(drop=True)
    print(f"  Manhattan zones after dissolve: {len(manhattan)} unique zones")

    # Compute centroids in projected CRS (EPSG:2263) for accuracy,
    # then reproject both polygons and centroid points to WGS84
    centroids_projected = manhattan.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(geometry=centroids_projected, crs=manhattan.crs)
    centroids_gdf = centroids_gdf.to_crs(epsg=4326)

    # Reproject polygons EPSG:2263 → EPSG:4326 (WGS84 / GPS coordinates)
    manhattan = manhattan.to_crs(epsg=4326)
    print(f"  Reprojected to: {manhattan.crs}")

    manhattan["lat"] = centroids_gdf.geometry.y.values
    manhattan["lon"] = centroids_gdf.geometry.x.values

    # Save GeoJSON for PyDeck
    geojson_path = ZONES_DIR / "manhattan_zones.geojson"
    manhattan[["LocationID", "zone", "borough", "geometry"]].to_file(
        geojson_path, driver="GeoJSON"
    )
    print(f"  Saved: {geojson_path.name}")

    # Save centroids CSV
    centroids_df = manhattan[["LocationID", "zone", "borough", "lat", "lon"]].copy()
    centroids_df = centroids_df.rename(columns={"zone": "zone_name"})
    centroids_csv = ZONES_DIR / "zone_centroids.csv"
    centroids_df.to_csv(centroids_csv, index=False)
    print(f"  Saved: {centroids_csv.name} ({len(centroids_df)} zones)")

    # Build zone_id ↔ index mapping (zone IDs are non-contiguous)
    zone_ids = sorted(centroids_df["LocationID"].tolist())
    zone_id_to_index = {zid: idx for idx, zid in enumerate(zone_ids)}
    index_to_zone_id = {idx: zid for idx, zid in enumerate(zone_ids)}
    mapping = {"zone_id_to_index": zone_id_to_index, "index_to_zone_id": index_to_zone_id}
    mapping_path = ZONES_DIR / "zone_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Saved: {mapping_path.name} ({len(zone_ids)} mappings)")

    # Compute pairwise Haversine distances (69×69 matrix)
    n = len(centroids_df)
    lats = centroids_df["lat"].values
    lons = centroids_df["lon"].values

    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        distances[i, :] = haversine_km(
            np.full(n, lats[i]), np.full(n, lons[i]), lats, lons
        )

    dist_path = ZONES_DIR / "zone_distances.npz"
    np.savez_compressed(dist_path, distances=distances, zone_ids=np.array(zone_ids))
    print(f"  Saved: {dist_path.name} ({n}x{n} matrix)")

    # Compute travel times (minutes) using zone-specific speeds
    travel_times = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                travel_times[i, j] = 0.0
                continue
            zid_i = zone_ids[i]
            zid_j = zone_ids[j]
            # Use highway speed if either endpoint is highway-adjacent
            if zid_i in HIGHWAY_ZONE_IDS or zid_j in HIGHWAY_ZONE_IDS:
                speed = HIGHWAY_SPEED_KPH
            else:
                speed = MANHATTAN_SPEED_KPH
            travel_times[i, j] = (distances[i, j] / speed) * 60.0  # km / (km/h) * 60 = minutes

    tt_path = ZONES_DIR / "zone_travel_times.npz"
    np.savez_compressed(tt_path, travel_times=travel_times, zone_ids=np.array(zone_ids))
    print(f"  Saved: {tt_path.name} ({n}x{n} matrix)")

    # Print verification info
    print("\n--- Zone Processing Summary ---")
    print(f"  Total Manhattan zones: {n}")

    # Verify known landmarks
    landmark_zones = {
        "Times Sq/Theatre District": (40.757, -73.986),
        "Midtown Center": (40.754, -73.984),
        "Financial District North": (40.709, -74.008),
        "Upper East Side South": (40.773, -73.959),
    }
    print("\n  Centroid verification (expected vs actual):")
    for zone_name, (exp_lat, exp_lon) in landmark_zones.items():
        row = centroids_df[centroids_df["zone_name"] == zone_name]
        if len(row) == 1:
            actual_lat = row["lat"].values[0]
            actual_lon = row["lon"].values[0]
            print(f"    {zone_name}:")
            print(f"      Expected: ({exp_lat:.3f}, {exp_lon:.3f})")
            print(f"      Actual:   ({actual_lat:.3f}, {actual_lon:.3f})")

    # Distance and travel time stats
    mask = ~np.eye(n, dtype=bool)
    print(f"\n  Pairwise distances (non-diagonal):")
    print(f"    Min: {distances[mask].min():.2f} km")
    print(f"    Max: {distances[mask].max():.2f} km")
    print(f"    Mean: {distances[mask].mean():.2f} km")

    print(f"\n  Travel times (non-diagonal):")
    print(f"    Min: {travel_times[mask].min():.1f} min")
    print(f"    Max: {travel_times[mask].max():.1f} min")
    print(f"    Mean: {travel_times[mask].mean():.1f} min")


TRIP_FILES = [
    "yellow_tripdata_2023-01.parquet",
    "yellow_tripdata_2023-06.parquet",
]

TRIP_OUTPUT_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_duration_seconds",
    "trip_distance",
    "pickup_hour",
    "day_of_week",
    "is_weekend",
    "fare_amount",
]


def process_trips() -> None:
    """Process raw trip parquet files: filter to Manhattan, clean, derive columns."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load Manhattan zone IDs from zone mapping
    mapping_path = ZONES_DIR / "zone_mapping.json"
    with open(mapping_path) as f:
        mapping = json.load(f)
    manhattan_zone_ids = [int(zid) for zid in mapping["zone_id_to_index"]]
    print(f"\nProcessing trips (Manhattan zone IDs: {len(manhattan_zone_ids)} zones)")

    # Columns we actually need from raw data
    needed_columns = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID",
        "trip_distance", "fare_amount", "passenger_count",
    ]

    # Load and concatenate all raw parquet files
    frames: list[pl.DataFrame] = []
    for filename in TRIP_FILES:
        path = RAW_DIR / filename
        print(f"\n  Loading {filename}...")
        df = pl.read_parquet(path)
        print(f"    Raw rows: {len(df):,}")
        # Normalize column names to handle casing differences between months
        df = df.rename({c: c.lower() for c in df.columns})
        # Select only needed columns (case-insensitive matched)
        col_map = {c.lower(): c for c in needed_columns}
        df = df.select([pl.col(k).alias(v) for k, v in col_map.items()])
        # Cast to consistent types across months (Int32 vs Int64 differences)
        df = df.cast({
            "PULocationID": pl.Int64,
            "DOLocationID": pl.Int64,
            "trip_distance": pl.Float64,
            "fare_amount": pl.Float64,
            "passenger_count": pl.Float64,
        })
        frames.append(df)

    raw = pl.concat(frames)
    total_raw = len(raw)
    print(f"\n  Combined raw rows: {total_raw:,}")

    # Filter: both pickup AND dropoff in Manhattan
    raw = raw.filter(
        pl.col("PULocationID").is_in(manhattan_zone_ids)
        & pl.col("DOLocationID").is_in(manhattan_zone_ids)
    )
    after_manhattan = len(raw)
    print(f"  After Manhattan filter: {after_manhattan:,} ({after_manhattan / total_raw * 100:.1f}%)")

    # Compute derived columns
    raw = raw.with_columns(
        (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
        .dt.total_seconds()
        .cast(pl.Int64)
        .alias("trip_duration_seconds"),
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        pl.col("tpep_pickup_datetime").dt.date().alias("pickup_date"),
        pl.col("tpep_pickup_datetime").dt.weekday().sub(1).alias("day_of_week"),
    )
    # Polars dt.weekday() returns 1=Monday...7=Sunday; subtract 1 for 0=Monday...6=Sunday
    raw = raw.with_columns(
        (pl.col("day_of_week") >= 5).alias("is_weekend"),
    )

    # Filter to expected months only (some trips have timestamps outside Jan/Jun 2023)
    raw = raw.filter(
        (pl.col("tpep_pickup_datetime").dt.strftime("%Y-%m").is_in(["2023-01", "2023-06"]))
    )
    print(f"  After month filter: {len(raw):,}")

    # Quality filters
    raw = raw.filter(
        (pl.col("trip_distance") > 0) & (pl.col("trip_distance") <= 50)
        & (pl.col("fare_amount") > 0) & (pl.col("fare_amount") <= 500)
        & (pl.col("trip_duration_seconds") > 60) & (pl.col("trip_duration_seconds") <= 3600)
        & ((pl.col("passenger_count") > 0) | pl.col("passenger_count").is_null())
    )
    after_quality = len(raw)
    print(f"  After quality filter: {after_quality:,} ({after_quality / after_manhattan * 100:.1f}% of Manhattan trips)")
    print(f"  Removed by quality: {after_manhattan - after_quality:,}")

    # Select final columns and save
    trips = raw.select(TRIP_OUTPUT_COLUMNS)
    trips_path = PROCESSED_DIR / "trips_manhattan.parquet"
    trips.write_parquet(trips_path)
    print(f"\n  Saved: {trips_path.name} ({len(trips):,} rows)")

    # Per-month stats
    print("\n  Trips per month:")
    monthly = (
        trips
        .with_columns(pl.col("tpep_pickup_datetime").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("count"))
        .sort("month")
    )
    for row in monthly.iter_rows(named=True):
        print(f"    {row['month']}: {row['count']:,}")

    # Build demand aggregation (15-minute buckets per zone)
    print("\n  Building 15-minute demand aggregation...")
    demand = _build_demand_aggregation(trips)
    demand_path = PROCESSED_DIR / "zone_demand_15min.parquet"
    demand.write_parquet(demand_path)
    print(f"  Saved: {demand_path.name} ({len(demand):,} rows)")

    # Print demand sample
    print("\n  Demand aggregation sample (top 10 rows by timestamp):")
    sample = demand.sort("timestamp").head(10)
    # Use to_pandas() string repr to avoid Windows cp1252 encoding issues with Polars box chars
    print(sample.to_pandas().to_string(index=False))

    # Print summary stats
    print(f"\n--- Trip Processing Summary ---")
    print(f"  Total raw trips: {total_raw:,}")
    print(f"  Manhattan-to-Manhattan: {after_manhattan:,}")
    print(f"  After quality filter: {after_quality:,}")
    print(f"  Demand aggregation rows: {len(demand):,}")
    print(f"  Unique zones in demand: {demand['zone_id'].n_unique()}")
    print(f"  Date range: {demand['timestamp'].min()} to {demand['timestamp'].max()}")
    dc = demand["demand_count"]
    print(f"  Demand per bucket: mean={dc.mean():.1f}, median={dc.median():.1f}, "
          f"min={dc.min()}, max={dc.max()}, std={dc.std():.1f}")


def _build_demand_aggregation(trips: pl.DataFrame) -> pl.DataFrame:
    """Group trips into 15-minute buckets per pickup zone.

    Args:
        trips: Cleaned trips dataframe.

    Returns:
        DataFrame with columns: timestamp, zone_id, demand_count, hour, day_of_week, is_weekend.
    """
    demand = (
        trips
        .with_columns(
            pl.col("tpep_pickup_datetime")
            .dt.truncate("15m")
            .alias("timestamp")
        )
        .group_by("timestamp", "PULocationID")
        .agg(pl.len().alias("demand_count"))
        .rename({"PULocationID": "zone_id"})
        .with_columns(
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().sub(1).alias("day_of_week"),
        )
        .with_columns(
            (pl.col("day_of_week") >= 5).alias("is_weekend"),
        )
        .sort("timestamp", "zone_id")
    )
    return demand


def main() -> None:
    """Run all preprocessing steps."""
    process_zones()
    process_trips()


if __name__ == "__main__":
    main()
