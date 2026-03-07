"""Exploratory data analysis and validation of processed trip data.

Prints summary statistics, top zones, hourly distributions,
and validates data quality.
"""

import json
from pathlib import Path

import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = DATA_DIR / "processed"
ZONES_DIR = DATA_DIR / "zones"


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pd.DataFrame, set[int]]:
    """Load trips, demand, centroids, and Manhattan zone IDs."""
    trips = pl.read_parquet(PROCESSED_DIR / "trips_manhattan.parquet")
    demand = pl.read_parquet(PROCESSED_DIR / "zone_demand_15min.parquet")
    centroids = pd.read_csv(ZONES_DIR / "zone_centroids.csv")

    with open(ZONES_DIR / "zone_mapping.json") as f:
        mapping = json.load(f)
    manhattan_ids = {int(zid) for zid in mapping["zone_id_to_index"]}

    return trips, demand, centroids, manhattan_ids


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def trips_per_month(trips: pl.DataFrame) -> None:
    """Print total trips per month."""
    print_section("1. Trips Per Month")
    monthly = (
        trips
        .with_columns(pl.col("tpep_pickup_datetime").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("trips"))
        .sort("month")
    )
    total = 0
    for row in monthly.iter_rows(named=True):
        print(f"  {row['month']}:  {row['trips']:>12,}")
        total += row["trips"]
    print(f"  {'TOTAL':<8} {total:>12,}")


def top_pickup_zones(trips: pl.DataFrame, centroids: pd.DataFrame) -> None:
    """Print top 10 pickup zones by trip volume."""
    print_section("2. Top 10 Pickup Zones")
    zone_counts = (
        trips
        .group_by("PULocationID")
        .agg(pl.len().alias("trips"))
        .sort("trips", descending=True)
        .head(10)
    )
    name_map = dict(zip(centroids["LocationID"], centroids["zone_name"]))
    total = len(trips)

    print(f"  {'Rank':<5} {'Zone':<40} {'Trips':>12} {'%':>7}")
    print(f"  {'-' * 5} {'-' * 40} {'-' * 12} {'-' * 7}")
    for i, row in enumerate(zone_counts.iter_rows(named=True), 1):
        name = name_map.get(row["PULocationID"], f"Zone {row['PULocationID']}")
        pct = row["trips"] / total * 100
        print(f"  {i:<5} {name:<40} {row['trips']:>12,} {pct:>6.1f}%")


def hourly_demand(trips: pl.DataFrame) -> None:
    """Print average trips per hour across all days."""
    print_section("3. Hourly Demand Distribution (avg trips/hour)")
    n_days = trips["tpep_pickup_datetime"].dt.date().n_unique()
    hourly = (
        trips
        .group_by("pickup_hour")
        .agg(pl.len().alias("total_trips"))
        .sort("pickup_hour")
        .with_columns((pl.col("total_trips") / n_days).round(0).cast(pl.Int64).alias("avg_per_day"))
    )

    print(f"  {'Hour':<6} {'Avg Trips/Day':>14} {'Total':>12}   Bar")
    print(f"  {'-' * 6} {'-' * 14} {'-' * 12}   {'-' * 30}")
    max_avg = hourly["avg_per_day"].max()
    for row in hourly.iter_rows(named=True):
        bar_len = int(row["avg_per_day"] / max_avg * 30) if max_avg > 0 else 0
        bar = "#" * bar_len
        hour_label = f"{row['pickup_hour']:02d}:00"
        print(f"  {hour_label:<6} {row['avg_per_day']:>14,} {row['total_trips']:>12,}   {bar}")


def trip_stats(trips: pl.DataFrame) -> None:
    """Print trip duration and distance statistics."""
    print_section("4. Trip Duration & Distance")
    dur = trips["trip_duration_seconds"] / 60.0  # to minutes
    dist = trips["trip_distance"]

    print(f"  Trip Duration (minutes):")
    print(f"    Mean:   {dur.mean():>8.1f}")
    print(f"    Median: {dur.median():>8.1f}")
    print(f"    Std:    {dur.std():>8.1f}")
    print(f"    Min:    {dur.min():>8.1f}")
    print(f"    Max:    {dur.max():>8.1f}")
    print()
    print(f"  Trip Distance (miles):")
    print(f"    Mean:   {dist.mean():>8.2f}")
    print(f"    Median: {dist.median():>8.2f}")
    print(f"    Std:    {dist.std():>8.2f}")
    print(f"    Min:    {dist.min():>8.2f}")
    print(f"    Max:    {dist.max():>8.2f}")


def weekday_vs_weekend(trips: pl.DataFrame) -> None:
    """Print weekend vs weekday average daily trips."""
    print_section("5. Weekend vs Weekday")
    daily = (
        trips
        .with_columns(pl.col("tpep_pickup_datetime").dt.date().alias("date"))
        .group_by("date", "is_weekend")
        .agg(pl.len().alias("trips"))
    )
    summary = (
        daily
        .group_by("is_weekend")
        .agg(
            pl.col("trips").mean().round(0).cast(pl.Int64).alias("avg_daily_trips"),
            pl.col("trips").median().round(0).cast(pl.Int64).alias("median_daily_trips"),
            pl.len().alias("n_days"),
        )
        .sort("is_weekend")
    )
    print(f"  {'Type':<12} {'Days':>6} {'Avg Trips/Day':>14} {'Median':>10}")
    print(f"  {'-' * 12} {'-' * 6} {'-' * 14} {'-' * 10}")
    for row in summary.iter_rows(named=True):
        label = "Weekend" if row["is_weekend"] else "Weekday"
        print(f"  {label:<12} {row['n_days']:>6} {row['avg_daily_trips']:>14,} {row['median_daily_trips']:>10,}")


def active_zones(trips: pl.DataFrame, centroids: pd.DataFrame) -> None:
    """Print number of zones with significant activity."""
    print_section("6. Active Manhattan Zones")
    zone_counts = (
        trips
        .group_by("PULocationID")
        .agg(pl.len().alias("trips"))
        .sort("trips", descending=True)
    )
    total_zones = zone_counts.height
    active = zone_counts.filter(pl.col("trips") > 100).height

    print(f"  Total zones with any pickups: {total_zones}")
    print(f"  Zones with >100 trips:        {active}")
    print(f"  Zones with <=100 trips:       {total_zones - active}")

    # Show the low-activity zones
    low = zone_counts.filter(pl.col("trips") <= 100)
    if low.height > 0:
        name_map = dict(zip(centroids["LocationID"], centroids["zone_name"]))
        print(f"\n  Low-activity zones:")
        for row in low.iter_rows(named=True):
            name = name_map.get(row["PULocationID"], f"Zone {row['PULocationID']}")
            print(f"    {name:<45} {row['trips']:>6} trips")


def demand_bucket_stats(demand: pl.DataFrame) -> None:
    """Print demand statistics from 15-minute bucketed data."""
    print_section("7. Demand Per 15-Min Bucket (across all zones)")
    dc = demand["demand_count"]
    print(f"  Total rows:    {len(demand):>12,}")
    print(f"  Unique zones:  {demand['zone_id'].n_unique():>12}")
    print(f"  Time buckets:  {demand['timestamp'].n_unique():>12,}")
    print()
    print(f"  Demand count per bucket:")
    print(f"    Mean:   {dc.mean():>8.1f}")
    print(f"    Median: {dc.median():>8.1f}")
    print(f"    Std:    {dc.std():>8.1f}")
    print(f"    Max:    {dc.max():>8}")
    print(f"    Min:    {dc.min():>8}")

    # Top zone-hours by demand
    print(f"\n  Top 5 busiest 15-min windows (any zone):")
    top = demand.sort("demand_count", descending=True).head(5)
    for row in top.iter_rows(named=True):
        print(f"    Zone {row['zone_id']:>3} at {row['timestamp']}: {row['demand_count']} trips")


def validation_checks(
    trips: pl.DataFrame, demand: pl.DataFrame, manhattan_ids: set[int]
) -> None:
    """Run data validation checks."""
    print_section("8. Validation Checks")
    errors = 0

    # Check no null PULocationIDs
    null_pu = trips["PULocationID"].null_count()
    status = "PASS" if null_pu == 0 else "FAIL"
    if null_pu > 0:
        errors += 1
    print(f"  [{status}] No null PULocationID: {null_pu} nulls")

    null_do = trips["DOLocationID"].null_count()
    status = "PASS" if null_do == 0 else "FAIL"
    if null_do > 0:
        errors += 1
    print(f"  [{status}] No null DOLocationID: {null_do} nulls")

    # Check all zone IDs are in Manhattan set
    pu_ids = set(trips["PULocationID"].unique().to_list())
    do_ids = set(trips["DOLocationID"].unique().to_list())
    extra_pu = pu_ids - manhattan_ids
    extra_do = do_ids - manhattan_ids
    status = "PASS" if len(extra_pu) == 0 else "FAIL"
    if len(extra_pu) > 0:
        errors += 1
    print(f"  [{status}] All PULocationIDs in Manhattan set: {len(extra_pu)} extra IDs")

    status = "PASS" if len(extra_do) == 0 else "FAIL"
    if len(extra_do) > 0:
        errors += 1
    print(f"  [{status}] All DOLocationIDs in Manhattan set: {len(extra_do)} extra IDs")

    # Check datetime ranges
    min_dt = trips["tpep_pickup_datetime"].min()
    max_dt = trips["tpep_pickup_datetime"].max()
    jan_ok = str(min_dt).startswith("2023-01")
    jun_ok = str(max_dt).startswith("2023-06")
    status = "PASS" if jan_ok else "FAIL"
    if not jan_ok:
        errors += 1
    print(f"  [{status}] Min datetime in Jan 2023: {min_dt}")

    status = "PASS" if jun_ok else "FAIL"
    if not jun_ok:
        errors += 1
    print(f"  [{status}] Max datetime in Jun 2023: {max_dt}")

    # Check no negative durations or distances
    neg_dur = trips.filter(pl.col("trip_duration_seconds") <= 0).height
    status = "PASS" if neg_dur == 0 else "FAIL"
    if neg_dur > 0:
        errors += 1
    print(f"  [{status}] No non-positive durations: {neg_dur} found")

    neg_dist = trips.filter(pl.col("trip_distance") <= 0).height
    status = "PASS" if neg_dist == 0 else "FAIL"
    if neg_dist > 0:
        errors += 1
    print(f"  [{status}] No non-positive distances: {neg_dist} found")

    # Check demand zone IDs
    demand_ids = set(demand["zone_id"].unique().to_list())
    extra_demand = demand_ids - manhattan_ids
    status = "PASS" if len(extra_demand) == 0 else "FAIL"
    if len(extra_demand) > 0:
        errors += 1
    print(f"  [{status}] All demand zone_ids in Manhattan set: {len(extra_demand)} extra")

    print(f"\n  Result: {8 - errors}/8 checks passed" + (" -- ALL CLEAR" if errors == 0 else f" -- {errors} FAILURES"))


def main() -> None:
    """Run all EDA analyses and validation."""
    trips, demand, centroids, manhattan_ids = load_data()
    print(f"Loaded {len(trips):,} trips, {len(demand):,} demand rows, {len(centroids)} zones")

    trips_per_month(trips)
    top_pickup_zones(trips, centroids)
    hourly_demand(trips)
    trip_stats(trips)
    weekday_vs_weekend(trips)
    active_zones(trips, centroids)
    demand_bucket_stats(demand)
    validation_checks(trips, demand, manhattan_ids)


if __name__ == "__main__":
    main()
