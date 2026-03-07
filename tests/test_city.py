"""Tests for CityGraph zone topology."""

from pathlib import Path

import pytest

from simulation.city import CityGraph

ZONES_DIR = Path(__file__).resolve().parents[1] / "data" / "zones"


@pytest.fixture
def city() -> CityGraph:
    return CityGraph(ZONES_DIR)


def test_n_zones(city: CityGraph) -> None:
    # 67 unique Manhattan zones (69 shapefile rows, but LocationID 103 has 3 polygons)
    assert city.n_zones == 67


def test_zone_ids_sorted(city: CityGraph) -> None:
    ids = city.zone_ids
    assert ids == sorted(ids)
    assert len(ids) == city.n_zones


def test_zone_indices(city: CityGraph) -> None:
    assert city.zone_indices == list(range(city.n_zones))


def test_bidirectional_mapping(city: CityGraph) -> None:
    for i in range(city.n_zones):
        zid = city.index_to_zone_id(i)
        assert city.zone_id_to_index(zid) == i

    for zid in city.zone_ids:
        idx = city.zone_id_to_index(zid)
        assert city.index_to_zone_id(idx) == zid


def test_centroid_times_square(city: CityGraph) -> None:
    # Times Sq/Theatre District is zone 230
    lat, lon = city.centroid(230)
    assert abs(lat - 40.757) < 0.01, f"Times Square lat {lat} too far from 40.757"
    assert abs(lon - (-73.986)) < 0.01, f"Times Square lon {lon} too far from -73.986"


def test_zone_name(city: CityGraph) -> None:
    assert city.zone_name(230) == "Times Sq/Theatre District"
    assert city.zone_id_to_name(230) == "Times Sq/Theatre District"


def test_name_to_zone_id_exact(city: CityGraph) -> None:
    assert city.name_to_zone_id("Times Sq/Theatre District") == 230


def test_name_to_zone_id_partial(city: CityGraph) -> None:
    zid = city.name_to_zone_id("times sq")
    assert zid == 230


def test_name_to_zone_id_missing(city: CityGraph) -> None:
    with pytest.raises(KeyError):
        city.name_to_zone_id("Nonexistent Zone 999")


def test_distance_reasonable(city: CityGraph) -> None:
    # Times Square (230) to Financial District North (87): ~6-7 km
    dist = city.distance_km(230, 87)
    assert 4.0 < dist < 10.0, f"Distance {dist} km not in expected range"

    # Self-distance should be zero
    assert city.distance_km(230, 230) == 0.0


def test_travel_time_reasonable(city: CityGraph) -> None:
    # Times Square to Financial District: should be ~10-20 min at Manhattan speeds
    tt = city.travel_time_minutes(230, 87)
    assert 5.0 < tt < 30.0, f"Travel time {tt} min not in expected range"

    # Self travel time should be zero
    assert city.travel_time_minutes(230, 230) == 0.0


def test_neighbors_nonempty(city: CityGraph) -> None:
    # Times Square should have neighbors within 2km
    nbrs = city.neighbors(230, radius_km=2.0)
    assert len(nbrs) > 0, "Times Square should have neighbors within 2km"
    # All neighbors should be valid zone IDs
    valid_ids = set(city.zone_ids)
    for n in nbrs:
        assert n in valid_ids
    # Self should not be in neighbors
    assert 230 not in nbrs


def test_neighbors_sorted_by_distance(city: CityGraph) -> None:
    nbrs = city.neighbors(230, radius_km=5.0)
    if len(nbrs) >= 2:
        dists = [city.distance_km(230, n) for n in nbrs]
        assert dists == sorted(dists), "Neighbors should be sorted by distance"


def test_neighbors_radius(city: CityGraph) -> None:
    # Small radius: fewer neighbors. Large radius: more neighbors.
    small = city.neighbors(230, radius_km=0.5)
    large = city.neighbors(230, radius_km=5.0)
    assert len(small) <= len(large)


def test_invalid_zone_id(city: CityGraph) -> None:
    with pytest.raises(KeyError):
        city.zone_name(9999)
    with pytest.raises(KeyError):
        city.centroid(9999)
    with pytest.raises(KeyError):
        city.zone_id_to_index(9999)
    with pytest.raises(KeyError):
        city.index_to_zone_id(9999)
