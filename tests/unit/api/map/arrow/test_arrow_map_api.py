"""Stress tests for ArrowMapAPI read operations, spatial queries, caching, and get_map_api_for_log."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from pathlib import Path

import pytest
import shapely.geometry as geom

from py123d.api.map.arrow.arrow_map_api import (
    ArrowMapAPI,
    get_lru_cached_map_api,
    get_map_api_for_log,
)
from py123d.datatypes import MapMetadata
from py123d.datatypes.map_objects.map_layer_types import (
    IntersectionType,
    LaneType,
    MapLayer,
    RoadEdgeType,
    RoadLineType,
    StopZoneType,
)
from py123d.datatypes.map_objects.map_objects import (
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    Lane,
    LaneGroup,
    RoadEdge,
    RoadLine,
    StopZone,
    Walkway,
)
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.geometry import Point2D, Point3D

from ..conftest import (
    make_carpark,
    make_crosswalk,
    make_generic_drivable,
    make_intersection,
    make_lane,
    make_lane_group,
    make_road_edge,
    make_road_line,
    make_stop_zone,
    make_walkway,
    write_and_read_map,
)


def _build_full_map_metadata() -> MapMetadata:
    return MapMetadata(
        dataset="test_dataset",
        location="boston",
        map_has_z=True,
        map_is_per_log=True,
        split="test_train",
        log_name="log_001",
    )


def _build_all_object_types() -> list:
    """Build one of every map object type with cross-references."""
    lane = make_lane(object_id=10, lane_group_id=100, speed_limit_mps=13.4)
    lane_group = make_lane_group(object_id=100, lane_ids=[10], intersection_id=1000)
    intersection = make_intersection(object_id=1000, lane_group_ids=[100])
    crosswalk = make_crosswalk(object_id=2000)
    walkway = make_walkway(object_id=3000)
    carpark = make_carpark(object_id=4000)
    generic_drivable = make_generic_drivable(object_id=5000)
    stop_zone = make_stop_zone(object_id=6000, lane_ids=[10])
    road_edge = make_road_edge(object_id=7000)
    road_line = make_road_line(object_id=8000)
    return [
        lane,
        lane_group,
        intersection,
        crosswalk,
        walkway,
        carpark,
        generic_drivable,
        stop_zone,
        road_edge,
        road_line,
    ]


class TestArrowMapAPIMetadata:
    """Test metadata round-trip."""

    def test_metadata_fields_preserved(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        objects = [make_lane(object_id=1)]
        api = write_and_read_map(tmp_path, metadata, objects)  # type: ignore

        read_meta = api.get_map_metadata()
        assert read_meta.dataset == metadata.dataset
        assert read_meta.location == metadata.location
        assert read_meta.map_has_z == metadata.map_has_z
        assert read_meta.map_is_per_log == metadata.map_is_per_log

    def test_metadata_version_is_string(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])
        assert isinstance(api.version, str)
        assert len(api.version) > 0


class TestArrowMapAPIAvailableLayers:
    """Test get_available_map_layers."""

    def test_all_layers_present(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        objects = _build_all_object_types()
        api = write_and_read_map(tmp_path, metadata, objects)

        layers = api.get_available_map_layers()
        assert len(layers) == 10
        for layer in MapLayer:
            assert layer in layers

    def test_subset_of_layers(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        objects = [make_lane(object_id=1), make_road_edge(object_id=2)]
        api = write_and_read_map(tmp_path, metadata, objects)

        layers = api.get_available_map_layers()
        assert MapLayer.LANE in layers
        assert MapLayer.ROAD_EDGE in layers
        assert MapLayer.CROSSWALK not in layers

    def test_empty_map_no_layers(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [])

        layers = api.get_available_map_layers()
        assert layers == []


class TestArrowMapAPIGetObject:
    """Test get_map_object_in_layer for each object type."""

    def test_get_lane(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(
            object_id=42,
            lane_group_id=100,
            lane_type=LaneType.FREEWAY,
            speed_limit_mps=30.0,
            predecessor_ids=[41],
            successor_ids=[43],
            left_lane_id=99,
            right_lane_id=101,
        )
        # Also write the referenced objects so IDs get remapped
        lg = make_lane_group(object_id=100, lane_ids=[42])
        pred = make_lane(object_id=41, y_offset=10.0)
        succ = make_lane(object_id=43, y_offset=20.0)
        left = make_lane(object_id=99, y_offset=30.0)
        right = make_lane(object_id=101, y_offset=40.0)
        api = write_and_read_map(tmp_path, metadata, [lane, lg, pred, succ, left, right])

        result = api.get_map_object_in_layer(42, MapLayer.LANE)
        assert isinstance(result, Lane)
        assert result.lane_type == LaneType.FREEWAY
        assert result.speed_limit_mps == pytest.approx(30.0)
        assert result.lane_group_id == 100
        assert result.left_lane_id == 99
        assert result.right_lane_id == 101
        assert 41 in result.predecessor_ids
        assert 43 in result.successor_ids
        # Boundary arrays should have data
        assert result.left_boundary.array.shape[1] == 3
        assert result.right_boundary.array.shape[1] == 3
        assert result.centerline.array.shape[1] == 3

    def test_get_lane_group(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1, lane_group_id=10)
        lg = make_lane_group(object_id=10, lane_ids=[1], intersection_id=100)
        inter = make_intersection(object_id=100, lane_group_ids=[10])
        api = write_and_read_map(tmp_path, metadata, [lane, lg, inter])

        result = api.get_map_object_in_layer(10, MapLayer.LANE_GROUP)
        assert isinstance(result, LaneGroup)
        assert 1 in result.lane_ids
        assert result.intersection_id == 100

    def test_get_intersection(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lg = make_lane_group(object_id=10)
        inter = make_intersection(object_id=100, lane_group_ids=[10], intersection_type=IntersectionType.STOP_SIGN)
        api = write_and_read_map(tmp_path, metadata, [lg, inter])

        result = api.get_map_object_in_layer(100, MapLayer.INTERSECTION)
        assert isinstance(result, Intersection)
        assert result.intersection_type == IntersectionType.STOP_SIGN
        assert 10 in result.lane_group_ids

    def test_get_crosswalk(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_crosswalk(object_id=5)])

        result = api.get_map_object_in_layer(5, MapLayer.CROSSWALK)
        assert isinstance(result, Crosswalk)
        assert result.outline.array.shape[1] == 3

    def test_get_carpark(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_carpark(object_id=6)])
        result = api.get_map_object_in_layer(6, MapLayer.CARPARK)
        assert isinstance(result, Carpark)

    def test_get_walkway(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_walkway(object_id=7)])
        result = api.get_map_object_in_layer(7, MapLayer.WALKWAY)
        assert isinstance(result, Walkway)

    def test_get_generic_drivable(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_generic_drivable(object_id=8)])
        result = api.get_map_object_in_layer(8, MapLayer.GENERIC_DRIVABLE)
        assert isinstance(result, GenericDrivable)

    def test_get_stop_zone(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1)
        sz = make_stop_zone(object_id=9, lane_ids=[1], stop_zone_type=StopZoneType.STOP_SIGN)
        api = write_and_read_map(tmp_path, metadata, [lane, sz])

        result = api.get_map_object_in_layer(9, MapLayer.STOP_ZONE)
        assert isinstance(result, StopZone)
        assert result.stop_zone_type == StopZoneType.STOP_SIGN
        assert 1 in result.lane_ids

    def test_get_road_edge(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_road_edge(object_id=10)])
        result = api.get_map_object_in_layer(10, MapLayer.ROAD_EDGE)
        assert isinstance(result, RoadEdge)
        assert result.road_edge_type == RoadEdgeType.ROAD_EDGE_BOUNDARY

    def test_get_road_line(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_road_line(object_id=11)])
        result = api.get_map_object_in_layer(11, MapLayer.ROAD_LINE)
        assert isinstance(result, RoadLine)
        assert result.road_line_type == RoadLineType.SOLID_WHITE

    def test_nonexistent_id_returns_none(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        result = api.get_map_object_in_layer(99999, MapLayer.LANE)
        assert result is None

    def test_id_in_wrong_layer_returns_none(self, tmp_path: Path) -> None:
        """Querying an ID in a layer that has no objects should return None."""
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        result = api.get_map_object_in_layer(1, MapLayer.CROSSWALK)
        assert result is None


class TestArrowMapAPIGetAllIds:
    def test_returns_correct_ids(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        objects = [make_lane(object_id=i, y_offset=i * 10.0) for i in range(5)]
        api = write_and_read_map(tmp_path, metadata, objects)  # type: ignore

        ids = api.get_all_map_object_ids_in_layer(MapLayer.LANE)
        assert len(ids) == 5
        assert set(ids) == {0, 1, 2, 3, 4}

    def test_missing_layer_returns_empty(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        ids = api.get_all_map_object_ids_in_layer(MapLayer.CROSSWALK)
        assert ids == []


class TestArrowMapAPIIterators:
    def test_single_layer_iterator(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        objects = [make_lane(object_id=i, y_offset=i * 10.0) for i in range(3)]
        api = write_and_read_map(tmp_path, metadata, objects)  # type: ignore

        items = list(api.get_all_map_objects_in_layer(MapLayer.LANE))
        assert len(items) == 3
        assert all(isinstance(item, Lane) for item in items)

    def test_multi_layer_iterator(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        objects = [make_lane(object_id=1), make_road_edge(object_id=2), make_road_line(object_id=3)]
        api = write_and_read_map(tmp_path, metadata, objects)

        items = list(api.get_all_map_objects_in_layers([MapLayer.LANE, MapLayer.ROAD_EDGE, MapLayer.ROAD_LINE]))
        assert len(items) == 3

    def test_empty_layer_iterator(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        items = list(api.get_all_map_objects_in_layer(MapLayer.CROSSWALK))
        assert items == []

    def test_iterator_consumed_twice(self, tmp_path: Path) -> None:
        """Calling the method twice should yield fresh iterators each time."""
        metadata = _build_full_map_metadata()
        objects = [make_lane(object_id=i, y_offset=i * 10.0) for i in range(3)]
        api = write_and_read_map(tmp_path, metadata, objects)  # type: ignore

        first = list(api.get_all_map_objects_in_layer(MapLayer.LANE))
        second = list(api.get_all_map_objects_in_layer(MapLayer.LANE))
        assert len(first) == 3
        assert len(second) == 3


class TestArrowMapAPISpatialQueries:
    """Test spatial query methods including 1 probes."""

    def test_radius_query_finds_nearby(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1, x_offset=0.0, y_offset=0.0)
        api = write_and_read_map(tmp_path, metadata, [lane])

        # Lane boundaries are at y=0..3, x=0..20 — query center of that
        result = api.get_map_objects_in_radius(Point2D(10.0, 1.5), radius=50.0, layers=[MapLayer.LANE])
        assert MapLayer.LANE in result
        assert len(result[MapLayer.LANE]) >= 1

    def test_radius_query_excludes_far(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1, x_offset=0.0, y_offset=0.0)
        api = write_and_read_map(tmp_path, metadata, [lane])

        # Query very far away with tiny radius
        result = api.get_map_objects_in_radius(Point2D(1000.0, 1000.0), radius=1.0, layers=[MapLayer.LANE])
        assert result[MapLayer.LANE] == []

    def test_radius_query_with_point3d(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, metadata, [lane])

        result = api.get_map_objects_in_radius(Point3D(10.0, 1.5, 0.0), radius=50.0, layers=[MapLayer.LANE])
        assert MapLayer.LANE in result
        assert len(result[MapLayer.LANE]) >= 1

    def test_query_single_geometry_intersects(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, metadata, [lane])

        box = geom.box(0, 0, 20, 5)
        result = api.query(geometry=box, layers=[MapLayer.LANE], predicate="intersects")
        assert MapLayer.LANE in result
        # Single geometry → should be a list
        assert isinstance(result[MapLayer.LANE], list)
        assert len(result[MapLayer.LANE]) >= 1

    def test_query_iterable_geometries_returns_dict(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, metadata, [lane])

        boxes = [geom.box(0, 0, 20, 5), geom.box(100, 100, 110, 110)]
        result = api.query(geometry=boxes, layers=[MapLayer.LANE], predicate="intersects")
        assert MapLayer.LANE in result
        # Iterable → should be a dict mapping int indices to lists
        assert isinstance(result[MapLayer.LANE], dict)

    def test_query_missing_layer_single_geometry_returns_list(self, tmp_path: Path) -> None:
        """Query on missing layer with single geometry should return empty list."""
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        box = geom.box(0, 0, 10, 10)
        result = api.query(geometry=box, layers=[MapLayer.CROSSWALK], predicate="intersects")
        crosswalk_result = result[MapLayer.CROSSWALK]

        assert isinstance(crosswalk_result, list)
        assert crosswalk_result == []

    def test_query_missing_layer_iterable_geometry_returns_dict(self, tmp_path: Path) -> None:
        """Query on missing layer with iterable geometry should return empty dict."""
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        boxes = [geom.box(0, 0, 10, 10), geom.box(20, 20, 30, 30)]
        result = api.query(geometry=boxes, layers=[MapLayer.CROSSWALK], predicate="intersects")
        crosswalk_result = result[MapLayer.CROSSWALK]

        assert isinstance(crosswalk_result, dict)
        assert crosswalk_result == {}

    def test_query_object_ids_missing_layer_single_returns_list(self, tmp_path: Path) -> None:
        """query_object_ids on missing layer with single geometry should return empty list."""
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        box = geom.box(0, 0, 10, 10)
        result = api.query_object_ids(geometry=box, layers=[MapLayer.CROSSWALK], predicate="intersects")
        crosswalk_result = result[MapLayer.CROSSWALK]

        assert isinstance(crosswalk_result, list)
        assert crosswalk_result == []

    def test_query_object_ids_missing_layer_iterable_returns_dict(self, tmp_path: Path) -> None:
        """query_object_ids on missing layer with iterable geometry should return empty dict."""
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        boxes = [geom.box(0, 0, 10, 10)]
        result = api.query_object_ids(geometry=boxes, layers=[MapLayer.CROSSWALK], predicate="intersects")
        crosswalk_result = result[MapLayer.CROSSWALK]

        assert isinstance(crosswalk_result, dict)
        assert crosswalk_result == {}

    def test_query_object_ids_single_geometry(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, metadata, [lane])

        box = geom.box(0, 0, 20, 5)
        result = api.query_object_ids(geometry=box, layers=[MapLayer.LANE], predicate="intersects")
        assert MapLayer.LANE in result
        assert isinstance(result[MapLayer.LANE], list)
        assert 1 in result[MapLayer.LANE]

    def test_query_object_ids_iterable_geometry(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, metadata, [lane])

        boxes = [geom.box(0, 0, 20, 5)]
        result = api.query_object_ids(geometry=boxes, layers=[MapLayer.LANE], predicate="intersects")
        assert MapLayer.LANE in result
        assert isinstance(result[MapLayer.LANE], dict)

    def test_multipolygon_not_iterable_in_shapely2(self, tmp_path: Path) -> None:
        """In shapely 2.x, MultiPolygon is NOT Iterable.

        This means the isinstance(geometry, Iterable) check in _query_layer
        correctly identifies MultiPolygon as a single geometry in shapely 2.x.
        But if someone passes a plain Python list of geometries, that IS Iterable.
        """
        metadata = _build_full_map_metadata()
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, metadata, [lane])

        multi = geom.MultiPolygon([geom.box(0, 0, 20, 5)])
        # In shapely 2.x, MultiPolygon is NOT Iterable
        assert not isinstance(multi, IterableABC), "In shapely 2.x, MultiPolygon should not be Iterable."

        # MultiPolygon as single geometry — should return list (like any single geometry)
        result = api.query(geometry=multi, layers=[MapLayer.LANE], predicate="intersects")
        lane_result = result[MapLayer.LANE]
        assert isinstance(lane_result, list)


class TestArrowMapAPIMapObjectLinks:
    """Test that map_api back-references are set on retrieved objects."""

    def test_lane_has_map_api_reference(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        lane = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert isinstance(lane, Lane)
        assert lane._map_api is api

    def test_lane_group_has_map_api_reference(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane_group(object_id=1)])

        lg = api.get_map_object_in_layer(1, MapLayer.LANE_GROUP)
        assert isinstance(lg, LaneGroup)
        assert lg._map_api is api

    def test_intersection_has_map_api_reference(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_intersection(object_id=1)])

        inter = api.get_map_object_in_layer(1, MapLayer.INTERSECTION)
        assert isinstance(inter, Intersection)
        assert inter._map_api is api


class TestArrowMapAPICaching:
    def test_same_object_returned_from_cache(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        api = write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        obj1 = api.get_map_object_in_layer(1, MapLayer.LANE)
        obj2 = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert obj1 is obj2, "LRU cache should return the same object instance"


class TestGetLruCachedMapApi:
    def setup_method(self) -> None:
        get_lru_cached_map_api.cache_clear()

    def test_returns_arrow_map_api(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        arrow_file = tmp_path / "logs" / "test_train" / "log_001" / "map.arrow"
        cached_api = get_lru_cached_map_api(arrow_file)
        assert isinstance(cached_api, ArrowMapAPI)

    def test_caches_same_path(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        arrow_file = tmp_path / "logs" / "test_train" / "log_001" / "map.arrow"
        api1 = get_lru_cached_map_api(arrow_file)
        api2 = get_lru_cached_map_api(arrow_file)
        assert api1 is api2

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(AssertionError, match="Arrow file of map not found"):
            get_lru_cached_map_api(Path("/nonexistent/path/map.arrow"))


class TestGetMapApiForLog:
    def setup_method(self) -> None:
        get_lru_cached_map_api.cache_clear()

    def test_per_log_map_found(self, tmp_path: Path) -> None:
        metadata = _build_full_map_metadata()
        write_and_read_map(tmp_path, metadata, [make_lane(object_id=1)])

        log_dir = tmp_path / "logs" / "test_train" / "log_001"
        log_metadata = LogMetadata(
            dataset="test_dataset",
            split="test_train",
            log_name="log_001",
            location="boston",
        )
        result = get_map_api_for_log(log_dir, log_metadata)
        assert result is not None
        assert isinstance(result, ArrowMapAPI)

    def test_no_map_returns_none(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "empty_log"
        log_dir.mkdir()
        log_metadata = LogMetadata(
            dataset="test_dataset",
            split="test_train",
            log_name="log_002",
            location=None,
        )
        result = get_map_api_for_log(log_dir, log_metadata)
        assert result is None

    def test_per_log_takes_precedence_over_global(self, tmp_path: Path) -> None:
        """When both per-log and global maps exist, per-log should be used."""
        # Create per-log map
        per_log_meta = _build_full_map_metadata()
        write_and_read_map(tmp_path, per_log_meta, [make_lane(object_id=1)])

        # Create global map (different object to distinguish)
        global_meta = MapMetadata(
            dataset="test_dataset",
            location="boston",
            map_has_z=True,
            map_is_per_log=False,
        )
        write_and_read_map(tmp_path, global_meta, [make_lane(object_id=2), make_crosswalk(object_id=3)])

        log_dir = tmp_path / "logs" / "test_train" / "log_001"
        log_metadata = LogMetadata(
            dataset="test_dataset",
            split="test_train",
            log_name="log_001",
            location="boston",
        )
        result = get_map_api_for_log(log_dir, log_metadata)
        assert result is not None

        # Per-log map had only 1 lane, no crosswalks
        # Global map had 1 lane + 1 crosswalk
        # If per-log takes precedence, no crosswalks should be found
        crosswalk_ids = result.get_all_map_object_ids_in_layer(MapLayer.CROSSWALK)
        assert crosswalk_ids == [], "Per-log map should take precedence — should have no crosswalks"
