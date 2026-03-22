"""End-to-end round-trip stress tests: write map objects with ArrowMapWriter, read back with ArrowMapAPI.

These tests are designed to find bugs, not just confirm happy paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely.geometry as geom

from py123d.datatypes import (
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    Lane,
    LaneGroup,
    MapMetadata,
    RoadEdge,
    RoadLine,
    StopZone,
    Walkway,
)
from py123d.datatypes.map_objects.map_layer_types import (
    IntersectionType,
    LaneType,
    MapLayer,
    RoadEdgeType,
    RoadLineType,
    StopZoneType,
)
from py123d.geometry import Point2D, Polyline2D, Polyline3D

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


def _meta() -> MapMetadata:
    return MapMetadata(
        dataset="test_dataset",
        location="boston",
        map_has_z=True,
        map_is_per_log=True,
        split="test_train",
        log_name="log_001",
    )


# =============================================================================
# Round-trip for every object type
# =============================================================================


class TestRoundTripBasic:
    """Write each object type, read back, verify all fields match."""

    def test_roundtrip_lane(self, tmp_path: Path) -> None:
        lane = make_lane(
            object_id=42,
            lane_group_id=100,
            lane_type=LaneType.BIKE_LANE,
            speed_limit_mps=8.33,
            predecessor_ids=[41],
            successor_ids=[43],
            left_lane_id=99,
            right_lane_id=101,
        )
        # Write referenced objects too so IDs survive remapping
        others = [
            make_lane(object_id=41, y_offset=10.0),
            make_lane(object_id=43, y_offset=20.0),
            make_lane(object_id=99, y_offset=30.0),
            make_lane(object_id=101, y_offset=40.0),
            make_lane_group(object_id=100, lane_ids=[42]),
        ]
        api = write_and_read_map(tmp_path, _meta(), [lane] + others)

        read_lane = api.get_map_object_in_layer(42, MapLayer.LANE)
        assert isinstance(read_lane, Lane)
        assert read_lane.object_id == 42
        assert read_lane.lane_type == LaneType.BIKE_LANE
        assert read_lane.lane_group_id == 100
        assert read_lane.speed_limit_mps == pytest.approx(8.33)
        assert read_lane.left_lane_id == 99
        assert read_lane.right_lane_id == 101
        assert 41 in read_lane.predecessor_ids
        assert 43 in read_lane.successor_ids

        # Boundary arrays
        np.testing.assert_allclose(read_lane.left_boundary.array, lane.left_boundary.array, atol=1e-6)
        np.testing.assert_allclose(read_lane.right_boundary.array, lane.right_boundary.array, atol=1e-6)
        np.testing.assert_allclose(read_lane.centerline.array, lane.centerline.array, atol=1e-6)

    def test_roundtrip_lane_group(self, tmp_path: Path) -> None:
        lane = make_lane(object_id=1, lane_group_id=10)
        lg = make_lane_group(object_id=10, lane_ids=[1], intersection_id=100, predecessor_ids=[9], successor_ids=[11])
        inter = make_intersection(object_id=100, lane_group_ids=[10])
        pred_lg = make_lane_group(object_id=9, x_offset=50.0)
        succ_lg = make_lane_group(object_id=11, x_offset=100.0)
        api = write_and_read_map(tmp_path, _meta(), [lane, lg, inter, pred_lg, succ_lg])

        read_lg = api.get_map_object_in_layer(10, MapLayer.LANE_GROUP)
        assert isinstance(read_lg, LaneGroup)
        assert 1 in read_lg.lane_ids
        assert read_lg.intersection_id == 100
        assert 9 in read_lg.predecessor_ids
        assert 11 in read_lg.successor_ids
        np.testing.assert_allclose(read_lg.left_boundary.array, lg.left_boundary.array, atol=1e-6)
        np.testing.assert_allclose(read_lg.right_boundary.array, lg.right_boundary.array, atol=1e-6)

    def test_roundtrip_intersection(self, tmp_path: Path) -> None:
        lg = make_lane_group(object_id=10)
        inter = make_intersection(object_id=100, lane_group_ids=[10], intersection_type=IntersectionType.STOP_SIGN)
        api = write_and_read_map(tmp_path, _meta(), [lg, inter])

        read = api.get_map_object_in_layer(100, MapLayer.INTERSECTION)
        assert isinstance(read, Intersection)
        assert read.intersection_type == IntersectionType.STOP_SIGN
        assert 10 in read.lane_group_ids

    def test_roundtrip_crosswalk(self, tmp_path: Path) -> None:
        cw = make_crosswalk(object_id=5)
        api = write_and_read_map(tmp_path, _meta(), [cw])

        read = api.get_map_object_in_layer(5, MapLayer.CROSSWALK)
        assert isinstance(read, Crosswalk)
        np.testing.assert_allclose(read.outline.array, cw.outline.array, atol=1e-6)

    def test_roundtrip_walkway(self, tmp_path: Path) -> None:
        ww = make_walkway(object_id=6)
        api = write_and_read_map(tmp_path, _meta(), [ww])
        read = api.get_map_object_in_layer(6, MapLayer.WALKWAY)
        assert isinstance(read, Walkway)

    def test_roundtrip_carpark(self, tmp_path: Path) -> None:
        cp = make_carpark(object_id=7)
        api = write_and_read_map(tmp_path, _meta(), [cp])
        read = api.get_map_object_in_layer(7, MapLayer.CARPARK)
        assert isinstance(read, Carpark)

    def test_roundtrip_generic_drivable(self, tmp_path: Path) -> None:
        gd = make_generic_drivable(object_id=8)
        api = write_and_read_map(tmp_path, _meta(), [gd])
        read = api.get_map_object_in_layer(8, MapLayer.GENERIC_DRIVABLE)
        assert isinstance(read, GenericDrivable)

    def test_roundtrip_stop_zone(self, tmp_path: Path) -> None:
        lane = make_lane(object_id=1)
        sz = make_stop_zone(object_id=9, lane_ids=[1], stop_zone_type=StopZoneType.YIELD_SIGN)
        api = write_and_read_map(tmp_path, _meta(), [lane, sz])

        read = api.get_map_object_in_layer(9, MapLayer.STOP_ZONE)
        assert isinstance(read, StopZone)
        assert read.stop_zone_type == StopZoneType.YIELD_SIGN
        assert 1 in read.lane_ids

    def test_roundtrip_road_edge(self, tmp_path: Path) -> None:
        re = make_road_edge(object_id=10, z_values=True)
        api = write_and_read_map(tmp_path, _meta(), [re])

        read = api.get_map_object_in_layer(10, MapLayer.ROAD_EDGE)
        assert isinstance(read, RoadEdge)
        assert read.road_edge_type == RoadEdgeType.ROAD_EDGE_BOUNDARY
        # XY should survive
        np.testing.assert_allclose(read.polyline.array[:, :2], re.polyline.array[:, :2], atol=1e-6)

    def test_roundtrip_road_line(self, tmp_path: Path) -> None:
        rl = make_road_line(object_id=11, z_values=True)
        api = write_and_read_map(tmp_path, _meta(), [rl])

        read = api.get_map_object_in_layer(11, MapLayer.ROAD_LINE)
        assert isinstance(read, RoadLine)
        assert read.road_line_type == RoadLineType.SOLID_WHITE
        np.testing.assert_allclose(read.polyline.array[:, :2], rl.polyline.array[:, :2], atol=1e-6)


# =============================================================================
# ID Remapping stress tests
# =============================================================================


class TestRoundTripIDRemapping:
    """Stress test that ID remapping preserves cross-references."""

    def test_string_ids_become_integers(self, tmp_path: Path) -> None:
        """Write objects with string IDs — after round-trip, IDs should be integers."""
        lane = make_lane(object_id="lane_a", lane_group_id="lg_1")
        lg = make_lane_group(object_id="lg_1", lane_ids=["lane_a"])
        api = write_and_read_map(tmp_path, _meta(), [lane, lg])

        # IDs should be integers after round-trip
        lane_ids = api.get_all_map_object_ids_in_layer(MapLayer.LANE)
        assert len(lane_ids) == 1
        assert isinstance(lane_ids[0], (int, np.integer))

        lg_ids = api.get_all_map_object_ids_in_layer(MapLayer.LANE_GROUP)
        assert len(lg_ids) == 1
        assert isinstance(lg_ids[0], (int, np.integer))

    def test_integer_ids_preserved(self, tmp_path: Path) -> None:
        """Write with explicit int IDs — they should survive unchanged."""
        lane = make_lane(object_id=42)
        api = write_and_read_map(tmp_path, _meta(), [lane])

        ids = api.get_all_map_object_ids_in_layer(MapLayer.LANE)
        assert 42 in ids

    def test_cross_reference_consistency(self, tmp_path: Path) -> None:
        """Lane→LaneGroup→Intersection cross-references must be consistent after remap."""
        lane = make_lane(object_id="L1", lane_group_id="LG1")
        lg = make_lane_group(object_id="LG1", lane_ids=["L1"], intersection_id="INT1")
        inter = make_intersection(object_id="INT1", lane_group_ids=["LG1"])
        api = write_and_read_map(tmp_path, _meta(), [lane, lg, inter])

        # Read all objects
        read_lane = list(api.get_all_map_objects_in_layer(MapLayer.LANE))[0]
        read_lg = list(api.get_all_map_objects_in_layer(MapLayer.LANE_GROUP))[0]
        read_inter = list(api.get_all_map_objects_in_layer(MapLayer.INTERSECTION))[0]

        assert isinstance(read_lane, Lane)
        assert isinstance(read_lg, LaneGroup)
        assert isinstance(read_inter, Intersection)

        # Cross-reference: lane.lane_group_id == lg.object_id
        assert read_lane.lane_group_id == read_lg.object_id

        # Cross-reference: lg.lane_ids contains lane.object_id
        assert read_lane.object_id in read_lg.lane_ids

        # Cross-reference: lg.intersection_id == inter.object_id
        assert read_lg.intersection_id == read_inter.object_id

        # Cross-reference: inter.lane_group_ids contains lg.object_id
        assert read_lg.object_id in read_inter.lane_group_ids

    def test_predecessor_chain_survives_remap(self, tmp_path: Path) -> None:
        """Write 3 lanes in chain A→B→C with string IDs, verify after remap."""
        lane_a = make_lane(object_id="A", successor_ids=["B"], y_offset=0.0)
        lane_b = make_lane(object_id="B", predecessor_ids=["A"], successor_ids=["C"], y_offset=10.0)
        lane_c = make_lane(object_id="C", predecessor_ids=["B"], y_offset=20.0)
        api = write_and_read_map(tmp_path, _meta(), [lane_a, lane_b, lane_c])

        lanes = {lane.object_id: lane for lane in api.get_all_map_objects_in_layer(MapLayer.LANE)}
        assert len(lanes) == 3

        # Find which lane is B (has both predecessor and successor)
        lane_b_read = None
        for lane in lanes.values():
            assert isinstance(lane, Lane)
            if len(lane.predecessor_ids) > 0 and len(lane.successor_ids) > 0:
                lane_b_read = lane
                break
        assert lane_b_read is not None, "Should find lane B with both predecessor and successor"

        # B's predecessor should be a valid lane ID
        assert lane_b_read.predecessor_ids[0] in lanes
        # B's successor should be a valid lane ID
        assert lane_b_read.successor_ids[0] in lanes

    def test_left_right_lane_adjacency_survives_remap(self, tmp_path: Path) -> None:
        """Two adjacent lanes referencing each other via left/right_lane_id."""
        lane_left = make_lane(object_id="LEFT", right_lane_id="RIGHT", y_offset=0.0)
        lane_right = make_lane(object_id="RIGHT", left_lane_id="LEFT", y_offset=10.0)
        api = write_and_read_map(tmp_path, _meta(), [lane_left, lane_right])

        lanes = list(api.get_all_map_objects_in_layer(MapLayer.LANE))
        assert len(lanes) == 2

        all_ids = {lane.object_id for lane in lanes}
        for lane in lanes:
            assert isinstance(lane, Lane)
            if lane.right_lane_id is not None:
                assert lane.right_lane_id in all_ids
            if lane.left_lane_id is not None:
                assert lane.left_lane_id in all_ids

    def test_stop_zone_lane_cross_refs(self, tmp_path: Path) -> None:
        """StopZone.lane_ids should reference actual lane IDs after remap."""
        lane = make_lane(object_id="L1")
        sz = make_stop_zone(object_id="SZ1", lane_ids=["L1"])
        api = write_and_read_map(tmp_path, _meta(), [lane, sz])

        read_lane = list(api.get_all_map_objects_in_layer(MapLayer.LANE))[0]
        read_sz = list(api.get_all_map_objects_in_layer(MapLayer.STOP_ZONE))[0]
        assert isinstance(read_sz, StopZone)
        assert read_lane.object_id in read_sz.lane_ids


# =============================================================================
# Coordinate preservation stress tests
# =============================================================================


class TestRoundTripCoordinatePreservation:
    def test_surface_outline_3d_coordinates(self, tmp_path: Path) -> None:
        """Surface object outline 3D coordinates should survive round-trip."""
        cw = make_crosswalk(object_id=1)
        api = write_and_read_map(tmp_path, _meta(), [cw])

        read = api.get_map_object_in_layer(1, MapLayer.CROSSWALK)
        assert isinstance(read, Crosswalk)
        np.testing.assert_allclose(read.outline.array, cw.outline.array, atol=1e-6)

    def test_lane_boundary_arrays_preserved(self, tmp_path: Path) -> None:
        """Lane boundaries, centerline arrays should match exactly."""
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, _meta(), [lane])

        read = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert isinstance(read, Lane)
        np.testing.assert_allclose(read.left_boundary.array, lane.left_boundary.array, atol=1e-10)
        np.testing.assert_allclose(read.right_boundary.array, lane.right_boundary.array, atol=1e-10)
        np.testing.assert_allclose(read.centerline.array, lane.centerline.array, atol=1e-10)

    def test_road_edge_z_coordinates_preserved(self, tmp_path: Path) -> None:
        """PROBE: RoadEdge 3D Z coordinates.

        The writer stores only WKB for line objects (not raw polyline array).
        The reader reconstructs via Polyline3D.from_linestring(linestring).
        Z coordinates may be lost if shapely doesn't preserve them in WKB.
        """
        n = 5
        xs = np.linspace(0, 30, n)
        ys = np.linspace(0, 10, n)
        zs = np.array([1.5, 2.0, 2.5, 3.0, 3.5])  # non-trivial Z values
        polyline = Polyline3D.from_array(np.column_stack([xs, ys, zs]))
        re = RoadEdge(object_id=1, road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY, polyline=polyline)

        api = write_and_read_map(tmp_path, _meta(), [re])
        read = api.get_map_object_in_layer(1, MapLayer.ROAD_EDGE)
        assert isinstance(read, RoadEdge)

        # XY should always survive
        np.testing.assert_allclose(read.polyline.array[:, :2], polyline.array[:, :2], atol=1e-6)

        # Z coordinates.
        # If Z is lost, this will fail and expose the bug
        np.testing.assert_allclose(
            read.polyline.array[:, 2],
            zs,
            atol=1e-6,
            err_msg="BUG: Road edge Z coordinates lost during round-trip. "
            "Writer stores only WKB, reader reconstructs via from_linestring. "
            "Z may be lost if shapely LineString WKB doesn't preserve Z.",
        )

    def test_bug2_road_line_z_coordinates_preserved(self, tmp_path: Path) -> None:
        """PROBE: Same test for RoadLine."""
        n = 5
        xs = np.linspace(0, 30, n)
        ys = np.linspace(0, 10, n)
        zs = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
        polyline = Polyline3D.from_array(np.column_stack([xs, ys, zs]))
        rl = RoadLine(object_id=1, road_line_type=RoadLineType.DOUBLE_SOLID_YELLOW, polyline=polyline)

        api = write_and_read_map(tmp_path, _meta(), [rl])
        read = api.get_map_object_in_layer(1, MapLayer.ROAD_LINE)
        assert isinstance(read, RoadLine)

        np.testing.assert_allclose(
            read.polyline.array[:, 2],
            zs,
            atol=1e-6,
            err_msg="BUG: Road line Z coordinates lost during round-trip.",
        )

    def test_very_small_coordinates(self, tmp_path: Path) -> None:
        """Stress: very small coordinates (1e-10) — precision loss in msgpack/WKB?"""
        small = 1e-10
        outline = Polyline3D.from_array(
            np.array(
                [
                    [0, 0, 0],
                    [small, 0, 0],
                    [small, small, 0],
                    [0, small, 0],
                    [0, 0, 0],
                ]
            )
        )
        cw = Crosswalk(object_id=1, outline=outline)
        api = write_and_read_map(tmp_path, _meta(), [cw])

        read = api.get_map_object_in_layer(1, MapLayer.CROSSWALK)
        assert isinstance(read, Crosswalk)
        np.testing.assert_allclose(read.outline.array, outline.array, atol=1e-15)

    def test_very_large_coordinates(self, tmp_path: Path) -> None:
        """Stress: very large coordinates (1e8) — float64 should handle this fine."""
        big = 1e8
        outline = Polyline3D.from_array(
            np.array(
                [
                    [big, big, 0],
                    [big + 10, big, 0],
                    [big + 10, big + 10, 0],
                    [big, big + 10, 0],
                    [big, big, 0],
                ]
            )
        )
        cw = Crosswalk(object_id=1, outline=outline)
        api = write_and_read_map(tmp_path, _meta(), [cw])

        read = api.get_map_object_in_layer(1, MapLayer.CROSSWALK)
        assert isinstance(read, Crosswalk)
        np.testing.assert_allclose(read.outline.array, outline.array, atol=1e-2)

    def test_negative_coordinates(self, tmp_path: Path) -> None:
        """Stress: negative coordinates."""
        outline = Polyline3D.from_array(
            np.array(
                [
                    [-10, -10, -1],
                    [10, -10, -1],
                    [10, 10, 1],
                    [-10, 10, 1],
                    [-10, -10, -1],
                ]
            )
        )
        cw = Crosswalk(object_id=1, outline=outline)
        api = write_and_read_map(tmp_path, _meta(), [cw])

        read = api.get_map_object_in_layer(1, MapLayer.CROSSWALK)
        assert isinstance(read, Crosswalk)
        np.testing.assert_allclose(read.outline.array, outline.array, atol=1e-6)

    def test_polygon_area_preserved(self, tmp_path: Path) -> None:
        """Shapely polygon area should be close to original after round-trip."""
        lane = make_lane(object_id=1)
        original_area = lane.shapely_polygon.area
        api = write_and_read_map(tmp_path, _meta(), [lane])

        read = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert isinstance(read, Lane)
        assert read.shapely_polygon.area == pytest.approx(original_area, rel=1e-6)


# =============================================================================
# Graph traversal tests
# =============================================================================


class TestRoundTripGraphTraversal:
    """Test that map_api back-references enable graph traversal after round-trip."""

    def test_full_graph_traversal(self, tmp_path: Path) -> None:
        """Lane → LaneGroup → Intersection → LaneGroup → Lane traversal."""
        lane = make_lane(object_id=1, lane_group_id=10)
        lg = make_lane_group(object_id=10, lane_ids=[1], intersection_id=100)
        inter = make_intersection(object_id=100, lane_group_ids=[10])
        api = write_and_read_map(tmp_path, _meta(), [lane, lg, inter])

        read_lane = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert isinstance(read_lane, Lane)

        # Lane → LaneGroup
        lane_group = read_lane.lane_group
        assert isinstance(lane_group, LaneGroup)
        assert lane_group.object_id == 10

        # LaneGroup → Intersection
        intersection = lane_group.intersection
        assert isinstance(intersection, Intersection)
        assert intersection.object_id == 100

        # Intersection → LaneGroups → Lanes
        lane_groups = intersection.lane_groups
        assert len(lane_groups) == 1
        lanes = lane_groups[0].lanes
        assert len(lanes) == 1
        assert lanes[0].object_id == 1

    def test_left_right_lane_traversal(self, tmp_path: Path) -> None:
        lane_left = make_lane(object_id=1, right_lane_id=2, y_offset=0.0)
        lane_right = make_lane(object_id=2, left_lane_id=1, y_offset=10.0)
        api = write_and_read_map(tmp_path, _meta(), [lane_left, lane_right])

        read_left = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert isinstance(read_left, Lane)
        right = read_left.right_lane
        assert isinstance(right, Lane)
        assert right.object_id == 2

        left_back = right.left_lane
        assert isinstance(left_back, Lane)
        assert left_back.object_id == 1

    def test_predecessor_successor_traversal(self, tmp_path: Path) -> None:
        lane_a = make_lane(object_id=1, successor_ids=[2], y_offset=0.0)
        lane_b = make_lane(object_id=2, predecessor_ids=[1], y_offset=10.0)
        api = write_and_read_map(tmp_path, _meta(), [lane_a, lane_b])

        read_a = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert isinstance(read_a, Lane)
        successors = read_a.successors
        assert len(successors) == 1
        assert successors[0].object_id == 2

        predecessors = successors[0].predecessors
        assert len(predecessors) == 1
        assert predecessors[0].object_id == 1

    def test_orphaned_reference_no_crash(self, tmp_path: Path) -> None:
        """Lane references a lane_group that doesn't exist — should return None, not crash."""
        lane = make_lane(object_id=1, lane_group_id=999)
        api = write_and_read_map(tmp_path, _meta(), [lane])

        read = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert isinstance(read, Lane)
        # lane_group_id was 999, but no lane_group with that ID exists
        # After ID remap, lane_group_id may become None (unknown ID filtered)
        # or it may remain as-is. Either way, lane.lane_group should be None.
        assert read.lane_group is None


# =============================================================================
# Edge cases
# =============================================================================


class TestRoundTripEdgeCases:
    def test_empty_map(self, tmp_path: Path) -> None:
        """Write zero objects, read back — all layers empty."""
        api = write_and_read_map(tmp_path, _meta(), [])
        assert api.get_available_map_layers() == []

    def test_single_object(self, tmp_path: Path) -> None:
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, _meta(), [lane])

        ids = api.get_all_map_object_ids_in_layer(MapLayer.LANE)
        assert len(ids) == 1

    def test_many_objects(self, tmp_path: Path) -> None:
        """Write 50 lanes, verify all retrieved with unique IDs."""
        lanes = [make_lane(object_id=i, y_offset=i * 10.0) for i in range(50)]
        api = write_and_read_map(tmp_path, _meta(), lanes)

        ids = api.get_all_map_object_ids_in_layer(MapLayer.LANE)
        assert len(ids) == 50
        assert len(set(ids)) == 50  # all unique

    def test_lane_with_all_optional_none(self, tmp_path: Path) -> None:
        """Lane with lane_group_id=None, left/right_lane_id=None, empty lists, speed_limit=None."""
        lane = make_lane(
            object_id=1,
            lane_group_id=None,
            left_lane_id=None,
            right_lane_id=None,
            predecessor_ids=[],
            successor_ids=[],
            speed_limit_mps=None,
        )
        api = write_and_read_map(tmp_path, _meta(), [lane])

        read = api.get_map_object_in_layer(1, MapLayer.LANE)
        assert isinstance(read, Lane)
        assert read.lane_group_id is None
        assert read.left_lane_id is None
        assert read.right_lane_id is None
        assert read.predecessor_ids == []
        assert read.successor_ids == []
        assert read.speed_limit_mps is None

    def test_duplicate_ids_in_same_layer_raises(self, tmp_path: Path) -> None:
        """Writer should reject two objects with the same ID in the same layer."""
        lane1 = make_lane(object_id=1, y_offset=0.0, speed_limit_mps=10.0)
        lane2 = make_lane(object_id=1, y_offset=50.0, speed_limit_mps=20.0)

        with pytest.raises(ValueError, match="Duplicate object ID 1 in layer LANE"):
            write_and_read_map(tmp_path, _meta(), [lane1, lane2])

    def test_polyline2d_input_roundtrip(self, tmp_path: Path) -> None:
        """Surface object created with Polyline2D outline — should survive round-trip."""
        outline_2d = Polyline2D.from_array(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]))
        cw = Crosswalk(object_id=1, outline=outline_2d)
        api = write_and_read_map(tmp_path, _meta(), [cw])

        read = api.get_map_object_in_layer(1, MapLayer.CROSSWALK)
        assert isinstance(read, Crosswalk)
        # After round-trip, outline is always 3D (writer zero-pads)
        assert read.outline.array.shape[1] == 3
        np.testing.assert_allclose(read.outline.array[:, :2], outline_2d.array, atol=1e-6)

    def test_surface_from_shapely_polygon_only(self, tmp_path: Path) -> None:
        """Surface created with only shapely_polygon (no explicit outline) — outline inferred."""
        polygon = geom.Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        cw = Crosswalk(object_id=1, shapely_polygon=polygon)
        api = write_and_read_map(tmp_path, _meta(), [cw])

        read = api.get_map_object_in_layer(1, MapLayer.CROSSWALK)
        assert isinstance(read, Crosswalk)
        assert read.outline.array.shape[1] == 3
        # Area should be preserved
        assert read.shapely_polygon.area == pytest.approx(polygon.area, rel=1e-6)

    def test_per_log_map_roundtrip(self, tmp_path: Path) -> None:
        meta = MapMetadata(
            dataset="test",
            location="boston",
            map_has_z=True,
            map_is_per_log=True,
            split="train",
            log_name="log_001",
        )
        api = write_and_read_map(tmp_path, meta, [make_lane(object_id=1)])
        assert api.map_is_per_log
        assert api.dataset == "test"

    def test_global_map_roundtrip(self, tmp_path: Path) -> None:
        meta = MapMetadata(
            dataset="test",
            location="boston",
            map_has_z=True,
            map_is_per_log=False,
        )
        api = write_and_read_map(tmp_path, meta, [make_lane(object_id=1)])
        assert not api.map_is_per_log


# =============================================================================
# Spatial queries after round-trip
# =============================================================================


class TestRoundTripSpatialQueries:
    def test_spatial_query_finds_object(self, tmp_path: Path) -> None:
        lane = make_lane(object_id=1, x_offset=0.0, y_offset=0.0)
        api = write_and_read_map(tmp_path, _meta(), [lane])

        box = geom.box(-5, -5, 25, 10)
        result = api.query(geometry=box, layers=[MapLayer.LANE], predicate="intersects")
        assert len(result[MapLayer.LANE]) >= 1

    def test_radius_query_finds_object(self, tmp_path: Path) -> None:
        lane = make_lane(object_id=1, x_offset=0.0, y_offset=0.0)
        api = write_and_read_map(tmp_path, _meta(), [lane])

        result = api.get_map_objects_in_radius(Point2D(10.0, 1.5), radius=50.0, layers=[MapLayer.LANE])
        assert len(result[MapLayer.LANE]) >= 1

    def test_negative_coordinates_spatial_query(self, tmp_path: Path) -> None:
        """Objects at negative coordinates should be findable."""
        outline = Polyline3D.from_array(
            np.array([[-20, -20, 0], [-10, -20, 0], [-10, -10, 0], [-20, -10, 0], [-20, -20, 0]])
        )
        cw = Crosswalk(object_id=1, outline=outline)
        api = write_and_read_map(tmp_path, _meta(), [cw])

        box = geom.box(-25, -25, -5, -5)
        result = api.query(geometry=box, layers=[MapLayer.CROSSWALK], predicate="intersects")
        assert len(result[MapLayer.CROSSWALK]) >= 1

    def test_overlapping_objects_both_found(self, tmp_path: Path) -> None:
        """Two overlapping lanes should both be found by a spatial query."""
        lane1 = make_lane(object_id=1, x_offset=0.0, y_offset=0.0)
        lane2 = make_lane(object_id=2, x_offset=5.0, y_offset=0.0)  # overlaps with lane1
        api = write_and_read_map(tmp_path, _meta(), [lane1, lane2])

        box = geom.box(0, 0, 25, 5)
        result = api.query(geometry=box, layers=[MapLayer.LANE], predicate="intersects")
        assert len(result[MapLayer.LANE]) == 2

    def test_radius_zero(self, tmp_path: Path) -> None:
        """Radius=0 — likely returns empty even if point is inside a polygon."""
        lane = make_lane(object_id=1)
        api = write_and_read_map(tmp_path, _meta(), [lane])

        result = api.get_map_objects_in_radius(Point2D(10.0, 1.5), radius=0.0, layers=[MapLayer.LANE])
        # With radius=0 the bbox is a point. This may or may not intersect.
        # Just verify it doesn't crash.
        assert isinstance(result, dict)


# =============================================================================
# Metadata round-trip
# =============================================================================


class TestRoundTripMetadata:
    def test_per_log_metadata_all_fields(self, tmp_path: Path) -> None:
        meta = MapMetadata(
            dataset="my_dataset",
            location="SF",
            map_has_z=True,
            map_is_per_log=True,
            split="train_split",
            log_name="log_42",
        )
        api = write_and_read_map(tmp_path, meta, [make_lane(object_id=1)])

        read = api.get_map_metadata()
        assert read.dataset == "my_dataset"
        assert read.location == "SF"
        assert read.map_has_z is True
        assert read.map_is_per_log is True
        assert isinstance(read.version, str)

    def test_global_metadata_none_split_log(self, tmp_path: Path) -> None:
        """Global map: split and log_name are typically None."""
        meta = MapMetadata(
            dataset="global_ds",
            location="NYC",
            map_has_z=False,
            map_is_per_log=False,
        )
        api = write_and_read_map(tmp_path, meta, [make_lane(object_id=1)])

        read = api.get_map_metadata()
        assert read.dataset == "global_ds"
        assert read.location == "NYC"
        assert read.map_has_z is False
        assert read.map_is_per_log is False
