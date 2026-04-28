"""Shared fixtures and factory functions for map API tests."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pytest

from py123d.api.map.arrow.arrow_map_api import ArrowMapAPI
from py123d.api.map.arrow.arrow_map_writer import ArrowMapWriter
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
    SpeedBump,
    StopZone,
    Walkway,
)
from py123d.datatypes.map_objects.base_map_objects import BaseMapObject, MapObjectIDType
from py123d.datatypes.map_objects.map_layer_types import (
    IntersectionType,
    LaneType,
    RoadEdgeType,
    RoadLineType,
    SpeedBumpType,
    StopZoneType,
)
from py123d.geometry import Polyline2D, Polyline3D

# ---------------------------------------------------------------------------
# Helper functions (not fixtures — so tests can customize arguments)
# ---------------------------------------------------------------------------


def make_polyline3d(
    n_points: int = 5,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
) -> Polyline3D:
    """Create a deterministic Polyline3D."""
    xs = np.linspace(0 + x_offset, 10 + x_offset, n_points)
    ys = np.linspace(0 + y_offset, 5 + y_offset, n_points)
    zs = np.linspace(0 + z_offset, 1 + z_offset, n_points)
    return Polyline3D.from_array(np.column_stack([xs, ys, zs]))


def make_polyline2d(
    n_points: int = 5,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> Polyline2D:
    """Create a deterministic Polyline2D."""
    xs = np.linspace(0 + x_offset, 10 + x_offset, n_points)
    ys = np.linspace(0 + y_offset, 5 + y_offset, n_points)
    return Polyline2D.from_array(np.column_stack([xs, ys]))


def make_square_outline(
    cx: float = 5.0,
    cy: float = 5.0,
    size: float = 10.0,
    z: float = 0.0,
) -> Polyline3D:
    """Create a valid closed-polygon Polyline3D (square)."""
    half = size / 2.0
    coords = np.array(
        [
            [cx - half, cy - half, z],
            [cx + half, cy - half, z],
            [cx + half, cy + half, z],
            [cx - half, cy + half, z],
            [cx - half, cy - half, z],  # close the polygon
        ]
    )
    return Polyline3D.from_array(coords)


def make_lane(
    object_id: MapObjectIDType = 0,
    lane_group_id: Optional[MapObjectIDType] = None,
    left_lane_id: Optional[MapObjectIDType] = None,
    right_lane_id: Optional[MapObjectIDType] = None,
    predecessor_ids: Optional[List[MapObjectIDType]] = None,
    successor_ids: Optional[List[MapObjectIDType]] = None,
    speed_limit_mps: Optional[float] = None,
    lane_type: LaneType = LaneType.SURFACE_STREET,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> Lane:
    """Create a Lane with valid parallel boundaries forming a non-degenerate polygon."""
    if predecessor_ids is None:
        predecessor_ids = []
    if successor_ids is None:
        successor_ids = []

    n = 5
    xs = np.linspace(0 + x_offset, 20 + x_offset, n)
    left_ys = np.full(n, 3.0 + y_offset)
    right_ys = np.full(n, 0.0 + y_offset)
    center_ys = np.full(n, 1.5 + y_offset)
    zs = np.linspace(0.0, 0.5, n)

    left_boundary = Polyline3D.from_array(np.column_stack([xs, left_ys, zs]))
    right_boundary = Polyline3D.from_array(np.column_stack([xs, right_ys, zs]))
    centerline = Polyline3D.from_array(np.column_stack([xs, center_ys, zs]))

    return Lane(
        object_id=object_id,
        lane_type=lane_type,
        lane_group_id=lane_group_id,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        centerline=centerline,
        left_lane_id=left_lane_id,
        right_lane_id=right_lane_id,
        predecessor_ids=predecessor_ids,
        successor_ids=successor_ids,
        speed_limit_mps=speed_limit_mps,
    )


def make_lane_group(
    object_id: MapObjectIDType = 0,
    lane_ids: Optional[List[MapObjectIDType]] = None,
    intersection_id: Optional[MapObjectIDType] = None,
    predecessor_ids: Optional[List[MapObjectIDType]] = None,
    successor_ids: Optional[List[MapObjectIDType]] = None,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> LaneGroup:
    """Create a LaneGroup with valid boundaries."""
    if lane_ids is None:
        lane_ids = []
    if predecessor_ids is None:
        predecessor_ids = []
    if successor_ids is None:
        successor_ids = []

    n = 5
    xs = np.linspace(0 + x_offset, 20 + x_offset, n)
    left_ys = np.full(n, 6.0 + y_offset)
    right_ys = np.full(n, 0.0 + y_offset)
    zs = np.linspace(0.0, 0.5, n)

    left_boundary = Polyline3D.from_array(np.column_stack([xs, left_ys, zs]))
    right_boundary = Polyline3D.from_array(np.column_stack([xs, right_ys, zs]))

    return LaneGroup(
        object_id=object_id,
        lane_ids=lane_ids,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        intersection_id=intersection_id,
        predecessor_ids=predecessor_ids,
        successor_ids=successor_ids,
    )


def make_intersection(
    object_id: MapObjectIDType = 0,
    lane_group_ids: Optional[List[MapObjectIDType]] = None,
    intersection_type: IntersectionType = IntersectionType.TRAFFIC_LIGHT,
    cx: float = 10.0,
    cy: float = 10.0,
) -> Intersection:
    """Create an Intersection with a valid outline."""
    if lane_group_ids is None:
        lane_group_ids = []
    outline = make_square_outline(cx, cy, size=20.0)
    return Intersection(
        object_id=object_id,
        intersection_type=intersection_type,
        lane_group_ids=lane_group_ids,
        outline=outline,
    )


def make_crosswalk(object_id: MapObjectIDType = 0, cx: float = 30.0, cy: float = 5.0) -> Crosswalk:
    outline = make_square_outline(cx, cy, size=4.0)
    return Crosswalk(object_id=object_id, outline=outline)


def make_walkway(object_id: MapObjectIDType = 0, cx: float = 40.0, cy: float = 5.0) -> Walkway:
    outline = make_square_outline(cx, cy, size=3.0)
    return Walkway(object_id=object_id, outline=outline)


def make_carpark(object_id: MapObjectIDType = 0, cx: float = 50.0, cy: float = 5.0) -> Carpark:
    outline = make_square_outline(cx, cy, size=8.0)
    return Carpark(object_id=object_id, outline=outline)


def make_generic_drivable(object_id: MapObjectIDType = 0, cx: float = 60.0, cy: float = 5.0) -> GenericDrivable:
    outline = make_square_outline(cx, cy, size=10.0)
    return GenericDrivable(object_id=object_id, outline=outline)


def make_stop_zone(
    object_id: MapObjectIDType = 0,
    lane_ids: Optional[Sequence[MapObjectIDType]] = None,
    stop_zone_type: StopZoneType = StopZoneType.TRAFFIC_LIGHT,
    cx: float = 70.0,
    cy: float = 5.0,
) -> StopZone:
    outline = make_square_outline(cx, cy, size=5.0)
    return StopZone(
        object_id=object_id,
        stop_zone_type=stop_zone_type,
        outline=outline,
        lane_ids=lane_ids,
    )


def make_speed_bump(
    object_id: MapObjectIDType = 0,
    speed_bump_type: SpeedBumpType = SpeedBumpType.UNKNOWN,
    cx: float = 80.0,
    cy: float = 5.0,
) -> SpeedBump:
    outline = make_square_outline(cx, cy, size=5.0)
    return SpeedBump(
        object_id=object_id,
        outline=outline,
        speed_bump_type=speed_bump_type,
    )


def make_road_edge(
    object_id: MapObjectIDType = 0,
    x_offset: float = 0.0,
    y_offset: float = 80.0,
    z_values: bool = True,
) -> RoadEdge:
    """Create a RoadEdge. If z_values=True, uses non-trivial Z coords to test Z-preservation."""
    n = 5
    xs = np.linspace(0 + x_offset, 30 + x_offset, n)
    ys = np.linspace(0 + y_offset, 10 + y_offset, n)
    zs = np.linspace(1.5, 3.5, n) if z_values else np.zeros(n)
    polyline = Polyline3D.from_array(np.column_stack([xs, ys, zs]))
    return RoadEdge(object_id=object_id, road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY, polyline=polyline)


def make_road_line(
    object_id: MapObjectIDType = 0,
    x_offset: float = 0.0,
    y_offset: float = 90.0,
    z_values: bool = True,
) -> RoadLine:
    """Create a RoadLine. If z_values=True, uses non-trivial Z coords to test Z-preservation."""
    n = 5
    xs = np.linspace(0 + x_offset, 30 + x_offset, n)
    ys = np.linspace(0 + y_offset, 10 + y_offset, n)
    zs = np.linspace(2.0, 4.0, n) if z_values else np.zeros(n)
    polyline = Polyline3D.from_array(np.column_stack([xs, ys, zs]))
    return RoadLine(object_id=object_id, road_line_type=RoadLineType.SOLID_WHITE, polyline=polyline)


# ---------------------------------------------------------------------------
# Core round-trip helper
# ---------------------------------------------------------------------------


def write_and_read_map(
    tmp_path: Path,
    metadata: MapMetadata,
    objects: List[BaseMapObject],
) -> ArrowMapAPI:
    """Write map objects with ArrowMapWriter, then read back with ArrowMapAPI."""
    maps_root = tmp_path / "maps"
    logs_root = tmp_path / "logs"
    maps_root.mkdir(exist_ok=True)
    logs_root.mkdir(exist_ok=True)

    writer = ArrowMapWriter(force_map_conversion=True, maps_root=maps_root, logs_root=logs_root)
    needs_writing = writer.reset(metadata)
    assert needs_writing, "Writer should indicate map needs writing"

    for obj in objects:
        writer.write_map_object(obj)
    writer.close()

    # Determine written file path
    if metadata.map_is_per_log:
        assert metadata.split is not None and metadata.log_name is not None, (
            "Per-log maps require split and log_name in metadata"
        )
        arrow_file = logs_root / metadata.split / metadata.log_name / "map.arrow"
    else:
        arrow_file = maps_root / metadata.dataset / f"{metadata.dataset}_{metadata.location}.arrow"

    assert arrow_file.exists(), f"Arrow file was not created at {arrow_file}"
    return ArrowMapAPI(arrow_file)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def per_log_map_metadata() -> MapMetadata:
    return MapMetadata(
        dataset="test_dataset",
        location="boston",
        map_has_z=True,
        map_is_per_log=True,
        split="test_train",
        log_name="log_001",
    )


@pytest.fixture
def global_map_metadata() -> MapMetadata:
    return MapMetadata(
        dataset="test_dataset",
        location="boston",
        map_has_z=True,
        map_is_per_log=False,
    )
