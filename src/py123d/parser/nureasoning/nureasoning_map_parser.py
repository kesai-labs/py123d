from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

from py123d.datatypes import (
    BaseMapObject,
    Crosswalk,
    Intersection,
    IntersectionType,
    Lane,
    LaneGroup,
    LaneType,
    MapMetadata,
    RoadEdge,
    RoadEdgeType,
    RoadLine,
    RoadLineType,
    StopZone,
    StopZoneType,
    TrafficLight,
)
from py123d.geometry import OccupancyMap2D, Point2D, Polyline2D, PoseSE3
from py123d.geometry.utils.polyline_utils import offset_points_perpendicular
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.nureasoning.utils.nureasoning_constants import (
    NUREASONING_INTERSECTION_TYPE_CONVERSION,
    NUREASONING_LANE_CONNECTOR_HALF_WIDTH,
    NUREASONING_LANE_TYPE_CONVERSION,
    NUREASONING_MAX_ROAD_EDGE_LENGTH,
    NUREASONING_ROAD_LINE_CONVERSION,
)
from py123d.parser.nureasoning.utils.nureasoning_map_utils import (
    order_lanes_left_to_right,
    split_polygon_into_boundaries,
)
from py123d.parser.nureasoning.utils.nureasoning_schema import load_schema_pickle, nuReasoningStaticMap
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)

logger = logging.getLogger(__name__)


class NureasoningMapParser(BaseMapParser):
    """Map parser for a single nuReasoning clip.

    nuReasoning maps are per-log: each clip ships a ``map.pkl`` that unpickles to a
    :class:`~py123d.parser.nureasoning.utils.nureasoning_schema.nuReasoningStaticMap`. All map geometry has
    ``z == 0``, so the map is stored in 2D (``map_has_z=False``).
    """

    def __init__(self, split: str, log_name: str, source_log_path: Path) -> None:
        self._split = split
        self._log_name = log_name
        self._source_log_path = source_log_path

    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return MapMetadata(
            dataset="nureasoning",
            split=self._split,
            log_name=self._log_name,
            location=self._get_clip_location(),
            map_has_z=False,
            map_is_per_log=True,
        )

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        static_map = load_schema_pickle(self._source_log_path / "map.pkl")
        assert isinstance(static_map, nuReasoningStaticMap), f"Expected nuReasoningStaticMap, got {type(static_map)}"

        lanes = _extract_nureasoning_lanes(static_map)
        lane_connectors = _extract_nureasoning_lane_connectors(static_map)
        intersections, connector_group_to_intersection = _extract_nureasoning_intersections(static_map, lane_connectors)
        lane_groups = _extract_nureasoning_lane_groups(lanes, lane_connectors, connector_group_to_intersection)
        road_edges = _extract_nureasoning_road_edges(static_map)
        road_lines = _extract_nureasoning_road_lines(static_map)
        crosswalks = _extract_nureasoning_crosswalks(static_map)
        stop_zones = _extract_nureasoning_stop_zones(static_map)
        traffic_lights = _extract_nureasoning_traffic_lights(static_map)

        yield from (
            lanes
            + lane_connectors
            + lane_groups
            + intersections
            + road_edges
            + road_lines
            + crosswalks
            + stop_zones
            + traffic_lights
        )

    def _get_clip_location(self) -> Optional[str]:
        """Reads the (normalized) clip location from the log's ``metadata.json``, if available."""
        location: Optional[str] = None
        metadata_json_path = self._source_log_path / "metadata.json"
        if metadata_json_path.is_file():
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                metadata_json = json.load(f)
            raw_location = metadata_json.get("clip_location", None)
            location = raw_location.replace(" ", "-") if raw_location else None
        return location


# ------------------------------------------------------------------------------------------------------------------
# Map object extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _extract_nureasoning_lanes(static_map: nuReasoningStaticMap) -> List[Lane]:
    """Extract regular (non-intersection) lanes, deriving left/right boundaries from the lane polygon.

    nuReasoning lanes provide only a closed polygon and a centerline, so the boundaries are derived by
    splitting the polygon around the centerline. Lanes with a degenerate centerline or an ambiguous split
    are skipped.
    """
    lanes: List[Lane] = []
    for lane in static_map.lanes:
        polygon_array = np.asarray(lane.polygon, dtype=np.float64)
        centerline_array = np.asarray(lane.centerline, dtype=np.float64) if lane.centerline is not None else None

        has_valid_geometry = (
            centerline_array is not None
            and centerline_array.ndim == 2
            and centerline_array.shape[0] >= 2
            and polygon_array.ndim == 2
            and polygon_array.shape[0] >= 4
        )
        if has_valid_geometry:
            centerline_xy = centerline_array[:, :2]
            polygon_xy = polygon_array[:, :2]
            # nuReasoning polygons are closed rings; drop the closing duplicate vertex before splitting.
            if np.allclose(polygon_xy[0], polygon_xy[-1]):
                polygon_xy = polygon_xy[:-1]

            left_boundary, right_boundary = split_polygon_into_boundaries(polygon_xy, centerline_xy)
            if left_boundary is not None and right_boundary is not None:
                lanes.append(
                    Lane(
                        object_id=int(lane.id),
                        lane_type=NUREASONING_LANE_TYPE_CONVERSION.get(lane.lane_type, LaneType.UNDEFINED),
                        left_boundary=left_boundary,
                        right_boundary=right_boundary,
                        centerline=Polyline2D.from_array(centerline_xy),
                        lane_group_id=int(lane.lane_group) if lane.lane_group is not None else None,
                        # NOTE: nuReasoning provides no lane topology; left/right neighbors and
                        # predecessors/successors are not available (lane_index direction is ambiguous).
                        left_lane_id=None,
                        right_lane_id=None,
                        predecessor_ids=[],
                        successor_ids=[],
                        speed_limit_mps=lane.speed_limit,
                        outline=None,
                        shapely_polygon=None,
                    )
                )

    return lanes


def _extract_nureasoning_lane_connectors(static_map: nuReasoningStaticMap) -> List[Lane]:
    """Extract intersection lane connectors as lanes, synthesizing boundaries via perpendicular offset.

    Lane connectors ship only a centerline (``geometry``), so left/right boundaries are offset from it by a
    fixed half-width (mirroring the nuScenes lane-connector handling).
    """
    lane_connectors: List[Lane] = []
    for connector in static_map.lane_connectors:
        centerline_array = np.asarray(connector.geometry, dtype=np.float64)
        if centerline_array.ndim == 2 and centerline_array.shape[0] >= 2:
            centerline_xy = centerline_array[:, :2]
            left_points = offset_points_perpendicular(centerline_xy, offset=NUREASONING_LANE_CONNECTOR_HALF_WIDTH)
            right_points = offset_points_perpendicular(centerline_xy, offset=-NUREASONING_LANE_CONNECTOR_HALF_WIDTH)

            lane_group_id = (
                int(connector.lane_group_connector_id) if connector.lane_group_connector_id is not None else None
            )
            lane_connectors.append(
                Lane(
                    object_id=int(connector.id),
                    lane_type=LaneType.UNDEFINED,
                    left_boundary=Polyline2D.from_array(left_points),
                    right_boundary=Polyline2D.from_array(right_points),
                    centerline=Polyline2D.from_array(centerline_xy),
                    lane_group_id=lane_group_id,
                    left_lane_id=None,
                    right_lane_id=None,
                    predecessor_ids=[],
                    successor_ids=[],
                    speed_limit_mps=connector.speed_limit,
                    outline=None,
                    shapely_polygon=None,
                )
            )

    return lane_connectors


def _extract_nureasoning_intersections(
    static_map: nuReasoningStaticMap, lane_connectors: List[Lane]
) -> Tuple[List[Intersection], Dict[int, int]]:
    """Extract intersections and assign connector lane-groups to them via spatial containment.

    nuReasoning's ``lane_connector.intersection_id`` lives in a different id space than ``intersection.id``,
    so the linkage is recovered spatially: a connector belongs to the intersection whose polygon contains its
    centerline midpoint (mirroring the nuScenes parser).

    :return: the intersections, and a ``lane_group_connector_id -> intersection_id`` assignment map.
    """
    # Midpoint of each connector centerline, plus a connector-id -> lane-group-id lookup.
    connector_midpoints: Dict[int, Point] = {}
    connector_to_group: Dict[int, int] = {}
    for connector in lane_connectors:
        if connector.lane_group_id is not None:
            midpoint = connector.centerline.interpolate(0.5, normalized=True)
            assert isinstance(midpoint, Point2D)
            connector_midpoints[connector.object_id] = midpoint.shapely_point
            connector_to_group[connector.object_id] = connector.lane_group_id

    occupancy_map = OccupancyMap2D.from_dict(connector_midpoints) if connector_midpoints else None

    intersections: List[Intersection] = []
    connector_group_to_intersection: Dict[int, int] = {}
    for intersection in static_map.intersections:
        geometry = np.asarray(intersection.geometry, dtype=np.float64)
        if geometry.ndim == 2 and geometry.shape[0] >= 3:
            intersection_id = int(intersection.id)
            polygon = Polygon(geometry[:, :2])

            lane_group_ids: List[int] = []
            if occupancy_map is not None and polygon.is_valid:
                inside_connector_ids = occupancy_map.intersects(polygon)
                lane_group_ids = sorted({connector_to_group[connector_id] for connector_id in inside_connector_ids})
                for lane_group_id in lane_group_ids:
                    connector_group_to_intersection[lane_group_id] = intersection_id

            intersections.append(
                Intersection(
                    object_id=intersection_id,
                    intersection_type=NUREASONING_INTERSECTION_TYPE_CONVERSION.get(
                        intersection.intersection_type, IntersectionType.DEFAULT
                    ),
                    lane_group_ids=lane_group_ids,
                    outline=None,
                    shapely_polygon=polygon,
                )
            )

    return intersections, connector_group_to_intersection


def _extract_nureasoning_lane_groups(
    lanes: List[Lane],
    lane_connectors: List[Lane],
    connector_group_to_intersection: Dict[int, int],
) -> List[LaneGroup]:
    """Derive lane groups by grouping lanes / connectors that share a lane-group id.

    For multi-lane groups, the group boundaries are the leftmost lane's left boundary and the rightmost
    lane's right boundary (ordered geometrically). Connector groups inherit their intersection from the
    spatial assignment computed in :func:`_extract_nureasoning_intersections`.
    """
    lanes_by_id: Dict[int, Lane] = {lane.object_id: lane for lane in lanes + lane_connectors}

    group_to_lane_ids: Dict[int, List[int]] = defaultdict(list)
    for lane in lanes + lane_connectors:
        if lane.lane_group_id is not None:
            group_to_lane_ids[lane.lane_group_id].append(lane.object_id)

    lane_groups: List[LaneGroup] = []
    for lane_group_id, lane_ids in group_to_lane_ids.items():
        if len(lane_ids) > 1:
            centerlines = [lanes_by_id[lane_id].centerline for lane_id in lane_ids]
            ordered = order_lanes_left_to_right(centerlines)
            left_boundary = lanes_by_id[lane_ids[ordered[0]]].left_boundary
            right_boundary = lanes_by_id[lane_ids[ordered[-1]]].right_boundary
        else:
            single_lane = lanes_by_id[lane_ids[0]]
            left_boundary = single_lane.left_boundary
            right_boundary = single_lane.right_boundary

        lane_groups.append(
            LaneGroup(
                object_id=lane_group_id,
                lane_ids=lane_ids,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                intersection_id=connector_group_to_intersection.get(lane_group_id),
                predecessor_ids=[],
                successor_ids=[],
                outline=None,
                shapely_polygon=None,
            )
        )

    return lane_groups


def _extract_nureasoning_road_edges(static_map: nuReasoningStaticMap) -> List[RoadEdge]:
    """Derive road edges from the outlines of the drivable areas (road blocks and intersections)."""
    drivable_polygons: List[Polygon] = []
    for element in list(static_map.road_blocks) + list(static_map.intersections):
        geometry = np.asarray(element.geometry, dtype=np.float64)
        if geometry.ndim == 2 and geometry.shape[0] >= 3:
            polygon = Polygon(geometry[:, :2])
            if polygon.is_valid:
                drivable_polygons.append(polygon)

    road_edges: List[RoadEdge] = []
    if drivable_polygons:
        road_edge_linear_rings = get_road_edge_linear_rings(drivable_polygons)
        road_edge_linestrings = split_line_geometry_by_max_length(
            road_edge_linear_rings, NUREASONING_MAX_ROAD_EDGE_LENGTH
        )
        for idx, linestring in enumerate(road_edge_linestrings):
            road_edges.append(
                RoadEdge(
                    object_id=idx,
                    road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
                    polyline=Polyline2D.from_linestring(linestring),
                )
            )

    return road_edges


def _extract_nureasoning_road_lines(static_map: nuReasoningStaticMap) -> List[RoadLine]:
    """Extract road lines (lane / road boundaries) from the map boundaries.

    A running index is used as the object id: the source boundary ids are 64-bit values that are not
    cross-referenced by any other map object.
    """
    road_lines: List[RoadLine] = []
    for idx, boundary in enumerate(static_map.boundaries):
        geometry = np.asarray(boundary.geometry, dtype=np.float64)
        if geometry.ndim == 2 and geometry.shape[0] >= 2:
            road_lines.append(
                RoadLine(
                    object_id=idx,
                    road_line_type=NUREASONING_ROAD_LINE_CONVERSION.get(boundary.type, RoadLineType.UNKNOWN),
                    polyline=Polyline2D.from_array(geometry[:, :2]),
                )
            )

    return road_lines


def _extract_nureasoning_crosswalks(static_map: nuReasoningStaticMap) -> List[Crosswalk]:
    """Extract crosswalks from the map crosswalk polygons."""
    crosswalks: List[Crosswalk] = []
    for crosswalk in static_map.crosswalks:
        geometry = np.asarray(crosswalk.geometry, dtype=np.float64)
        if geometry.ndim == 2 and geometry.shape[0] >= 3:
            crosswalks.append(Crosswalk(object_id=int(crosswalk.id), shapely_polygon=Polygon(geometry[:, :2])))

    return crosswalks


def _extract_nureasoning_stop_zones(static_map: nuReasoningStaticMap) -> List[StopZone]:
    """Extract stop zones from the map stop polygons.

    The source carries no stop-zone type, so all stop zones use :attr:`StopZoneType.UNKNOWN`.
    """
    stop_zones: List[StopZone] = []
    for stop_polygon in static_map.stop_polygons:
        geometry = np.asarray(stop_polygon.geometry, dtype=np.float64)
        if geometry.ndim == 2 and geometry.shape[0] >= 3:
            stop_zones.append(
                StopZone(
                    object_id=int(stop_polygon.id),
                    stop_zone_type=StopZoneType.UNKNOWN,
                    shapely_polygon=Polygon(geometry[:, :2]),
                )
            )

    return stop_zones


def _extract_nureasoning_traffic_lights(static_map: nuReasoningStaticMap) -> List[TrafficLight]:
    """Extract traffic lights from the map.

    The source provides a single 3D point per light (the height is above ground, or 0 in some clips) and no
    orientation, extent or lane association, so the default housing dimensions are used and ``lane_ids`` is empty.
    """
    traffic_lights: List[TrafficLight] = []
    for traffic_light in static_map.traffic_lights:
        geometry = np.asarray(traffic_light.geometry, dtype=np.float64)
        if geometry.ndim != 2 or geometry.shape[0] == 0:
            continue
        x, y, z = geometry[0, :3] if geometry.shape[1] >= 3 else (*geometry[0, :2], 0.0)
        traffic_lights.append(
            TrafficLight.from_pose(
                object_id=int(traffic_light.id),
                pose_se3=PoseSE3(float(x), float(y), float(z), 1.0, 0.0, 0.0, 0.0),
            )
        )
    return traffic_lights
