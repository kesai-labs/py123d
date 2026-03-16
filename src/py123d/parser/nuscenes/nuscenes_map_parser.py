from collections import defaultdict
from pathlib import Path
from typing import Dict, Final, Iterator, List, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    BaseMapObject,
    Carpark,
    Crosswalk,
    GenericDrivable,
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
    Walkway,
)
from py123d.geometry import OccupancyMap2D, Point2D, Polyline2D, Polyline3D
from py123d.geometry.utils.polyline_utils import offset_points_perpendicular
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.nuscenes.utils.nuscenes_constants import NUSCENES_LANE_TYPE_MAPPING, NUSCENES_MAP_LOCATIONS
from py123d.parser.nuscenes.utils.nuscenes_map_utils import (
    extract_lane_and_boundaries,
    extract_nuscenes_centerline,
    order_lanes_left_to_right,
)
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
    split_polygon_by_grid,
)

check_dependencies(["nuscenes"], optional_name="nuscenes")
from nuscenes.map_expansion.map_api import NuScenesMap

# TODO @DanielDauner: Add to config.
MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # [m]
MAX_LANE_WIDTH: Final[float] = 4.0  # [m]
MIN_LANE_WIDTH: Final[float] = 1.0  # [m]


class NuScenesMapParser(BaseMapParser):
    """Map parser for nuScenes maps, using the nuscenes-devkit's NuScenesMap API."""

    def __init__(self, nuscenes_maps_root: Path, location: str) -> None:
        assert location in NUSCENES_MAP_LOCATIONS, (
            f"Map name {location} is not supported. Supported maps: {NUSCENES_MAP_LOCATIONS}"
        )
        self._nuscenes_maps_root = nuscenes_maps_root
        self._location = location

    def get_map_metadata(self) -> MapMetadata:
        return MapMetadata(
            dataset="nuscenes",
            split=None,
            log_name=None,
            location=self._location,
            map_has_z=False,
            map_is_per_log=False,
        )

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        nuscenes_map = NuScenesMap(dataroot=str(self._nuscenes_maps_root), map_name=self._location)

        # 1. extract road edges (used later to determine lane connector widths)
        road_edges = _extract_nuscenes_road_edges(nuscenes_map)

        # 2. extract lanes
        lanes = _extract_nuscenes_lanes(nuscenes_map)

        # 3. extract lane connectors (i.e. lanes on intersections)
        lane_connectors = _extract_nuscenes_lane_connectors(nuscenes_map, road_edges)

        # 4. extract intersections (and store lane-connector to intersection assignment for lane groups)
        intersections, intersection_assignment = _extract_intersections_and_assignment(nuscenes_map, lane_connectors)

        # 5. extract lane groups
        lane_groups = _extract_nuscenes_lane_groups(lanes, lane_connectors, intersection_assignment)

        # Yield all map objects
        yield from lanes
        yield from lane_connectors
        yield from road_edges
        yield from intersections
        yield from lane_groups
        yield from _extract_nuscenes_crosswalks(nuscenes_map)
        yield from _extract_nuscenes_walkways(nuscenes_map)
        yield from _extract_nuscenes_carparks(nuscenes_map)
        yield from _extract_nuscenes_generic_drivables(nuscenes_map)
        yield from _extract_nuscenes_stop_zones(nuscenes_map)
        yield from _extract_nuscenes_road_lines(nuscenes_map)


def _extract_nuscenes_lanes(nuscenes_map: NuScenesMap) -> List[Lane]:
    """Helper function to extract lanes from a nuScenes map."""

    # NOTE: nuScenes does not explicitly provide lane groups and does not assign lanes to roadblocks.
    # Therefore, we query the roadblocks given the middle-point of the centerline to assign lanes to a road block.
    # Unlike road segments, road blocks outline a lane group going in the same direction.
    # In case a roadblock cannot be assigned, e.g. because the lane is not located within any roadblock, or the
    # roadblock data is invalid [1], we assign a new lane group with only this lane.
    # [1] https://github.com/nutonomy/nuscenes-devkit/issues/862

    road_blocks_invalid = nuscenes_map.map_name in {"singapore-queenstown", "singapore-hollandvillage"}

    road_block_dict: Dict[str, Polygon] = {}
    if not road_blocks_invalid:
        road_block_dict: Dict[str, Polygon] = {
            road_block["token"]: nuscenes_map.extract_polygon(road_block["polygon_token"])
            for road_block in nuscenes_map.road_block
        }

    road_block_map = OccupancyMap2D.from_dict(road_block_dict)  # type: ignore
    lanes: List[Lane] = []
    for lane_record in nuscenes_map.lane:
        token = lane_record["token"]

        # 1. Extract centerline and boundaries
        centerline, left_boundary, right_boundary = extract_lane_and_boundaries(nuscenes_map, lane_record)

        if left_boundary is None or right_boundary is None:
            continue  # skip lanes without valid boundaries

        # 2. Query road block for lane group assignment
        lane_group_id: str = token  # default to self, override if road block found
        if not road_blocks_invalid:
            query_point = centerline.interpolate(0.5, normalized=True).shapely_point  # type: ignore
            intersecting_roadblock = road_block_map.query_nearest(query_point, max_distance=0.1, all_matches=False)

            # NOTE: if a lane cannot be assigned to a road block, we assume a new lane group with only this lane.
            # The lane group id is set to be the same as the lane id in this case.
            if len(intersecting_roadblock) > 0:
                lane_group_id = road_block_map.ids[intersecting_roadblock[0]]  # type: ignore

        # Get topology
        incoming = nuscenes_map.get_incoming_lane_ids(token)
        outgoing = nuscenes_map.get_outgoing_lane_ids(token)

        lane_type = NUSCENES_LANE_TYPE_MAPPING.get(lane_record.get("lane_type", ""), LaneType.UNDEFINED)

        lanes.append(
            Lane(
                object_id=token,
                lane_type=lane_type,
                lane_group_id=lane_group_id,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                centerline=centerline,
                left_lane_id=None,
                right_lane_id=None,
                predecessor_ids=incoming,  # type: ignore
                successor_ids=outgoing,  # type: ignore
                speed_limit_mps=None,
                outline=None,
                shapely_polygon=None,
            )
        )

    return lanes


def _extract_nuscenes_lane_connectors(nuscenes_map: NuScenesMap, road_edges: List[RoadEdge]) -> List[Lane]:
    """Helper function to extract lane connectors from a nuScenes map."""

    # TODO @DanielDauner: consider using connected lanes to estimate the lane width

    road_edge_map = OccupancyMap2D(geometries=[road_edge.shapely_linestring for road_edge in road_edges])
    lane_connectors: List[Lane] = []
    for lane_record in nuscenes_map.lane_connector:
        lane_connector_token: str = lane_record["token"]

        centerline = extract_nuscenes_centerline(nuscenes_map, lane_record)

        _, nearest_edge_distances = road_edge_map.query_nearest(
            centerline.linestring, return_distance=True, all_matches=False
        )
        road_edge_distance = nearest_edge_distances[0] if nearest_edge_distances else float("inf")

        lane_half_width = np.clip(road_edge_distance, MIN_LANE_WIDTH / 2.0, MAX_LANE_WIDTH / 2.0)

        left_pts = offset_points_perpendicular(centerline.array, offset=lane_half_width)
        right_pts = offset_points_perpendicular(centerline.array, offset=-lane_half_width)

        predecessor_ids = nuscenes_map.get_incoming_lane_ids(lane_connector_token)
        successor_ids = nuscenes_map.get_outgoing_lane_ids(lane_connector_token)

        lane_group_id = lane_connector_token
        lane_connectors.append(
            Lane(
                object_id=lane_connector_token,
                lane_type=LaneType.UNDEFINED,
                lane_group_id=lane_group_id,
                left_boundary=Polyline2D.from_array(left_pts),
                right_boundary=Polyline2D.from_array(right_pts),
                centerline=centerline,
                left_lane_id=None,  # Not directly available in nuscenes
                right_lane_id=None,  # Not directly available in nuscenes
                predecessor_ids=predecessor_ids,  # type: ignore
                successor_ids=successor_ids,  # type: ignore
                speed_limit_mps=None,  # Default value
                outline=None,
                shapely_polygon=None,
            )
        )

    return lane_connectors


def _extract_nuscenes_lane_groups(
    lanes: List[Lane],
    lane_connectors: List[Lane],
    intersection_assignment: Dict[str, int],
) -> List[LaneGroup]:
    """Helper function to extract lane groups from a nuScenes map."""

    lane_groups = []
    lanes_dict = {lane.object_id: lane for lane in lanes + lane_connectors}

    # 1. Gather all lane group ids that were previously assigned in the lanes (either roadblocks of lane themselves)
    lane_group_lane_dict: Dict[str, List[str]] = defaultdict(list)
    for lane in lanes + lane_connectors:
        if lane.lane_group_id is not None:
            lane_group_lane_dict[lane.lane_group_id].append(lane.object_id)  # type: ignore

    for lane_group_id, lane_ids in lane_group_lane_dict.items():
        if len(lane_ids) > 1:
            lane_centerlines: List[Polyline2D] = [lanes_dict[lane_id].centerline for lane_id in lane_ids]  # type: ignore
            ordered_lane_indices = order_lanes_left_to_right(lane_centerlines)
            left_boundary = lanes_dict[lane_ids[ordered_lane_indices[0]]].left_boundary
            right_boundary = lanes_dict[lane_ids[ordered_lane_indices[-1]]].right_boundary

        else:
            lane_id = lane_ids[0]
            lane = lanes_dict[lane_id]
            left_boundary = lane.left_boundary
            right_boundary = lane.right_boundary

        # 2. For each lane group, gather predecessor and successor lane groups
        predecessor_ids = set()
        successor_ids = set()
        for lane_id in lane_ids:
            lane = lanes_dict[lane_id]
            if lane is None:
                continue
            for pred_id in lane.predecessor_ids:
                pred_lane = lanes_dict.get(pred_id)
                if pred_lane is not None and pred_lane.lane_group_id is not None:
                    predecessor_ids.add(pred_lane.lane_group_id)
            for succ_id in lane.successor_ids:
                succ_lane = lanes_dict.get(succ_id)
                if succ_lane is not None and succ_lane.lane_group_id is not None:
                    successor_ids.add(succ_lane.lane_group_id)

        intersection_ids = set(
            [int(intersection_assignment[lane_id]) for lane_id in lane_ids if lane_id in intersection_assignment]
        )
        assert len(intersection_ids) <= 1, "A lane group cannot belong to multiple intersections."
        intersection_id = None if len(intersection_ids) == 0 else intersection_ids.pop()

        lane_groups.append(
            LaneGroup(
                object_id=lane_group_id,
                lane_ids=lane_ids,  # type: ignore
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                intersection_id=intersection_id,
                predecessor_ids=list(predecessor_ids),
                successor_ids=list(successor_ids),
                outline=None,
                shapely_polygon=None,
            )
        )

    return lane_groups


def _extract_intersections_and_assignment(
    nuscenes_map: NuScenesMap, lane_connectors: List[Lane]
) -> Tuple[List[Intersection], Dict[str, int]]:
    """Return intersection data and lane-connector to intersection assignment."""

    intersections: List[Intersection] = []
    intersection_assignment = {}

    # 1. Extract intersections and corresponding polygons
    intersection_polygons = []
    for road_segment in nuscenes_map.road_segment:
        if road_segment["is_intersection"]:
            if "polygon_token" in road_segment:
                polygon = nuscenes_map.extract_polygon(road_segment["polygon_token"])
                intersection_polygons.append(polygon)

    # 2. Find lane connectors within each intersection polygon
    # For this, we collect all mid-points of lane connects and check whether they are located in a intersection polygon.
    lane_connector_center_point_dict = {}
    for lane_connector in lane_connectors:
        lane_connector_midpoint = lane_connector.centerline.interpolate(0.5, normalized=True)
        assert isinstance(lane_connector_midpoint, Point2D)
        lane_connector_center_point_dict[lane_connector.object_id] = lane_connector_midpoint.shapely_point

    centerpoint_map = OccupancyMap2D.from_dict(lane_connector_center_point_dict)  # type: ignore
    for idx, intersection_polygon in enumerate(intersection_polygons):
        intersecting_lane_connector_ids = centerpoint_map.intersects(intersection_polygon)
        for lane_connector_id in intersecting_lane_connector_ids:
            intersection_assignment[lane_connector_id] = idx

        intersections.append(
            Intersection(
                object_id=idx,
                intersection_type=IntersectionType.DEFAULT,
                lane_group_ids=intersecting_lane_connector_ids,  # type: ignore
                outline=None,
                shapely_polygon=intersection_polygon,
            )
        )

    return intersections, intersection_assignment


def _extract_nuscenes_crosswalks(nuscenes_map: NuScenesMap) -> List[Crosswalk]:
    """Extract crosswalk data from a nuScenes map."""
    crosswalks: List[Crosswalk] = []
    for idx, crossing in enumerate(nuscenes_map.ped_crossing):
        if "polygon_token" in crossing:
            polygon = nuscenes_map.extract_polygon(crossing["polygon_token"])
            crosswalks.append(Crosswalk(object_id=idx, shapely_polygon=polygon))
    return crosswalks


def _extract_nuscenes_walkways(nuscenes_map: NuScenesMap) -> List[Walkway]:
    """Extract walkway data from a nuScenes map."""
    walkways: List[Walkway] = []
    for idx, walkway_record in enumerate(nuscenes_map.walkway):
        if "polygon_token" in walkway_record:
            polygon = nuscenes_map.extract_polygon(walkway_record["polygon_token"])
            walkways.append(Walkway(object_id=idx, shapely_polygon=polygon))
    return walkways


def _extract_nuscenes_carparks(nuscenes_map: NuScenesMap) -> List[Carpark]:
    """Extract carpark data from a nuScenes map."""
    carparks: List[Carpark] = []
    for idx, carpark_record in enumerate(nuscenes_map.carpark_area):
        if "polygon_token" in carpark_record:
            polygon = nuscenes_map.extract_polygon(carpark_record["polygon_token"])
            carparks.append(Carpark(object_id=idx, shapely_polygon=polygon))
    return carparks


def _extract_nuscenes_generic_drivables(nuscenes_map: NuScenesMap) -> List[GenericDrivable]:
    """Extract generic drivable area data from a nuScenes map."""
    cell_size = 20.0
    drivable_polygons = []
    for drivable_area_record in nuscenes_map.drivable_area:
        drivable_area = nuscenes_map.get("drivable_area", drivable_area_record["token"])
        for polygon_token in drivable_area["polygon_tokens"]:
            polygon = nuscenes_map.extract_polygon(polygon_token)
            split_polygons = split_polygon_by_grid(polygon, cell_size=cell_size)
            drivable_polygons.extend(split_polygons)

    return [GenericDrivable(object_id=idx, shapely_polygon=geometry) for idx, geometry in enumerate(drivable_polygons)]


def _extract_nuscenes_stop_zones(nuscenes_map: NuScenesMap) -> List[StopZone]:
    """Extract stop zone data from a nuScenes map."""
    NUSCENES_STOP_CUES_TO_STOP_ZONE_TYPE = {
        "PED_CROSSING": StopZoneType.PEDESTRIAN_CROSSING,
        "TURN_STOP": StopZoneType.TURN_STOP,
        "TRAFFIC_LIGHT": StopZoneType.TRAFFIC_LIGHT,
        "STOP_SIGN": StopZoneType.STOP_SIGN,
        "YIELD": StopZoneType.YIELD_SIGN,
    }
    stop_zones: List[StopZone] = []
    for stop_line in nuscenes_map.stop_line:
        token = stop_line["token"]
        if "polygon_token" not in stop_line:
            continue
        polygon = nuscenes_map.extract_polygon(stop_line["polygon_token"])
        if not polygon.is_valid:
            continue

        if "stop_line_type" in stop_line.keys():
            stop_zone_type = NUSCENES_STOP_CUES_TO_STOP_ZONE_TYPE.get(stop_line["stop_line_type"], StopZoneType.UNKNOWN)
        else:
            stop_zone_type = StopZoneType.UNKNOWN

        stop_zones.append(
            StopZone(
                object_id=token,
                stop_zone_type=stop_zone_type,
                shapely_polygon=polygon,
            )
        )
    return stop_zones


def _extract_nuscenes_road_lines(nuscenes_map: NuScenesMap) -> List[RoadLine]:
    """Extract road line data (dividers) from a nuScenes map."""
    line_token_to_type = _build_line_token_to_type_mapping(nuscenes_map)

    road_lines: List[RoadLine] = []
    running_idx = 0

    # Process road dividers
    for divider in nuscenes_map.road_divider:
        line = nuscenes_map.extract_line(divider["line_token"])
        line_type = _get_road_line_type(divider["line_token"], line_token_to_type)
        road_lines.append(
            RoadLine(
                object_id=running_idx,
                road_line_type=line_type,
                polyline=Polyline3D.from_linestring(LineString(line.coords)),
            )
        )
        running_idx += 1

    # Process lane dividers
    for divider in nuscenes_map.lane_divider:
        line = nuscenes_map.extract_line(divider["line_token"])
        line_type = _get_road_line_type(divider["line_token"], line_token_to_type)
        road_lines.append(
            RoadLine(
                object_id=running_idx,
                road_line_type=line_type,
                polyline=Polyline3D.from_linestring(LineString(line.coords)),
            )
        )
        running_idx += 1

    return road_lines


def _extract_nuscenes_road_edges(nuscenes_map: NuScenesMap) -> List[RoadEdge]:
    """Helper function to extract road edges from a nuScenes map."""
    drivable_polygons = []
    for drivable_area_record in nuscenes_map.drivable_area:
        drivable_area = nuscenes_map.get("drivable_area", drivable_area_record["token"])
        for polygon_token in drivable_area["polygon_tokens"]:
            polygon = nuscenes_map.extract_polygon(polygon_token)
            drivable_polygons.append(polygon)

    road_edge_linear_rings = get_road_edge_linear_rings(drivable_polygons)
    road_edges_linestrings = split_line_geometry_by_max_length(road_edge_linear_rings, MAX_ROAD_EDGE_LENGTH)  # type: ignore

    road_edges_cache: List[RoadEdge] = []
    for idx in range(len(road_edges_linestrings)):
        road_edges_cache.append(
            RoadEdge(
                object_id=idx,
                road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
                polyline=Polyline2D.from_linestring(road_edges_linestrings[idx]),
            )
        )

    return road_edges_cache


_NUSCENES_TO_ROAD_LINE_TYPE: Dict[str, RoadLineType] = {
    "SINGLE_SOLID_WHITE": RoadLineType.SOLID_WHITE,
    "DOUBLE_DASHED_WHITE": RoadLineType.DOUBLE_DASH_WHITE,
    "SINGLE_SOLID_YELLOW": RoadLineType.SOLID_YELLOW,
}


def _build_line_token_to_type_mapping(nuscenes_map: NuScenesMap) -> Dict[str, str]:
    """Builds a mapping from line tokens to their segment types, constructed once per map."""
    line_token_to_type: Dict[str, str] = {}
    for lane_record in nuscenes_map.lane:
        for seg in lane_record.get("left_lane_divider_segments", []):
            token = seg.get("line_token")
            seg_type = seg.get("segment_type")
            if token and seg_type:
                line_token_to_type[token] = seg_type

        for seg in lane_record.get("right_lane_divider_segments", []):
            token = seg.get("line_token")
            seg_type = seg.get("segment_type")
            if token and seg_type:
                line_token_to_type[token] = seg_type

    return line_token_to_type


def _get_road_line_type(line_token: str, line_token_to_type: Dict[str, str]) -> RoadLineType:
    """Maps a nuScenes line token to a RoadLineType using a pre-built mapping."""
    nuscenes_type = line_token_to_type.get(line_token, "UNKNOWN")
    return _NUSCENES_TO_ROAD_LINE_TYPE.get(nuscenes_type, RoadLineType.UNKNOWN)
