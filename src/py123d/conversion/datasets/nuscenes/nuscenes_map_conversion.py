from collections import defaultdict
from pathlib import Path
from typing import Dict, Final, List

import numpy as np
from shapely.geometry import LineString, Polygon

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.datasets.nuscenes.utils.nuscenes_constants import NUSCENES_MAPS
from py123d.conversion.datasets.nuscenes.utils.nuscenes_map_utils import (
    extract_lane_and_boundaries,
    extract_nuscenes_centerline,
    order_lanes_left_to_right,
)
from py123d.conversion.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
    split_polygon_by_grid,
)
from py123d.datatypes.map_objects.map_layer_types import RoadEdgeType, RoadLineType, StopZoneType
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
from py123d.geometry import OccupancyMap2D, Polyline2D, Polyline3D
from py123d.geometry.utils.polyline_utils import offset_points_perpendicular

check_dependencies(["nuscenes"], optional_name="nuscenes")
from nuscenes.map_expansion.map_api import NuScenesMap

MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # [m]
MAX_LANE_WIDTH: Final[float] = 4.0  # [m]
MIN_LANE_WIDTH: Final[float] = 1.0  # [m]


def write_nuscenes_map(nuscenes_maps_root: Path, location: str, map_writer: AbstractMapWriter) -> None:
    """Converts the nuScenes map types to the 123D format, and sends elements to the map writer.

    :param nuscenes_maps_root: Path to the nuScenes maps root directory
    :param location: Name of the specific map location to convert
    :param map_writer: Map writer instance to write the converted elements
    """

    assert location in NUSCENES_MAPS, f"Map name {location} is not supported."
    nuscenes_map = NuScenesMap(dataroot=str(nuscenes_maps_root), map_name=location)

    # 1. extract road edges (used later to determine lane connector widths)
    road_edges = _extract_nuscenes_road_edges(nuscenes_map)

    # 2. extract lanes
    lanes = _extract_nuscenes_lanes(nuscenes_map)

    # 3. extract lane connectors (i.e. lanes on intersections)
    lane_connectors = _extract_nuscenes_lane_connectors(nuscenes_map, road_edges)

    # 4. extract intersections (and store lane-connector to intersection assignment for lane groups)
    intersection_assignment = _write_nuscenes_intersections(nuscenes_map, lane_connectors, map_writer)

    # 5. extract lane groups
    lane_groups = _extract_nuscenes_lane_groups(nuscenes_map, lanes, lane_connectors, intersection_assignment)

    # Write remaining map elements
    _write_nuscenes_crosswalks(nuscenes_map, map_writer)
    _write_nuscenes_walkways(nuscenes_map, map_writer)
    _write_nuscenes_carparks(nuscenes_map, map_writer)
    _write_nuscenes_generic_drivables(nuscenes_map, map_writer)
    _write_nuscenes_stop_zones(nuscenes_map, map_writer)
    _write_nuscenes_road_lines(nuscenes_map, map_writer)

    for lane in lanes + lane_connectors:
        map_writer.write_lane(lane)

    for road_edge in road_edges:
        map_writer.write_road_edge(road_edge)

    for lane_group in lane_groups:
        map_writer.write_lane_group(lane_group)


def _extract_nuscenes_lanes(nuscenes_map: NuScenesMap) -> List[Lane]:
    """Helper function to extract lanes from a nuScenes map."""

    # NOTE: nuScenes does not provide explicitly provide lane groups and does not assign lanes to roadblocks.
    # Therefore, we query the roadblocks given the middle-point of the centerline to assign lanes to a road block.
    # Unlike road segments, road blocks outline a lane group going in the same direction.
    # In case a roadblock cannot be assigned, e.g. because the lane is not located within any roadblock, or the
    # roadblock data is invalid [1], we assign a new lane group with only this lane.
    # [1] https://github.com/nutonomy/nuscenes-devkit/issues/862

    road_blocks_invalid = nuscenes_map.map_name in ["singapore-queenstown", "singapore-hollandvillage"]

    road_block_dict: Dict[str, Polygon] = {}
    if not road_blocks_invalid:
        road_block_dict: Dict[str, Polygon] = {
            road_block["token"]: nuscenes_map.extract_polygon(road_block["polygon_token"])
            for road_block in nuscenes_map.road_block
        }

    road_block_map = OccupancyMap2D.from_dict(road_block_dict)
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
            query_point = centerline.interpolate(0.5, normalized=True).shapely_point
            intersecting_roadblock = road_block_map.query_nearest(query_point, max_distance=0.1, all_matches=False)

            # NOTE: if a lane cannot be assigned to a road block, we assume a new lane group with only this lane.
            # The lane group id is set to be the same as the lane id in this case.
            if len(intersecting_roadblock) > 0:
                lane_group_id = road_block_map.ids[intersecting_roadblock[0]]

        # Get topology
        incoming = nuscenes_map.get_incoming_lane_ids(token)
        outgoing = nuscenes_map.get_outgoing_lane_ids(token)

        lanes.append(
            Lane(
                object_id=token,
                lane_group_id=lane_group_id,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                centerline=centerline,
                left_lane_id=None,
                right_lane_id=None,
                predecessor_ids=incoming,
                successor_ids=outgoing,
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
                lane_group_id=lane_group_id,
                left_boundary=Polyline2D.from_array(left_pts),
                right_boundary=Polyline2D.from_array(right_pts),
                centerline=centerline,
                left_lane_id=None,  # Not directly available in nuscenes
                right_lane_id=None,  # Not directly available in nuscenes
                predecessor_ids=predecessor_ids,
                successor_ids=successor_ids,
                speed_limit_mps=None,  # Default value
                outline=None,
                shapely_polygon=None,
            )
        )

    return lane_connectors


def _extract_nuscenes_lane_groups(
    nuscenes_map: NuScenesMap, lanes: List[Lane], lane_connectors: List[Lane], intersection_assignment: Dict[str, int]
) -> List[LaneGroup]:
    """Helper function to extract lane groups from a nuScenes map."""

    lane_groups = []
    lanes_dict = {lane.object_id: lane for lane in lanes + lane_connectors}

    # 1. Gather all lane group ids that were previously assigned in the lanes (either roadblocks of lane themselves)
    lane_group_lane_dict: Dict[str, List[str]] = defaultdict(list)
    for lane in lanes + lane_connectors:
        lane_group_lane_dict[lane.lane_group_id].append(lane.object_id)

    for lane_group_id, lane_ids in lane_group_lane_dict.items():
        if len(lane_ids) > 1:
            lane_centerlines: List[Polyline2D] = [lanes_dict[lane_id].centerline for lane_id in lane_ids]
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
                if pred_lane is not None:
                    predecessor_ids.add(pred_lane.lane_group_id)
            for succ_id in lane.successor_ids:
                succ_lane = lanes_dict.get(succ_id)
                if succ_lane is not None:
                    successor_ids.add(succ_lane.lane_group_id)

        intersection_ids = set(
            [int(intersection_assignment[lane_id]) for lane_id in lane_ids if lane_id in intersection_assignment]
        )
        assert len(intersection_ids) <= 1, "A lane group cannot belong to multiple intersections."
        intersection_id = None if len(intersection_ids) == 0 else intersection_ids.pop()

        lane_groups.append(
            LaneGroup(
                object_id=lane_group_id,
                lane_ids=lane_ids,
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


def _write_nuscenes_intersections(
    nuscenes_map: NuScenesMap, lane_connectors: List[Lane], map_writer: AbstractMapWriter
) -> None:
    """Write intersection data to map_writer and return lane-connector to intersection assignment."""

    intersection_assignment = {}

    # 1. Extract intersections and corresponding polygons
    intersection_polygons = []
    for road_segment in nuscenes_map.road_segment:
        if road_segment["is_intersection"]:
            if "polygon_token" in road_segment:
                polygon = nuscenes_map.extract_polygon(road_segment["polygon_token"])
                intersection_polygons.append(polygon)

    # 2. Find lane connectors within each intersection polygon
    lane_connector_center_point_dict = {
        lane_connector.object_id: lane_connector.centerline.interpolate(0.5, normalized=True).shapely_point
        for lane_connector in lane_connectors
    }
    centerpoint_map = OccupancyMap2D.from_dict(lane_connector_center_point_dict)
    for idx, intersection_polygon in enumerate(intersection_polygons):
        intersecting_lane_connector_ids = centerpoint_map.intersects(intersection_polygon)
        for lane_connector_id in intersecting_lane_connector_ids:
            intersection_assignment[lane_connector_id] = idx

        map_writer.write_intersection(
            Intersection(
                object_id=idx,
                lane_group_ids=intersecting_lane_connector_ids,
                outline=None,
                shapely_polygon=intersection_polygon,
            )
        )

    return intersection_assignment


def _write_nuscenes_crosswalks(nuscenes_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """Write crosswalk data to map_writer."""

    crosswalk_polygons = []
    for crossing in nuscenes_map.ped_crossing:
        if "polygon_token" in crossing:
            polygon = nuscenes_map.extract_polygon(crossing["polygon_token"])
            crosswalk_polygons.append(polygon)

    for idx, polygon in enumerate(crosswalk_polygons):
        map_writer.write_crosswalk(Crosswalk(object_id=idx, shapely_polygon=polygon))


def _write_nuscenes_walkways(nuscenes_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """Write walkway data to map_writer."""
    walkway_polygons = []
    for walkway_record in nuscenes_map.walkway:
        if "polygon_token" in walkway_record:
            polygon = nuscenes_map.extract_polygon(walkway_record["polygon_token"])
            walkway_polygons.append(polygon)

    for idx, polygon in enumerate(walkway_polygons):
        map_writer.write_walkway(Walkway(object_id=idx, shapely_polygon=polygon))


def _write_nuscenes_carparks(nuscenes_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """Write carpark data to map_writer."""
    carpark_polygons = []
    for carpark_record in nuscenes_map.carpark_area:
        if "polygon_token" in carpark_record:
            polygon = nuscenes_map.extract_polygon(carpark_record["polygon_token"])
            carpark_polygons.append(polygon)

    for idx, polygon in enumerate(carpark_polygons):
        map_writer.write_carpark(Carpark(object_id=idx, shapely_polygon=polygon))


def _write_nuscenes_generic_drivables(nuscenes_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """Write generic drivable area data to map_writer."""
    cell_size = 20.0
    drivable_polygons = []
    for drivable_area_record in nuscenes_map.drivable_area:
        drivable_area = nuscenes_map.get("drivable_area", drivable_area_record["token"])
        for polygon_token in drivable_area["polygon_tokens"]:
            polygon = nuscenes_map.extract_polygon(polygon_token)

            split_polygons = split_polygon_by_grid(polygon, cell_size=cell_size)
            drivable_polygons.extend(split_polygons)
            # drivable_polygons.append(polygon)

    for idx, geometry in enumerate(drivable_polygons):
        map_writer.write_generic_drivable(GenericDrivable(object_id=idx, shapely_polygon=geometry))


def _write_nuscenes_stop_zones(nuscenes_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """Write stop line data to map_writer."""

    NUSCENES_STOP_CUES_TO_STOP_ZONE_TYPE = {
        "PED_CROSSING": StopZoneType.PEDESTRIAN_CROSSING,
        "TURN_STOP": StopZoneType.TURN_STOP,
        "TRAFFIC_LIGHT": StopZoneType.TRAFFIC_LIGHT,
        "STOP_SIGN": StopZoneType.STOP_SIGN,
        "YIELD": StopZoneType.YIELD_SIGN,
    }
    for stop_line in nuscenes_map.stop_line:
        token = stop_line["token"]
        if "polygon_token" in stop_line:
            polygon = nuscenes_map.extract_polygon(stop_line["polygon_token"])
        else:
            continue
        if not polygon.is_valid:
            continue

        # Note: Stop lines are written as generic drivable for compatibility
        if "stop_line_type" in stop_line.keys():
            stop_zone_type = NUSCENES_STOP_CUES_TO_STOP_ZONE_TYPE.get(stop_line["stop_line_type"], StopZoneType.UNKNOWN)
        else:
            stop_zone_type = StopZoneType.UNKNOWN

        map_writer.write_stop_zone(
            StopZone(
                object_id=token,
                stop_zone_type=stop_zone_type,
                shapely_polygon=polygon,
            )
        )


def _write_nuscenes_road_lines(nuscenes_map: NuScenesMap, map_writer: AbstractMapWriter) -> None:
    """Write road line data (dividers) to map_writer."""
    # Process road dividers
    road_dividers = nuscenes_map.road_divider
    running_idx = 0
    for divider in road_dividers:
        line = nuscenes_map.extract_line(divider["line_token"])

        # Determine line type
        line_type = _get_road_line_type(divider["line_token"], nuscenes_map)

        map_writer.write_road_line(
            RoadLine(
                object_id=running_idx,
                road_line_type=line_type,
                polyline=Polyline3D.from_linestring(LineString(line.coords)),
            )
        )
        running_idx += 1

    # Process lane dividers
    lane_dividers = nuscenes_map.lane_divider
    for divider in lane_dividers:
        line = nuscenes_map.extract_line(divider["line_token"])
        line_type = _get_road_line_type(divider["line_token"], nuscenes_map)

        map_writer.write_road_line(
            RoadLine(
                object_id=running_idx,
                road_line_type=line_type,
                polyline=Polyline3D.from_linestring(LineString(line.coords)),
            )
        )
        running_idx += 1


def _extract_nuscenes_road_edges(nuscenes_map: NuScenesMap) -> List[RoadEdge]:
    """Helper function to extract road edges from a nuScenes map."""
    drivable_polygons = []
    for drivable_area_record in nuscenes_map.drivable_area:
        drivable_area = nuscenes_map.get("drivable_area", drivable_area_record["token"])
        for polygon_token in drivable_area["polygon_tokens"]:
            polygon = nuscenes_map.extract_polygon(polygon_token)
            drivable_polygons.append(polygon)

    road_edge_linear_rings = get_road_edge_linear_rings(drivable_polygons)
    road_edges_linestrings = split_line_geometry_by_max_length(road_edge_linear_rings, MAX_ROAD_EDGE_LENGTH)

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


def _get_road_line_type(line_token: str, nuscenes_map: NuScenesMap) -> RoadLineType:
    """Map nuscenes line type to RoadLineType."""

    # FIXME @DanielDauner: Store token to type mapping. Creating mapping for every call is not ideal.
    nuscenes_to_road_line_type = {
        "SINGLE_SOLID_WHITE": RoadLineType.SOLID_WHITE,
        "DOUBLE_DASHED_WHITE": RoadLineType.DOUBLE_DASH_WHITE,
        "SINGLE_SOLID_YELLOW": RoadLineType.SOLID_YELLOW,
    }

    line_token_to_type = {}
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

    nuscenes_type = line_token_to_type.get(line_token, "UNKNOWN")
    return nuscenes_to_road_line_type.get(nuscenes_type, RoadLineType.UNKNOWN)
