import logging
from pathlib import Path
from typing import Dict, Final, List

import numpy as np
import shapely

from py123d.api.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.conversion.utils.map_utils.opendrive.parser.lane import XODRRoadMark
from py123d.conversion.utils.map_utils.opendrive.parser.opendrive import XODR, Junction
from py123d.conversion.utils.map_utils.opendrive.utils.collection import collect_element_helpers
from py123d.conversion.utils.map_utils.opendrive.utils.id_system import lane_section_id_from_lane_group_id
from py123d.conversion.utils.map_utils.opendrive.utils.lane_helper import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
)
from py123d.conversion.utils.map_utils.opendrive.utils.objects_helper import OpenDriveObjectHelper
from py123d.conversion.utils.map_utils.opendrive.utils.stop_zone_helper import create_stop_zones_from_signals
from py123d.conversion.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)
from py123d.conversion.utils.map_utils.road_edge.road_edge_3d_utils import (
    get_road_edges_3d_from_drivable_surfaces,
    lift_outlines_to_3d,
)
from py123d.datatypes.map_objects import (
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    Lane,
    LaneGroup,
    RoadEdge,
    RoadEdgeType,
    RoadLine,
    RoadLineType,
    Walkway,
)
from py123d.geometry.geometry_index import Point3DIndex
from py123d.geometry.polyline import Polyline3D

logger = logging.getLogger(__name__)

MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # [m]


def convert_xodr_map(
    xordr_file: Path,
    map_writer: AbstractMapWriter,
    interpolation_step_size: float = 1.0,
    connection_distance_threshold: float = 0.1,
    internal_only: bool = True,
) -> None:
    """Converts an OpenDRIVE map file and the map objects to an 123D map using a map writer.

    :param xordr_file: Path to the OpenDRIVE (.xodr) file.
    :param map_writer: Map writer to write the extracted map objects.
    :param interpolation_step_size: Step size for interpolating polylines, defaults to 1.0
    :param connection_distance_threshold: Distance threshold for connecting road elements, defaults to 0.1
    :param internal_only: If True, only write internal road lines (center + between lanes), defaults to True
    """

    opendrive = XODR.parse_from_file(xordr_file)

    (
        _,
        junction_dict,
        lane_helper_dict,
        lane_group_helper_dict,
        object_helper_dict,
        center_lane_marks_dict,
        signal_dict,
    ) = collect_element_helpers(opendrive, interpolation_step_size, connection_distance_threshold)

    # Collect data frames and store (needed for road edge/line extraction)
    lanes = _extract_and_write_lanes(lane_group_helper_dict, map_writer)
    lane_groups = _extract_and_write_lane_groups(lane_group_helper_dict, map_writer)
    car_parks = _extract_and_write_carparks(lane_helper_dict, map_writer)
    generic_drivables = _extract_and_write_generic_drivables(lane_helper_dict, map_writer)

    # Write other map elements
    _write_walkways(lane_helper_dict, map_writer)
    _write_intersections(junction_dict, lane_group_helper_dict, map_writer)
    _write_crosswalks(object_helper_dict, map_writer)
    _write_stop_zones(signal_dict, lane_helper_dict, map_writer)

    # Extract polyline elements that are inferred of other road surfaces.
    _write_road_lines(lane_helper_dict, lane_groups, center_lane_marks_dict, map_writer, internal_only)
    _write_road_edges(
        lanes=lanes,
        lane_groups=lane_groups,
        car_parks=car_parks,
        generic_drivables=generic_drivables,
        map_writer=map_writer,
    )


def _extract_and_write_lanes(
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper],
    map_writer: AbstractMapWriter,
) -> List[Lane]:
    """Extracts lanes from lane group helpers and writes them using the map writer."""

    lanes: List[Lane] = []
    for lane_group_helper in lane_group_helper_dict.values():
        lane_group_id = lane_group_helper.lane_group_id
        lane_helpers = lane_group_helper.lane_helpers
        num_lanes = len(lane_helpers)
        # NOTE: Lanes are going left to right, ie. inner to outer
        for lane_idx, lane_helper in enumerate(lane_helpers):
            left_lane_id = lane_helpers[lane_idx - 1].lane_id if lane_idx > 0 else None
            right_lane_id = lane_helpers[lane_idx + 1].lane_id if lane_idx < num_lanes - 1 else None
            lane = Lane(
                object_id=lane_helper.lane_id,
                lane_group_id=lane_group_id,
                left_boundary=lane_helper.inner_polyline_3d,
                right_boundary=lane_helper.outer_polyline_3d,
                centerline=lane_helper.center_polyline_3d,
                left_lane_id=left_lane_id,
                right_lane_id=right_lane_id,
                predecessor_ids=lane_helper.predecessor_lane_ids,
                successor_ids=lane_helper.successor_lane_ids,
                speed_limit_mps=lane_helper.speed_limit_mps,
                outline=lane_helper.outline_polyline_3d,
            )
            lanes.append(lane)
            map_writer.write_lane(lane)

    return lanes


def _extract_and_write_lane_groups(
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper], map_writer: AbstractMapWriter
) -> List[LaneGroup]:
    """Extracts lane groups from lane group helpers and writes them using the map writer."""

    lane_groups: List[LaneGroup] = []
    for lane_group_helper in lane_group_helper_dict.values():
        lane_group_helper: OpenDriveLaneGroupHelper
        lane_group = LaneGroup(
            object_id=lane_group_helper.lane_group_id,
            lane_ids=[lane_helper.lane_id for lane_helper in lane_group_helper.lane_helpers],
            left_boundary=lane_group_helper.inner_polyline_3d,
            right_boundary=lane_group_helper.outer_polyline_3d,
            intersection_id=lane_group_helper.junction_id,
            predecessor_ids=lane_group_helper.predecessor_lane_group_ids,
            successor_ids=lane_group_helper.successor_lane_group_ids,
            outline=lane_group_helper.outline_polyline_3d,
        )
        lane_groups.append(lane_group)
        map_writer.write_lane_group(lane_group)

    return lane_groups


def _write_walkways(lane_helper_dict: Dict[str, OpenDriveLaneHelper], map_writer: AbstractMapWriter) -> None:
    """Writes walkways from lane helpers using the map writer."""
    for lane_helper in lane_helper_dict.values():
        if lane_helper.type == "sidewalk":
            map_writer.write_walkway(
                Walkway(
                    object_id=lane_helper.lane_id,
                    outline=lane_helper.outline_polyline_3d,
                )
            )


def _extract_and_write_carparks(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper], map_writer: AbstractMapWriter
) -> List[Carpark]:
    """Extracts carparks from lane helpers and writes them using the map writer."""
    carparks: List[Carpark] = []
    for lane_helper in lane_helper_dict.values():
        if lane_helper.type == "parking":
            carpark = Carpark(
                object_id=lane_helper.lane_id,
                outline=lane_helper.outline_polyline_3d,
            )
            carparks.append(carpark)
            map_writer.write_carpark(carpark)
    return carparks


def _extract_and_write_generic_drivables(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper], map_writer: AbstractMapWriter
) -> List[GenericDrivable]:
    """Extracts generic drivables from lane helpers and writes them using the map writer."""

    generic_drivables: List[GenericDrivable] = []
    for lane_helper in lane_helper_dict.values():
        if lane_helper.type in ["border", "bidirectional"]:
            generic_drivable = GenericDrivable(
                object_id=lane_helper.lane_id,
                outline=lane_helper.outline_polyline_3d,
            )
            generic_drivables.append(generic_drivable)
            map_writer.write_generic_drivable(generic_drivable)
    return generic_drivables


def _write_intersections(
    junction_dict: Dict[str, Junction],
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper],
    map_writer: AbstractMapWriter,
) -> None:
    def _find_lane_group_helpers_with_junction_id(junction_id: int) -> List[OpenDriveLaneGroupHelper]:
        return [
            lane_group_helper
            for lane_group_helper in lane_group_helper_dict.values()
            if lane_group_helper.junction_id == junction_id
        ]

    for junction in junction_dict.values():
        lane_group_helpers = _find_lane_group_helpers_with_junction_id(junction.id)
        lane_group_ids_ = [lane_group_helper.lane_group_id for lane_group_helper in lane_group_helpers]
        if len(lane_group_ids_) == 0:
            logger.debug(f"Skipping Junction {junction.id} without lane groups!")
            continue

        # TODO @DanielDauner: Create a method that extracts 3D outlines of intersections.
        outline = _extract_intersection_outline(lane_group_helpers, junction.id)
        map_writer.write_intersection(
            Intersection(
                object_id=junction.id,
                lane_group_ids=lane_group_ids_,
                outline=outline,
            )
        )


def _write_crosswalks(object_helper_dict: Dict[int, OpenDriveObjectHelper], map_writer: AbstractMapWriter) -> None:
    for object_helper in object_helper_dict.values():
        map_writer.write_crosswalk(
            Crosswalk(
                object_id=object_helper.object_id,
                outline=object_helper.outline_polyline_3d,
            )
        )


def _write_stop_zones(
    signal_dict: Dict, lane_helper_dict: Dict[str, OpenDriveLaneHelper], map_writer: AbstractMapWriter
) -> None:
    """Write stop zones from signal helpers using the map writer."""
    stop_zones = create_stop_zones_from_signals(signal_dict, lane_helper_dict)
    for stop_zone in stop_zones.values():
        map_writer.write_stop_zone(stop_zone)


def _map_road_mark_to_type(mark_type: str, color: str) -> RoadLineType:
    # TODO: Verify that "none" doesn't mean something different in OpenDRIVE spec.
    if mark_type == "none":
        mark_type = "dashed"

    if color == "standard":
        color = "white"

    mapping = {
        ("solid", "white"): RoadLineType.SOLID_WHITE,
        ("solid", "yellow"): RoadLineType.SOLID_YELLOW,
        ("broken", "white"): RoadLineType.DASHED_WHITE,
        ("broken", "yellow"): RoadLineType.DASHED_YELLOW,
        ("solid solid", "white"): RoadLineType.DOUBLE_SOLID_WHITE,
        ("solid solid", "yellow"): RoadLineType.DOUBLE_SOLID_YELLOW,
        ("broken broken", "white"): RoadLineType.DOUBLE_DASH_WHITE,
        ("broken broken", "yellow"): RoadLineType.DOUBLE_DASH_YELLOW,
        ("solid broken", "white"): RoadLineType.SOLID_DASH_WHITE,
        ("solid broken", "yellow"): RoadLineType.SOLID_DASH_YELLOW,
        ("broken solid", "white"): RoadLineType.DASH_SOLID_WHITE,
        ("broken solid", "yellow"): RoadLineType.DASH_SOLID_YELLOW,
    }
    return mapping.get((mark_type.lower(), (color or "white").lower()), RoadLineType.DASHED_WHITE)


def _extract_polyline_segment(
    polyline: Polyline3D,
    lane_length: float,
    s_start: float,
    s_end: float,
    step_size: float = 1.0,
) -> Polyline3D | None:
    """Extract segment of polyline between s_start and s_end (relative to lane section)."""
    total_length = polyline.length
    scale = total_length / lane_length if lane_length > 0 else 1.0
    dist_start = max(0.0, s_start * scale)
    dist_end = min(total_length, s_end * scale)

    if dist_end - dist_start < 0.1:
        return None

    num_points = max(2, int((dist_end - dist_start) / step_size) + 1)
    distances = np.linspace(dist_start, dist_end, num_points)
    points = polyline.interpolate(distances)
    return Polyline3D.from_array(points)


def _write_road_lines(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    lane_groups: List[LaneGroup],
    center_lane_marks_dict: Dict[str, List[XODRRoadMark]],
    map_writer: AbstractMapWriter,
    internal_only: bool = False,
) -> None:
    """Write road lines from OpenDRIVE road mark data.

    Road lines are created from:
    - Center lines: using center lane (id=0) road marks at lane_group.left_boundary
    - Lane boundaries: using lane road marks at lane.right_boundary
    """
    lane_group_on_intersection = {lg.object_id: lg.intersection_id is not None for lg in lane_groups}
    lane_group_dict = {lg.object_id: lg for lg in lane_groups}

    running_id = 0

    # A. Center lines (separating opposite directions)
    processed_lane_sections = set()
    for lane_group in lane_groups:
        if lane_group_on_intersection.get(lane_group.object_id, False):
            continue

        lane_section_id = lane_section_id_from_lane_group_id(lane_group.object_id)
        if lane_section_id in processed_lane_sections:
            continue
        processed_lane_sections.add(lane_section_id)

        center_marks = center_lane_marks_dict.get(lane_section_id, [])
        if not center_marks:
            continue

        polyline = lane_group.left_boundary
        lane_length = polyline.length

        for i, mark in enumerate(center_marks):
            s_start = mark.s_offset
            s_end = center_marks[i + 1].s_offset if i + 1 < len(center_marks) else lane_length

            segment = _extract_polyline_segment(polyline, lane_length, s_start, s_end)
            if segment is None:
                continue

            road_line_type = _map_road_mark_to_type(mark.type, mark.color)
            map_writer.write_road_line(RoadLine(object_id=running_id, road_line_type=road_line_type, polyline=segment))
            running_id += 1

    # B. Lane boundaries (between lanes in same group)
    for lane_id, lane_helper in lane_helper_dict.items():
        if lane_helper.type != "driving":
            continue

        lane_group_id = lane_id.rsplit("_", 1)[0]
        if lane_group_on_intersection.get(lane_group_id, False):
            continue

        lane_group = lane_group_dict.get(lane_group_id)
        if lane_group is None:
            continue

        # Find lane in lane_group to check adjacency
        lane_idx_in_group = None
        for idx, lid in enumerate(lane_group.lane_ids):
            if lid == lane_id:
                lane_idx_in_group = idx
                break
        if lane_idx_in_group is None:
            continue

        num_lanes = len(lane_group.lane_ids)
        has_right_neighbor = lane_idx_in_group < num_lanes - 1
        is_internal = has_right_neighbor

        if internal_only and not is_internal:
            continue

        road_marks = lane_helper.open_drive_lane.road_marks
        if not road_marks:
            continue

        polyline = lane_helper.outer_polyline_3d
        lane_length = lane_helper.s_range[1] - lane_helper.s_range[0]

        for i, mark in enumerate(road_marks):
            s_start = mark.s_offset
            s_end = road_marks[i + 1].s_offset if i + 1 < len(road_marks) else lane_length

            segment = _extract_polyline_segment(polyline, lane_length, s_start, s_end)
            if segment is None:
                continue

            road_line_type = _map_road_mark_to_type(mark.type or "none", mark.color or "white")
            map_writer.write_road_line(RoadLine(object_id=running_id, road_line_type=road_line_type, polyline=segment))
            running_id += 1


def _write_road_edges(
    lanes: List[Lane],
    lane_groups: List[LaneGroup],
    car_parks: List[Carpark],
    generic_drivables: List[GenericDrivable],
    map_writer: AbstractMapWriter,
) -> None:
    road_edges_ = get_road_edges_3d_from_drivable_surfaces(
        lanes=lanes,
        lane_groups=lane_groups,
        car_parks=car_parks,
        generic_drivables=generic_drivables,
    )
    road_edge_linestrings = split_line_geometry_by_max_length(
        [road_edges.linestring for road_edges in road_edges_], MAX_ROAD_EDGE_LENGTH
    )

    running_id = 0
    for road_edge_linestring in road_edge_linestrings:
        #  TODO @DanielDauner: Figure out if other types should/could be assigned here.
        map_writer.write_road_edge(
            RoadEdge(
                object_id=running_id,
                road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
                polyline=Polyline3D.from_linestring(road_edge_linestring),
            )
        )
        running_id += 1


def _extract_intersection_outline(lane_group_helpers: List[OpenDriveLaneGroupHelper], junction_id: str) -> Polyline3D:
    """Helper method to extract intersection outline in 3D from lane group helpers."""

    # 1. Extract the intersection outlines in 2D
    intersection_polygons: List[shapely.Polygon] = [
        lane_group_helper.shapely_polygon for lane_group_helper in lane_group_helpers
    ]
    intersection_edges = get_road_edge_linear_rings(
        intersection_polygons,
        buffer_distance=0.25,
        add_interiors=False,
    )

    # 2. Lift the 2D outlines to 3D
    lane_group_outlines: List[Polyline3D] = [
        lane_group_helper.outline_polyline_3d for lane_group_helper in lane_group_helpers
    ]
    intersection_outlines = lift_outlines_to_3d(intersection_edges, lane_group_outlines)

    # NOTE: When the intersection has multiple non-overlapping outlines, we cannot return a single outline in 3D.
    # For now, we return the longest outline.
    valid_outlines = [outline for outline in intersection_outlines if outline.array.shape[0] > 2]
    if len(valid_outlines) == 0:
        logger.warning(
            f"Could not extract valid outline for intersection {junction_id} with {len(intersection_edges)} edges!"
        )
        longest_outline_2d = max(intersection_edges, key=lambda outline: outline.length)
        average_z = sum(outline.array[:, 2].mean() for outline in intersection_outlines) / len(intersection_outlines)

        outline_3d_array = np.zeros((len(longest_outline_2d.coords), 3))
        outline_3d_array[:, Point3DIndex.XY] = np.array(longest_outline_2d.coords)
        outline_3d_array[:, Point3DIndex.Z] = average_z
        longest_outline = Polyline3D.from_array(outline_3d_array)
    else:
        longest_outline = max(valid_outlines, key=lambda outline: outline.length)

    return longest_outline
