import logging
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional

import numpy as np
import shapely

from py123d.datatypes import (
    BaseMapObject,
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
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.geometry.geometry_index import Point3DIndex
from py123d.geometry.polyline import Polyline3D
from py123d.parser.abstract_dataset_parser import MapParser
from py123d.parser.opendrive.utils.collection import collect_element_helpers
from py123d.parser.opendrive.utils.id_system import lane_section_id_from_lane_group_id
from py123d.parser.opendrive.utils.lane_helper import (
    OpenDriveLaneGroupHelper,
    OpenDriveLaneHelper,
)
from py123d.parser.opendrive.utils.objects_helper import OpenDriveObjectHelper
from py123d.parser.opendrive.utils.stop_zone_helper import create_stop_zones_from_signals
from py123d.parser.opendrive.xodr_parser.lane import XODRRoadMark
from py123d.parser.opendrive.xodr_parser.opendrive import XODR, Junction
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)
from py123d.parser.utils.map_utils.road_edge.road_edge_3d_utils import (
    get_road_edges_3d_from_drivable_surfaces,
    lift_outlines_to_3d,
)

logger = logging.getLogger(__name__)

MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # [m]


class OpenDriveMapParser(MapParser):
    """Lightweight, picklable handle to one OpenDRIVE map."""

    def __init__(
        self,
        xodr_path: Path,
        location: Optional[str] = None,
        interpolation_step_size: float = 1.0,
        connection_distance_threshold: float = 0.1,
        internal_only: bool = True,
    ) -> None:
        self._xodr_path = xodr_path
        self._location = location
        self._interpolation_step_size = interpolation_step_size
        self._connection_distance_threshold = connection_distance_threshold
        self._internal_only = internal_only

    def get_map_metadata(self) -> MapMetadata:
        """Returns metadata for this OpenDRIVE map."""
        return MapMetadata(
            dataset="opendrive",
            location=self._location,
            map_has_z=True,
            map_is_per_log=False,
        )

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Yields map objects lazily from the XODR file."""
        yield from iter_xodr_map_objects(
            xodr_file=self._xodr_path,
            interpolation_step_size=self._interpolation_step_size,
            connection_distance_threshold=self._connection_distance_threshold,
            internal_only=self._internal_only,
        )


def iter_xodr_map_objects(
    xodr_file: Path,
    interpolation_step_size: float = 1.0,
    connection_distance_threshold: float = 0.1,
    internal_only: bool = True,
) -> Iterator[BaseMapObject]:
    """Yields all map objects extracted from an OpenDRIVE (.xodr) file.

    :param xodr_file: Path to the OpenDRIVE (.xodr) file.
    :param interpolation_step_size: Step size for interpolating polylines, defaults to 1.0
    :param connection_distance_threshold: Distance threshold for connecting road elements, defaults to 0.1
    :param internal_only: If True, only yield internal road lines (center + between lanes), defaults to True
    """
    opendrive = XODR.parse_from_file(xodr_file)

    (
        _,
        junction_dict,
        lane_helper_dict,
        lane_group_helper_dict,
        object_helper_dict,
        center_lane_marks_dict,
        signal_dict,
    ) = collect_element_helpers(opendrive, interpolation_step_size, connection_distance_threshold)

    # Extract and yield primary surfaces (also collected for road edge/line inference)
    lanes = _extract_lanes(lane_group_helper_dict)
    yield from lanes

    lane_groups = _extract_lane_groups(lane_group_helper_dict)
    yield from lane_groups

    car_parks = _extract_carparks(lane_helper_dict)
    yield from car_parks

    generic_drivables = _extract_generic_drivables(lane_helper_dict)
    yield from generic_drivables

    # Yield other map elements
    yield from _iter_walkways(lane_helper_dict)
    yield from _iter_intersections(junction_dict, lane_group_helper_dict)
    yield from _iter_crosswalks(object_helper_dict)
    yield from _iter_stop_zones(signal_dict, lane_helper_dict)

    # Yield polyline elements that are inferred from other road surfaces
    yield from _iter_road_lines(lane_helper_dict, lane_groups, center_lane_marks_dict, internal_only)
    yield from _iter_road_edges(lanes, lane_groups, car_parks, generic_drivables)


# ------------------------------------------------------------------------------------------------------------------
# Extraction helpers (collect into lists, needed for road edge/line inference)
# ------------------------------------------------------------------------------------------------------------------


def _extract_lanes(lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper]) -> List[Lane]:
    """Extracts lanes from lane group helpers."""
    lanes: List[Lane] = []
    for lane_group_helper in lane_group_helper_dict.values():
        lane_group_id = lane_group_helper.lane_group_id
        lane_helpers = lane_group_helper.lane_helpers
        num_lanes = len(lane_helpers)
        # NOTE: Lanes are going left to right, ie. inner to outer
        for lane_idx, lane_helper in enumerate(lane_helpers):
            left_lane_id = lane_helpers[lane_idx - 1].lane_id if lane_idx > 0 else None
            right_lane_id = lane_helpers[lane_idx + 1].lane_id if lane_idx < num_lanes - 1 else None
            lanes.append(
                Lane(
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
            )
    return lanes


def _extract_lane_groups(lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper]) -> List[LaneGroup]:
    """Extracts lane groups from lane group helpers."""
    lane_groups: List[LaneGroup] = []
    for lane_group_helper in lane_group_helper_dict.values():
        lane_group_helper: OpenDriveLaneGroupHelper
        lane_groups.append(
            LaneGroup(
                object_id=lane_group_helper.lane_group_id,
                lane_ids=[lane_helper.lane_id for lane_helper in lane_group_helper.lane_helpers],
                left_boundary=lane_group_helper.inner_polyline_3d,
                right_boundary=lane_group_helper.outer_polyline_3d,
                intersection_id=lane_group_helper.junction_id,
                predecessor_ids=lane_group_helper.predecessor_lane_group_ids,
                successor_ids=lane_group_helper.successor_lane_group_ids,
                outline=lane_group_helper.outline_polyline_3d,
            )
        )
    return lane_groups


def _extract_carparks(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> List[Carpark]:
    """Extracts carparks from lane helpers."""
    return [
        Carpark(object_id=lh.lane_id, outline=lh.outline_polyline_3d)
        for lh in lane_helper_dict.values()
        if lh.type == "parking"
    ]


def _extract_generic_drivables(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> List[GenericDrivable]:
    """Extracts generic drivables from lane helpers."""
    return [
        GenericDrivable(object_id=lh.lane_id, outline=lh.outline_polyline_3d)
        for lh in lane_helper_dict.values()
        if lh.type in {"border", "bidirectional"}
    ]


# ------------------------------------------------------------------------------------------------------------------
# Iterator helpers (yield map objects one at a time)
# ------------------------------------------------------------------------------------------------------------------


def _iter_walkways(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> Iterator[Walkway]:
    """Yields walkways from lane helpers."""
    for lane_helper in lane_helper_dict.values():
        if lane_helper.type == "sidewalk":
            yield Walkway(object_id=lane_helper.lane_id, outline=lane_helper.outline_polyline_3d)


def _iter_intersections(
    junction_dict: Dict[str, Junction],
    lane_group_helper_dict: Dict[str, OpenDriveLaneGroupHelper],
) -> Iterator[Intersection]:
    """Yields intersection objects from junctions."""

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

        outline = _extract_intersection_outline(lane_group_helpers, junction.id)
        yield Intersection(object_id=junction.id, lane_group_ids=lane_group_ids_, outline=outline)


def _iter_crosswalks(object_helper_dict: Dict[int, OpenDriveObjectHelper]) -> Iterator[Crosswalk]:
    """Yields crosswalk objects."""
    for object_helper in object_helper_dict.values():
        yield Crosswalk(object_id=object_helper.object_id, outline=object_helper.outline_polyline_3d)


def _iter_stop_zones(signal_dict: Dict, lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> Iterator:
    """Yields stop zone objects."""
    stop_zones = create_stop_zones_from_signals(signal_dict, lane_helper_dict)
    yield from stop_zones.values()


def _iter_road_lines(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    lane_groups: List[LaneGroup],
    center_lane_marks_dict: Dict[str, List[XODRRoadMark]],
    internal_only: bool = False,
) -> Iterator[RoadLine]:
    """Yields road lines from OpenDRIVE road mark data.

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
            yield RoadLine(object_id=running_id, road_line_type=road_line_type, polyline=segment)
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
            yield RoadLine(object_id=running_id, road_line_type=road_line_type, polyline=segment)
            running_id += 1


def _iter_road_edges(
    lanes: List[Lane],
    lane_groups: List[LaneGroup],
    car_parks: List[Carpark],
    generic_drivables: List[GenericDrivable],
) -> Iterator[RoadEdge]:
    """Yields road edge objects."""
    road_edges_ = get_road_edges_3d_from_drivable_surfaces(
        lanes=lanes,
        lane_groups=lane_groups,
        car_parks=car_parks,
        generic_drivables=generic_drivables,
    )
    road_edge_linestrings = split_line_geometry_by_max_length(
        [road_edges.linestring for road_edges in road_edges_], MAX_ROAD_EDGE_LENGTH
    )

    for running_id, road_edge_linestring in enumerate(road_edge_linestrings):
        # TODO @DanielDauner: Figure out if other types should/could be assigned here.
        yield RoadEdge(
            object_id=running_id,
            road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
            polyline=Polyline3D.from_linestring(road_edge_linestring),
        )


# ------------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------------------------------------------------------


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
