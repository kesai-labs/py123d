import logging
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional, Tuple

import numpy as np
import shapely

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
    RoadEdge,
    RoadEdgeType,
    NoneLane,
    RoadLine,
    RoadLineType,
    Shoulder,
    Walkway,
)
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.geometry.geometry_index import Point3DIndex
from py123d.geometry.polyline import Polyline3D
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.opendrive.utils.collection import collect_element_helpers
from py123d.parser.opendrive.utils.id_system import build_lane_id, lane_section_id_from_lane_group_id
from py123d.parser.opendrive.utils.lane_helper import OpenDriveLaneGroupHelper, OpenDriveLaneHelper
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
MIN_ROAD_EDGE_HOLE_WIDTH: Final[float] = 1.0  # [m] drops sliver holes between adjacent surfaces at junction corners


class OpenDriveMapParser(BaseMapParser):
    """Lightweight, picklable handle to one OpenDRIVE map."""

    def __init__(
        self,
        xodr_path: Path,
        location: Optional[str] = None,
        interpolation_step_size: float = 1.0,
        connection_distance_threshold: float = 0.1,
        internal_only: bool = True,
        road_edge_fill_hole_points: Optional[List[List[float]]] = None,
        road_edge_non_drivable_points: Optional[List[List[float]]] = None,
        non_drivable_none_lane_min_width: Optional[float] = None,
    ) -> None:
        self._xodr_path = xodr_path
        self._location = location
        self._interpolation_step_size = interpolation_step_size
        self._connection_distance_threshold = connection_distance_threshold
        self._internal_only = internal_only
        self._road_edge_fill_hole_points = road_edge_fill_hole_points
        self._road_edge_non_drivable_points = road_edge_non_drivable_points
        self._non_drivable_none_lane_min_width = non_drivable_none_lane_min_width

    def get_map_metadata(self) -> MapMetadata:
        """Returns metadata for this OpenDRIVE map."""
        # If location is not provided, use the file name as location
        _location = self._location or self._xodr_path.name.removesuffix("".join(self._xodr_path.suffixes))
        return MapMetadata(
            dataset="opendrive",
            location=_location,
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
            road_edge_fill_hole_points=self._road_edge_fill_hole_points,
            road_edge_non_drivable_points=self._road_edge_non_drivable_points,
            non_drivable_none_lane_min_width=self._non_drivable_none_lane_min_width,
        )


def iter_xodr_map_objects(
    xodr_file: Path,
    interpolation_step_size: float = 1.0,
    connection_distance_threshold: float = 0.1,
    internal_only: bool = True,
    road_edge_fill_hole_points: Optional[List[List[float]]] = None,
    road_edge_non_drivable_points: Optional[List[List[float]]] = None,
    non_drivable_none_lane_min_width: Optional[float] = None,
) -> Iterator[BaseMapObject]:
    """Yields all map objects extracted from an OpenDRIVE (.xodr) file.

    :param xodr_file: Path to the OpenDRIVE (.xodr) file.
    :param interpolation_step_size: Step size for interpolating polylines, defaults to 1.0
    :param connection_distance_threshold: Distance threshold for connecting road elements, defaults to 0.1
    :param internal_only: If True, only yield internal road lines (center + between lanes), defaults to True
    :param road_edge_fill_hole_points: Road-edge holes containing one of these (x, y) points are filled,
        patching known bugs in the source map, defaults to None
    :param road_edge_non_drivable_points: Shoulder/none-lane surfaces containing one of these (x, y)
        points are carved out of the drivable envelope, marking areas that are not drivable in
        reality, defaults to None
    :param non_drivable_none_lane_min_width: None lanes on non-junction roads at least this
        wide are median strips and carved out of the drivable envelope, defaults to None (disabled)
    """
    opendrive = XODR.parse_from_file(xodr_file)

    (
        road_dict,
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

    car_parks, junction_parking_drivables = _extract_carparks(lane_helper_dict, road_dict)
    yield from car_parks

    generic_drivables = _extract_generic_drivables(lane_helper_dict) + junction_parking_drivables
    yield from generic_drivables

    shoulders = _extract_shoulders(lane_helper_dict)
    yield from shoulders

    none_lanes, curbed_none_lanes = _extract_none_lanes(lane_helper_dict)
    yield from none_lanes
    yield from curbed_none_lanes

    # Curbed none lanes are carve candidates; auto-carving is disabled pending per-town validation.
    non_drivable_polygons = _match_non_drivable_surfaces(shoulders + none_lanes, road_edge_non_drivable_points)
    non_drivable_polygons += _collect_median_polygons(lane_helper_dict, road_dict, non_drivable_none_lane_min_width)
    non_drivable_polygons = _subtract_lane_coverage(non_drivable_polygons, lanes)

    # Yield other map elements
    yield from _iter_walkways(lane_helper_dict)
    yield from _iter_intersections(junction_dict, lane_group_helper_dict)
    yield from _iter_crosswalks(object_helper_dict)
    yield from _iter_stop_zones(signal_dict, lane_helper_dict)

    # Yield polyline elements that are inferred from other road surfaces
    yield from _iter_road_lines(lane_helper_dict, lane_groups, center_lane_marks_dict, internal_only)
    yield from _iter_road_edges(
        lanes,
        lane_groups,
        car_parks,
        generic_drivables + shoulders + none_lanes,
        road_edge_fill_hole_points,
        non_drivable_polygons,
    )


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
                    lane_type=LaneType.SURFACE_STREET,
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


def _extract_carparks(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper], road_dict
) -> Tuple[List[Carpark], List[GenericDrivable]]:
    """Extracts carparks from lane helpers.
    Parking lanes on junction roads are connector pavement dragged through the intersection by the
    map generator, not real carparks; they are returned as generic drivables instead.
    """
    car_parks: List[Carpark] = []
    junction_parking: List[GenericDrivable] = []
    for lane_id, lane_helper in lane_helper_dict.items():
        if lane_helper.type != "parking":
            continue
        road = road_dict.get(int(lane_id.split("_")[0]))
        on_junction = road is not None and road.junction is not None and str(road.junction) not in ("-1", "None")
        if on_junction:
            junction_parking.append(GenericDrivable(object_id=lane_helper.lane_id, outline=lane_helper.outline_polyline_3d))
        else:
            car_parks.append(Carpark(object_id=lane_helper.lane_id, outline=lane_helper.outline_polyline_3d))
    return car_parks, junction_parking


def _extract_generic_drivables(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> List[GenericDrivable]:
    """Extracts generic drivables from lane helpers."""
    return [
        GenericDrivable(object_id=lh.lane_id, outline=lh.outline_polyline_3d)
        for lh in lane_helper_dict.values()
        if lh.type in {"border", "bidirectional"}
    ]


def _extract_shoulders(lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> List[Shoulder]:
    """Extracts shoulders from lane helpers."""
    return [
        Shoulder(object_id=lh.lane_id, outline=lh.outline_polyline_3d)
        for lh in lane_helper_dict.values()
        if lh.type == "shoulder"
    ]


def _extract_none_lanes(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
) -> Tuple[List[NoneLane], List[NoneLane]]:
    """Extracts none/restricted-type lane surfaces from lane helpers, split into (flat, curbed).
    A none lane behind a curb road mark on its inner boundary is raised and not drivable;
    curbed areas are excluded from road-edge inference so they stay outside the drivable envelope.
    """
    flat_areas: List[NoneLane] = []
    curbed_areas: List[NoneLane] = []
    for lane_id, lane_helper in lane_helper_dict.items():
        if lane_helper.type not in {"none", "restricted"}:
            continue
        none_lane = NoneLane(object_id=lane_helper.lane_id, outline=lane_helper.outline_polyline_3d)
        if _has_inner_curb(lane_id, lane_helper_dict):
            curbed_areas.append(none_lane)
        else:
            flat_areas.append(none_lane)
    return flat_areas, curbed_areas


def _subtract_lane_coverage(
    non_drivable_polygons: List[shapely.Polygon], lanes: List[Lane]
) -> List[shapely.Polygon]:
    """Removes driving-lane surface from carve polygons: lanes crossing an island stay drivable."""
    if not non_drivable_polygons:
        return []
    lane_union = shapely.union_all([lane.shapely_polygon for lane in lanes]).buffer(0.05, join_style=2)
    carved: List[shapely.Polygon] = []
    for polygon in non_drivable_polygons:
        remainder = polygon.difference(lane_union)
        parts = remainder.geoms if remainder.geom_type == "MultiPolygon" else [remainder]
        carved.extend(part for part in parts if part.geom_type == "Polygon" and part.area > 1.0)
    return carved


def _match_non_drivable_surfaces(surfaces, points: Optional[List[List[float]]]) -> List[shapely.Polygon]:
    """Returns polygons of the surfaces that contain one of the given (x, y) points."""
    if not points:
        return []
    shapely_points = [shapely.Point(p) for p in points]
    matched = []
    for surface in surfaces:
        polygon = shapely.Polygon(surface.outline.array[:, :2])
        if any(polygon.contains(point) for point in shapely_points):
            matched.append(polygon)
    return matched


def _collect_median_polygons(
    lane_helper_dict: Dict[str, OpenDriveLaneHelper],
    road_dict,
    min_width: Optional[float],
) -> List[shapely.Polygon]:
    """Polygons of wide none lanes on non-junction roads: median strips, not drivable."""
    if min_width is None:
        return []
    median_polygons = []
    for lane_id, lane_helper in lane_helper_dict.items():
        if lane_helper.type not in {"none", "restricted"}:
            continue
        road = road_dict.get(int(lane_id.split("_")[0]))
        if road is None or (road.junction is not None and str(road.junction) not in ("-1", "None")):
            continue
        polygon = shapely.Polygon(lane_helper.outline_polyline_3d.array[:, :2]).buffer(0)
        length = lane_helper.center_polyline_se2.length
        if length < 1e-3 or polygon.area / length < min_width:
            continue
        median_polygons.append(polygon)
    return median_polygons


def _has_inner_curb(lane_id: str, lane_helper_dict: Dict[str, OpenDriveLaneHelper]) -> bool:
    """True if the next lane toward the road center carries a curb road mark on the shared boundary."""
    road_idx, lane_section_idx, _, lane_idx = lane_id.split("_")
    lane_idx = int(lane_idx)
    inner_lane_idx = lane_idx - 1 if lane_idx > 0 else lane_idx + 1
    if inner_lane_idx == 0:
        return False
    inner_lane_id = build_lane_id(int(road_idx), int(lane_section_idx), inner_lane_idx)
    inner_helper = lane_helper_dict.get(inner_lane_id)
    if inner_helper is None or not inner_helper.open_drive_lane.road_marks:
        return False
    return any(mark.type == "curb" for mark in inner_helper.open_drive_lane.road_marks)


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

    # Pre-build junction_id -> lane_group_helpers mapping
    junction_to_lane_groups: Dict[int, List[OpenDriveLaneGroupHelper]] = {}
    for lane_group_helper in lane_group_helper_dict.values():
        if lane_group_helper.junction_id is not None:
            junction_to_lane_groups.setdefault(lane_group_helper.junction_id, []).append(lane_group_helper)

    for junction in junction_dict.values():
        lane_group_helpers = junction_to_lane_groups.get(junction.id, [])
        lane_group_ids_ = [lane_group_helper.lane_group_id for lane_group_helper in lane_group_helpers]
        if len(lane_group_ids_) == 0:
            logger.debug(f"Skipping Junction {junction.id} without lane groups!")
            continue

        outline = _extract_intersection_outline(lane_group_helpers, junction.id)
        yield Intersection(
            object_id=junction.id,
            intersection_type=IntersectionType.DEFAULT,
            lane_group_ids=lane_group_ids_,
            outline=outline,
        )


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

    # Pre-build lane_id -> (group_id, index) mapping for O(1) lookup
    lane_id_to_group_index: Dict[str, int] = {}
    for lg in lane_groups:
        for idx, lid in enumerate(lg.lane_ids):
            lane_id_to_group_index[lid] = idx

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

        lane_idx_in_group = lane_id_to_group_index.get(lane_id)
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
    fill_hole_points: Optional[List[List[float]]] = None,
    non_drivable_polygons: Optional[List[shapely.Polygon]] = None,
) -> Iterator[RoadEdge]:
    """Yields road edge objects."""
    road_edges_ = get_road_edges_3d_from_drivable_surfaces(
        lanes=lanes,
        lane_groups=lane_groups,
        car_parks=car_parks,
        generic_drivables=generic_drivables,
        min_interior_width=MIN_ROAD_EDGE_HOLE_WIDTH,
        fill_hole_points=fill_hole_points,
        non_drivable_polygons=non_drivable_polygons,
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
) -> Optional[Polyline3D]:
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
    valid_outlines = [
        outline for outline in intersection_outlines if outline.array.shape[0] > 2 and outline.array.shape[1] >= 3
    ]
    if len(valid_outlines) == 0:
        logger.warning(
            f"Could not extract valid outline for intersection {junction_id} with {len(intersection_edges)} edges!"
        )
        longest_outline_2d = max(intersection_edges, key=lambda outline: outline.length)
        outlines_with_z = [o for o in intersection_outlines if o.array.shape[1] >= 3 and o.array.shape[0] > 0]
        average_z = (
            sum(outline.array[:, 2].mean() for outline in outlines_with_z) / len(outlines_with_z)
            if outlines_with_z
            else 0.0
        )

        outline_3d_array = np.zeros((len(longest_outline_2d.coords), 3))
        outline_3d_array[:, Point3DIndex.XY] = np.array(longest_outline_2d.coords)
        outline_3d_array[:, Point3DIndex.Z] = average_z
        longest_outline = Polyline3D.from_array(outline_3d_array)
    else:
        longest_outline = max(valid_outlines, key=lambda outline: outline.length)

    return longest_outline
