from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
import shapely.geometry as geom

from py123d.datatypes import (
    BaseMapObject,
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
)
from py123d.geometry import OccupancyMap2D, Point3DIndex, Polyline2D, Polyline3D
from py123d.parser.av2.utils.av2_constants import AV2_LANE_TYPE_MAPPING, AV2_ROAD_LINE_TYPE_MAPPING
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)
from py123d.parser.utils.map_utils.road_edge.road_edge_3d_utils import lift_road_edges_to_3d

LANE_GROUP_MARK_TYPES: List[str] = [
    "DASHED_WHITE",
    "DOUBLE_DASH_WHITE",
    "DASH_SOLID_WHITE",
    "SOLID_DASH_WHITE",
    "SOLID_WHITE",
]
MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0

logger = logging.getLogger(__name__)


# Map parser
# ----------------------------------------------------------------------------------------------------------------------


class Av2MapParser(BaseMapParser):
    """Lightweight, picklable handle to one AV2 map.

    Shared across AV2 dataset variants (sensor, motion, etc.) since they use the same map format.
    """

    def __init__(self, source_log_path: Path, split: str, dataset: str = "av2-sensor") -> None:
        self._source_log_path = source_log_path
        self._split = split
        self._dataset = dataset

    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return get_av2_map_metadata(self._split, self._source_log_path, dataset=self._dataset)

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        source_log_path = self._source_log_path

        map_folder = source_log_path / "map"
        log_map_archive_path = next(map_folder.glob("log_map_archive_*.json"))

        with open(log_map_archive_path, "r", encoding="utf-8") as f:
            log_map_archive = json.load(f)

        # Parse drivable areas
        drivable_areas: Dict[int, Polyline3D] = {}
        for drivable_area_id, drivable_area_dict in log_map_archive["drivable_areas"].items():
            drivable_areas[int(drivable_area_id)] = _extract_polyline(
                drivable_area_dict["area_boundary"], close=True, source_log_path=source_log_path
            )

        # Parse lane boundaries into Polyline3D
        for _, lane_segment_dict in log_map_archive["lane_segments"].items():
            lane_segment_dict["left_lane_boundary"] = _extract_polyline(
                lane_segment_dict["left_lane_boundary"], source_log_path=source_log_path
            )
            lane_segment_dict["right_lane_boundary"] = _extract_polyline(
                lane_segment_dict["right_lane_boundary"], source_log_path=source_log_path
            )

        # Parse crosswalk outlines
        for _, crosswalk_dict in log_map_archive["pedestrian_crossings"].items():
            p1, p2 = np.array([[p["x"], p["y"], p["z"]] for p in crosswalk_dict["edge1"]], dtype=np.float64)
            p3, p4 = np.array([[p["x"], p["y"], p["z"]] for p in crosswalk_dict["edge2"]], dtype=np.float64)
            crosswalk_dict["outline"] = Polyline3D.from_array(np.array([p1, p2, p4, p3, p1], dtype=np.float64))

        # Build derived structures (lane groups, intersections)
        lane_group_dict = _extract_lane_group_dict(log_map_archive["lane_segments"])
        intersection_dict = _extract_intersection_dict(log_map_archive["lane_segments"], lane_group_dict)

        # Yield all map objects
        yield from _iter_av2_lanes(log_map_archive["lane_segments"])
        yield from _iter_av2_lane_groups(lane_group_dict)
        yield from _iter_av2_intersections(intersection_dict)
        yield from _iter_av2_crosswalks(log_map_archive["pedestrian_crossings"])
        yield from _iter_av2_generic_drivables(drivable_areas)
        yield from _iter_av2_road_edges(drivable_areas)
        yield from _iter_av2_road_lines(log_map_archive["lane_segments"])


# Public helpers (used by dataset parsers for metadata)
# ----------------------------------------------------------------------------------------------------------------------


def get_av2_map_metadata(split: str, source_log_path: Path, dataset: str = "av2-sensor") -> MapMetadata:
    """Get map metadata for an AV2 log.

    :param split: Dataset split name, e.g. "av2-sensor_train".
    :param source_log_path: Path to the AV2 source log folder.
    :param dataset: Dataset name, defaults to "av2-sensor".
    :return: MapMetadata for this log.
    """
    map_folder = source_log_path / "map"
    log_map_archive_path = next(map_folder.glob("log_map_archive_*.json"), None)
    assert log_map_archive_path is not None, f"Log map archive file not found in {map_folder}."
    location = log_map_archive_path.name.split("____")[1].split("_")[0]
    return MapMetadata(
        dataset=dataset,
        split=split,
        log_name=source_log_path.name,
        location=location,
        map_has_z=True,
        map_is_per_log=True,
    )


# Polyline / geometry helpers
# ----------------------------------------------------------------------------------------------------------------------


def _extract_polyline(
    data: List[Dict[str, float]], close: bool = False, source_log_path: Optional[Path] = None
) -> Polyline3D:
    """Instantiate a Polyline3D from AV2 coordinate dicts."""
    polyline = np.array([[p["x"], p["y"], p["z"]] for p in data], dtype=np.float64)
    if close:
        polyline = np.vstack([polyline, polyline[0]])

    # NOTE @DanielDauner: AV2 map can have NaN values in the Z axis.
    # In this case we replace NaNs with the median (or zeros).
    if np.isnan(polyline).any():
        median_z = np.nanmedian(polyline[:, 2], axis=-1)
        logger.warning(
            f"Found NaN values in {source_log_path} polyline data: Replacing NaNs of z-axis with median height."
        )
        for i in range(polyline.shape[0]):
            if pd.isna(polyline[i, 2]):
                polyline[i, 2] = median_z if not pd.isna(median_z) else 0.0

    return Polyline3D.from_array(polyline)


def _get_centerline_from_boundaries(
    left_boundary: Polyline3D,
    right_boundary: Polyline3D,
    resolution: float = 0.1,
) -> Polyline3D:
    """Compute centerline from left and right lane boundaries."""
    points_per_meter = 1 / resolution
    num_points = int(np.ceil(max([right_boundary.length, left_boundary.length]) * points_per_meter))
    right_array = right_boundary.interpolate(np.linspace(0, right_boundary.length, num_points, endpoint=True))
    left_array = left_boundary.interpolate(np.linspace(0, left_boundary.length, num_points, endpoint=True))
    return Polyline3D.from_array(np.mean([right_array, left_array], axis=0))


# Map object iterators
# ----------------------------------------------------------------------------------------------------------------------


def _iter_av2_lanes(lanes: Dict[str, Any]) -> Iterator[Lane]:
    """Yield Lane objects from AV2 lane segments."""
    for lane_id, lane_dict in lanes.items():
        centerline = _get_centerline_from_boundaries(
            left_boundary=lane_dict["left_lane_boundary"],
            right_boundary=lane_dict["right_lane_boundary"],
        )

        # NOTE @DanielDauner: Some neighbor lane IDs might not be present in the dataset.
        left_lane_id = lane_dict["left_neighbor_id"] if str(lane_dict["left_neighbor_id"]) in lanes else None
        right_lane_id = lane_dict["right_neighbor_id"] if str(lane_dict["right_neighbor_id"]) in lanes else None

        lane_type = AV2_LANE_TYPE_MAPPING.get(lane_dict.get("lane_type", ""), LaneType.UNDEFINED)

        yield Lane(
            object_id=lane_id,
            lane_type=lane_type,
            lane_group_id=lane_dict["lane_group_id"],
            left_boundary=lane_dict["left_lane_boundary"],
            right_boundary=lane_dict["right_lane_boundary"],
            centerline=centerline,
            left_lane_id=left_lane_id,
            right_lane_id=right_lane_id,
            predecessor_ids=lane_dict["predecessors"],
            successor_ids=lane_dict["successors"],
            speed_limit_mps=None,
            outline=None,
            shapely_polygon=None,
        )


def _iter_av2_lane_groups(lane_group_dict: Dict[int, Any]) -> Iterator[LaneGroup]:
    """Yield LaneGroup objects."""
    for lane_group_id, lane_group_values in lane_group_dict.items():
        yield LaneGroup(
            object_id=lane_group_id,
            lane_ids=lane_group_values["lane_ids"],
            left_boundary=lane_group_values["left_boundary"],
            right_boundary=lane_group_values["right_boundary"],
            intersection_id=lane_group_values["intersection_id"],
            predecessor_ids=lane_group_values["predecessor_ids"],
            successor_ids=lane_group_values["successor_ids"],
            outline=None,
            shapely_polygon=None,
        )


def _iter_av2_intersections(intersection_dict: Dict[int, Any]) -> Iterator[Intersection]:
    """Yield Intersection objects."""
    for intersection_id, intersection_values in intersection_dict.items():
        yield Intersection(
            object_id=intersection_id,
            intersection_type=IntersectionType.DEFAULT,
            lane_group_ids=intersection_values["lane_group_ids"],
            outline=intersection_values["outline_3d"],
        )


def _iter_av2_crosswalks(crosswalks: Dict[str, Any]) -> Iterator[Crosswalk]:
    """Yield Crosswalk objects."""
    for crosswalk_id, crosswalk_dict in crosswalks.items():
        yield Crosswalk(
            object_id=crosswalk_id,
            outline=crosswalk_dict["outline"],
        )


def _iter_av2_generic_drivables(drivable_areas: Dict[int, Polyline3D]) -> Iterator[GenericDrivable]:
    """Yield GenericDrivable objects."""
    for drivable_area_id, drivable_area_outline in drivable_areas.items():
        yield GenericDrivable(
            object_id=drivable_area_id,
            outline=drivable_area_outline,
        )


def _iter_av2_road_edges(drivable_areas: Dict[int, Polyline3D]) -> Iterator[RoadEdge]:
    """Yield RoadEdge objects derived from drivable area boundaries.

    NOTE @DanielDauner: We merge all drivable areas in 2D and lift the outlines to 3D.
    Currently the method assumes that the drivable areas do not overlap and all road surfaces are included.
    """
    drivable_polygons = [geom.Polygon(da.array[:, :2]) for da in drivable_areas.values()]
    road_edges_2d = get_road_edge_linear_rings(drivable_polygons)
    non_conflicting_road_edges = lift_road_edges_to_3d(road_edges_2d, list(drivable_areas.values()))
    non_conflicting_road_edges_linestrings = [polyline.linestring for polyline in non_conflicting_road_edges]
    road_edges = split_line_geometry_by_max_length(non_conflicting_road_edges_linestrings, MAX_ROAD_EDGE_LENGTH)

    for idx, road_edge in enumerate(road_edges):
        # TODO @DanielDauner: Figure out if other road edge types should/could be assigned here.
        yield RoadEdge(
            object_id=idx,
            road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
            polyline=Polyline3D.from_linestring(road_edge),
        )


def _iter_av2_road_lines(lanes: Dict[str, Any]) -> Iterator[RoadLine]:
    """Yield RoadLine objects from lane boundary markings."""
    running_road_line_id = 0
    for lane in lanes.values():
        for side in ["left", "right"]:
            # NOTE @DanielDauner: We currently ignore lane markings that are NONE in the AV2 dataset.
            if lane[f"{side}_lane_mark_type"] == "NONE":
                continue
            yield RoadLine(
                object_id=running_road_line_id,
                road_line_type=AV2_ROAD_LINE_TYPE_MAPPING[lane[f"{side}_lane_mark_type"]],
                polyline=lane[f"{side}_lane_boundary"],
            )
            running_road_line_id += 1


# Lane group & intersection extraction
# ----------------------------------------------------------------------------------------------------------------------


def _extract_lane_group_dict(lanes: Dict[str, Any]) -> Dict[int, Any]:
    """Collect lane groups from neighboring lanes by traversing neighbor links.

    :param lanes: Dictionary of lane segment information.
    :return: Dictionary of lane group information.
    """
    lane_group_sets = _extract_lane_groups(lanes)
    lane_group_set_dict = {i: lane_group for i, lane_group in enumerate(lane_group_sets)}
    lane_group_dict: Dict[int, Dict[str, Any]] = {}

    def _get_lane_group_ids_of_lane_ids(lane_ids: List[str]) -> List[int]:
        lane_group_ids_: List[int] = []
        for lane_group_id_, lane_group_set_ in lane_group_set_dict.items():
            if any(str(lane_id) in lane_group_set_ for lane_id in lane_ids):
                lane_group_ids_.append(lane_group_id_)
        return list(set(lane_group_ids_))

    for lane_group_id, lane_group_set in lane_group_set_dict.items():
        lane_group_dict[lane_group_id] = {}
        lane_group_dict[lane_group_id]["id"] = lane_group_id
        lane_group_dict[lane_group_id]["lane_ids"] = [int(lane_id) for lane_id in lane_group_set]

        successor_lanes: List[str] = []
        predecessor_lanes: List[str] = []
        for lane_id in lane_group_set:
            lane_dict = lanes[str(lane_id)]
            lane_dict["lane_group_id"] = lane_group_id
            successor_lanes.extend(lane_dict["successors"])
            predecessor_lanes.extend(lane_dict["predecessors"])

        left_boundary = lanes[lane_group_set[0]]["left_lane_boundary"]
        right_boundary = lanes[lane_group_set[-1]]["right_lane_boundary"]

        lane_group_dict[lane_group_id]["intersection_id"] = None
        lane_group_dict[lane_group_id]["predecessor_ids"] = _get_lane_group_ids_of_lane_ids(predecessor_lanes)
        lane_group_dict[lane_group_id]["successor_ids"] = _get_lane_group_ids_of_lane_ids(successor_lanes)
        lane_group_dict[lane_group_id]["left_boundary"] = left_boundary
        lane_group_dict[lane_group_id]["right_boundary"] = right_boundary
        outline_array = np.vstack(
            [
                left_boundary.array[:, :3],
                right_boundary.array[:, :3][::-1],
                left_boundary.array[0, :3][None, ...],
            ]
        )
        lane_group_dict[lane_group_id]["outline"] = Polyline3D.from_array(outline_array)

    return lane_group_dict


def _extract_lane_groups(lanes: Dict[str, Any]) -> List[List[str]]:
    """Extract lane groups by traversing neighboring lanes.

    :param lanes: Dictionary of lane information.
    :return: List of lane groups, where each lane group is a list of lane IDs.
    """
    visited: set[str] = set()
    lane_groups: List[List[str]] = []

    def _get_valid_neighbor_id(lane_data: Dict[str, Any], direction: str) -> Optional[str]:
        neighbor_id = str(lane_data.get(f"{direction}_neighbor_id"))
        mark_type = lane_data.get(f"{direction}_lane_mark_type", None)
        if (neighbor_id is not None) and (neighbor_id in lanes) and (mark_type in LANE_GROUP_MARK_TYPES):
            return neighbor_id
        return None

    def _traverse_group(start_lane_id: str) -> List[str]:
        group = [start_lane_id]
        queue = [start_lane_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            left_neighbor = _get_valid_neighbor_id(lanes[current_id], "left")
            if left_neighbor is not None and left_neighbor not in visited:
                queue.append(left_neighbor)
                group = [left_neighbor] + group

            right_neighbor = _get_valid_neighbor_id(lanes[current_id], "right")
            if right_neighbor is not None and right_neighbor not in visited:
                queue.append(right_neighbor)
                group += [right_neighbor]

        return group

    for lane_id in lanes:
        if lane_id not in visited:
            group = _traverse_group(lane_id)
            lane_groups.append(group)

    return lane_groups


def _extract_intersection_dict(
    lanes: Dict[str, Any],
    lane_group_dict: Dict[int, Any],
    max_distance: float = 0.01,
) -> Dict[int, Any]:
    """Extract intersection outlines from lane groups.

    :param lanes: Dictionary of lane information.
    :param lane_group_dict: Dictionary of lane group information.
    :param max_distance: Maximum distance to consider for intersection boundaries.
    :return: Dictionary of intersection information.
    """

    def _interpolate_z_on_segment(point: shapely.Point, segment_coords: npt.NDArray[np.float64]) -> float:
        p1, p2 = segment_coords[0], segment_coords[1]
        segment_vec = p2[:2] - p1[:2]
        point_vec = np.array([point.x, point.y]) - p1[:2]
        segment_length_sq = np.dot(segment_vec, segment_vec)
        if segment_length_sq == 0:
            return p1[2]
        t = np.clip(np.dot(point_vec, segment_vec) / segment_length_sq, 0, 1)
        return p1[2] + t * (p2[2] - p1[2])

    # 1. Collect lane groups where at least one lane is an intersection.
    lane_group_intersection_dict: Dict[int, Any] = {}
    for lane_group_id, lane_group in lane_group_dict.items():
        is_intersection_lanes = [lanes[str(lane_id)]["is_intersection"] for lane_id in lane_group["lane_ids"]]
        if any(is_intersection_lanes):
            lane_group_intersection_dict[lane_group_id] = lane_group

    # 2. Merge intersection polygons in 2D.
    lane_group_intersection_geometry: Dict[int, shapely.Polygon] = {}
    for lane_group_id, lane_group in lane_group_intersection_dict.items():
        lane_group_polygon_2d = shapely.Polygon(lane_group["outline"].array[:, Point3DIndex.XY])
        if lane_group_polygon_2d.is_valid:
            lane_group_intersection_geometry[lane_group_id] = lane_group_polygon_2d

    intersection_polygons = gpd.GeoSeries(lane_group_intersection_geometry).union_all()

    # 3. Collect intersection polygons and their lane group IDs.
    geometries: List[shapely.Polygon] = []
    if isinstance(intersection_polygons, geom.Polygon):
        geometries.append(intersection_polygons)
    elif isinstance(intersection_polygons, geom.MultiPolygon):
        geometries.extend(intersection_polygons.geoms)

    intersection_dict: Dict[int, Any] = {}
    for intersection_idx, intersection_polygon in enumerate(geometries):
        if intersection_polygon.is_empty:
            continue
        lane_group_ids = [
            lg_id
            for lg_id, lg_polygon in lane_group_intersection_geometry.items()
            if intersection_polygon.intersects(lg_polygon)
        ]
        for lg_id in lane_group_ids:
            lane_group_dict[lg_id]["intersection_id"] = intersection_idx

        intersection_dict[intersection_idx] = {
            "id": intersection_idx,
            "outline_2d": Polyline2D.from_array(np.array(list(intersection_polygon.exterior.coords), dtype=np.float64)),
            "lane_group_ids": lane_group_ids,
        }

    # 4. Lift intersection outlines to 3D.
    boundary_segments_list: List[npt.NDArray[np.float64]] = []
    for lane_group in lane_group_intersection_dict.values():
        coords = np.array(lane_group["outline"].linestring.coords, dtype=np.float64).reshape(-1, 1, 3)
        segment_coords_boundary = np.concatenate([coords[:-1], coords[1:]], axis=1)
        boundary_segments_list.append(segment_coords_boundary)

    if len(boundary_segments_list) >= 1:
        boundary_segments = np.concatenate(boundary_segments_list, axis=0)
        boundary_segment_linestrings = shapely.creation.linestrings(boundary_segments)
        occupancy_map = OccupancyMap2D(boundary_segment_linestrings)  # type: ignore[arg-type]

        for intersection_id, intersection_data in intersection_dict.items():
            points_2d = intersection_data["outline_2d"].array
            points_3d = np.zeros((len(points_2d), 3), dtype=np.float64)
            points_3d[:, :2] = points_2d

            query_points = shapely.creation.points(points_2d)
            results = occupancy_map.query_nearest(query_points, max_distance=max_distance, exclusive=True)
            for query_idx, geometry_idx in zip(*results):
                query_point = query_points[query_idx]
                segment_coords = boundary_segments[geometry_idx]
                best_z = _interpolate_z_on_segment(query_point, segment_coords)
                points_3d[query_idx, 2] = best_z

            intersection_dict[intersection_id]["outline_3d"] = Polyline3D.from_array(points_3d)

    return intersection_dict
