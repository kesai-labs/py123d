from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.geometry as geom

from py123d.datatypes.map_objects.map_layer_types import LaneType
from py123d.datatypes.map_objects.map_objects import RoadEdge, RoadLine
from py123d.geometry import OccupancyMap2D, Point3D, Polyline3D, PolylineSE2, PoseSE2, Vector2D
from py123d.geometry.transform.transform_se2 import translate_se2_along_body_frame
from py123d.geometry.utils.rotation_utils import normalize_angle

MAX_LANE_WIDTH = 25.0  # meters
MIN_LANE_WIDTH = 2.0
DEFAULT_LANE_WIDTH = 3.7
BOUNDARY_STEP_SIZE = 0.25  # meters
MAX_Z_DISTANCE = 1.0  # meters

PERP_START_OFFSET = 0.1  # meters

MIN_HIT_DISTANCE = 0.1  # meters
MIN_AVERAGE_DISTANCE = 1.5
MAX_AVERAGE_DISTANCE = 7.0


def get_type_and_id_from_token(token: str) -> Tuple[str, int]:
    """Extract type and id from token."""
    line_type, line_id = token.split("_")
    return line_type, int(line_id)


def get_polyline_from_token(polyline_dict: Dict[str, Dict[int, Polyline3D]], token: str) -> Polyline3D:
    """Extract polyline from token."""
    line_type, line_id = get_type_and_id_from_token(token)
    return polyline_dict[line_type][line_id]


@dataclass
class WaymoLaneData:
    """Helper class to store lane data."""

    # Regular lane features
    # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L142
    object_id: int
    centerline: Polyline3D
    predecessor_ids: List[int]
    successor_ids: List[int]
    speed_limit_mps: Optional[float]
    lane_type: LaneType

    # Waymo allows multiple left/right neighbors
    # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L111
    left_neighbors: List[Dict[str, int]]
    right_neighbors: List[Dict[str, int]]

    # To be filled
    left_boundary: Optional[Polyline3D] = None
    right_boundary: Optional[Polyline3D] = None


@dataclass
class PerpendicularHit:
    """Helper class to store perpendicular hit data."""

    distance_along_perp_2d: float
    hit_point_3d: Point3D
    hit_polyline_token: str
    centerline_hit_crossing: bool
    heading_error: Optional[float] = None

    @property
    def hit_polyline_id(self) -> int:
        """Extract polyline id from token."""
        return get_type_and_id_from_token(self.hit_polyline_token)[1]

    @property
    def hit_polyline_type(self) -> int:
        """Extract polyline id from token."""
        return get_type_and_id_from_token(self.hit_polyline_token)[0]


def _collect_perpendicular_hits(
    lane_query_se2: PoseSE2,
    lane_token: str,
    polyline_dict: Dict[str, Dict[int, Polyline3D]],
    lane_polyline_se2_dict: Dict[int, PolylineSE2],
    occupancy_2d: OccupancyMap2D,
    sign: float,
) -> List[PerpendicularHit]:
    assert sign in [1.0, -1.0], "Sign must be either 1.0 (left) or -1.0 (right)"
    # perp_start_point = translate_along_yaw(lane_query_se2, Vector2D(0.0, sign * PERP_START_OFFSET))
    perp_start_point = lane_query_se2
    perp_end_point = translate_se2_along_body_frame(lane_query_se2, Vector2D(0.0, sign * MAX_LANE_WIDTH / 2.0))
    perp_linestring = geom.LineString([[perp_start_point.x, perp_start_point.y], [perp_end_point.x, perp_end_point.y]])

    lane_linestring = occupancy_2d.geometries[occupancy_2d.id_to_idx[lane_token]]

    # 1. find intersecting lines, compute 3D distance
    intersecting_tokens = occupancy_2d.intersects(perp_linestring)

    perpendicular_hits: List[PerpendicularHit] = []
    for intersecting_token in intersecting_tokens:
        intersecting_polyline_3d = get_polyline_from_token(polyline_dict, intersecting_token)
        intersecting_linestring = occupancy_2d.geometries[occupancy_2d.id_to_idx[intersecting_token]]
        centerline_hit_crossing: bool = (
            lane_linestring.intersects(intersecting_linestring) if intersecting_token.startswith("lane_") else False
        )

        intersecting_geom_points: List[geom.Point] = []
        intersecting_geometries = perp_linestring.intersection(intersecting_linestring)
        if isinstance(intersecting_geometries, geom.Point):
            intersecting_geom_points.append(intersecting_geometries)
        elif isinstance(intersecting_geometries, geom.MultiPoint):
            intersecting_geom_points.extend([geom for geom in intersecting_geometries.geoms])

        for intersecting_geom_point in intersecting_geom_points:
            distance_along_perp_2d = perp_linestring.project(intersecting_geom_point)

            distance_along_intersecting_norm = intersecting_linestring.project(intersecting_geom_point, normalized=True)
            intersecting_point_3d = intersecting_polyline_3d.interpolate(
                distance_along_intersecting_norm * intersecting_polyline_3d.length
            )

            heading_error = None
            if intersecting_token.startswith("lane_"):
                # Compute heading error if the intersecting token is a lane
                intersecting_polyline_se2 = lane_polyline_se2_dict[intersecting_token]
                lane_heading = intersecting_polyline_se2.interpolate(
                    distance_along_intersecting_norm * intersecting_polyline_se2.length
                )
                heading_error = normalize_angle(lane_query_se2.yaw - lane_heading.yaw)

            perpendicular_hits.append(
                PerpendicularHit(
                    distance_along_perp_2d=distance_along_perp_2d,
                    hit_point_3d=intersecting_point_3d,
                    hit_polyline_token=intersecting_token,
                    centerline_hit_crossing=centerline_hit_crossing,
                    heading_error=heading_error,
                )
            )

    return perpendicular_hits


def _filter_perpendicular_hits(
    perpendicular_hits: List[PerpendicularHit],
    lane_point_3d: Point3D,
) -> List[PerpendicularHit]:
    filtered_hits = []
    for hit in perpendicular_hits:
        # 1. filter hits too far in the vertical direction
        z_distance = np.abs(hit.hit_point_3d.z - lane_point_3d.z)
        if z_distance > MAX_Z_DISTANCE:
            continue

        # 2. filter hits that are too close and not with the road edge (e.g. close lane lines)
        if hit.distance_along_perp_2d < MIN_HIT_DISTANCE and hit.hit_polyline_type != "road-edge":
            continue

        filtered_hits.append(hit)

    # Sort hits by distance_along_perp_2d
    filtered_hits.sort(key=lambda hit: hit.distance_along_perp_2d)

    return filtered_hits


def fill_lane_boundaries(
    lane_data_dict: Dict[int, WaymoLaneData], road_lines: List[RoadLine], road_edges: List[RoadEdge]
) -> Tuple[Dict[str, Polyline3D], Dict[str, Polyline3D]]:
    """Welcome to insanity.

    :param lane_data: List of of WaymoLaneData helper class
    :param road_lines: List of RoadLine objects
    :param road_edges: List of RoadEdge objects
    :return: Tuple of left and right lane boundaries as 3D polylines
    """

    polyline_dict: Dict[str, Dict[int, Polyline3D]] = {"lane": {}, "road-line": {}, "road-edge": {}}
    lane_polyline_se2_dict: Dict[int, PolylineSE2] = {}

    for lane_id, lane in lane_data_dict.items():
        polyline_dict["lane"][lane_id] = lane.centerline
        lane_polyline_se2_dict[f"lane_{lane_id}"] = lane.centerline.polyline_se2

    # for road_line in road_lines:
    #     polyline_dict["road-line"][road_line.object_id] = road_line.polyline_3d

    for road_edge in road_edges:
        polyline_dict["road-edge"][road_edge.object_id] = road_edge.polyline_3d

    geometries = []
    tokens = []
    for line_type, polylines in polyline_dict.items():
        for polyline_id, polyline in polylines.items():
            geometries.append(polyline.polyline_2d.linestring)
            tokens.append(f"{line_type}_{polyline_id}")

    occupancy_2d = OccupancyMap2D(geometries, tokens)

    left_boundaries = {}
    right_boundaries = {}

    for lane_id, lane_polyline in polyline_dict["lane"].items():
        current_lane_token = f"lane_{lane_id}"
        lane_polyline_se2 = lane_polyline_se2_dict[current_lane_token]

        # 1. sample poses along centerline
        num_samples = int(lane_polyline.length / BOUNDARY_STEP_SIZE) + 1

        distances_se2 = np.linspace(0, lane_polyline_se2.length, num_samples, endpoint=True)
        lane_queries_se2 = [
            PoseSE2.from_array(pose_se2_array) for pose_se2_array in lane_polyline_se2.interpolate(distances_se2)
        ]
        distances_3d = np.linspace(0, lane_polyline.length, num_samples, endpoint=True)
        lane_queries_3d = [
            Point3D.from_array(point_3d_array) for point_3d_array in lane_polyline.interpolate(distances_3d)
        ]
        assert len(lane_queries_se2) == len(lane_queries_3d), (
            f"Number of sampled SE2 poses {len(lane_queries_se2)} and 3D points {len(lane_queries_3d)} must be the same"
        )

        for sign in [1.0, -1.0]:
            boundary_points_3d: List[Optional[Point3D]] = []
            for lane_query_se2, lane_query_3d in zip(lane_queries_se2, lane_queries_3d):
                perpendicular_hits = _collect_perpendicular_hits(
                    lane_query_se2=lane_query_se2,
                    lane_token=current_lane_token,
                    polyline_dict=polyline_dict,
                    lane_polyline_se2_dict=lane_polyline_se2_dict,
                    occupancy_2d=occupancy_2d,
                    sign=sign,
                )
                perpendicular_hits = _filter_perpendicular_hits(
                    perpendicular_hits=perpendicular_hits, lane_point_3d=lane_query_3d
                )

                boundary_point_3d: Optional[Point3D] = None
                # 1. First, try to find the boundary point from the perpendicular hits
                if len(perpendicular_hits) > 0:
                    first_hit = perpendicular_hits[0]

                    # 1.1. If the first hit is a road edge, use it as the boundary point
                    if first_hit.hit_polyline_type == "road-edge":
                        boundary_point_3d = first_hit.hit_point_3d
                    elif first_hit.hit_polyline_type == "road-line":
                        boundary_point_3d = first_hit.hit_point_3d
                    elif first_hit.hit_polyline_type == "lane":
                        for hit in perpendicular_hits:
                            if hit.hit_polyline_type == "road-edge":
                                continue
                            if hit.hit_polyline_type == "lane":
                                lane_data_dict[lane_id].predecessor_ids

                                has_same_predecessor = (
                                    len(
                                        set(lane_data_dict[hit.hit_polyline_id].predecessor_ids)
                                        & set(lane_data_dict[lane_id].predecessor_ids)
                                    )
                                    > 0
                                )
                                has_same_successor = (
                                    len(
                                        set(lane_data_dict[hit.hit_polyline_id].successor_ids)
                                        & set(lane_data_dict[lane_id].successor_ids)
                                    )
                                    > 0
                                )
                                heading_min = np.pi / 8.0
                                invalid_heading_error = heading_min < abs(hit.heading_error) < (np.pi - heading_min)
                                if (
                                    not has_same_predecessor
                                    and not has_same_successor
                                    and not hit.centerline_hit_crossing
                                    and MAX_AVERAGE_DISTANCE > hit.distance_along_perp_2d
                                    and MIN_AVERAGE_DISTANCE < hit.distance_along_perp_2d
                                    and not invalid_heading_error
                                ):
                                    # 2. if first hit is lane line, use it as boundary point
                                    boundary_point_3d = Point3D.from_array(
                                        (hit.hit_point_3d.array + lane_query_3d.array) / 2.0
                                    )
                                    break

                boundary_points_3d.append(boundary_point_3d)

            no_boundary_ratio = boundary_points_3d.count(None) / len(boundary_points_3d)
            final_boundary_points_3d = []

            def _get_default_boundary_point_3d(lane_query_se2: PoseSE2, lane_query_3d: Point3D, sign: float) -> Point3D:
                perp_boundary_distance = DEFAULT_LANE_WIDTH / 2.0
                boundary_point_se2 = translate_se2_along_body_frame(
                    lane_query_se2, Vector2D(0.0, sign * perp_boundary_distance)
                )
                return Point3D(boundary_point_se2.x, boundary_point_se2.y, lane_query_3d.z)

            if no_boundary_ratio > 0.8:
                for lane_query_se2, lane_query_3d in zip(lane_queries_se2, lane_queries_3d):
                    boundary_point_3d = _get_default_boundary_point_3d(lane_query_se2, lane_query_3d, sign)
                    final_boundary_points_3d.append(boundary_point_3d.array)

            else:
                for boundary_idx, (lane_query_se2, lane_query_3d) in enumerate(zip(lane_queries_se2, lane_queries_3d)):
                    if boundary_points_3d[boundary_idx] is None:
                        boundary_point_3d = _get_default_boundary_point_3d(lane_query_se2, lane_query_3d, sign)
                    else:
                        boundary_point_3d = boundary_points_3d[boundary_idx]
                    final_boundary_points_3d.append(boundary_point_3d.array)

                # # 2. If no boundary point was found, use the lane query point as the boundary point
                # if boundary_point_3d is None:

            if len(final_boundary_points_3d) > 1:
                if sign == 1.0:
                    lane_data_dict[lane_id].left_boundary = Polyline3D.from_array(
                        np.array(final_boundary_points_3d, dtype=np.float64)
                    )
                else:
                    lane_data_dict[lane_id].right_boundary = Polyline3D.from_array(
                        np.array(final_boundary_points_3d, dtype=np.float64)
                    )

    return left_boundaries, right_boundaries
