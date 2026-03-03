from typing import Dict, List, Optional

import numpy as np

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.datasets.wod.utils.wod_boundary_utils import WaymoLaneData, fill_lane_boundaries
from py123d.conversion.datasets.wod.utils.wod_constants import (
    WAYMO_LANE_TYPE_CONVERSION,
    WAYMO_ROAD_EDGE_TYPE_CONVERSION,
    WAYMO_ROAD_LINE_TYPE_CONVERSION,
)
from py123d.datatypes.map_objects.map_layer_types import LaneType, RoadEdgeType, RoadLineType
from py123d.datatypes.map_objects.map_objects import Carpark, Crosswalk, Lane, LaneGroup, RoadEdge, RoadLine
from py123d.geometry import Polyline3D
from py123d.geometry.utils.units import mph_to_mps

check_dependencies(modules=["waymo_open_dataset"], optional_name="waymo")
from waymo_open_dataset.protos import map_pb2

# TODO:
# - Implement stop signs
# - Implement speed bumps
# - Implement driveways with a different semantic type if needed
# - Implement intersections and lane group logic


def convert_wod_map(map_features: List[map_pb2.MapFeature], map_writer: AbstractMapWriter) -> None:
    # We first extract all road lines, road edges, and lanes, and write them to the map writer.
    # NOTE: road lines and edges are used needed to extract lane boundaries.
    road_lines = _write_and_get_waymo_road_lines(map_features, map_writer)
    road_edges = _write_and_get_waymo_road_edges(map_features, map_writer)
    lanes = _write_and_get_waymo_lanes(map_features, road_lines, road_edges, map_writer)

    # Write lane groups based on the extracted lanes
    _write_waymo_lane_groups(lanes, map_writer)

    # Write miscellaneous surfaces (carparks, crosswalks, stop zones, etc.) directly from the Waymo frame proto
    _write_waymo_misc_surfaces(map_features, map_writer)


def _write_and_get_waymo_road_lines(
    map_features: List[map_pb2.MapFeature], map_writer: AbstractMapWriter
) -> List[RoadLine]:
    """Helper function to extract road lines from a Waymo frame proto."""

    road_lines: List[RoadLine] = []
    for map_feature in map_features:
        if map_feature.HasField("road_line"):
            polyline = _extract_polyline_waymo_proto(map_feature.road_line)
            if polyline is not None:
                road_line_type = WAYMO_ROAD_LINE_TYPE_CONVERSION.get(map_feature.road_line.type, RoadLineType.UNKNOWN)
                road_lines.append(
                    RoadLine(
                        object_id=map_feature.id,
                        road_line_type=road_line_type,
                        polyline=polyline,
                    )
                )

    for road_line in road_lines:
        map_writer.write_road_line(road_line)

    return road_lines


def _write_and_get_waymo_road_edges(
    map_features: List[map_pb2.MapFeature], map_writer: AbstractMapWriter
) -> List[RoadEdge]:
    """Helper function to extract road edges from a Waymo frame proto."""

    road_edges: List[RoadEdge] = []
    for map_feature in map_features:
        if map_feature.HasField("road_edge"):
            polyline = _extract_polyline_waymo_proto(map_feature.road_edge)
            if polyline is not None:
                road_edge_type = WAYMO_ROAD_EDGE_TYPE_CONVERSION.get(map_feature.road_edge.type, RoadEdgeType.UNKNOWN)
                road_edges.append(
                    RoadEdge(
                        object_id=map_feature.id,
                        road_edge_type=road_edge_type,
                        polyline=polyline,
                    )
                )

    for road_edge in road_edges:
        map_writer.write_road_edge(road_edge)

    return road_edges


def _write_and_get_waymo_lanes(
    map_features: List[map_pb2.MapFeature],
    road_lines: List[RoadLine],
    road_edges: List[RoadEdge],
    map_writer: AbstractMapWriter,
) -> List[Lane]:
    # 1. Load lane data from Waymo frame proto
    lane_data_dict: Dict[int, WaymoLaneData] = {}
    for map_feature in map_features:
        if map_feature.HasField("lane"):
            centerline = _extract_polyline_waymo_proto(map_feature.lane)

            # In case of a invalid lane, skip it
            if centerline is None:
                continue

            speed_limit_mps = mph_to_mps(map_feature.lane.speed_limit_mph)
            speed_limit_mps = speed_limit_mps if speed_limit_mps > 0.0 else None

            lane_data_dict[map_feature.id] = WaymoLaneData(
                object_id=map_feature.id,
                centerline=centerline,
                predecessor_ids=[int(lane_id_) for lane_id_ in map_feature.lane.entry_lanes],
                successor_ids=[int(lane_id_) for lane_id_ in map_feature.lane.exit_lanes],
                speed_limit_mps=speed_limit_mps,
                lane_type=WAYMO_LANE_TYPE_CONVERSION.get(map_feature.lane.type, LaneType.UNDEFINED),
                left_neighbors=_extract_lane_neighbors(map_feature.lane.left_neighbors),
                right_neighbors=_extract_lane_neighbors(map_feature.lane.right_neighbors),
            )

    # 2. Process lane data to fill in left/right boundaries
    fill_lane_boundaries(lane_data_dict, road_lines, road_edges)

    def _get_majority_neighbor(neighbors: List[Dict[str, int]]) -> Optional[int]:
        if len(neighbors) == 0:
            return None
        length = {
            neighbor["lane_id"]: neighbor["self_end_index"] - neighbor["self_start_index"] for neighbor in neighbors
        }
        return str(max(length, key=length.get))

    lanes: List[Lane] = []
    for lane_data in lane_data_dict.values():
        # Skip lanes without boundaries
        if lane_data.left_boundary is None or lane_data.right_boundary is None:
            continue

        lanes.append(
            Lane(
                object_id=lane_data.object_id,
                lane_group_id=lane_data.object_id,
                left_boundary=lane_data.left_boundary,
                right_boundary=lane_data.right_boundary,
                centerline=lane_data.centerline,
                left_lane_id=_get_majority_neighbor(lane_data.left_neighbors),
                right_lane_id=_get_majority_neighbor(lane_data.right_neighbors),
                predecessor_ids=lane_data.predecessor_ids,
                successor_ids=lane_data.successor_ids,
                speed_limit_mps=lane_data.speed_limit_mps,
            )
        )

    for lane in lanes:
        map_writer.write_lane(lane)

    return lanes


def _write_waymo_lane_groups(lanes: List[Lane], map_writer: AbstractMapWriter) -> None:
    # NOTE: WOD Perception does not provide lane groups, so we create a lane group for each lane.
    for lane in lanes:
        map_writer.write_lane_group(
            LaneGroup(
                object_id=lane.object_id,
                lane_ids=[lane.object_id],
                left_boundary=lane.left_boundary,
                right_boundary=lane.right_boundary,
                intersection_id=None,
                predecessor_ids=lane.predecessor_ids,
                successor_ids=lane.successor_ids,
                outline=lane.outline_3d,
            )
        )


def _write_waymo_misc_surfaces(map_features: List[map_pb2.MapFeature], map_writer: AbstractMapWriter) -> None:
    for map_feature in map_features:
        if map_feature.HasField("driveway"):
            # NOTE: We currently only handle classify driveways as carparks.
            outline = _extract_outline_from_waymo_proto(map_feature.driveway)
            if outline is not None:
                map_writer.write_carpark(Carpark(object_id=map_feature.id, outline=outline))
        elif map_feature.HasField("crosswalk"):
            outline = _extract_outline_from_waymo_proto(map_feature.crosswalk)
            if outline is not None:
                map_writer.write_crosswalk(Crosswalk(object_id=map_feature.id, outline=outline))

        elif map_feature.HasField("stop_sign"):
            pass  # TODO: Implement stop signs
        elif map_feature.HasField("speed_bump"):
            pass  # TODO: Implement speed bumps


def _extract_polyline_waymo_proto(data) -> Optional[Polyline3D]:
    polyline: Optional[Polyline3D] = None
    polyline_array = np.array([[p.x, p.y, p.z] for p in data.polyline], dtype=np.float64)
    if polyline_array.ndim == 2 and polyline_array.shape[1] == 3 and len(polyline_array) >= 2:
        # NOTE: A valid polyline must have at least 2 points, be 3D, and be non-empty
        polyline = Polyline3D.from_array(polyline_array)
    return polyline


def _extract_outline_from_waymo_proto(data) -> Optional[Polyline3D]:
    outline: Optional[Polyline3D] = None
    outline_array = np.array([[p.x, p.y, p.z] for p in data.polygon], dtype=np.float64)
    if outline_array.ndim == 2 and outline_array.shape[0] >= 3 and outline_array.shape[1] == 3:
        # NOTE: A valid polygon outline must have at least 3 points, be 3D, and be non-empty
        outline = Polyline3D.from_array(outline_array)
    return outline


def _extract_lane_neighbors(data) -> List[Dict[str, int]]:
    neighbors = []
    for neighbor in data:
        neighbors.append(
            {
                "lane_id": neighbor.feature_id,
                "self_start_index": neighbor.self_start_index,
                "self_end_index": neighbor.self_end_index,
                "neighbor_start_index": neighbor.neighbor_start_index,
                "neighbor_end_index": neighbor.neighbor_end_index,
            }
        )
    return neighbors
