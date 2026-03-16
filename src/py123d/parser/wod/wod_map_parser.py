from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from py123d.common.utils.dependencies import check_dependencies

check_dependencies(modules=["google.protobuf"], optional_name="waymo")

from py123d.datatypes import BaseMapObject, MapMetadata
from py123d.datatypes.map_objects.map_layer_types import LaneType, RoadEdgeType, RoadLineType, StopZoneType
from py123d.datatypes.map_objects.map_objects import Carpark, Crosswalk, Lane, LaneGroup, RoadEdge, RoadLine, StopZone
from py123d.geometry import Polyline3D
from py123d.geometry.utils.units import mph_to_mps
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.wod.utils.wod_boundary_utils import WaymoLaneData, fill_lane_boundaries
from py123d.parser.wod.utils.wod_constants import (
    WAYMO_LANE_TYPE_CONVERSION,
    WAYMO_ROAD_EDGE_TYPE_CONVERSION,
    WAYMO_ROAD_LINE_TYPE_CONVERSION,
)
from py123d.parser.wod.waymo_open_dataset.protos import map_pb2

# TODO:
# - Implement speed bumps
# - Implement driveways with a different semantic type if needed
# - Implement intersections and lane group logic

STOP_ZONE_DEPTH = 0.5  # Depth of synthesized stop zone polygons in meters


class WODMapParser(BaseMapParser):
    """Shared map parser for WOD Perception and WOD Motion datasets.

    Both datasets share the same map feature protobuf format, so this parser
    handles map extraction for both. The map features are loaded lazily from
    the TFRecord file when :meth:`iter_map_objects` is called.

    :param dataset: Dataset identifier, e.g. ``"wod_perception"`` or ``"wod-motion"``.
    :param split: Split name, e.g. ``"wod-perception_train"``.
    :param log_name: Log or scenario name used as the map identifier.
    :param source_tf_record_path: Path to the TFRecord file containing map features.
    :param scenario_id: If provided, load a specific scenario from the TFRecord (WOD Motion).
        If ``None``, load the initial frame (WOD Perception).
    :param add_dummy_lane_groups: Whether to add dummy lane groups. \
        If True, creates a lane group for each lane since WOD does not provide lane groups.
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        log_name: str,
        source_tf_record_path: Path,
        scenario_id: Optional[str] = None,
        add_dummy_lane_groups: bool = False,
    ) -> None:
        self._dataset = dataset
        self._split = split
        self._log_name = log_name
        self._source_tf_record_path = source_tf_record_path
        self._scenario_id = scenario_id
        self._add_dummy_lane_groups = add_dummy_lane_groups

    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return MapMetadata(
            dataset=self._dataset,
            split=self._split,
            log_name=self._log_name,
            location=None,  # TODO: Add location information.
            map_has_z=True,
            map_is_per_log=True,
        )

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        map_features = self._load_map_features()
        yield from iter_wod_map_objects(map_features, self._add_dummy_lane_groups)

    def _load_map_features(self) -> List[map_pb2.MapFeature]:
        """Loads map features from the TFRecord file."""
        import importlib

        check_dependencies(modules=["tensorflow"], optional_name="waymo")
        import tensorflow as tf

        # Proto dependencies must be loaded in dependency order
        _proto_base = "py123d.parser.wod.waymo_open_dataset.protos"

        if self._scenario_id is not None:
            # WOD Motion: load specific scenario by ID
            for _proto in (
                "vector_pb2",
                "map_pb2",
                "keypoint_pb2",
                "label_pb2",
                "dataset_pb2",
                "camera_tokens_pb2",
                "compressed_lidar_pb2",
            ):
                importlib.import_module(f"{_proto_base}.{_proto}")
            from py123d.parser.wod.waymo_open_dataset.protos import scenario_pb2

            dataset = tf.data.TFRecordDataset(str(self._source_tf_record_path), compression_type="")
            for data in dataset:
                scenario = scenario_pb2.Scenario.FromString(data.numpy())
                if str(scenario.scenario_id) == self._scenario_id:
                    return list(scenario.map_features)
            raise ValueError(f"Scenario ID {self._scenario_id} not found in Waymo file: {self._source_tf_record_path}")
        else:
            # WOD Perception: load initial frame
            for _proto in ("vector_pb2", "keypoint_pb2", "label_pb2", "map_pb2"):
                importlib.import_module(f"{_proto_base}.{_proto}")
            from py123d.parser.wod.waymo_open_dataset.protos import dataset_pb2

            dataset = tf.data.TFRecordDataset(self._source_tf_record_path, compression_type="")
            for data in dataset:
                initial_frame = dataset_pb2.Frame()
                initial_frame.ParseFromString(data.numpy())
                return list(initial_frame.map_features)
            raise ValueError(f"No frames found in TFRecord: {self._source_tf_record_path}")


def iter_wod_map_objects(
    map_features: List[map_pb2.MapFeature], add_dummy_lane_groups: bool
) -> Iterator[BaseMapObject]:
    """Yields all map objects from WOD map features.

    :param map_features: Protobuf map features from a WOD frame or scenario.
    :param add_dummy_lane_groups: Whether to add dummy lane groups. If True, creates a lane group for each lane since WOD does not provide lane groups.
    :yields: BaseMapObject instances (RoadLine, RoadEdge, Lane, LaneGroup, Carpark, Crosswalk, StopZone).
    """
    # We first extract all road lines, road edges, and lanes.
    # NOTE: road lines and edges are needed to extract lane boundaries.
    road_lines = _get_waymo_road_lines(map_features)
    road_edges = _get_waymo_road_edges(map_features)
    lanes = _get_waymo_lanes(map_features, road_lines, road_edges, add_dummy_lane_groups)

    yield from road_lines
    yield from road_edges
    yield from lanes

    if add_dummy_lane_groups:
        # Yield lane groups based on the extracted lanes
        yield from _get_waymo_lane_groups(lanes)

    # Yield miscellaneous surfaces (carparks, crosswalks, stop zones, etc.)
    lane_dict = {lane.object_id: lane for lane in lanes}
    yield from _get_waymo_misc_surfaces(map_features, lane_dict)


def _get_waymo_road_lines(map_features: List[map_pb2.MapFeature]) -> List[RoadLine]:
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
    return road_lines


def _get_waymo_road_edges(map_features: List[map_pb2.MapFeature]) -> List[RoadEdge]:
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
    return road_edges


def _get_waymo_lanes(
    map_features: List[map_pb2.MapFeature],
    road_lines: List[RoadLine],
    road_edges: List[RoadEdge],
    add_dummy_lane_groups: bool,
) -> List[Lane]:
    """Extracts lanes from Waymo map features with computed boundaries.

    Uses road lines and road edges to determine left/right lane boundaries via
    perpendicular ray casting (see :func:`fill_lane_boundaries`).

    :param map_features: Protobuf map features from a WOD frame or scenario.
    :param road_lines: Previously extracted road lines (used for boundary computation).
    :param road_edges: Previously extracted road edges (used for boundary computation).
    :param add_dummy_lane_groups: Whether to assign each lane its own lane group ID.
    :return: List of Lane objects with computed boundaries.
    """
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

    def _get_majority_neighbor(neighbors: List[Dict[str, int]]) -> Optional[str]:
        """Returns the lane ID of the neighbor with the longest overlap, or None if no neighbors."""
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
                lane_type=lane_data.lane_type,
                left_boundary=lane_data.left_boundary,
                right_boundary=lane_data.right_boundary,
                centerline=lane_data.centerline,
                lane_group_id=lane_data.object_id if add_dummy_lane_groups else None,
                left_lane_id=_get_majority_neighbor(lane_data.left_neighbors),
                right_lane_id=_get_majority_neighbor(lane_data.right_neighbors),
                predecessor_ids=lane_data.predecessor_ids,
                successor_ids=lane_data.successor_ids,
                speed_limit_mps=lane_data.speed_limit_mps,
            )
        )

    return lanes


def _get_waymo_lane_groups(lanes: List[Lane]) -> List[LaneGroup]:
    """Creates a dummy lane group for each lane since WOD does not provide lane groups."""
    lane_groups: List[LaneGroup] = []
    for lane in lanes:
        lane_groups.append(
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
    return lane_groups


def _get_waymo_misc_surfaces(map_features: List[map_pb2.MapFeature], lane_dict: Dict[int, Lane]) -> List[BaseMapObject]:
    """Extracts miscellaneous map surfaces (driveways, crosswalks, stop zones) from Waymo map features."""
    surfaces: List[BaseMapObject] = []
    for map_feature in map_features:
        if map_feature.HasField("driveway"):
            # NOTE: We currently only classify driveways as carparks.
            outline = _extract_outline_from_waymo_proto(map_feature.driveway)
            if outline is not None:
                surfaces.append(Carpark(object_id=map_feature.id, outline=outline))
        elif map_feature.HasField("crosswalk"):
            outline = _extract_outline_from_waymo_proto(map_feature.crosswalk)
            if outline is not None:
                surfaces.append(Crosswalk(object_id=map_feature.id, outline=outline))
        elif map_feature.HasField("stop_sign"):
            stop_zone = _create_stop_zone_from_stop_sign(map_feature, lane_dict)
            if stop_zone is not None:
                surfaces.append(stop_zone)
        elif map_feature.HasField("speed_bump"):
            pass  # TODO: Implement speed bumps
    return surfaces


def _create_stop_zone_from_stop_sign(map_feature: map_pb2.MapFeature, lane_dict: Dict[int, Lane]) -> Optional[StopZone]:
    """Synthesize a StopZone polygon from a Waymo stop sign and its controlled lanes.

    For each controlled lane, a small rectangle (STOP_ZONE_DEPTH wide) is created at
    the lane entry using the first points of the left/right boundaries. The per-lane
    rectangles are merged into a single polygon via shapely union.

    :param map_feature: Waymo MapFeature containing a stop_sign.
    :param lane_dict: Dictionary mapping lane IDs to Lane objects.
    :return: A StopZone if synthesis succeeds, None otherwise.
    """
    from shapely import MultiPolygon, Polygon, union_all

    stop_sign = map_feature.stop_sign
    controlled_lane_ids = [int(lid) for lid in stop_sign.lane]

    # Find lanes that exist in the parsed lane dict
    controlled_lanes = [lane_dict[lid] for lid in controlled_lane_ids if lid in lane_dict]
    if not controlled_lanes:
        return None

    # Create a small rectangle at the entry of each controlled lane
    polygons: List[Polygon] = []
    all_z: List[float] = []
    for lane in controlled_lanes:
        left_arr = lane.left_boundary.array
        right_arr = lane.right_boundary.array

        # Entry = first points; create a rectangle STOP_ZONE_DEPTH deep along the lane
        # Use the first two points of each boundary to determine lane direction
        n_pts = min(len(left_arr), len(right_arr))
        if n_pts < 2:
            continue

        # Find the index corresponding to STOP_ZONE_DEPTH along the centerline
        centerline_arr = lane.centerline.array
        cumulative_dist = np.cumsum(np.linalg.norm(np.diff(centerline_arr[:, :2], axis=0), axis=1))
        depth_idx = np.searchsorted(cumulative_dist, STOP_ZONE_DEPTH) + 1
        depth_idx = min(depth_idx, n_pts - 1)

        # Rectangle corners: left[0], right[0], right[depth_idx], left[depth_idx]
        coords_2d = [
            (left_arr[0, 0], left_arr[0, 1]),
            (right_arr[0, 0], right_arr[0, 1]),
            (right_arr[depth_idx, 0], right_arr[depth_idx, 1]),
            (left_arr[depth_idx, 0], left_arr[depth_idx, 1]),
        ]
        poly = Polygon(coords_2d)
        if poly.is_valid and poly.area > 1e-6:
            polygons.append(poly)
            all_z.extend([left_arr[0, 2], right_arr[0, 2], left_arr[depth_idx, 2], right_arr[depth_idx, 2]])

    if not polygons:
        return None

    # Merge per-lane rectangles
    merged = union_all(polygons)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda g: g.area)
    if not isinstance(merged, Polygon) or merged.is_empty:
        return None

    # Create 3D outline from merged polygon exterior
    avg_z = float(np.mean(all_z))
    xy = np.array(merged.exterior.coords)
    z = np.full((xy.shape[0], 1), avg_z)
    outline = Polyline3D.from_array(np.hstack([xy, z]))

    lane_ids_str = [str(lid) for lid in controlled_lane_ids if lid in lane_dict]
    return StopZone(
        object_id=map_feature.id,
        stop_zone_type=StopZoneType.STOP_SIGN,
        outline=outline,
        lane_ids=lane_ids_str,
    )


def _extract_polyline_waymo_proto(data: Any) -> Optional[Polyline3D]:
    """Extracts a 3D polyline from a Waymo protobuf message with a ``polyline`` field."""
    polyline: Optional[Polyline3D] = None
    polyline_array = np.array([[p.x, p.y, p.z] for p in data.polyline], dtype=np.float64)
    if polyline_array.ndim == 2 and polyline_array.shape[1] == 3 and len(polyline_array) >= 2:
        # NOTE: A valid polyline must have at least 2 points, be 3D, and be non-empty
        polyline = Polyline3D.from_array(polyline_array)
    return polyline


def _extract_outline_from_waymo_proto(data: Any) -> Optional[Polyline3D]:
    """Extracts a 3D polygon outline from a Waymo protobuf message with a ``polygon`` field."""
    outline: Optional[Polyline3D] = None
    outline_array = np.array([[p.x, p.y, p.z] for p in data.polygon], dtype=np.float64)
    if outline_array.ndim == 2 and outline_array.shape[0] >= 3 and outline_array.shape[1] == 3:
        # NOTE: A valid polygon outline must have at least 3 points, be 3D, and be non-empty
        outline = Polyline3D.from_array(outline_array)
    return outline


def _extract_lane_neighbors(data: Any) -> List[Dict[str, int]]:
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
