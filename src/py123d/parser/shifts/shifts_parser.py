"""Parser for the Shifts vehicle motion prediction dataset (the SDC task of the Shifts benchmark).

Every scene is one serialized ``Scene`` protobuf: 25 past and 25 future frames at 5 Hz with the ego track,
vehicle and pedestrian tracks, a per-scene vector map (lane centerlines, crosswalks, road polygons),
traffic light states for the past frames and scene tags. py123d converts each scene into one 10 s log
with a per-log map. There are no sensors.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np
import shapely.geometry as geom
from shapely.ops import unary_union

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    CustomModality,
    CustomModalityMetadata,
    EgoStateSE3,
    LogMetadata,
    MapMetadata,
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
    TrafficLightStatus,
)
from py123d.datatypes.map_objects.base_map_objects import BaseMapObject
from py123d.datatypes.map_objects.map_layer_types import LaneType, RoadEdgeType
from py123d.datatypes.map_objects.map_objects import Crosswalk, GenericDrivable, Lane, RoadEdge
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.geometry import BoundingBoxSE3, EulerAngles, Polyline3D, PoseSE3, Vector3D
from py123d.parser.base_dataset_parser import BaseDatasetParser, BaseLogParser, BaseMapParser, ModalitiesSync
from py123d.parser.registry import ShiftsBoxDetectionLabel
from py123d.parser.shifts.shifts_constants import (
    SHIFTS_ARROW_SECTION_MAPPING,
    SHIFTS_AUX_MODALITY_ID,
    SHIFTS_AVAILABLE_SPLITS,
    SHIFTS_DATASET,
    SHIFTS_EGO_STATE_SE3_METADATA,
    SHIFTS_FRAME_INTERVAL_US,
    SHIFTS_SPLIT_TO_DIRECTORY,
    SHIFTS_TRAFFIC_LIGHT_MAPPING,
)
from py123d.parser.shifts.shifts_download import ShiftsDownloader
from py123d.parser.wod.utils.wod_boundary_utils import WaymoLaneData, fill_lane_boundaries

if TYPE_CHECKING:
    from py123d.parser.shifts.protos import dataset_pb2, geom_pb2, map_pb2

logger = logging.getLogger(__name__)

SHIFTS_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=ShiftsBoxDetectionLabel)

_MIN_POLYLINE_LENGTH_M = 1e-3
_MIN_PEDESTRIAN_SPEED_MPS = 1e-3


def read_scene(scene_path: Union[str, Path]) -> "dataset_pb2.Scene":
    """Deserialize one Shifts ``Scene`` protobuf file."""
    check_dependencies(modules=["google.protobuf"], optional_name="shifts")
    from py123d.parser.shifts.protos import dataset_pb2

    scene = dataset_pb2.Scene()
    scene.ParseFromString(Path(scene_path).read_bytes())
    return scene


def _scene_location(scene: "dataset_pb2.Scene") -> Optional[str]:
    """The ``track`` tag (the city the scene was recorded in), or ``None`` when unset."""
    from py123d.parser.shifts.protos import tags_pb2

    name = tags_pb2.Track.Name(scene.scene_tags.track)
    return None if name.startswith("_") else name


class ShiftsParser(BaseDatasetParser):
    """Dataset parser for the Shifts vehicle motion prediction dataset."""

    def __init__(
        self,
        splits: List[str],
        shifts_data_root: Optional[Union[str, Path]] = None,
        skip_maps: bool = False,
        skip_logs: bool = False,
        downloader: Optional[ShiftsDownloader] = None,
    ) -> None:
        """Initialize the Shifts parser.

        :param splits: Splits to process, one or more of :data:`SHIFTS_AVAILABLE_SPLITS`
            (``shifts_train``, ``shifts_val``, ``shifts_test``). Shifts calls the validation set
            "development" and the test set "evaluation".
        :param shifts_data_root: Root with ``train/``, ``development/`` and ``evaluation/`` directories of
            ``.pb`` scene files. Required when ``downloader`` is ``None``, ignored otherwise.
        :param skip_maps: If ``True``, ``get_map_parsers()`` returns an empty list.
        :param skip_logs: If ``True``, ``get_log_parsers()`` returns an empty list.
        :param downloader: Optional :class:`ShiftsDownloader` run at construction time; its
            ``output_dir`` then serves as the data root.
        """
        for split in splits:
            assert split in SHIFTS_AVAILABLE_SPLITS, (
                f"Split {split} is not available. Available splits: {SHIFTS_AVAILABLE_SPLITS}"
            )
        self._splits: List[str] = list(splits)
        self._skip_maps = skip_maps
        self._skip_logs = skip_logs

        if downloader is not None:
            downloader._splits = list(self._splits)
            downloader.download()
            assert downloader.output_dir is not None
            self._shifts_data_root = downloader.output_dir
        else:
            assert shifts_data_root is not None, "`shifts_data_root` must be provided when `downloader` is None."
            self._shifts_data_root = Path(shifts_data_root)
            assert self._shifts_data_root.exists(), f"The `shifts_data_root` {shifts_data_root} does not exist."

        self._split_scene_paths: List[Tuple[str, Path]] = self._collect_scene_paths()

    def _collect_scene_paths(self) -> List[Tuple[str, Path]]:
        split_scene_paths: List[Tuple[str, Path]] = []
        for split in self._splits:
            split_dir = self._shifts_data_root / SHIFTS_SPLIT_TO_DIRECTORY[split]
            assert split_dir.is_dir(), f"Split directory {split_dir} for {split} does not exist."
            scene_paths = sorted(split_dir.rglob("*.pb"))
            if not scene_paths:
                logger.warning("No scene files found for %s in %s", split, split_dir)
            split_scene_paths.extend((split, scene_path) for scene_path in scene_paths)
        return split_scene_paths

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        if self._skip_maps:
            return []
        return [ShiftsMapParser(split=split, scene_path=scene_path) for split, scene_path in self._split_scene_paths]

    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        if self._skip_logs:
            return []
        return [ShiftsLogParser(split=split, scene_path=scene_path) for split, scene_path in self._split_scene_paths]


class ShiftsMapParser(BaseMapParser):
    """Per-scene map parser: the ``path_graph`` of one Shifts scene."""

    def __init__(self, split: str, scene_path: Path) -> None:
        self._split = split
        self._scene_path = scene_path

    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        scene = read_scene(self._scene_path)
        return MapMetadata(
            dataset=SHIFTS_DATASET,
            split=self._split,
            log_name=scene.id,
            location=_scene_location(scene),
            map_has_z=True,
            map_is_per_log=True,
        )

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        scene = read_scene(self._scene_path)
        yield from iter_shifts_map_objects(scene.path_graph)


class ShiftsLogParser(BaseLogParser):
    """Lightweight, picklable handle to one Shifts scene."""

    def __init__(self, split: str, scene_path: Path) -> None:
        self._split = split
        self._scene_path = scene_path

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        scene = read_scene(self._scene_path)
        return LogMetadata(
            dataset=SHIFTS_DATASET,
            split=self._split,
            log_name=scene.id,
            location=_scene_location(scene),
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        scene = read_scene(self._scene_path)
        num_history = len(scene.past_vehicle_tracks)
        num_future = len(scene.future_vehicle_tracks)
        all_timestamps = extract_all_timestamps(num_history + num_future)

        all_ego_states = extract_all_ego_states(scene, all_timestamps)
        all_box_detections = extract_all_box_detections(scene, all_timestamps)
        all_traffic_lights = extract_all_traffic_lights(scene, all_timestamps)
        _, all_aux_modalities = extract_all_custom_data(scene, all_timestamps)

        for time_idx, timestamp in enumerate(all_timestamps):
            modalities: List[BaseModality] = [
                modality
                for modality in (
                    all_ego_states[time_idx],
                    all_box_detections[time_idx],
                    all_traffic_lights[time_idx],
                    all_aux_modalities[time_idx],
                )
                if modality is not None
            ]
            yield ModalitiesSync(timestamp=timestamp, modalities=modalities)


def extract_all_timestamps(num_frames: int) -> List[Timestamp]:
    """Relative timestamps at 5 Hz, starting at 0 s (the scene protobuf carries no absolute time)."""
    return [Timestamp.from_us(frame_idx * SHIFTS_FRAME_INTERVAL_US) for frame_idx in range(num_frames)]


def _iter_ego_tracks(scene: "dataset_pb2.Scene") -> Iterator[Optional["dataset_pb2.VehicleTrack"]]:
    """Ego track per frame, past then future; ``None`` where the scene has no ego state."""
    for frame_idx in range(len(scene.past_vehicle_tracks)):
        yield scene.past_ego_track[frame_idx] if frame_idx < len(scene.past_ego_track) else None
    for frame_idx in range(len(scene.future_vehicle_tracks)):
        yield scene.future_ego_track[frame_idx] if frame_idx < len(scene.future_ego_track) else None


def _yaw_to_pose_se3(x: float, y: float, z: float, yaw: float) -> PoseSE3:
    quaternion = EulerAngles(roll=0.0, pitch=0.0, yaw=yaw).quaternion
    return PoseSE3(x=x, y=y, z=z, qw=quaternion.qw, qx=quaternion.qx, qy=quaternion.qy, qz=quaternion.qz)


def extract_all_ego_states(scene: "dataset_pb2.Scene", all_timestamps: List[Timestamp]) -> List[Optional[EgoStateSE3]]:
    """Ego states from the past and future ego tracks.

    The ego position is reported at ground level (z = 0) while agent positions are box centres, so the
    ego box centre is lifted by half the vehicle height. Ego acceleration is not in the data; dynamics
    are left to the writer (``infer_ego_dynamics``).
    """
    metadata = SHIFTS_EGO_STATE_SE3_METADATA
    all_ego_states: List[Optional[EgoStateSE3]] = []
    for track, timestamp in zip(_iter_ego_tracks(scene), all_timestamps):
        if track is None:
            all_ego_states.append(None)
            continue
        center_se3 = _yaw_to_pose_se3(
            x=track.position.x,
            y=track.position.y,
            z=track.position.z + metadata.height / 2.0,
            yaw=track.yaw,
        )
        all_ego_states.append(
            EgoStateSE3.from_center(
                center_se3=center_se3, metadata=metadata, timestamp=timestamp, dynamic_state_se3=None
            )
        )
    return all_ego_states


def _pedestrian_yaw(track: "dataset_pb2.PedestrianTrack") -> float:
    """Pedestrians carry no orientation; like the official API, the heading follows the velocity (0 at rest)."""
    velocity = track.linear_velocity
    if math.hypot(velocity.x, velocity.y) > _MIN_PEDESTRIAN_SPEED_MPS:
        return math.atan2(velocity.y, velocity.x)
    return 0.0


def _track_to_box_detection(
    track: Union["dataset_pb2.VehicleTrack", "dataset_pb2.PedestrianTrack"],
    label: ShiftsBoxDetectionLabel,
    yaw: float,
) -> BoxDetectionSE3:
    """Convert a vehicle or pedestrian track state into a box detection.

    Track tokens are prefixed by ``v``/``p`` so the two id spaces cannot collide.
    """
    token = f"v{track.track_id}" if label == ShiftsBoxDetectionLabel.VEHICLE else f"p{track.track_id}"

    center_se3 = _yaw_to_pose_se3(x=track.position.x, y=track.position.y, z=track.position.z, yaw=yaw)
    return BoxDetectionSE3(
        attributes=BoxDetectionAttributes(label=label, track_token=token),
        bounding_box_se3=BoundingBoxSE3(
            center_se3=center_se3,
            length=track.dimensions.x,
            width=track.dimensions.y,
            height=track.dimensions.z,
        ),
        velocity_3d=Vector3D(x=track.linear_velocity.x, y=track.linear_velocity.y, z=track.linear_velocity.z),
    )


def extract_all_box_detections(scene: "dataset_pb2.Scene", all_timestamps: List[Timestamp]) -> List[BoxDetectionsSE3]:
    """Vehicle and pedestrian boxes per frame, past then future (velocities are in the global frame)."""
    num_history = len(scene.past_vehicle_tracks)
    all_box_detections: List[BoxDetectionsSE3] = []
    for time_idx, timestamp in enumerate(all_timestamps):
        if time_idx < num_history:
            vehicles = scene.past_vehicle_tracks[time_idx].tracks
            pedestrians = (
                scene.past_pedestrian_tracks[time_idx].tracks if time_idx < len(scene.past_pedestrian_tracks) else []
            )
        else:
            future_idx = time_idx - num_history
            vehicles = scene.future_vehicle_tracks[future_idx].tracks
            pedestrians = (
                scene.future_pedestrian_tracks[future_idx].tracks
                if future_idx < len(scene.future_pedestrian_tracks)
                else []
            )
        detections = [_track_to_box_detection(track, ShiftsBoxDetectionLabel.VEHICLE, track.yaw) for track in vehicles]
        detections += [
            _track_to_box_detection(track, ShiftsBoxDetectionLabel.PEDESTRIAN, _pedestrian_yaw(track))
            for track in pedestrians
        ]
        all_box_detections.append(
            BoxDetectionsSE3(
                box_detections=detections, timestamp=timestamp, metadata=SHIFTS_BOX_DETECTIONS_SE3_METADATA
            )
        )
    return all_box_detections


def lane_traffic_light_status(
    lane: "map_pb2.Lane", section_to_state: Mapping[int, int]
) -> Optional[TrafficLightStatus]:
    """Traffic light status of one lane from the section states of a frame.

    Follows ``get_lane_availability`` in the official ``ysdc_dataset_api.utils.map``: a lane without
    sections is unregulated (``None``); a lane with an arrow section is governed by that section unless
    its main light is switched off; otherwise the main section decides.
    """
    section_ids = lane.traffic_light_section_ids
    arrow_id = section_ids.left_section_id or section_ids.right_section_id
    if not (section_ids.main_section_id or arrow_id):
        return None

    main_state = section_to_state.get(section_ids.main_section_id)
    if arrow_id:
        if main_state is not None and SHIFTS_TRAFFIC_LIGHT_MAPPING.get(main_state) == TrafficLightStatus.OFF:
            return TrafficLightStatus.OFF
        arrow_state = section_to_state.get(arrow_id)
        if arrow_state is None:
            return TrafficLightStatus.UNKNOWN
        return SHIFTS_ARROW_SECTION_MAPPING.get(
            arrow_state, SHIFTS_TRAFFIC_LIGHT_MAPPING.get(arrow_state, TrafficLightStatus.UNKNOWN)
        )

    if main_state is None:
        return TrafficLightStatus.UNKNOWN
    return SHIFTS_TRAFFIC_LIGHT_MAPPING.get(main_state, TrafficLightStatus.UNKNOWN)


def extract_all_traffic_lights(
    scene: "dataset_pb2.Scene", all_timestamps: List[Timestamp]
) -> List[Optional[TrafficLightDetections]]:
    """Per-lane traffic light states for the frames that have them (the past frames); ``None`` otherwise.

    The ``lane_id`` is the lane's index in ``scene.path_graph.lanes``, which is also its map object id.
    """
    all_traffic_lights: List[Optional[TrafficLightDetections]] = []
    for time_idx, timestamp in enumerate(all_timestamps):
        if time_idx >= len(scene.traffic_lights):
            all_traffic_lights.append(None)
            continue
        section_to_state = {section.id: section.state for section in scene.traffic_lights[time_idx].sections}
        detections = []
        for lane_idx, lane in enumerate(scene.path_graph.lanes):
            status = lane_traffic_light_status(lane, section_to_state)
            if status is not None:
                detections.append(TrafficLightDetection(lane_id=lane_idx, status=status))
        all_traffic_lights.append(TrafficLightDetections(detections=detections, timestamp=timestamp))
    return all_traffic_lights


def extract_all_custom_data(
    scene: "dataset_pb2.Scene", all_timestamps: List[Timestamp]
) -> Tuple[CustomModalityMetadata, List[CustomModality]]:
    """Scene tags and prediction requests as the static ``aux`` metadata, plus a per-frame phase marker."""
    from py123d.parser.shifts.protos import tags_pb2

    num_history = len(scene.past_vehicle_tracks)
    tags = scene.scene_tags
    static_metadata = {
        "scene_id": scene.id,
        "scene_tags": {
            "day_time": tags_pb2.DayTimeTag.Name(tags.day_time),
            "season": tags_pb2.SeasonTag.Name(tags.season),
            "track": tags_pb2.Track.Name(tags.track),
            "sun_phase": tags_pb2.SunPhaseTag.Name(tags.sun_phase),
            "precipitation": tags_pb2.Precipitation.Name(tags.precipitation),
        },
        "prediction_requests": {
            f"v{request.track_id}": [tags_pb2.TrajectoryTag.Name(tag) for tag in request.trajectory_tags]
            for request in scene.prediction_requests
        },
        "num_history_frames": num_history,
        "num_future_frames": len(scene.future_vehicle_tracks),
        "t0_frame_index": num_history - 1,
        "lanes_give_way": [idx for idx, lane in enumerate(scene.path_graph.lanes) if lane.gives_way_to_some_lane],
        "timestamps_are_relative": True,
    }
    modality_metadata = CustomModalityMetadata(modality_id=SHIFTS_AUX_MODALITY_ID, metadata=static_metadata)

    aux_modalities: List[CustomModality] = []
    for time_idx, timestamp in enumerate(all_timestamps):
        if time_idx < num_history - 1:
            phase = "history"
        elif time_idx == num_history - 1:
            phase = "current"
        else:
            phase = "future"
        aux_modalities.append(
            CustomModality(
                data={"frame_index": time_idx, "phase": phase}, metadata=modality_metadata, timestamp=timestamp
            )
        )
    return modality_metadata, aux_modalities


def _points_to_array(points) -> np.ndarray:
    """Protobuf ``Vector3`` list to an (N, 3) array with consecutive duplicates removed."""
    array = np.array([[point.x, point.y, point.z] for point in points], dtype=np.float64).reshape(-1, 3)
    if len(array) > 1:
        keep = np.concatenate(([True], np.any(np.diff(array, axis=0) != 0.0, axis=1)))
        array = array[keep]
    return array


def _polyline_from_points(points) -> Optional[Polyline3D]:
    array = _points_to_array(points)
    if len(array) < 2 or np.sum(np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1)) < _MIN_POLYLINE_LENGTH_M:
        return None
    return Polyline3D.from_array(array)


def _outline_from_polygon(polygon: "geom_pb2.Polygon") -> Optional[Polyline3D]:
    array = _points_to_array(polygon.points)
    if len(array) >= 2 and np.all(array[0] == array[-1]):
        array = array[:-1]
    if len(array) < 3:
        return None
    return Polyline3D.from_array(array)


def _union_road_polygons(path_graph: "map_pb2.PathGraph") -> List[geom.Polygon]:
    """Merge the road polygons of a scene into disjoint surfaces (invalid rings are repaired with ``buffer(0)``)."""
    polygons: List[geom.Polygon] = []
    for road_polygon in path_graph.road_polygons:
        outline = _outline_from_polygon(road_polygon.geometry)
        if outline is None:
            continue
        polygon = geom.Polygon(outline.array[:, :2])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    if not polygons:
        return []
    union = unary_union(polygons)
    parts = union.geoms if isinstance(union, geom.MultiPolygon) else [union]
    return [part for part in parts if isinstance(part, geom.Polygon) and not part.is_empty]


def iter_shifts_map_objects(path_graph: "map_pb2.PathGraph") -> Iterator[BaseMapObject]:
    """Map objects of one scene.

    Shifts lanes are centerlines without boundaries, so the boundaries are synthesized with the same ray
    casting as for WOD (:func:`fill_lane_boundaries`), using the outlines of the merged road polygons as
    road edges: a lane ends where the road surface ends or halfway to a neighbouring centerline, with a
    3.7 m default. Object ids are the lane index (shared with the traffic light ``lane_id``), followed by the
    crosswalks, then the merged road surfaces and their edge rings.
    """
    num_lanes = len(path_graph.lanes)
    num_crosswalks = len(path_graph.crosswalks)

    # The road polygons are tiles that touch and partly overlap, so each tile outline would put a road edge
    # across the road. Their union gives the drivable surface per scene, with traffic islands as holes.
    road_edges: List[RoadEdge] = []
    drivables: List[GenericDrivable] = []
    next_object_id = num_lanes + num_crosswalks
    for surface in _union_road_polygons(path_graph):
        drivables.append(GenericDrivable(object_id=next_object_id, shapely_polygon=surface))
        next_object_id += 1
        for ring in (surface.exterior, *surface.interiors):
            road_edges.append(
                RoadEdge(
                    object_id=next_object_id,
                    road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
                    polyline=Polyline3D.from_linestring(geom.LineString(ring.coords)),
                )
            )
            next_object_id += 1

    lane_data_dict: Dict[int, WaymoLaneData] = {}
    for lane_idx, lane in enumerate(path_graph.lanes):
        centerline = _polyline_from_points(lane.centers)
        if centerline is None:
            continue
        lane_data_dict[lane_idx] = WaymoLaneData(
            object_id=lane_idx,
            centerline=centerline,
            predecessor_ids=[],
            successor_ids=[],
            speed_limit_mps=float(lane.max_velocity) if lane.max_velocity > 0.0 else None,
            lane_type=LaneType.SURFACE_STREET,
            left_neighbors=[],
            right_neighbors=[],
        )
    if lane_data_dict:
        fill_lane_boundaries(lane_data_dict, [], road_edges)

    yield from road_edges
    for lane_data in lane_data_dict.values():
        if lane_data.left_boundary is None or lane_data.right_boundary is None:
            continue
        yield Lane(
            object_id=lane_data.object_id,
            lane_type=lane_data.lane_type,
            left_boundary=lane_data.left_boundary,
            right_boundary=lane_data.right_boundary,
            centerline=lane_data.centerline,
            speed_limit_mps=lane_data.speed_limit_mps,
        )
    for crosswalk_idx, crosswalk in enumerate(path_graph.crosswalks):
        outline = _outline_from_polygon(crosswalk.geometry)
        if outline is not None:
            yield Crosswalk(object_id=num_lanes + crosswalk_idx, outline=outline)
    yield from drivables
