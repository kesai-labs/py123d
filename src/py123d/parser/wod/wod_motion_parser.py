from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    EgoStateSE3,
    EgoStateSE3Metadata,
    LogMetadata,
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
)
from py123d.geometry import (
    BoundingBoxSE3,
    EulerAngles,
    PoseSE3,
    Vector3D,
)
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
)
from py123d.parser.registry import WODMotionBoxDetectionLabel
from py123d.parser.wod.utils.wod_constants import WOD_MOTION_AVAILABLE_SPLITS, WOD_MOTION_TRAFFIC_LIGHT_MAPPING
from py123d.parser.wod.wod_map_parser import WODMapParser

if TYPE_CHECKING:
    from py123d.datatypes.modalities.base_modality import BaseModality
    from py123d.parser.wod.waymo_open_dataset.protos import scenario_pb2

logger = logging.getLogger(__name__)

# NOTE: These parameters are estimates based on the vehicle model used in the WOD Motion dataset.
# The vehicle is a Chrysler Pacifica (minivan).
# [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
WOD_MOTION_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="wod-motion_chrysler_pacifica",
    width=2.3320000171661377,
    length=5.285999774932861,
    height=2.3299999237060547,
    wheel_base=3.089,
    center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=2.3299999237060547 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)

WOD_MOTION_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(
    box_detection_label_class=WODMotionBoxDetectionLabel,
)


def _lazy_import_tf_and_scenario_pb2():
    """Lazy import of tensorflow and scenario_pb2 to avoid import errors at module load time."""
    import importlib
    import os

    from py123d.common.utils.dependencies import check_dependencies

    check_dependencies(modules=["tensorflow"], optional_name="waymo")
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))

    # Proto dependencies must be loaded in dependency order before scenario_pb2
    for _proto in (
        "vector_pb2",
        "map_pb2",
        "keypoint_pb2",
        "label_pb2",
        "dataset_pb2",
        "camera_tokens_pb2",
        "compressed_lidar_pb2",
    ):
        importlib.import_module(f"py123d.parser.wod.waymo_open_dataset.protos.{_proto}")
    from py123d.parser.wod.waymo_open_dataset.protos import scenario_pb2

    return tf, scenario_pb2


def _get_all_tfrecord_scenario_ids(tf_record_path: Path) -> List[str]:
    """Helper to get all scenario IDs from a WOD-Motion TFRecord file."""
    tf, scenario_pb2 = _lazy_import_tf_and_scenario_pb2()
    dataset = tf.data.TFRecordDataset(str(tf_record_path), compression_type="")
    scenario_ids: List[str] = []
    for data in dataset:
        scenario = scenario_pb2.Scenario.FromString(data.numpy())
        scenario_ids.append(str(scenario.scenario_id))
    return scenario_ids


def _get_scenario_from_tfrecord(tf_record_path: Path, scenario_id: str) -> Optional[scenario_pb2.Scenario]:
    """Helper to get a specific scenario from a WOD-Motion TFRecord file by scenario ID."""
    tf, scenario_pb2 = _lazy_import_tf_and_scenario_pb2()
    dataset = tf.data.TFRecordDataset(str(tf_record_path), compression_type="")
    for data in dataset:
        scenario = scenario_pb2.Scenario.FromString(data.numpy())
        if str(scenario.scenario_id) == scenario_id:
            return scenario
    return None


class WODMotionParser(BaseDatasetParser):
    """Dataset parser for the Waymo Open Dataset - Motion (WODM)."""

    def __init__(
        self,
        splits: List[str],
        wod_motion_data_root: Union[str, Path],
        add_dummy_lane_groups: bool,
    ) -> None:
        assert wod_motion_data_root is not None, "The variable `wod_motion_data_root` must be provided."
        assert Path(wod_motion_data_root).exists(), (
            f"The provided `wod_motion_data_root` path {wod_motion_data_root} does not exist."
        )
        for split in splits:
            assert split in WOD_MOTION_AVAILABLE_SPLITS, (
                f"Split {split} is not available. Available splits: {WOD_MOTION_AVAILABLE_SPLITS}"
            )

        self._splits: List[str] = splits
        self._wod_motion_data_root: Path = Path(wod_motion_data_root)
        self._add_dummy_lane_groups: bool = add_dummy_lane_groups

        self._split_tf_record_pairs: List[Tuple[str, Path, str]] = self._collect_split_tf_record_pairs()

    def _collect_split_tf_record_pairs(self) -> List[Tuple[str, Path, str]]:
        """Helper to collect the pairings of the split names and the corresponding tf record file."""
        split_tf_record_pairs: List[Tuple[str, Path, str]] = []
        split_name_mapping: Dict[str, str] = {
            "wod-motion_train": "training",
            "wod-motion_val": "validation",
            "wod-motion_test": "testing",
        }

        for split in self._splits:
            assert split in split_name_mapping.keys()
            split_folder = self._wod_motion_data_root / split_name_mapping[split]
            source_log_paths = [log_file for log_file in split_folder.iterdir() if ".tfrecord" in log_file.name]
            for source_log_path in source_log_paths:
                scenario_ids = _get_all_tfrecord_scenario_ids(source_log_path)
                for scenario_id in scenario_ids:
                    split_tf_record_pairs.append((split, source_log_path, scenario_id))

        return split_tf_record_pairs

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        return [
            WODMapParser(
                dataset="wod-motion",
                split=split,
                log_name=scenario_id,
                source_tf_record_path=source_tf_record_path,
                scenario_id=scenario_id,
                add_dummy_lane_groups=self._add_dummy_lane_groups,
            )
            for split, source_tf_record_path, scenario_id in self._split_tf_record_pairs
        ]

    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        return [
            WODMotionLogParser(
                split=split,
                source_tf_record_path=source_tf_record_path,
                scenario_id=scenario_id,
            )
            for split, source_tf_record_path, scenario_id in self._split_tf_record_pairs
        ]


class WODMotionLogParser(BaseLogParser):
    """Lightweight, picklable handle to one WOD Motion log."""

    def __init__(
        self,
        split: str,
        source_tf_record_path: Path,
        scenario_id: str,
    ) -> None:
        self._split = split
        self._source_tf_record_path = source_tf_record_path
        self._scenario_id = scenario_id

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset="wod-motion",
            split=self._split,
            log_name=self._scenario_id,
            location=None,
            timestep_seconds=0.1,
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        ego_metadata = WOD_MOTION_EGO_STATE_SE3_METADATA
        box_detections_metadata = WOD_MOTION_BOX_DETECTIONS_SE3_METADATA

        scenario = _get_scenario_from_tfrecord(self._source_tf_record_path, self._scenario_id)
        assert scenario is not None, (
            f"Scenario ID {self._scenario_id} not found in Waymo file: {self._source_tf_record_path}"
        )

        all_timestamps = _extract_all_timestamps(scenario)
        all_ego_states = _extract_all_ego_states(scenario, all_timestamps, ego_metadata)
        all_box_detections = _extract_all_wod_motion_box_detections(scenario, all_timestamps, box_detections_metadata)
        all_traffic_lights = _extract_all_traffic_lights(scenario)

        assert len(all_timestamps) == len(all_ego_states) == len(all_box_detections) == len(all_traffic_lights), (
            "All extracted data lists must have the same length."
        )

        for time_idx in range(len(all_timestamps)):
            yield ModalitiesSync(
                timestamp=all_timestamps[time_idx],
                modalities=[
                    all_ego_states[time_idx],
                    all_box_detections[time_idx],
                    all_traffic_lights[time_idx],
                ],
            )

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        raise NotImplementedError("Async per-modality iteration is not implemented for WOD Motion.")


def _extract_all_timestamps(scenario: scenario_pb2.Scenario) -> List[Timestamp]:
    return [Timestamp.from_s(ts) for ts in scenario.timestamps_seconds]


def _extract_all_ego_states(
    scenario: scenario_pb2.Scenario,
    all_timestamps: List[Timestamp],
    ego_metadata: EgoStateSE3Metadata,
) -> List[EgoStateSE3]:
    all_ego_states: List[EgoStateSE3] = []
    for track_idx, track in enumerate(scenario.tracks):
        if scenario.sdc_track_index != track_idx:
            continue

        for state in track.states:
            assert state.valid, "Ego state is not valid."
            quaternion = EulerAngles(roll=0.0, pitch=0.0, yaw=state.heading).quaternion
            center_se3 = PoseSE3(
                x=state.center_x,
                y=state.center_y,
                z=state.center_z,
                qw=quaternion.qw,
                qx=quaternion.qx,
                qy=quaternion.qy,
                qz=quaternion.qz,
            )
            assert ego_metadata.length == state.length, "Ego vehicle length does not match vehicle parameters."
            assert ego_metadata.width == state.width, "Ego vehicle width does not match vehicle parameters."
            assert ego_metadata.height == state.height, "Ego vehicle height does not match vehicle parameters."
            ego_state = EgoStateSE3.from_center(
                center_se3=center_se3,
                metadata=ego_metadata,
                dynamic_state_se3=None,
                timestamp=all_timestamps[len(all_ego_states)],
            )
            all_ego_states.append(ego_state)

    assert len(all_ego_states) == len(scenario.timestamps_seconds), (
        f"Ego states length (={len(all_ego_states)}) does not match timestamps length (={len(scenario.timestamps_seconds)})."
    )
    return all_ego_states


def _extract_all_wod_motion_box_detections(
    scenario: scenario_pb2.Scenario,
    all_timestamps: List[Timestamp],
    box_detections_metadata: BoxDetectionsSE3Metadata,
) -> List[BoxDetectionsSE3]:
    """Extracts all box detections from the WOD-Motion scenario."""

    # We first collect all tracks over all timesteps in a dictionary, where the key is the track ID
    tracks_collection: Dict[str, List[Optional[BoxDetectionSE3]]] = {}
    for track_idx, track in enumerate(scenario.tracks):
        # NOTE: We skip the track of the ego vehicle and include in the ego state extraction
        if scenario.sdc_track_index == track_idx:
            continue

        track_id = str(track.id)
        tracks_collection[track_id] = []
        label = WODMotionBoxDetectionLabel(track.object_type)
        for state in track.states:
            if state.valid:
                quaternion = EulerAngles(roll=0.0, pitch=0.0, yaw=state.heading).quaternion
                center_se3 = PoseSE3(
                    x=state.center_x,
                    y=state.center_y,
                    z=state.center_z,
                    qw=quaternion.qw,
                    qx=quaternion.qx,
                    qy=quaternion.qy,
                    qz=quaternion.qz,
                )
                bounding_box_se3 = BoundingBoxSE3(
                    center_se3=center_se3,
                    length=state.length,
                    width=state.width,
                    height=state.height,
                )
                box_detection = BoxDetectionSE3(
                    attributes=BoxDetectionAttributes(
                        label=label,
                        track_token=track_id,
                    ),
                    bounding_box_se3=bounding_box_se3,
                    velocity_3d=Vector3D(x=state.velocity_x, y=state.velocity_y, z=0.0),
                )
                tracks_collection[track_id].append(box_detection)
            else:
                tracks_collection[track_id].append(None)

    # Check if all tracks have the same number of timesteps
    num_timesteps = len(scenario.timestamps_seconds)
    assert all(len(detections) == num_timesteps for detections in tracks_collection.values()), (
        "Not all tracks have the same number of timesteps."
    )

    # Next, accumulate all detections per timestep
    all_box_detections: List[BoxDetectionsSE3] = []
    for time_idx in range(num_timesteps):
        box_detections_at_time_idx: List[BoxDetectionSE3] = []
        for track_id, detections in tracks_collection.items():
            detection = detections[time_idx]
            if detection is not None:
                box_detections_at_time_idx.append(detection)
        all_box_detections.append(
            BoxDetectionsSE3(
                box_detections=box_detections_at_time_idx,
                timestamp=all_timestamps[time_idx],
                metadata=box_detections_metadata,
            )
        )

    assert len(all_box_detections) == num_timesteps, (
        "Number of box detection timesteps does not match number of scenario timesteps."
    )

    return all_box_detections


def _extract_all_traffic_lights(scenario: scenario_pb2.Scenario) -> List[TrafficLightDetections]:
    """Extracts all traffic light detections from the WOD-Motion scenario."""
    assert len(scenario.dynamic_map_states) == len(scenario.timestamps_seconds), (
        "Number of traffic light detection timesteps does not match number of scenario timesteps."
    )

    all_traffic_lights: List[TrafficLightDetections] = []

    for dynamic_map_state, ts in zip(scenario.dynamic_map_states, scenario.timestamps_seconds):
        detections: List[TrafficLightDetection] = []
        for lane_state in dynamic_map_state.lane_states:
            traffic_light_status = WOD_MOTION_TRAFFIC_LIGHT_MAPPING[lane_state.state]
            detections.append(TrafficLightDetection(lane_id=lane_state.lane, status=traffic_light_status))

        all_traffic_lights.append(TrafficLightDetections(detections=detections, timestamp=Timestamp.from_s(ts)))

    return all_traffic_lights
