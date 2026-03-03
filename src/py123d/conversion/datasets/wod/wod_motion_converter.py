import logging
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.api.scene.abstract_log_writer import AbstractLogWriter
from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.wod.utils.wod_constants import (
    WOD_MOTION_AVAILABLE_SPLITS,
    WOD_MOTION_TRAFFIC_LIGHT_MAPPING,
)
from py123d.conversion.datasets.wod.wod_map_conversion import convert_wod_map
from py123d.conversion.registry.box_detection_label_registry import WODMotionBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionMetadata, BoxDetectionSE3, BoxDetectionsSE3
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetection, TrafficLights
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors import (
    LidarID,
    LidarMetadata,
)
from py123d.datatypes.time.time_point import Timestamp
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.vehicle_parameters import (
    get_wod_motion_chrysler_pacifica_parameters,
)
from py123d.geometry import (
    BoundingBoxSE3,
    EulerAngles,
    PoseSE3,
    Vector3D,
)

check_dependencies(modules=["tensorflow"], optional_name="waymo")
import tensorflow as tf

from py123d.conversion.datasets.wod.waymo_open_dataset.protos import scenario_pb2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))

logger = logging.getLogger(__name__)


def _get_all_tfrecord_scenario_ids(tf_record_path: Path) -> List[str]:
    """Helper to get all scenario IDs from a WOD-Motion TFRecord file."""
    dataset = tf.data.TFRecordDataset(str(tf_record_path), compression_type="")
    scenario_ids: List[str] = []
    for data in dataset:
        scenario = scenario_pb2.Scenario.FromString(data.numpy())
        scenario_ids.append(str(scenario.scenario_id))
    return scenario_ids


def _get_scenario_from_tfrecord(tf_record_path: Path, scenario_id: str) -> Optional[scenario_pb2.Scenario]:
    """Helper to get a specific scenario from a WOD-Motion TFRecord file by scenario ID."""
    dataset = tf.data.TFRecordDataset(str(tf_record_path), compression_type="")
    for data in dataset:
        scenario = scenario_pb2.Scenario.FromString(data.numpy())
        if str(scenario.scenario_id) == scenario_id:
            return scenario
    return None


class WODMotionConverter(AbstractDatasetConverter):
    """Converter for the Waymo Open Dataset - Motion (WODM)."""

    def __init__(
        self,
        splits: List[str],
        wod_motion_data_root: Union[str, Path],
        dataset_converter_config: DatasetConverterConfig,
    ) -> None:
        super().__init__(dataset_converter_config)
        for split in splits:
            assert split in WOD_MOTION_AVAILABLE_SPLITS, (
                f"Split {split} is not available. Available splits: {WOD_MOTION_AVAILABLE_SPLITS}"
            )

        self._splits: List[str] = splits
        self._wod_motion_data_root: Path = Path(wod_motion_data_root)
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

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(self._split_tf_record_pairs)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return len(self._split_tf_record_pairs)

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        split, source_tf_record_path, scenario_id = self._split_tf_record_pairs[map_index]
        scenario = _get_scenario_from_tfrecord(source_tf_record_path, scenario_id)
        assert scenario is not None, f"Scenario ID {scenario_id} not found in Waymo file: {source_tf_record_path}"
        map_metadata = _get_wod_motion_map_metadata(scenario, split)
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)
        if map_needs_writing:
            convert_wod_map(scenario.map_features, map_writer)
        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""

        split, source_tf_record_path, scenario_id = self._split_tf_record_pairs[log_index]
        try:
            scenario = _get_scenario_from_tfrecord(source_tf_record_path, scenario_id)
            assert scenario is not None, f"Scenario ID {scenario_id} not found in Waymo file: {source_tf_record_path}"

            # 1. Initialize Metadata
            log_metadata = LogMetadata(
                dataset="wod-motion",
                split=split,
                log_name=str(scenario.scenario_id),
                location=None,
                timestep_seconds=0.1,
                vehicle_parameters=get_wod_motion_chrysler_pacifica_parameters(),
                box_detection_label_class=WODMotionBoxDetectionLabel,
                map_metadata=_get_wod_motion_map_metadata(scenario, split),
            )

            # 2. Prepare log writer
            log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

            # 3. Process source log data
            if log_needs_writing:
                all_timestamps = _extract_all_timestamps(scenario)
                all_ego_states = _extract_all_ego_states(scenario)
                all_box_detections = _extract_all_wod_motion_box_detections(scenario)
                all_traffic_lights = _extract_all_traffic_lights(scenario)

                assert (
                    len(all_timestamps) == len(all_ego_states) == len(all_box_detections) == len(all_traffic_lights)
                ), "All extracted data lists must have the same length."

                for time_idx in range(len(all_timestamps)):
                    log_writer.write(
                        timestamp=all_timestamps[time_idx],
                        ego_state=all_ego_states[time_idx],
                        box_detections=all_box_detections[time_idx],
                        traffic_lights=all_traffic_lights[time_idx],
                    )

            log_writer.close()
        except Exception as e:
            logger.error(f"Error processing log {source_tf_record_path}: {e}")
            traceback.print_exc()


def _get_wod_motion_map_metadata(scenario: scenario_pb2.Scenario, split: str) -> MapMetadata:
    """Gets the WOD-Motion map metadata from the initial frame."""
    map_metadata = MapMetadata(
        dataset="wod-motion",
        split=split,
        log_name=str(scenario.scenario_id),
        location=None,  # TODO: Add location information.
        map_has_z=True,
        map_is_local=True,
    )
    return map_metadata


def _extract_all_timestamps(scenario: scenario_pb2.Scenario) -> List[Timestamp]:
    return [Timestamp.from_s(ts) for ts in scenario.timestamps_seconds]


def _extract_all_ego_states(scenario: scenario_pb2.Scenario) -> List[EgoStateSE3]:
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
            vehicle_parameters = get_wod_motion_chrysler_pacifica_parameters()
            assert vehicle_parameters.length == state.length, "Ego vehicle length does not match vehicle parameters."
            assert vehicle_parameters.width == state.width, "Ego vehicle width does not match vehicle parameters."
            assert vehicle_parameters.height == state.height, "Ego vehicle height does not match vehicle parameters."
            ego_state = EgoStateSE3.from_center(
                center_se3=center_se3,
                vehicle_parameters=vehicle_parameters,
                dynamic_state_se3=None,
            )
            all_ego_states.append(ego_state)

    assert len(all_ego_states) == len(scenario.timestamps_seconds), (
        f"Ego states length (={len(all_ego_states)}) does not match timestamps length (={len(scenario.timestamps_seconds)})."
    )
    return all_ego_states


def _extract_all_wod_motion_box_detections(scenario: scenario_pb2.Scenario) -> List[BoxDetectionsSE3]:
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
                    metadata=BoxDetectionMetadata(
                        label=label,
                        timestamp=None,
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
        all_box_detections.append(BoxDetectionsSE3(box_detections=box_detections_at_time_idx))  # type: ignore

    assert len(all_box_detections) == num_timesteps, (
        "Number of box detection timesteps does not match number of scenario timesteps."
    )

    return all_box_detections


def _extract_all_traffic_lights(scenario: scenario_pb2.Scenario) -> List[TrafficLights]:
    """Extracts all traffic light detections from the WOD-Motion scenario."""

    all_traffic_lights: List[TrafficLights] = []

    for dynamic_map_state in scenario.dynamic_map_states:
        traffic_light_detections: List[TrafficLightDetection] = []
        for lane_state in dynamic_map_state.lane_states:
            traffic_light_status = WOD_MOTION_TRAFFIC_LIGHT_MAPPING[lane_state.state]
            traffic_light_detections.append(TrafficLightDetection(lane_id=lane_state.lane, status=traffic_light_status))

        all_traffic_lights.append(TrafficLights(traffic_light_detections=traffic_light_detections))
    assert len(all_traffic_lights) == len(scenario.timestamps_seconds), (
        "Number of traffic light detection timesteps does not match number of scenario timesteps."
    )

    return all_traffic_lights


def _get_wod_motion_lidar_metadata(
    scenario: scenario_pb2.Scenario,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LidarID, LidarMetadata]:
    raise NotImplementedError("WOD-Motion Lidar metadata extraction is not yet implemented.")


def _extract_wod_motion_lidars(
    scenario: scenario_pb2.Scenario,
) -> None:
    raise NotImplementedError("WOD-Motion Lidar extraction is not yet implemented.")
