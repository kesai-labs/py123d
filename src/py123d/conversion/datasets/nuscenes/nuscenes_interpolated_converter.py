import bisect
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pyquaternion import Quaternion

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.api.scene.abstract_log_writer import AbstractLogWriter, CameraData, LidarData
from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.nuscenes.nuscenes_map_conversion import NUSCENES_MAPS, write_nuscenes_map
from py123d.conversion.datasets.nuscenes.utils.nuscenes_constants import (
    NUSCENES_CAMERA_IDS,
    NUSCENES_DATABASE_VERSION_MAPPING,
    NUSCENES_DETECTION_NAME_DICT,
    NUSCENES_INTERPOLATED_DATA_SPLITS,
    TARGET_DT,
)
from py123d.conversion.registry.box_detection_label_registry import NuScenesBoxDetectionLabel
from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    DynamicStateSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    MapMetadata,
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.vehicle_state.ego_metadata import get_nuscenes_renault_zoe_parameters
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D

check_dependencies(["nuscenes"], "nuscenes")
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes

# Target time interval in microseconds (10Hz = 100ms)
_TARGET_DT_US: int = int(TARGET_DT * 1e6)

# Tolerance for matching cameras to lidar sweep timestamps.
# Since we select the last camera *before* the lidar timestamp, the offset can be up to one full
# camera period (~83 ms at ~12 Hz).  We use 100 ms to be consistent with the keyframe extraction.
_CAMERA_TIMESTAMP_TOLERANCE_US: int = 100_000


class NuScenesInterpolatedConverter(AbstractDatasetConverter):
    """Dataset converter for the nuScenes dataset that interpolates to 10Hz.

    In contrast to :class:`NuScenesConverter`, this converter upsamples the native 2Hz keyframe
    rate to 10Hz by leveraging intermediate lidar sweeps (~20Hz, selecting every 2nd for ~10Hz),
    real ego poses from those sweeps, and linearly interpolated bounding box detections between
    keyframes (with SLERP for rotations).
    """

    def __init__(
        self,
        splits: List[str],
        nuscenes_data_root: Union[Path, str],
        nuscenes_map_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
        nuscenes_dbs: Optional[Dict[str, NuScenes]] = None,
    ) -> None:
        """Initializes the :class:`NuScenesInterpolatedConverter`.

        :param splits: List of splits to include in the conversion, e.g., \
            ["nuscenes-interpolated_train", "nuscenes-interpolated_val"]
        :param nuscenes_data_root: Path to the root directory of the nuScenes dataset
        :param nuscenes_map_root: Path to the root directory of the nuScenes map data
        :param dataset_converter_config: Configuration for the dataset converter
        """
        super().__init__(dataset_converter_config)

        assert nuscenes_data_root is not None, "The variable `nuscenes_data_root` must be provided."
        assert nuscenes_map_root is not None, "The variable `nuscenes_map_root` must be provided."
        for split in splits:
            assert split in NUSCENES_INTERPOLATED_DATA_SPLITS, (
                f"Split {split} is not available. Available splits: {NUSCENES_INTERPOLATED_DATA_SPLITS}"
            )

        self._splits: List[str] = splits

        self._nuscenes_data_root: Path = Path(nuscenes_data_root)
        self._nuscenes_map_root: Path = Path(nuscenes_map_root)

        self._nuscenes_dbs: Dict[str, NuScenes] = nuscenes_dbs if nuscenes_dbs is not None else {}
        self._scene_tokens_per_split: Dict[str, List[str]] = self._collect_scene_tokens()

    def __reduce__(self):
        return (
            self.__class__,
            (
                self._splits,
                self._nuscenes_data_root,
                self._nuscenes_map_root,
                self.dataset_converter_config,
                self._nuscenes_dbs,
            ),
        )

    def _collect_scene_tokens(self) -> Dict[str, List[str]]:
        """Collects scene tokens for the specified splits."""

        scene_tokens_per_split: Dict[str, List[str]] = {}
        # Conversion from nuScenes internal split names to our split names
        nuscenes_split_name_mapping = {
            "nuscenes-interpolated_train": "train",
            "nuscenes-interpolated_val": "val",
            "nuscenes-interpolated_test": "test",
            "nuscenes-interpolated-mini_train": "mini_train",
            "nuscenes-interpolated-mini_val": "mini_val",
        }

        # Loads the mapping from split names to scene names in nuScenes
        scene_splits = create_splits_scenes()

        # Iterate over split names,
        for split in self._splits:
            database_version = NUSCENES_DATABASE_VERSION_MAPPING[split]
            nusc = self._nuscenes_dbs.get(database_version)
            if nusc is None:
                nusc = NuScenes(
                    version=database_version,
                    dataroot=str(self._nuscenes_data_root),
                    verbose=False,
                )
                self._nuscenes_dbs[database_version] = nusc

            available_scenes = [scene for scene in nusc.scene]
            nuscenes_split = nuscenes_split_name_mapping[split]
            scene_names = scene_splits.get(nuscenes_split, [])

            # get token
            scene_tokens = [scene["token"] for scene in available_scenes if scene["name"] in scene_names]
            scene_tokens_per_split[split] = scene_tokens
        return scene_tokens_per_split

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(NUSCENES_MAPS)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return sum(len(scene_tokens) for scene_tokens in self._scene_tokens_per_split.values())

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        map_name = NUSCENES_MAPS[map_index]

        map_metadata = _get_nuscenes_map_metadata(map_name)
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)

        if map_needs_writing:
            write_nuscenes_map(nuscenes_maps_root=self._nuscenes_map_root, location=map_name, map_writer=map_writer)

        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass.

        Converts a nuScenes scene to a log at 10Hz by using intermediate lidar sweeps
        for ego poses and lidar data, interpolating box detections between 2Hz keyframes,
        and finding nearest camera data for each sweep timestamp.
        """
        # Find the scene token for the given log index
        all_scene_tokens = []
        for split, scene_tokens in self._scene_tokens_per_split.items():
            all_scene_tokens.extend([(split, token) for token in scene_tokens])

        if log_index >= len(all_scene_tokens):
            raise ValueError(f"Log index {log_index} is out of range. Total logs: {len(all_scene_tokens)}")

        split, scene_token = all_scene_tokens[log_index]

        database_version = NUSCENES_DATABASE_VERSION_MAPPING[split]
        nusc = self._nuscenes_dbs[database_version]
        scene = nusc.get("scene", scene_token)
        log_record = nusc.get("log", scene["log_token"])

        # 1. Initialize log metadata (use TARGET_DT for interpolated 10Hz output)
        log_metadata = LogMetadata(
            dataset="nuscenes",
            split=split,
            log_name=scene["name"],
            location=log_record["location"],
            timestep_seconds=TARGET_DT,
            vehicle_parameters=get_nuscenes_renault_zoe_parameters(),
            box_detection_label_class=NuScenesBoxDetectionLabel,
            pinhole_camera_metadata=_get_nuscenes_pinhole_camera_metadata(nusc, scene, self.dataset_converter_config),
            lidar_metadata=_get_nuscenes_lidar_metadata(nusc, scene, self.dataset_converter_config),
            map_metadata=_get_nuscenes_map_metadata(log_record["location"]),
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        if log_needs_writing:
            can_bus = NuScenesCanBus(dataroot=str(self._nuscenes_data_root))
            scene_name = scene["name"]

            # 3. Collect all lidar sweeps (keyframes + non-keyframes) for the scene
            lidar_timeline = _collect_lidar_sweep_timeline(nusc, scene)

            # 4. Collect keyframe data: ordered samples and their box detections
            keyframe_samples = _collect_keyframe_samples(nusc, scene)
            keyframe_timestamps = [s["timestamp"] for s in keyframe_samples]
            keyframe_detections: Dict[str, BoxDetectionsSE3] = {}
            for sample in keyframe_samples:
                keyframe_detections[sample["token"]] = _extract_nuscenes_box_detections(nusc, sample)

            # 5. Select ~10Hz sweeps from the lidar timeline
            selected_sweeps = _select_10hz_sweeps(lidar_timeline, keyframe_timestamps)

            # 6. Build camera timelines for non-keyframe camera lookup
            camera_timelines: Dict[str, List[Dict[str, Any]]] = {}
            if self.dataset_converter_config.include_pinhole_cameras:
                camera_timelines = _collect_camera_timelines(nusc, scene)

            # 7. Iterate over selected 10Hz sweeps and write frames
            for sweep in selected_sweeps:
                timestamp = Timestamp.from_us(sweep["timestamp"])

                if sweep["is_key_frame"]:
                    # Keyframe: use original extraction (annotations, synchronized sensors)
                    sample = nusc.get("sample", sweep["sample_token"])
                    log_writer.write(
                        timestamp=timestamp,
                        ego_state=_extract_ego_state_from_sample_data(nusc, sweep, can_bus, scene_name),
                        box_detections=keyframe_detections[sweep["sample_token"]],
                        pinhole_cameras=_extract_nuscenes_cameras(
                            nusc=nusc,
                            sample=sample,
                            nuscenes_data_root=self._nuscenes_data_root,
                            dataset_converter_config=self.dataset_converter_config,
                        ),
                        lidars=[_l]
                        if (
                            _l := _extract_lidar_from_sample_data(
                                sweep,
                                nuscenes_data_root=self._nuscenes_data_root,
                                dataset_converter_config=self.dataset_converter_config,
                            )
                        )
                        is not None
                        else None,
                    )
                else:
                    # Non-keyframe: interpolated boxes, real ego pose, nearest sensors
                    ego_state = _extract_ego_state_from_sample_data(nusc, sweep, can_bus, scene_name)

                    # Find surrounding keyframes and interpolate boxes
                    prev_kf, next_kf = _find_surrounding_keyframes(sweep["timestamp"], keyframe_samples)
                    if prev_kf is not None and next_kf is not None:
                        delta = next_kf["timestamp"] - prev_kf["timestamp"]
                        t = (sweep["timestamp"] - prev_kf["timestamp"]) / delta
                        box_detections = _interpolate_box_detections(
                            keyframe_detections[prev_kf["token"]],
                            keyframe_detections[next_kf["token"]],
                            t,
                            timestamp,
                        )
                    elif prev_kf is not None:
                        box_detections = keyframe_detections[prev_kf["token"]]
                    else:
                        box_detections = BoxDetectionsSE3(box_detections=[], timestamp=timestamp)

                    # Find nearest cameras for this sweep timestamp
                    cameras = _find_nearest_cameras_for_sweep(
                        nusc=nusc,
                        target_timestamp=sweep["timestamp"],
                        camera_timelines=camera_timelines,
                        nuscenes_data_root=self._nuscenes_data_root,
                        dataset_converter_config=self.dataset_converter_config,
                    )

                    # Lidar from this sweep
                    lidar = _extract_lidar_from_sample_data(
                        sweep,
                        nuscenes_data_root=self._nuscenes_data_root,
                        dataset_converter_config=self.dataset_converter_config,
                    )

                    log_writer.write(
                        timestamp=timestamp,
                        ego_state=ego_state,
                        box_detections=box_detections,
                        pinhole_cameras=cameras,
                        lidars=[lidar] if lidar is not None else None,
                    )

        log_writer.close()
        del nusc
        gc.collect()


# ---------------------------------------------------------------------------
# Lidar sweep timeline helpers
# ---------------------------------------------------------------------------


def _collect_lidar_sweep_timeline(nusc: NuScenes, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collects all LIDAR_TOP sample_data records for a scene (keyframes + sweeps).

    Walks the sample_data linked list starting from the first keyframe's lidar token
    forward through the entire scene.

    :param nusc: The NuScenes database instance.
    :param scene: The scene record.
    :return: Chronologically ordered list of lidar sweep dicts with keys:
        token, timestamp, ego_pose_token, filename, is_key_frame, sample_token.
    """
    first_sample = nusc.get("sample", scene["first_sample_token"])
    last_sample = nusc.get("sample", scene["last_sample_token"])
    last_kf_timestamp = last_sample["timestamp"]

    lidar_sd_token = first_sample["data"]["LIDAR_TOP"]
    timeline: List[Dict[str, Any]] = []

    current = nusc.get("sample_data", lidar_sd_token)
    while current:
        # Stop if we've gone past the last keyframe
        if current["timestamp"] > last_kf_timestamp and not current["is_key_frame"]:
            break

        timeline.append(
            {
                "token": current["token"],
                "timestamp": current["timestamp"],
                "ego_pose_token": current["ego_pose_token"],
                "filename": current["filename"],
                "is_key_frame": current["is_key_frame"],
                "sample_token": current.get("sample_token", ""),
            }
        )

        if current["next"]:
            current = nusc.get("sample_data", current["next"])
        else:
            break

    return timeline


def _collect_keyframe_samples(nusc: NuScenes, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collects all keyframe sample records for a scene in chronological order.

    :param nusc: The NuScenes database instance.
    :param scene: The scene record.
    :return: Ordered list of sample dicts.
    """
    samples: List[Dict[str, Any]] = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        samples.append(sample)
        sample_token = sample["next"] if sample["next"] else None
    return samples


def _select_10hz_sweeps(
    lidar_timeline: List[Dict[str, Any]],
    keyframe_timestamps: List[int],
) -> List[Dict[str, Any]]:
    """Selects approximately 10Hz sweeps from the ~20Hz lidar timeline.

    Between each pair of consecutive keyframes, distributes intermediate timestamps at
    regular intervals and picks the closest lidar sweep for each. Keyframes are always included.

    :param lidar_timeline: Full lidar sweep timeline from :func:`_collect_lidar_sweep_timeline`.
    :param keyframe_timestamps: Sorted list of keyframe timestamps in microseconds.
    :return: Selected sweeps at approximately 10Hz, sorted by timestamp.
    """
    if not lidar_timeline:
        return []

    sweep_timestamps = np.array([s["timestamp"] for s in lidar_timeline])
    kf_set = set(keyframe_timestamps)
    selected_tokens: set = set()
    selected: List[Dict[str, Any]] = []

    # Always include all keyframe sweeps
    for sweep in lidar_timeline:
        if sweep["is_key_frame"] and sweep["timestamp"] in kf_set:
            selected_tokens.add(sweep["token"])
            selected.append(sweep)

    # Between each pair of consecutive keyframes, add intermediate sweeps at ~10Hz
    kf_sorted = sorted(keyframe_timestamps)
    for kf_idx in range(len(kf_sorted) - 1):
        kf_ts = kf_sorted[kf_idx]
        next_kf_ts = kf_sorted[kf_idx + 1]
        delta = next_kf_ts - kf_ts
        n_intervals = max(1, round(delta / _TARGET_DT_US))

        for i in range(1, n_intervals):
            target_ts = kf_ts + i * (delta / n_intervals)

            # Binary search for closest lidar sweep
            idx = int(np.searchsorted(sweep_timestamps, target_ts))
            candidates = []
            if idx > 0:
                candidates.append(idx - 1)
            if idx < len(sweep_timestamps):
                candidates.append(idx)

            if not candidates:
                continue

            best_idx = min(candidates, key=lambda j: abs(int(sweep_timestamps[j]) - target_ts))
            sweep = lidar_timeline[best_idx]

            if sweep["token"] not in selected_tokens:
                selected_tokens.add(sweep["token"])
                selected.append(sweep)

    # Sort by timestamp
    selected.sort(key=lambda s: s["timestamp"])
    return selected


# ---------------------------------------------------------------------------
# Ego state extraction from sample_data (works for both keyframes and sweeps)
# ---------------------------------------------------------------------------


def _extract_ego_state_from_sample_data(
    nusc: NuScenes,
    sweep: Dict[str, Any],
    can_bus: NuScenesCanBus,
    scene_name: str,
) -> EgoStateSE3:
    """Extracts the ego state from a lidar sample_data record (keyframe or non-keyframe).

    Uses the real ego pose from the sample_data's ego_pose_token and matches
    CAN bus data for dynamic state (velocity, acceleration, angular velocity).

    :param nusc: The NuScenes database instance.
    :param sweep: A lidar sweep dict from the timeline.
    :param can_bus: The NuScenes CAN bus API.
    :param scene_name: The scene name for CAN bus lookup.
    :return: The ego state.
    """
    ego_pose = nusc.get("ego_pose", sweep["ego_pose_token"])
    quat = Quaternion(ego_pose["rotation"])
    vehicle_parameters = get_nuscenes_renault_zoe_parameters()

    imu_pose = PoseSE3(
        x=ego_pose["translation"][0],
        y=ego_pose["translation"][1],
        z=ego_pose["translation"][2],
        qw=quat.w,
        qx=quat.x,
        qy=quat.y,
        qz=quat.z,
    )

    try:
        pose_msgs = can_bus.get_messages(scene_name, "pose")
    except Exception:
        pose_msgs = []

    if pose_msgs:
        closest_msg = None
        min_time_diff = float("inf")
        for msg in pose_msgs:
            time_diff = abs(msg["utime"] - sweep["timestamp"])
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_msg = msg

        if closest_msg and min_time_diff < 500_000:
            velocity = [*closest_msg["vel"]]
            acceleration = [*closest_msg["accel"]]
            angular_velocity = [*closest_msg["rotation_rate"]]
        else:
            velocity = acceleration = angular_velocity = [0.0, 0.0, 0.0]
    else:
        velocity = acceleration = angular_velocity = [0.0, 0.0, 0.0]

    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(*velocity),
        acceleration=Vector3D(*acceleration),
        angular_velocity=Vector3D(*angular_velocity),
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_pose,
        dynamic_state_se3=dynamic_state,
        vehicle_parameters=vehicle_parameters,
    )


# ---------------------------------------------------------------------------
# Box detection interpolation
# ---------------------------------------------------------------------------


def _find_surrounding_keyframes(
    timestamp: int,
    keyframe_samples: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Finds the previous and next keyframe samples surrounding a given timestamp.

    :param timestamp: Target timestamp in microseconds.
    :param keyframe_samples: Ordered list of keyframe sample dicts.
    :return: Tuple of (previous_keyframe, next_keyframe). Either may be None at boundaries.
    """
    kf_timestamps = [s["timestamp"] for s in keyframe_samples]
    idx = bisect.bisect_right(kf_timestamps, timestamp)

    prev_kf = keyframe_samples[idx - 1] if idx > 0 else None
    next_kf = keyframe_samples[idx] if idx < len(keyframe_samples) else None
    return prev_kf, next_kf


def _interpolate_box_detections(
    prev_detections: BoxDetectionsSE3,
    next_detections: BoxDetectionsSE3,
    t: float,
    interpolated_timestamp: Timestamp,
) -> BoxDetectionsSE3:
    """Interpolates box detections between two keyframes.

    Matches detections by track token (instance_token). For matched pairs:
    - Position: linear interpolation
    - Rotation: SLERP
    - Dimensions: linear interpolation
    - Velocity: linear interpolation

    Detections that only appear in one keyframe are excluded from interpolated frames.

    :param prev_detections: Box detections from the previous keyframe.
    :param next_detections: Box detections from the next keyframe.
    :param t: Interpolation ratio in [0, 1].
    :param interpolated_timestamp: Timestamp for the interpolated frame.
    :return: Interpolated box detections.
    """
    # Build lookup by track token for the next keyframe
    next_by_track: Dict[str, BoxDetectionSE3] = {}
    for det in next_detections:
        next_by_track[det.metadata.track_token] = det

    interpolated: List[BoxDetectionSE3] = []
    for prev_det in prev_detections:
        track_token = prev_det.metadata.track_token
        next_det = next_by_track.get(track_token)
        if next_det is None:
            continue  # Track doesn't exist in next keyframe, skip at interpolated frame

        # Interpolate position (linear)
        prev_center = prev_det.bounding_box_se3.center_se3
        next_center = next_det.bounding_box_se3.center_se3
        interp_x = prev_center.x + t * (next_center.x - prev_center.x)
        interp_y = prev_center.y + t * (next_center.y - prev_center.y)
        interp_z = prev_center.z + t * (next_center.z - prev_center.z)

        # Interpolate rotation (SLERP)
        q_prev = Quaternion(prev_center.qw, prev_center.qx, prev_center.qy, prev_center.qz)
        q_next = Quaternion(next_center.qw, next_center.qx, next_center.qy, next_center.qz)
        q_interp = Quaternion.slerp(q_prev, q_next, t)

        center = PoseSE3(
            x=interp_x,
            y=interp_y,
            z=interp_z,
            qw=q_interp.w,
            qx=q_interp.x,
            qy=q_interp.y,
            qz=q_interp.z,
        )

        # Interpolate dimensions (linear)
        prev_bb = prev_det.bounding_box_se3
        next_bb = next_det.bounding_box_se3
        length = prev_bb.length + t * (next_bb.length - prev_bb.length)
        width = prev_bb.width + t * (next_bb.width - prev_bb.width)
        height = prev_bb.height + t * (next_bb.height - prev_bb.height)

        bounding_box = BoundingBoxSE3(center_se3=center, length=length, width=width, height=height)

        # Interpolate velocity (linear)
        velocity_3d = None
        if prev_det.velocity_3d is not None and next_det.velocity_3d is not None:
            vx = prev_det.velocity_3d.x + t * (next_det.velocity_3d.x - prev_det.velocity_3d.x)
            vy = prev_det.velocity_3d.y + t * (next_det.velocity_3d.y - prev_det.velocity_3d.y)
            vz = prev_det.velocity_3d.z + t * (next_det.velocity_3d.z - prev_det.velocity_3d.z)
            velocity_3d = Vector3D(x=vx, y=vy, z=vz)
        elif prev_det.velocity_3d is not None:
            velocity_3d = prev_det.velocity_3d
        elif next_det.velocity_3d is not None:
            velocity_3d = next_det.velocity_3d

        metadata = BoxDetectionAttributes(
            label=prev_det.metadata.label,
            track_token=track_token,
            num_lidar_points=0,
        )

        interpolated.append(
            BoxDetectionSE3(
                metadata=metadata,
                bounding_box_se3=bounding_box,
                velocity_3d=velocity_3d,
            )
        )

    return BoxDetectionsSE3(box_detections=interpolated, timestamp=interpolated_timestamp)


# ---------------------------------------------------------------------------
# Camera data for non-keyframe timestamps
# ---------------------------------------------------------------------------


def _collect_camera_timelines(
    nusc: NuScenes,
    scene: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Collects all sample_data records for each camera channel in a scene.

    :param nusc: The NuScenes database instance.
    :param scene: The scene record.
    :return: Dict mapping camera channel name to its chronological list of sample_data records.
    """
    first_sample = nusc.get("sample", scene["first_sample_token"])
    last_sample = nusc.get("sample", scene["last_sample_token"])
    last_kf_timestamp = last_sample["timestamp"]

    timelines: Dict[str, List[Dict[str, Any]]] = {}

    for _camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
        timeline: List[Dict[str, Any]] = []
        sd_token = first_sample["data"][camera_channel]
        current = nusc.get("sample_data", sd_token)

        while current:
            if current["timestamp"] > last_kf_timestamp and not current["is_key_frame"]:
                break
            timeline.append(current)
            if current["next"]:
                current = nusc.get("sample_data", current["next"])
            else:
                break

        timelines[camera_channel] = timeline

    return timelines


def _find_nearest_cameras_for_sweep(
    nusc: NuScenes,
    target_timestamp: int,
    camera_timelines: Dict[str, List[Dict[str, Any]]],
    nuscenes_data_root: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> List[CameraData]:
    """Finds the last camera observation at or before a given sweep timestamp for each channel.

    This mirrors the keyframe convention where ``sample["data"]["CAM_*"]`` points to the camera
    image captured during (just before completion of) the lidar sweep.  For consistency, at
    non-keyframe timestamps we select the most recent camera record whose timestamp is
    <= the target lidar sweep timestamp, within a tolerance of 100 ms.

    :param nusc: The NuScenes database instance.
    :param target_timestamp: Target timestamp in microseconds (lidar sweep time).
    :param camera_timelines: Camera timelines from :func:`_collect_camera_timelines`.
    :param nuscenes_data_root: Path to the nuScenes dataset root.
    :param dataset_converter_config: Dataset converter configuration.
    :return: List of CameraData for cameras within tolerance of the target timestamp.
    """
    camera_data_list: List[CameraData] = []
    if not dataset_converter_config.include_pinhole_cameras:
        return camera_data_list

    for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
        timeline = camera_timelines.get(camera_channel, [])
        if not timeline:
            continue

        # Find the last camera record at or before the target timestamp.
        # bisect_right gives the insertion point *after* any equal entries,
        # so idx-1 is the last entry with timestamp <= target_timestamp.
        timestamps = [sd["timestamp"] for sd in timeline]
        idx = bisect.bisect_right(timestamps, target_timestamp)

        if idx == 0:
            continue  # no camera record at or before the target

        best_idx = idx - 1
        if target_timestamp - timestamps[best_idx] > _CAMERA_TIMESTAMP_TOLERANCE_US:
            continue

        cam_data = timeline[best_idx]
        calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        translation_array = np.array(calib["translation"], dtype=np.float64)
        rotation_array = np.array(calib["rotation"], dtype=np.float64)
        extrinsic = PoseSE3.from_R_t(rotation=rotation_array, translation=translation_array)

        cam_path = nuscenes_data_root / str(cam_data["filename"])
        if cam_path.exists() and cam_path.is_file():
            camera_data_list.append(
                CameraData(
                    camera_name=camera_channel,
                    camera_id=camera_type,
                    extrinsic=extrinsic,
                    relative_path=cam_path.relative_to(nuscenes_data_root),
                    dataset_root=nuscenes_data_root,
                    timestamp=Timestamp.from_us(cam_data["timestamp"]),
                )
            )

    return camera_data_list


# ---------------------------------------------------------------------------
# Lidar extraction from sample_data (for non-keyframe sweeps)
# ---------------------------------------------------------------------------


def _extract_lidar_from_sample_data(
    sweep: Dict[str, Any],
    nuscenes_data_root: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> Optional[LidarData]:
    """Extracts lidar data from a sample_data record (works for keyframes and sweeps).

    :param sweep: A lidar sweep dict from the timeline.
    :param nuscenes_data_root: Path to the nuScenes dataset root.
    :param dataset_converter_config: Dataset converter configuration.
    :return: Optional LidarData.
    """
    lidar: Optional[LidarData] = None
    if dataset_converter_config.include_lidars:
        absolute_lidar_path = nuscenes_data_root / sweep["filename"]
        if absolute_lidar_path.exists() and absolute_lidar_path.is_file():
            lidar = LidarData(
                lidar_name="LIDAR_TOP",
                lidar_type=LidarID.LIDAR_TOP,
                relative_path=absolute_lidar_path.relative_to(nuscenes_data_root),
                dataset_root=nuscenes_data_root,
                timestamp=Timestamp.from_us(sweep["timestamp"]),
            )
    return lidar


# ---------------------------------------------------------------------------
# Keyframe box detection extraction (unchanged from NuScenesConverter)
# ---------------------------------------------------------------------------


def _extract_nuscenes_box_detections(nusc: NuScenes, sample: Dict[str, Any]) -> BoxDetectionsSE3:
    """Extracts the box detections from a nuScenes keyframe sample."""
    box_detections: List[BoxDetectionSE3] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))

        # Create PoseSE3 for box center and orientation
        center_quat = box.orientation
        center = PoseSE3(
            box.center[0],
            box.center[1],
            box.center[2],
            center_quat.w,
            center_quat.x,
            center_quat.y,
            center_quat.z,
        )
        bounding_box = BoundingBoxSE3(
            center_se3=center,
            length=box.wlh[1],
            width=box.wlh[0],
            height=box.wlh[2],
        )
        # Get detection type
        category = ann["category_name"]
        label = NUSCENES_DETECTION_NAME_DICT[category]

        # Get velocity if available
        velocity = nusc.box_velocity(ann_token)
        velocity_3d = Vector3D(x=velocity[0], y=velocity[1], z=velocity[2] if len(velocity) > 2 else 0.0)

        metadata = BoxDetectionAttributes(
            label=label,
            track_token=ann["instance_token"],
            num_lidar_points=ann.get("num_lidar_pts", 0),
        )
        box_detection = BoxDetectionSE3(
            metadata=metadata,
            bounding_box_se3=bounding_box,
            velocity_3d=velocity_3d,
        )
        box_detections.append(box_detection)
    return BoxDetectionsSE3(box_detections=box_detections, timestamp=Timestamp.from_us(sample["timestamp"]))  # type: ignore


# ---------------------------------------------------------------------------
# Camera extraction for keyframes (unchanged from NuScenesConverter)
# ---------------------------------------------------------------------------


def _extract_nuscenes_cameras(
    nusc: NuScenes,
    sample: Dict[str, Any],
    nuscenes_data_root: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> List[CameraData]:
    """Extracts the pinhole camera data from a nuScenes keyframe sample."""
    camera_data_list: List[CameraData] = []
    if dataset_converter_config.include_pinhole_cameras:
        for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
            cam_token = sample["data"][camera_channel]
            cam_data = nusc.get("sample_data", cam_token)

            # Check timestamp synchronization (within 100ms)
            if abs(cam_data["timestamp"] - sample["timestamp"]) > 100000:
                continue

            calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            translation_array = np.array(calib["translation"], dtype=np.float64)  # array of shape (3,)
            rotation_array = np.array(calib["rotation"], dtype=np.float64)  # array of shape (4,)
            extrinsic = PoseSE3.from_R_t(rotation=rotation_array, translation=translation_array)

            cam_path = nuscenes_data_root / str(cam_data["filename"])
            if cam_path.exists() and cam_path.is_file():
                camera_data_list.append(
                    CameraData(
                        camera_name=camera_channel,
                        camera_id=camera_type,
                        extrinsic=extrinsic,
                        relative_path=cam_path.relative_to(nuscenes_data_root),
                        dataset_root=nuscenes_data_root,
                        timestamp=Timestamp.from_us(cam_data["timestamp"]),
                    )
                )

    return camera_data_list


# ---------------------------------------------------------------------------
# Metadata helpers (unchanged from NuScenesConverter)
# ---------------------------------------------------------------------------


def _get_nuscenes_pinhole_camera_metadata(
    nusc: NuScenes,
    scene: Dict[str, Any],
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Extracts the pinhole camera metadata from a nuScenes scene."""
    camera_metadata: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    if dataset_converter_config.include_pinhole_cameras:
        first_sample_token = scene["first_sample_token"]
        first_sample = nusc.get("sample", first_sample_token)
        for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
            cam_token = first_sample["data"][camera_channel]
            cam_data = nusc.get("sample_data", cam_token)
            calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

            # Intrinsic & distortion parameters
            intrinsic_matrix = np.array(calib["camera_intrinsic"])
            intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsic_matrix)
            distortion = PinholeDistortion.from_array(np.zeros(5), copy=False)

            # Extrinsic parameters
            translation_array = np.array(calib["translation"], dtype=np.float64)  # array of shape (3,)
            rotation_array = np.array(calib["rotation"], dtype=np.float64)  # array of shape (4,)
            extrinsic = PoseSE3.from_R_t(rotation=rotation_array, translation=translation_array)

            camera_metadata[camera_type] = PinholeCameraMetadata(
                camera_name=camera_channel,
                camera_id=camera_type,
                width=cam_data["width"],
                height=cam_data["height"],
                intrinsics=intrinsic,
                distortion=distortion,
                camera_to_imu_se3=extrinsic,
                is_undistorted=True,
            )

    return camera_metadata


def _get_nuscenes_lidar_metadata(
    nusc: NuScenes,
    scene: Dict[str, Any],
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LidarID, LidarMetadata]:
    """Extracts the Lidar metadata from a nuScenes scene."""
    metadata: Dict[LidarID, LidarMetadata] = {}
    if dataset_converter_config.include_lidars:
        first_sample_token = scene["first_sample_token"]
        first_sample = nusc.get("sample", first_sample_token)
        lidar_token = first_sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)
        calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        translation = np.array(calib["translation"])
        rotation = Quaternion(calib["rotation"]).rotation_matrix
        lidar_to_imu_se3 = np.eye(4)
        lidar_to_imu_se3[:3, :3] = rotation
        lidar_to_imu_se3[:3, 3] = translation
        lidar_to_imu_se3 = PoseSE3.from_transformation_matrix(lidar_to_imu_se3)
        metadata[LidarID.LIDAR_TOP] = LidarMetadata(
            lidar_name="LIDAR_TOP",
            lidar_id=LidarID.LIDAR_TOP,
            lidar_to_imu_se3=lidar_to_imu_se3,
        )
    return metadata


def _get_nuscenes_map_metadata(location: str) -> MapMetadata:
    """Creates nuScenes map metadata for a given location."""
    return MapMetadata(
        dataset="nuscenes",
        split=None,
        log_name=None,
        location=location,
        map_has_z=False,
        map_is_local=False,
    )
