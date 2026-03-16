"""Shared extraction functions for nuScenes parsers (2Hz and 10Hz).

This module contains all sensor data extraction, timeline collection, and interpolation
functions used by :class:`NuScenesParser` for both 2Hz and 10Hz modes.
"""

from __future__ import annotations

import bisect
import gc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    DynamicStateSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.sensors.lidar import LidarMergedMetadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import ParsedCamera, ParsedLidar
from py123d.parser.nuscenes.utils.nuscenes_constants import (
    NUSCENES_BOX_DETECTIONS_SE3_METADATA,
    NUSCENES_CAMERA_IDS,
    NUSCENES_DETECTION_NAME_DICT,
    NUSCENES_LIDAR_SWEEP_DURATION_US,
    TARGET_DT,
)

check_dependencies(["nuscenes"], "nuscenes")
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

# Target time interval in microseconds (10Hz = 100ms)
_TARGET_DT_US: int = int(TARGET_DT * 1e6)

# Tolerance for matching cameras to lidar sweep timestamps.
# Since we select the last camera *before* the lidar timestamp, the offset can be up to one full
# camera period (~83 ms at ~12 Hz). We use 100 ms to be consistent with the keyframe extraction.
_CAMERA_TIMESTAMP_TOLERANCE_US: int = 100_000


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def get_nuscenes_pinhole_camera_metadata_from_scene(
    load_nusc_fn: Callable[[], NuScenes],
    scene_token: str,
) -> Optional[Dict[CameraID, PinholeCameraMetadata]]:
    """Extracts the pinhole camera metadata from a nuScenes scene."""
    nusc = load_nusc_fn()
    try:
        scene = nusc.get("scene", scene_token)
        return get_nuscenes_pinhole_camera_metadata(nusc, scene) or None
    finally:
        del nusc
        gc.collect()


def get_nuscenes_pinhole_camera_metadata(
    nusc: NuScenes,
    scene: Dict[str, Any],
) -> Dict[CameraID, PinholeCameraMetadata]:
    """Extracts the pinhole camera metadata from a nuScenes scene."""
    camera_metadata: Dict[CameraID, PinholeCameraMetadata] = {}
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)
    for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
        cam_token = first_sample["data"][camera_channel]
        cam_data = nusc.get("sample_data", cam_token)
        calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

        intrinsic_matrix = np.array(calib["camera_intrinsic"])
        intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsic_matrix)
        distortion = PinholeDistortion.from_array(np.zeros(5), copy=False)

        translation_array = np.array(calib["translation"], dtype=np.float64)
        rotation_array = np.array(calib["rotation"], dtype=np.float64)
        camera_to_imu_se3 = PoseSE3.from_R_t(rotation=rotation_array, translation=translation_array)

        camera_metadata[camera_type] = PinholeCameraMetadata(
            camera_name=camera_channel,
            camera_id=camera_type,
            width=cam_data["width"],
            height=cam_data["height"],
            intrinsics=intrinsic,
            distortion=distortion,
            camera_to_imu_se3=camera_to_imu_se3,
            is_undistorted=True,
        )

    return camera_metadata


def get_nuscenes_lidar_metadata_from_scene(
    load_nusc_fn: Callable[[], NuScenes],
    scene_token: str,
) -> LidarMergedMetadata:
    """Extracts the Lidar merged metadata from a nuScenes scene."""
    nusc = load_nusc_fn()
    try:
        scene = nusc.get("scene", scene_token)
        return get_nuscenes_lidar_merged_metadata(nusc, scene)
    finally:
        del nusc
        gc.collect()


def get_nuscenes_lidar_merged_metadata(
    nusc: NuScenes,
    scene: Dict[str, Any],
) -> LidarMergedMetadata:
    """Extracts the Lidar merged metadata from a nuScenes scene."""
    metadata: Dict[LidarID, LidarMetadata] = {}
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)
    lidar_token = first_sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    lidar_to_imu_se3 = PoseSE3.from_R_t(
        rotation=np.array(calib["rotation"], dtype=np.float64),
        translation=np.array(calib["translation"], dtype=np.float64),
    )
    metadata[LidarID.LIDAR_TOP] = LidarMetadata(
        lidar_name="LIDAR_TOP",
        lidar_id=LidarID.LIDAR_TOP,
        lidar_to_imu_se3=lidar_to_imu_se3,
    )
    return LidarMergedMetadata(metadata)


# ------------------------------------------------------------------------------------------------------------------
# Timeline collection helpers
# ------------------------------------------------------------------------------------------------------------------


def collect_lidar_sweep_timeline(nusc: NuScenes, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def collect_camera_timelines(
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


def collect_keyframe_samples(nusc: NuScenes, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
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


# ------------------------------------------------------------------------------------------------------------------
# CAN bus dynamic state helper
# ------------------------------------------------------------------------------------------------------------------


def _find_dynamic_state_from_can_bus(
    can_bus: NuScenesCanBus,
    scene_name: str,
    timestamp_us: int,
    max_time_diff_us: int = 500_000,
) -> DynamicStateSE3:
    """Finds the closest CAN bus dynamic state for a given timestamp.

    Uses binary search (bisect) for efficient lookup. Falls back to zero dynamics
    if the CAN bus has no data for this scene or the closest message exceeds the tolerance.

    :param can_bus: The NuScenes CAN bus API.
    :param scene_name: The scene name for CAN bus lookup.
    :param timestamp_us: Target timestamp in microseconds.
    :param max_time_diff_us: Maximum allowed time difference in microseconds (default 500ms).
    :return: The dynamic state (velocity, acceleration, angular velocity).
    """
    # NOTE: The nuscenes-devkit CAN bus API raises a bare Exception (not a typed subclass)
    # when a scene is not available in the CAN bus data. We must catch Exception here.
    try:
        pose_msgs = can_bus.get_messages(scene_name, "pose")
    except Exception:
        pose_msgs = []

    if not pose_msgs:
        return DynamicStateSE3(
            velocity=Vector3D(0.0, 0.0, 0.0),
            acceleration=Vector3D(0.0, 0.0, 0.0),
            angular_velocity=Vector3D(0.0, 0.0, 0.0),
        )

    # Binary search for the closest CAN message by timestamp
    utimes = [msg["utime"] for msg in pose_msgs]
    idx = bisect.bisect_left(utimes, timestamp_us)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(utimes):
        candidates.append(idx)

    best_idx = min(candidates, key=lambda i: abs(utimes[i] - timestamp_us))
    min_time_diff = abs(utimes[best_idx] - timestamp_us)

    if min_time_diff < max_time_diff_us:
        msg = pose_msgs[best_idx]
        return DynamicStateSE3(
            velocity=Vector3D(*msg["vel"]),
            acceleration=Vector3D(*msg["accel"]),
            angular_velocity=Vector3D(*msg["rotation_rate"]),
        )

    return DynamicStateSE3(
        velocity=Vector3D(0.0, 0.0, 0.0),
        acceleration=Vector3D(0.0, 0.0, 0.0),
        angular_velocity=Vector3D(0.0, 0.0, 0.0),
    )


# ------------------------------------------------------------------------------------------------------------------
# Ego state extraction
# ------------------------------------------------------------------------------------------------------------------


def extract_ego_state_from_sample(
    nusc: NuScenes, sample: Dict[str, Any], can_bus: NuScenesCanBus, ego_metadata: EgoStateSE3Metadata
) -> EgoStateSE3:
    """Extracts the ego state from a nuScenes keyframe sample (via its LIDAR_TOP data)."""
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    imu_pose = PoseSE3.from_R_t(
        rotation=np.array(ego_pose["rotation"], dtype=np.float64),
        translation=np.array(ego_pose["translation"], dtype=np.float64),
    )

    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    dynamic_state = _find_dynamic_state_from_can_bus(can_bus, scene_name, sample["timestamp"])

    return EgoStateSE3.from_imu(
        imu_se3=imu_pose,
        dynamic_state_se3=dynamic_state,
        metadata=ego_metadata,
        timestamp=Timestamp.from_us(sample["timestamp"]),
    )


def extract_ego_state_from_sample_data(
    nusc: NuScenes,
    sweep: Dict[str, Any],
    can_bus: NuScenesCanBus,
    scene_name: str,
    ego_metadata: EgoStateSE3Metadata,
) -> EgoStateSE3:
    """Extracts the ego state from a lidar sample_data record (keyframe or non-keyframe).

    Uses the real ego pose from the sample_data's ego_pose_token and matches
    CAN bus data for dynamic state (velocity, acceleration, angular velocity).

    :param nusc: The NuScenes database instance.
    :param sweep: A lidar sweep dict from the timeline.
    :param can_bus: The NuScenes CAN bus API.
    :param scene_name: The scene name for CAN bus lookup.
    :param ego_metadata: Vehicle parameters for ego state construction.
    :return: The ego state.
    """
    ego_pose = nusc.get("ego_pose", sweep["ego_pose_token"])

    imu_pose = PoseSE3.from_R_t(
        rotation=np.array(ego_pose["rotation"], dtype=np.float64),
        translation=np.array(ego_pose["translation"], dtype=np.float64),
    )

    dynamic_state = _find_dynamic_state_from_can_bus(can_bus, scene_name, sweep["timestamp"])

    return EgoStateSE3.from_imu(
        imu_se3=imu_pose,
        dynamic_state_se3=dynamic_state,
        metadata=ego_metadata,
        timestamp=Timestamp.from_us(sweep["timestamp"]),
    )


# ------------------------------------------------------------------------------------------------------------------
# Box detection extraction
# ------------------------------------------------------------------------------------------------------------------


def extract_nuscenes_box_detections(
    nusc: NuScenes, sample: Dict[str, Any], box_detections_metadata: BoxDetectionsSE3Metadata
) -> BoxDetectionsSE3:
    """Extracts the box detections from a nuScenes keyframe sample.

    Handles NaN velocities (returned by nuscenes-devkit for boundary annotations) by replacing
    them with zero vectors.
    """
    box_detections: List[BoxDetectionSE3] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        center_se3 = PoseSE3.from_R_t(
            rotation=np.array(ann["rotation"], dtype=np.float64),
            translation=np.array(ann["translation"], dtype=np.float64),
        )
        width, length, height = ann["size"]
        bounding_box = BoundingBoxSE3(
            center_se3=center_se3,
            length=length,
            width=width,
            height=height,
        )
        category = ann["category_name"]
        label = NUSCENES_DETECTION_NAME_DICT[category]

        velocity = nusc.box_velocity(ann_token)
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3)
        velocity_3d = Vector3D(x=float(velocity[0]), y=float(velocity[1]), z=float(velocity[2]))

        attributes = BoxDetectionAttributes(
            label=label,
            track_token=ann["instance_token"],
            num_lidar_points=ann.get("num_lidar_pts", 0),
        )
        box_detection = BoxDetectionSE3(
            attributes=attributes,
            bounding_box_se3=bounding_box,
            velocity_3d=velocity_3d,
        )
        box_detections.append(box_detection)
    return BoxDetectionsSE3(
        box_detections=box_detections,
        timestamp=Timestamp.from_us(sample["timestamp"]),
        metadata=box_detections_metadata,
    )


# ------------------------------------------------------------------------------------------------------------------
# Camera extraction
# ------------------------------------------------------------------------------------------------------------------


def _extract_camera_from_sample_data(
    nusc: NuScenes,
    cam_data: Dict[str, Any],
    camera_type: CameraID,
    nuscenes_data_root: Path,
    pinhole_cameras_metadata: Dict[CameraID, PinholeCameraMetadata],
) -> Optional[ParsedCamera]:
    """Extracts a single ParsedCamera from a camera sample_data record.

    Computes the correct camera_to_global_se3 by composing the ego pose with the static
    camera_to_imu extrinsic from the metadata.

    :param nusc: The NuScenes database instance.
    :param cam_data: A camera sample_data record.
    :param camera_type: The CameraID enum for this camera.
    :param nuscenes_data_root: Path to the nuScenes dataset root.
    :param pinhole_cameras_metadata: Camera metadata dict keyed by CameraID.
    :return: ParsedCamera, or None if the file doesn't exist.
    """
    cam_path = nuscenes_data_root / str(cam_data["filename"])
    if not (cam_path.exists() and cam_path.is_file()):
        return None

    ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
    ego_to_global = PoseSE3.from_R_t(
        rotation=np.array(ego_pose["rotation"], dtype=np.float64),
        translation=np.array(ego_pose["translation"], dtype=np.float64),
    )
    camera_to_global_se3 = rel_to_abs_se3(
        origin=ego_to_global,
        pose_se3=pinhole_cameras_metadata[camera_type].camera_to_imu_se3,
    )

    return ParsedCamera(
        metadata=pinhole_cameras_metadata[camera_type],
        timestamp=Timestamp.from_us(cam_data["timestamp"]),
        camera_to_global_se3=camera_to_global_se3,
        dataset_root=nuscenes_data_root,
        relative_path=cam_path.relative_to(nuscenes_data_root),
    )


def extract_nuscenes_cameras(
    nusc: NuScenes,
    sample: Dict[str, Any],
    nuscenes_data_root: Path,
    pinhole_cameras_metadata: Optional[Dict[CameraID, PinholeCameraMetadata]],
) -> List[ParsedCamera]:
    """Extracts the pinhole camera data from a nuScenes keyframe sample.

    For each camera, the camera_to_global_se3 is computed by composing the ego pose
    (from the camera's ego_pose_token) with the static camera_to_imu extrinsic.
    """
    if pinhole_cameras_metadata is None:
        return []

    camera_data_list: List[ParsedCamera] = []
    for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
        if camera_type not in pinhole_cameras_metadata:
            continue

        cam_token = sample["data"][camera_channel]
        cam_data = nusc.get("sample_data", cam_token)

        # Check timestamp synchronization (within 100ms)
        if abs(cam_data["timestamp"] - sample["timestamp"]) > _CAMERA_TIMESTAMP_TOLERANCE_US:
            continue

        parsed_camera = _extract_camera_from_sample_data(
            nusc, cam_data, camera_type, nuscenes_data_root, pinhole_cameras_metadata
        )
        if parsed_camera is not None:
            camera_data_list.append(parsed_camera)

    return camera_data_list


def extract_cameras_from_timeline(
    nusc: NuScenes,
    cam_data: Dict[str, Any],
    camera_type: CameraID,
    nuscenes_data_root: Path,
    pinhole_cameras_metadata: Dict[CameraID, PinholeCameraMetadata],
) -> Optional[ParsedCamera]:
    """Extracts a single ParsedCamera from a camera timeline record (for async iteration)."""
    return _extract_camera_from_sample_data(nusc, cam_data, camera_type, nuscenes_data_root, pinhole_cameras_metadata)


def find_nearest_cameras_for_sweep(
    nusc: NuScenes,
    target_timestamp: int,
    camera_timelines: Dict[str, List[Dict[str, Any]]],
    nuscenes_data_root: Path,
    pinhole_cameras_metadata: Optional[Dict[CameraID, PinholeCameraMetadata]],
) -> List[ParsedCamera]:
    """Finds the last camera observation at or before a given sweep timestamp for each channel.

    This mirrors the keyframe convention where ``sample["data"]["CAM_*"]`` points to the camera
    image captured during (just before completion of) the lidar sweep. For consistency, at
    non-keyframe timestamps we select the most recent camera record whose timestamp is
    <= the target lidar sweep timestamp, within a tolerance of 100 ms.

    :param nusc: The NuScenes database instance.
    :param target_timestamp: Target timestamp in microseconds (lidar sweep time).
    :param camera_timelines: Camera timelines from :func:`collect_camera_timelines`.
    :param nuscenes_data_root: Path to the nuScenes dataset root.
    :param pinhole_cameras_metadata: Camera metadata dict keyed by camera ID.
    :return: List of ParsedCamera for cameras within tolerance of the target timestamp.
    """
    if pinhole_cameras_metadata is None:
        return []

    camera_data_list: List[ParsedCamera] = []

    for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
        if camera_type not in pinhole_cameras_metadata:
            continue

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
        parsed_camera = _extract_camera_from_sample_data(
            nusc, cam_data, camera_type, nuscenes_data_root, pinhole_cameras_metadata
        )
        if parsed_camera is not None:
            camera_data_list.append(parsed_camera)

    return camera_data_list


# ------------------------------------------------------------------------------------------------------------------
# Lidar extraction
# ------------------------------------------------------------------------------------------------------------------


def extract_nuscenes_lidar(
    nusc: NuScenes,
    sample: Dict[str, Any],
    nuscenes_data_root: Path,
    lidar_metadata: LidarMergedMetadata,
) -> Optional[ParsedLidar]:
    """Extracts the Lidar data from a nuScenes keyframe sample."""
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    absolute_lidar_path = nuscenes_data_root / lidar_data["filename"]
    if absolute_lidar_path.exists() and absolute_lidar_path.is_file():
        # The nuScenes lidar timestamp marks the end of the sweep (full rotation).
        # The sweep covers the 1/20s (50ms) period before that timestamp.
        end_timestamp = Timestamp.from_us(sample["timestamp"])
        start_timestamp = Timestamp.from_us(sample["timestamp"] - NUSCENES_LIDAR_SWEEP_DURATION_US)
        return ParsedLidar(
            metadata=lidar_metadata,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            relative_path=absolute_lidar_path.relative_to(nuscenes_data_root),
            dataset_root=nuscenes_data_root,
            iteration=lidar_data.get("iteration"),
        )
    return None


def extract_lidar_from_sample_data(
    sweep: Dict[str, Any],
    nuscenes_data_root: Path,
    lidar_metadata: LidarMergedMetadata,
) -> Optional[ParsedLidar]:
    """Extracts lidar data from a sample_data record (works for keyframes and sweeps).

    :param sweep: A lidar sweep dict from the timeline.
    :param nuscenes_data_root: Path to the nuScenes dataset root.
    :param lidar_metadata: Lidar merged metadata.
    :return: Optional ParsedLidar.
    """
    absolute_lidar_path = nuscenes_data_root / sweep["filename"]
    if absolute_lidar_path.exists() and absolute_lidar_path.is_file():
        # The nuScenes lidar timestamp marks the end of the sweep (full rotation).
        # The sweep covers the 1/20s (50ms) period before that timestamp.
        end_timestamp = Timestamp.from_us(sweep["timestamp"])
        start_timestamp = Timestamp.from_us(sweep["timestamp"] - NUSCENES_LIDAR_SWEEP_DURATION_US)
        return ParsedLidar(
            metadata=lidar_metadata,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            relative_path=absolute_lidar_path.relative_to(nuscenes_data_root),
            dataset_root=nuscenes_data_root,
        )
    return None


# ------------------------------------------------------------------------------------------------------------------
# 10Hz interpolation helpers
# ------------------------------------------------------------------------------------------------------------------


def select_10hz_sweeps(
    lidar_timeline: List[Dict[str, Any]],
    keyframe_timestamps: List[int],
) -> List[Dict[str, Any]]:
    """Selects approximately 10Hz sweeps from the ~20Hz lidar timeline.

    Between each pair of consecutive keyframes, distributes intermediate timestamps at
    regular intervals and picks the closest lidar sweep for each. Keyframes are always included.

    :param lidar_timeline: Full lidar sweep timeline from :func:`collect_lidar_sweep_timeline`.
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


def find_surrounding_keyframes(
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


def interpolate_box_detections(
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
        next_by_track[det.attributes.track_token] = det

    interpolated: List[BoxDetectionSE3] = []
    for prev_det in prev_detections:
        track_token = prev_det.attributes.track_token
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

        attributes = BoxDetectionAttributes(
            label=prev_det.attributes.label,
            track_token=track_token,
            num_lidar_points=0,
        )

        interpolated.append(
            BoxDetectionSE3(
                attributes=attributes,
                bounding_box_se3=bounding_box,
                velocity_3d=velocity_3d,
            )
        )

    return BoxDetectionsSE3(
        box_detections=interpolated, timestamp=interpolated_timestamp, metadata=NUSCENES_BOX_DETECTIONS_SE3_METADATA
    )
