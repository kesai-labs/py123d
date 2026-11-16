"""KITScenes Multimodal on-disk point cloud loaders for py123d path-based lidar and radar I/O."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from py123d.datatypes import LidarFeature
from py123d.datatypes.sensors.radar import RADAR_FEATURE_DTYPES, RadarFeature
from py123d.geometry import PoseSE3
from py123d.geometry.transform import reframe_points_3d_array
from py123d.parser.kitscenes.kitscenes_constants import CALIBRATION_FILE, LIDAR_ID_MAPPING, RADAR_ID_MAPPING

# Lidar coordinates are stored as int32 in units of the ``discretization_resolution`` file metadata (5 mm).
_DEFAULT_DISCRETIZATION_RESOLUTION = 0.005
_LIDAR_COLUMNS = ["x", "y", "z", "reflectivity", "timestamp", "ring"]
_RADAR_COLUMNS = ["x", "y", "z", "rcs", "range_rate", "azimuth_std", "elevation_std", "timestamp", "detection_id"]


@lru_cache(maxsize=64)
def _load_sensor_to_imu_se3(calibration_path: Path, sensor_name: str) -> PoseSE3:
    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)
    return PoseSE3.from_transformation_matrix(np.array(calibration[sensor_name]["T_to_reference"], dtype=np.float64))


def load_kitscenes_lidar_timestamps_us(parquet_path: Union[Path, str]) -> npt.NDArray[np.int64]:
    """Load the per-point timestamps of all valid points of a lidar sweep, in microseconds."""
    table = pq.read_table(parquet_path, columns=["x", "y", "z", "timestamp"])
    valid = _valid_point_mask(table)
    return _seconds_to_us(table.column("timestamp").to_numpy()[valid])


def load_kitscenes_point_cloud_data_from_path(
    parquet_path: Union[Path, str],
) -> Tuple[npt.NDArray[np.float32], Dict[str, np.ndarray]]:
    """Load a KITScenes lidar sweep into the py123d point cloud layout.

    Points are stored in the sensor frame without ego-motion compensation. They are transformed to the ego frame
    using the extrinsic from the scene's ``calib.json``. Invalid returns, stored as points at the origin, are removed.

    :param parquet_path: Absolute path to ``<scene>/<lidar_name>/<frame_index>.parquet``.
    :return: ``(point_cloud_3d, point_cloud_features)`` with xyz float32 in the ego frame and feature arrays.
    """
    parquet_path = Path(parquet_path)
    lidar_name = parquet_path.parent.name
    lidar_id = LIDAR_ID_MAPPING[lidar_name]
    lidar_to_imu_se3 = _load_sensor_to_imu_se3(parquet_path.parent.parent / CALIBRATION_FILE, lidar_name)

    table = pq.read_table(parquet_path, columns=_LIDAR_COLUMNS)
    file_metadata = table.schema.metadata or {}
    resolution = float(file_metadata.get(b"discretization_resolution", _DEFAULT_DISCRETIZATION_RESOLUTION))

    valid = _valid_point_mask(table)
    xyz = np.stack([table.column(axis).to_numpy()[valid] for axis in ("x", "y", "z")], axis=-1)
    point_cloud_3d = reframe_points_3d_array(
        from_origin=lidar_to_imu_se3,
        to_origin=PoseSE3.identity(),
        points_3d_array=xyz.astype(np.float64) * resolution,
    ).astype(np.float32)

    # Reflectivity is ~[0, 1] for diffuse surfaces and exceeds 1 for retroreflectors (e.g. signs, plates).
    reflectivity = table.column("reflectivity").to_numpy()[valid]
    point_cloud_features = {
        LidarFeature.IDS.serialize(): np.full(len(point_cloud_3d), int(lidar_id), dtype=np.uint8),
        LidarFeature.INTENSITY.serialize(): (np.clip(reflectivity, 0.0, 1.0) * 255.0).astype(np.uint8),
        LidarFeature.CHANNEL.serialize(): table.column("ring").to_numpy()[valid].astype(np.uint8),
        LidarFeature.TIMESTAMPS.serialize(): _seconds_to_us(table.column("timestamp").to_numpy()[valid]),
    }
    return point_cloud_3d, point_cloud_features


def load_kitscenes_radar_timestamp_us(parquet_path: Union[Path, str]) -> Optional[int]:
    """Load the timestamp of a radar sweep in microseconds. All detections of a sweep share one timestamp."""
    timestamps_s = pq.read_table(parquet_path, columns=["timestamp"]).column("timestamp").to_numpy()
    if len(timestamps_s) == 0:
        return None
    return int(_seconds_to_us(np.median(timestamps_s)))


def load_kitscenes_radar_point_cloud_data_from_path(
    parquet_path: Union[Path, str],
) -> Tuple[npt.NDArray[np.float32], Dict[str, np.ndarray]]:
    """Load a KITScenes radar sweep (Continental ARS548 detections) into the py123d point cloud layout.

    Detections are stored in the sensor frame and transformed to the ego frame with the extrinsic from the scene's
    ``calib.json``. The raw radial velocity (``range_rate``) is kept and additionally projected onto the ego-frame
    ray direction, as in the other radar parsers. It is not ego-motion compensated.

    :param parquet_path: Absolute path to ``<scene>/<radar_name>/<frame_index>.parquet``.
    :return: ``(point_cloud_3d, point_cloud_features)`` with xyz float32 in the ego frame and feature arrays.
    """
    parquet_path = Path(parquet_path)
    radar_name = parquet_path.parent.name
    radar_id = RADAR_ID_MAPPING[radar_name]
    radar_to_imu_se3 = _load_sensor_to_imu_se3(parquet_path.parent.parent / CALIBRATION_FILE, radar_name)

    table = pq.read_table(parquet_path, columns=_RADAR_COLUMNS)
    xyz_sensor = np.stack([table.column(axis).to_numpy() for axis in ("x", "y", "z")], axis=-1).astype(np.float64)
    point_cloud_3d = reframe_points_3d_array(
        from_origin=radar_to_imu_se3,
        to_origin=PoseSE3.identity(),
        points_3d_array=xyz_sensor,
    ).astype(np.float32)

    radial_velocity = table.column("range_rate").to_numpy()
    direction_ego = xyz_sensor @ radar_to_imu_se3.rotation_matrix.T
    direction_norm = np.linalg.norm(direction_ego, axis=-1, keepdims=True)
    unit_direction_ego = np.divide(
        direction_ego, direction_norm, out=np.zeros_like(direction_ego), where=direction_norm > 0
    )
    velocity_ego = unit_direction_ego * radial_velocity[:, None]

    features = {
        RadarFeature.IDS: np.full(len(point_cloud_3d), int(radar_id)),
        RadarFeature.TIMESTAMPS: _seconds_to_us(table.column("timestamp").to_numpy()),
        RadarFeature.CLUSTER_ID: table.column("detection_id").to_numpy(),
        RadarFeature.RCS: table.column("rcs").to_numpy(),
        RadarFeature.RADIAL_VELOCITY: radial_velocity,
        RadarFeature.VELOCITY_X: velocity_ego[:, 0],
        RadarFeature.VELOCITY_Y: velocity_ego[:, 1],
        RadarFeature.AZIMUTH_STD: table.column("azimuth_std").to_numpy(),
        RadarFeature.ELEVATION_STD: table.column("elevation_std").to_numpy(),
    }
    point_cloud_features = {
        feature.serialize(): values.astype(RADAR_FEATURE_DTYPES[feature]) for feature, values in features.items()
    }
    return point_cloud_3d, point_cloud_features


def _valid_point_mask(table) -> npt.NDArray[np.bool_]:
    """Invalid returns are stored as points exactly at the sensor origin."""
    return (
        (table.column("x").to_numpy() != 0) | (table.column("y").to_numpy() != 0) | (table.column("z").to_numpy() != 0)
    )


def _seconds_to_us(timestamps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    return np.round(timestamps_s * 1e6).astype(np.int64)
