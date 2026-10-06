"""KITScenes Multimodal on-disk point cloud loaders for py123d path-based lidar I/O."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from py123d.datatypes import LidarFeature
from py123d.geometry import PoseSE3
from py123d.geometry.transform import reframe_points_3d_array
from py123d.parser.kitscenes.kitscenes_constants import CALIBRATION_FILE, LIDAR_ID_MAPPING

# Lidar coordinates are stored as int32 in units of the ``discretization_resolution`` file metadata (5 mm).
_DEFAULT_DISCRETIZATION_RESOLUTION = 0.005
_LIDAR_COLUMNS = ["x", "y", "z", "reflectivity", "timestamp", "ring"]


@lru_cache(maxsize=64)
def _load_lidar_to_imu_se3(calibration_path: Path, lidar_name: str) -> PoseSE3:
    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)
    return PoseSE3.from_transformation_matrix(np.array(calibration[lidar_name]["T_to_reference"], dtype=np.float64))


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
    lidar_to_imu_se3 = _load_lidar_to_imu_se3(parquet_path.parent.parent / CALIBRATION_FILE, lidar_name)

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


def _valid_point_mask(table) -> npt.NDArray[np.bool_]:
    """Invalid returns are stored as points exactly at the sensor origin."""
    return (
        (table.column("x").to_numpy() != 0) | (table.column("y").to_numpy() != 0) | (table.column("z").to_numpy() != 0)
    )


def _seconds_to_us(timestamps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    return np.round(timestamps_s * 1e6).astype(np.int64)
