from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Union

import DracoPy
import numpy as np
import numpy.typing as npt
import pandas as pd

from py123d.datatypes.sensors.lidar import LidarFeature, LidarID, LidarMetadata
from py123d.geometry.transform import rel_to_abs_points_3d_array


@lru_cache(maxsize=4)
def _read_lidar_parquet(parquet_path: str) -> pd.DataFrame:
    """Read and cache a LiDAR parquet file. Cached to avoid re-reading the same ~200MB file per spin."""
    return pd.read_parquet(parquet_path)


def decode_physical_ai_av_spin(
    draco_blob: bytes, spin_start_us: int, lidar_to_imu_se3
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray]]:
    """Decode one lidar spin (Draco point cloud) into ego-frame points and per-point features.

    The current dataset revision stores three named generic attributes besides the positions: ``intensity`` (uint8),
    ``timestamp`` (uint32, the absolute point time in us, in the time base of ``spin_start_timestamp``) and
    ``model_element`` (2 x uint16: the beam index 0..127 and the column index 0..3599). They map to the features
    INTENSITY, TIMESTAMPS, CHANNEL and COLUMN. Older revisions stored intensity and a relative 16-bit time in the
    Draco colors (red = intensity, green + 256 * blue = time code); those are read when the named attributes are
    absent.

    :param draco_blob: The Draco-encoded point cloud of the spin.
    :param spin_start_us: The spin start time in us (for the time code of older revisions).
    :param lidar_to_imu_se3: The lidar extrinsic; the points are transformed from the sensor frame to the ego frame.
    :return: Points in the ego frame (N x 3, float32) and the features.
    """
    mesh = DracoPy.decode(draco_blob)
    points = np.asarray(mesh.points, dtype=np.float64).reshape(-1, 3)
    points = rel_to_abs_points_3d_array(origin=lidar_to_imu_se3, points_3d_array=points).astype(np.float32)

    named: Dict[str, npt.NDArray] = {}
    for attribute in getattr(mesh, "attributes", None) or []:
        name = attribute.get("name") if isinstance(attribute, dict) else None
        if name:
            named[name] = np.asarray(attribute["data"])

    features: Dict[str, npt.NDArray] = {}
    if "intensity" in named:
        features[LidarFeature.INTENSITY.serialize()] = named["intensity"].reshape(-1).astype(np.uint8)
    if "timestamp" in named:
        features[LidarFeature.TIMESTAMPS.serialize()] = named["timestamp"].reshape(-1).astype(np.int64)
    if "model_element" in named:
        element = named["model_element"].reshape(-1, 2)
        assert int(element[:, 0].max(initial=0)) <= np.iinfo(np.uint8).max, "beam index exceeds uint8"
        features[LidarFeature.CHANNEL.serialize()] = element[:, 0].astype(np.uint8)
        features[LidarFeature.COLUMN.serialize()] = element[:, 1].astype(np.uint16)
    if not named and mesh.colors is not None:
        colors = np.asarray(mesh.colors, dtype=np.uint8)
        features[LidarFeature.INTENSITY.serialize()] = colors[:, 0]
        raw = (colors[:, 2].astype(np.int32) << 8) | colors[:, 1].astype(np.int32)
        relative_us = 105000.0 - raw * (210000.0 / 65535.0)
        features[LidarFeature.TIMESTAMPS.serialize()] = int(spin_start_us) + np.round(relative_us).astype(np.int64)
    return points, features


def spin_start_column(columns) -> str:
    """The spin start column of a lidar table: ``spin_start_timestamp`` (current revisions) or
    ``reference_timestamp`` (older revisions)."""
    for column in ("spin_start_timestamp", "reference_timestamp", "timestamp"):
        if column in columns:
            return column
    raise ValueError(f"No spin start column in the lidar table; columns: {list(columns)}")


def load_physical_ai_av_point_cloud_data_from_path(
    parquet_path: Union[Path, str],
    spin_index: int,
    lidar_metadatas: Dict[LidarID, LidarMetadata],
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray]]:
    """Load a single LiDAR spin from a Physical AI AV parquet file (see :func:`decode_physical_ai_av_spin`).

    :param parquet_path: Path to the LiDAR parquet file.
    :param spin_index: Index of the spin to decode (row index in the parquet).
    :param lidar_metadatas: Dictionary mapping LidarID to LidarMetadata (contains extrinsics).
    :return: Tuple of (xyz in ego frame as Nx3 float32 array, features dict).
    """
    lidar_df = _read_lidar_parquet(str(parquet_path))
    row = lidar_df.iloc[spin_index]
    spin_start_us = int(row[spin_start_column(lidar_df.columns)])
    return decode_physical_ai_av_spin(
        row["draco_encoded_pointcloud"], spin_start_us, lidar_metadatas[LidarID.LIDAR_TOP].lidar_to_imu_se3
    )
