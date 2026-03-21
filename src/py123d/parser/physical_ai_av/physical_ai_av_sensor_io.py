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


def load_physical_ai_av_point_cloud_data_from_path(
    parquet_path: Union[Path, str],
    spin_index: int,
    lidar_metadatas: Dict[LidarID, LidarMetadata],
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray]]:
    """Load a single LiDAR spin from a Physical AI AV parquet file.

    Points are decoded from Draco and transformed from the lidar sensor frame
    to the ego (vehicle/rig) frame using the lidar-to-vehicle extrinsic.

    :param parquet_path: Path to the LiDAR parquet file.
    :param spin_index: Index of the spin to decode (row index in the parquet).
    :param lidar_metadatas: Dictionary mapping LidarID to LidarMetadata (contains extrinsics).
    :return: Tuple of (xyz in ego frame as Nx3 float32 array, features dict).
    """
    lidar_df = _read_lidar_parquet(str(parquet_path))
    row = lidar_df.iloc[spin_index]

    draco_blob = row["draco_encoded_pointcloud"]
    mesh = DracoPy.decode(draco_blob)
    points = np.array(mesh.points, dtype=np.float64).reshape(-1, 3)

    # Transform from sensor frame to ego frame
    lidar_to_ego = lidar_metadatas[LidarID.LIDAR_TOP].lidar_to_imu_se3
    points = rel_to_abs_points_3d_array(origin=lidar_to_ego, points_3d_array=points).astype(np.float32)

    # Extract per-point features from Draco color channels.
    # Physical AI AV encodes: Red = intensity (uint8), Green+Blue = 16-bit relative timestamp.
    # Formula: raw = (Blue << 8) | Green, relative_us = 105000 - raw * 210000 / 65535
    # Absolute timestamp = reference_timestamp + relative_us
    lidar_features: Dict[str, npt.NDArray] = {}
    colors = mesh.colors
    if colors is not None:
        colors_arr = np.array(colors, dtype=np.uint8)
        lidar_features[LidarFeature.INTENSITY.serialize()] = colors_arr[:, 0]

        green = colors_arr[:, 1].astype(np.int32)
        blue = colors_arr[:, 2].astype(np.int32)
        raw_ts = (blue << 8) | green
        relative_us = 105000.0 - raw_ts * (210000.0 / 65535.0)
        reference_timestamp = int(row["reference_timestamp"])
        absolute_us = reference_timestamp + np.round(relative_us).astype(np.int64)
        lidar_features[LidarFeature.TIMESTAMPS.serialize()] = absolute_us

    return points, lidar_features
