import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from py123d.datatypes import LidarFeature, LidarID, LidarMetadatas
from py123d.geometry import PoseSE3
from py123d.geometry.transform import reframe_points_3d_array


def load_kitti360_point_cloud_data_from_path(
    filepath: Path, lidar_metadatas: LidarMetadatas
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads KITTI-360 Lidar point clouds the original binary files."""

    if not filepath.exists():
        logging.warning(f"Lidar file does not exist: {filepath}. Returning empty point cloud.")
        return np.empty((0, 3), dtype=np.float32), {}

    # NOTE @DanielDauner: KITTI-360 stores point clouds is binary files, that need to be reshaped to (N,4).
    # Indices: x,y,z and intensity. Intensity is stored as a float, but we will convert it to uint8 in the Lidar data structure.
    lidar_extrinsic = lidar_metadatas[LidarID.LIDAR_TOP].extrinsic
    lidar_data = np.fromfile(filepath, dtype=np.float32).reshape([-1, 4])
    lidar_ids = np.zeros(lidar_data.shape[0], dtype=np.uint8)  # nuScenes only has a top lidar.
    lidar_ids[:] = int(LidarID.LIDAR_TOP)

    assert lidar_extrinsic is not None, "Lidar extrinsic must be available in log metadata."
    assert lidar_data.shape[1] == 4, (
        f"Expected Lidar data to have 4 columns (x, y, z, intensity), but got {lidar_data.shape[1]} columns."
    )
    point_cloud_3d = reframe_points_3d_array(
        from_origin=lidar_extrinsic,
        to_origin=PoseSE3.identity(),
        points_3d_array=lidar_data[..., :3],  # type: ignore
    )
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): (lidar_data[:, 3] * 255.0).astype(np.uint8),
        LidarFeature.IDS.serialize(): lidar_ids,
    }

    return point_cloud_3d.astype(np.float32), point_cloud_features
