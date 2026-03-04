from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from py123d.datatypes import LidarFeature, LidarID, LidarMetadatas
from py123d.geometry import PoseSE3
from py123d.geometry.transform import reframe_points_3d_array


def load_nuscenes_point_cloud_data_from_path(
    pcd_path: Path, lidar_metadatas: LidarMetadatas
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads nuScenes Lidar point clouds from the original binary files."""

    lidar_data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)  # Indices: x, y, z, intensity, ring
    assert lidar_data.ndim == 2 and lidar_data.shape[1] == 5, (
        f"Expected Lidar data to have shape (N, 5) for nuScenes, but got shape {lidar_data.shape}."
    )
    lidar_extrinsic = lidar_metadatas[LidarID.LIDAR_TOP].extrinsic
    assert lidar_extrinsic is not None, "Lidar extrinsic must be available in log metadata for nuScenes Lidar loading."

    lidar_ids = np.zeros(lidar_data.shape[0], dtype=np.uint8)  # nuScenes only has a top lidar.
    lidar_ids[:] = int(LidarID.LIDAR_TOP)

    # convert lidar to ego frame
    point_cloud_3d = reframe_points_3d_array(
        from_origin=lidar_extrinsic,
        to_origin=PoseSE3.identity(),
        points_3d_array=lidar_data[..., :3],  # type: ignore
    )
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): lidar_data[..., 3].astype(np.uint8),
        LidarFeature.CHANNEL.serialize(): lidar_data[..., 4].astype(np.uint8),
        LidarFeature.IDS.serialize(): lidar_ids,
    }

    return point_cloud_3d.astype(np.float32), point_cloud_features
