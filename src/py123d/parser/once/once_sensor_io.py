from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from py123d.datatypes import LidarFeature, LidarID
from py123d.parser.once.utils.once_constants import ONCE_TO_ISO_ROTATION


def load_once_point_cloud_data_from_path(bin_path: Union[Path, str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads an ONCE lidar sweep and rotates it from the ONCE lidar frame into the ISO 8855 ego frame.

    ONCE stores each sweep as a float32 ``(N, 4)`` binary of ``x, y, z, intensity`` in the lidar frame
    (x left, y backward, z up). The roof lidar is the ego/IMU origin, so only the axis change is applied.

    :param bin_path: Absolute path to the ``lidar_roof/<frame_id>.bin`` file.
    :return: Tuple of (point_cloud_3d [N, 3] float32 in ego frame, features dict).
    """
    bin_path = Path(bin_path)
    assert bin_path.exists(), f"ONCE lidar file not found: {bin_path}"

    lidar_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    point_cloud_3d = lidar_data[:, :3].astype(np.float64) @ ONCE_TO_ISO_ROTATION.T

    # Intensity is stored as a raw float reflectance, mostly in [0, 255].
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): np.clip(lidar_data[:, 3], 0.0, 255.0).astype(np.uint8),
        LidarFeature.IDS.serialize(): np.full(len(lidar_data), int(LidarID.LIDAR_TOP), dtype=np.uint8),
    }
    return point_cloud_3d.astype(np.float32), point_cloud_features
