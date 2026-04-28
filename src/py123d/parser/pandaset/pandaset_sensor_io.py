from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from py123d.datatypes import LidarFeature, LidarID
from py123d.geometry.transform import abs_to_rel_points_3d_array
from py123d.parser.pandaset.utils.pandaset_constants import PANDASET_CAMERA_EXTRINSICS
from py123d.parser.pandaset.utils.pandaset_utils import (
    compute_global_main_lidar_from_camera,
    global_main_lidar_to_global_imu,
    pandaset_pose_dict_to_pose_se3,
    read_json,
    read_pkl_gz,
)


def load_pandaset_point_cloud_data_from_path(
    pkl_gz_path: Union[Path, str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads PandaSet lidar point clouds from a gzip-pickle file and converts them to ego frame.

    The iteration index is derived from the filename (e.g. ``03.pkl.gz`` → iteration 3)
    and used to look up the ego pose from the sibling ``poses.json`` file.

    :param pkl_gz_path: Absolute path to the ``{iteration:02d}.pkl.gz`` lidar file.
    :return: Tuple of (point_cloud_3d [N, 3] float32 in ego frame, features dict).
    """
    pkl_gz_path = Path(pkl_gz_path)
    assert pkl_gz_path.exists(), f"Pandaset Lidar file not found: {pkl_gz_path}"

    # Derive iteration from filename: "03.pkl.gz" → 3
    iteration = int(pkl_gz_path.name.split(".")[0])

    # NOTE @DanielDauner: Pickled pandas DataFrame with columns:
    #  - PC: "x", "y", "z",
    #  - Features: "i" = Intensity [0,255], "t" = Time in absolute seconds, "d" = Lidar ID (0 for top, 1 for front)
    all_lidar_df = read_pkl_gz(pkl_gz_path)

    # Use float64 precision for global coordinates.
    point_cloud_3d_global_frame = all_lidar_df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    # Derive lidar-to-world from front camera pose + extrinsic (lidar poses.json is unreliable).
    log_path = pkl_gz_path.parent.parent
    front_camera_poses = read_json(log_path / "camera" / "front_camera" / "poses.json")
    global_lidar = compute_global_main_lidar_from_camera(
        camera_pose=pandaset_pose_dict_to_pose_se3(front_camera_poses[iteration]),
        camera_extrinsic=PANDASET_CAMERA_EXTRINSICS["front_camera"],
    )
    ego_pose = global_main_lidar_to_global_imu(global_lidar)
    point_cloud_3d = abs_to_rel_points_3d_array(ego_pose, point_cloud_3d_global_frame)

    # Convert lidar ids of PandaSet to 123D LidarIDs.
    lidar_id = np.zeros(len(point_cloud_3d), dtype=np.uint8)
    lidar_id[all_lidar_df["d"] == 0] = int(LidarID.LIDAR_TOP)
    lidar_id[all_lidar_df["d"] == 1] = int(LidarID.LIDAR_FRONT)

    # Load lidar features.
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): all_lidar_df["i"].to_numpy(dtype=np.uint8),
        LidarFeature.TIMESTAMPS.serialize(): (all_lidar_df["t"].to_numpy(dtype=np.float64) * 1e6).astype(np.int64),
        LidarFeature.IDS.serialize(): lidar_id,
    }

    return point_cloud_3d.astype(np.float32), point_cloud_features
