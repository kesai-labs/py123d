from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from py123d.datatypes.sensors.lidar import LidarFeature, LidarID
from py123d.datatypes.time.timestamp import Timestamp


def load_av2_sensor_point_cloud_data_from_path(
    feather_path: Union[Path, str],
) -> Tuple[npt.NDArray, Dict[str, npt.NDArray]]:
    """Loads AV2 sensor Lidar point clouds from a feather file."""

    # NOTE: The AV2 dataset stores both top and down Lidar data in the same feather file.
    # All coordinates are in ego frame.

    # Columns: "x", "y", "z", "intensity", "laser_number", "offset_ns"
    all_lidar_df = pd.read_feather(feather_path)
    lidar_xyz = all_lidar_df[["x", "y", "z"]].to_numpy()

    # We need to separate them based on the laser_number field.
    # See here: https://github.com/argoverse/av2-api/issues/77#issuecomment-1178040867
    lidar_id = np.zeros(len(lidar_xyz), dtype=np.uint8)
    lidar_id[all_lidar_df["laser_number"] < 32] = int(LidarID.LIDAR_TOP)
    lidar_id[all_lidar_df["laser_number"] >= 32] = int(LidarID.LIDAR_DOWN)

    # Separate channel ids of the two 32 channel lidars
    lidar_channel = all_lidar_df["laser_number"].to_numpy(dtype=np.uint8) % 32

    # We can get the timestamps in ns from the file name.
    timestamp_initial = Timestamp.from_ns(int(Path(feather_path).stem))

    lidar_timestamps = all_lidar_df["offset_ns"].to_numpy(dtype=np.int64) // 1000
    lidar_timestamps += timestamp_initial.time_us

    # Store features
    lidar_features = {
        LidarFeature.IDS.serialize(): lidar_id.astype(np.uint8),
        LidarFeature.INTENSITY.serialize(): all_lidar_df["intensity"].to_numpy(dtype=np.uint8),
        LidarFeature.CHANNEL.serialize(): lidar_channel.astype(np.uint8),
        LidarFeature.TIMESTAMPS.serialize(): lidar_timestamps.astype(np.int64),
    }
    return lidar_xyz, lidar_features
