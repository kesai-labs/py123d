"""Loading a lidar spin of Physical AI AV from its parquet file."""

from pathlib import Path

import DracoPy
import numpy as np
import pandas as pd
import pytest

from py123d.datatypes.sensors.lidar import LidarFeature, LidarID, LidarMetadata
from py123d.geometry import PoseSE3
from py123d.parser.physical_ai_av.physical_ai_av_sensor_io import load_physical_ai_av_point_cloud_data_from_path

SPIN_START_US = 89_971
LIDAR_METADATAS = {
    LidarID.LIDAR_TOP: LidarMetadata(
        lidar_name="lidar_top_360fov", lidar_id=LidarID.LIDAR_TOP, lidar_to_imu_se3=PoseSE3.identity()
    )
}


@pytest.mark.parametrize("spin_start_column", ["reference_timestamp", "spin_start_timestamp"])
def test_point_timestamps_follow_the_spin_start(tmp_path: Path, spin_start_column: str):
    """The per-point times are offsets from the spin start, whichever name the snapshot gives that column."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    # Red is the intensity; green and blue are the low and high byte of the relative time.
    colors = np.array([[7, 0, 0], [9, 255, 255]], dtype=np.uint8)
    blob = DracoPy.encode(points, colors=colors, preserve_order=True)
    parquet_path = tmp_path / "clip.lidar_top_360fov.parquet"
    pd.DataFrame({spin_start_column: [SPIN_START_US], "draco_encoded_pointcloud": [blob]}).to_parquet(parquet_path)

    xyz, features = load_physical_ai_av_point_cloud_data_from_path(parquet_path, 0, LIDAR_METADATAS)

    np.testing.assert_allclose(xyz, points, atol=1e-2)
    np.testing.assert_array_equal(features[LidarFeature.INTENSITY.serialize()], [7, 9])
    np.testing.assert_array_equal(
        features[LidarFeature.TIMESTAMPS.serialize()], [SPIN_START_US + 105_000, SPIN_START_US - 105_000]
    )
