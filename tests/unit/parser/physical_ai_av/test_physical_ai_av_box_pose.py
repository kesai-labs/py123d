import numpy as np
import pandas as pd
import pytest

from py123d.parser.physical_ai_av.physical_ai_av_parser import _build_box_detections
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA


def test_a_box_is_placed_with_the_ego_pose_at_its_own_timestamp():
    """An obstacle row carries the time the LiDAR scanned the object and is given in the rig frame of that time."""
    timestamps = np.arange(0, 300_001, 10_000, dtype=np.int64)  # the ego drives along +x at 10 m/s, poses at 100 Hz
    ego_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "x": timestamps * 1e-5,
            "y": 0.0,
            "z": 0.0,
        }
    )
    scanned_at = 154_000  # in the middle of the sweep that starts at 100 ms, between two ego poses
    obstacles = pd.DataFrame(
        [{"timestamp_us": scanned_at, "track_id": "7", "center_x": 10.0, "center_y": 0.0, "center_z": 0.0, "size_x": 4.5,
          "size_y": 1.9, "size_z": 1.6, "orientation_x": 0.0, "orientation_y": 0.0, "orientation_z": 0.0,
          "orientation_w": 1.0, "label_class": "automobile"}]
    )  # fmt: skip

    detections = _build_box_detections(
        obstacles, 100_000, ego_df, timestamps, None, PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA
    )

    assert detections.box_detections[0].bounding_box_se3.center_se3.x == pytest.approx(11.54)
