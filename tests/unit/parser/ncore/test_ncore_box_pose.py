from types import SimpleNamespace

import numpy as np
import pytest

from py123d.geometry.pose import PoseSE3
from py123d.parser.ncore.ncore_parser import _build_box_detections_in_window
from py123d.parser.ncore.utils.ncore_constants import NCORE_BOX_DETECTIONS_SE3_METADATA


def _observation(timestamp_us: int, x_in_rig: float) -> SimpleNamespace:
    bbox3 = SimpleNamespace(centroid=(x_in_rig, 0.0, 0.0), dim=(4.5, 1.9, 1.6), rot=(0.0, 0.0, 0.0))
    return SimpleNamespace(
        timestamp_us=timestamp_us, reference_frame_id="rig", bbox3=bbox3, class_id="automobile", track_id="7"
    )


def test_a_box_is_placed_with_the_rig_pose_at_its_own_timestamp():
    """NCore stamps a cuboid with the time the LiDAR scanned the object and gives it in the rig frame of that time."""
    rig_poses_ts = np.array([0, 100_000, 200_000], dtype=np.int64)
    rig_poses_se3 = np.tile(np.eye(4), (3, 1, 1))
    rig_poses_se3[:, 0, 3] = [0.0, 1.0, 2.0]  # the ego drives along +x at 10 m/s
    cuboids = [_observation(150_000, x_in_rig=10.0)]  # scanned in the middle of the sweep, 10 m ahead

    detections = _build_box_detections_in_window(
        cuboids,
        np.array([150_000], dtype=np.int64),
        sweep_start_us=100_000,
        sweep_end_us=200_000,
        reference_to_rig={"rig": PoseSE3.identity()},
        rig_poses_se3=rig_poses_se3,
        rig_poses_ts=rig_poses_ts,
        metadata=NCORE_BOX_DETECTIONS_SE3_METADATA,
    )

    assert detections.box_detections[0].bounding_box_se3.center_se3.x == pytest.approx(11.5)
