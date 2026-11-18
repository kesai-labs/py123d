from typing import Dict, List

import numpy as np

from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.sensors.lidar import LidarID, LidarMergedMetadata, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import CameraID
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import PoseSE3
from py123d.parser.registry import OnceBoxDetectionLabel

ONCE_SPLITS: List[str] = ["once_train", "once_val", "once_test"]

# Sequence ids per split, from ``ImageSets/{train,val,test}.txt`` of the official devkit:
# https://github.com/once-for-auto-driving/once_devkit/tree/master/ImageSets
ONCE_SEQUENCES_BY_SPLIT: Dict[str, List[str]] = {
    "once_train": ["000076", "000080", "000092", "000104", "000113", "000121"],
    "once_val": ["000027", "000028", "000112", "000201"],
    "once_test": [
        "000034",
        "000077",
        "000168",
        "000200",
        "000273",
        "000275",
        "000303",
        "000318",
        "000322",
        "000334",
    ],
}

# Camera folder name -> 123D camera id. The devkit tags are cam01 "top" (wide, ~120° HFOV), cam03 "top2"
# (narrow, ~60° HFOV), cam05 "left_back", cam06 "left_front", cam07 "right_front", cam08 "right_back", cam09 "back".
ONCE_CAMERA_MAPPING: Dict[str, CameraID] = {
    "cam01": CameraID.PCAM_F0,
    "cam03": CameraID.PCAM_F1,
    "cam06": CameraID.PCAM_L0,
    "cam05": CameraID.PCAM_L1,
    "cam07": CameraID.PCAM_R0,
    "cam08": CameraID.PCAM_R1,
    "cam09": CameraID.PCAM_B0,
}
ONCE_CAMERA_NAMES: List[str] = list(ONCE_CAMERA_MAPPING.keys())

ONCE_LIDAR_NAME = "lidar_roof"

# ONCE releases every fifth sweep of the 10 Hz lidar (2 Hz) and annotates every second released frame (1 Hz).
ONCE_FRAME_PERIOD_US = 500_000
ONCE_LIDAR_SWEEP_DURATION_US = 100_000
ONCE_MAX_INTERPOLATION_GAP_US = 1_500_000

ONCE_BOX_DETECTION_FROM_STR: Dict[str, OnceBoxDetectionLabel] = {
    "Car": OnceBoxDetectionLabel.CAR,
    "Truck": OnceBoxDetectionLabel.TRUCK,
    "Bus": OnceBoxDetectionLabel.BUS,
    "Pedestrian": OnceBoxDetectionLabel.PEDESTRIAN,
    "Cyclist": OnceBoxDetectionLabel.CYCLIST,
}

# Gates for associating boxes of consecutive annotated frames (1 s apart) into tracks, in meters per second.
ONCE_TRACK_MAX_SPEED_MPS: Dict[OnceBoxDetectionLabel, float] = {
    OnceBoxDetectionLabel.CAR: 25.0,
    OnceBoxDetectionLabel.TRUCK: 25.0,
    OnceBoxDetectionLabel.BUS: 25.0,
    OnceBoxDetectionLabel.CYCLIST: 12.0,
    OnceBoxDetectionLabel.PEDESTRIAN: 4.0,
}

# ONCE lidar frame: x left, y backward, z up. ISO 8855: x forward, y left, z up.
# ``p_iso = ONCE_TO_ISO_ROTATION @ p_once``
ONCE_TO_ISO_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],  # ISO x (forward) = -ONCE y
        [1.0, 0.0, 0.0],  # ISO y (left) = ONCE x
        [0.0, 0.0, 1.0],  # ISO z (up) = ONCE z
    ],
    dtype=np.float64,
)
ONCE_TO_ISO_TRANSFORM = np.eye(4, dtype=np.float64)
ONCE_TO_ISO_TRANSFORM[:3, :3] = ONCE_TO_ISO_ROTATION

# NOTE: ONCE does not publish the vehicle model or its dimensions. The values are estimates for a mid-size car with
# the roof lidar as IMU origin, about 1.8 m above the ground and roughly above the vehicle center.
ONCE_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="once_vehicle",
    width=1.9,
    length=4.9,
    height=1.6,
    wheel_base=2.9,
    center_to_imu_se3=PoseSE3(x=0.0, y=0.0, z=-1.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
    rear_axle_to_imu_se3=PoseSE3(x=-1.45, y=0.0, z=-1.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
)

ONCE_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=OnceBoxDetectionLabel)

ONCE_LIDAR_MERGED_METADATA = LidarMergedMetadata(
    {
        LidarID.LIDAR_TOP: LidarMetadata(
            lidar_name=ONCE_LIDAR_NAME,
            lidar_id=LidarID.LIDAR_TOP,
            lidar_to_imu_se3=PoseSE3.identity(),
        )
    }
)
