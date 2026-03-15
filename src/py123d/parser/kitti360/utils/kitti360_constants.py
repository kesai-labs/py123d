from typing import Dict, Final, List, Set

from py123d.datatypes import CameraID
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry.pose import PoseSE3
from py123d.parser.registry import KITTI360BoxDetectionLabel

KITTI360_SPLITS: Set[str] = {"kitti360_train", "kitti360_val", "kitti360_test"}

KITTI360_DT: Final[float] = 0.1
KITTI360_LIDAR_NAME: Final[str] = "velodyne_points"
KITTI360_LIDAR_SWEEP_DURATION_US: Final[int] = 100_000  # 1/10s = 100ms, one full Velodyne HDL-64E rotation

KITTI360_PINHOLE_CAMERA_IDS: Dict[CameraID, str] = {
    CameraID.PCAM_STEREO_L: "image_00",
    CameraID.PCAM_STEREO_R: "image_01",
}

KITTI360_FISHEYE_MEI_CAMERA_IDS: Dict[CameraID, str] = {
    CameraID.FMCAM_L: "image_02",
    CameraID.FMCAM_R: "image_03",
}

KITTI360_ALL_SEQUENCES: Final[List[str]] = [
    "2013_05_28_drive_0000_sync",
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    "2013_05_28_drive_0008_sync",
    "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
    "2013_05_28_drive_0018_sync",
]

# NOTE: The parameters in KITTI-360 are estimates based on the vehicle model used in the dataset.
# Uses a 2006 VW Passat Variant B6 [1]. Vertical distance is estimated based on the Lidar.
# KITTI-360 is currently the only dataset where the IMU has a lateral offset to the rear axle [2].
# The rear axle is at (0.05, -0.32, 0.0) from the IMU in the body frame.
# [1] https://en.wikipedia.org/wiki/Volkswagen_Passat_(B6)
# [2] https://www.cvlibs.net/datasets/kitti-360/documentation.php
_REAR_AXLE_TO_CENTER_LONGITUDINAL = 1.3369
_REAR_AXLE_TO_CENTER_VERTICAL = 1.516 / 2 - 0.9
_REAR_AXLE_IN_IMU_X = 0.05
_REAR_AXLE_IN_IMU_Y = -0.32

KITTI360_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="kitti360_vw_passat",
    width=1.820,
    length=4.775,
    height=1.516,
    wheel_base=2.709,
    center_to_imu_se3=PoseSE3(
        x=_REAR_AXLE_TO_CENTER_LONGITUDINAL + _REAR_AXLE_IN_IMU_X,
        y=_REAR_AXLE_IN_IMU_Y,
        z=_REAR_AXLE_TO_CENTER_VERTICAL,
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
    ),
    rear_axle_to_imu_se3=PoseSE3(x=_REAR_AXLE_IN_IMU_X, y=_REAR_AXLE_IN_IMU_Y, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
)

KITTI360_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=KITTI360BoxDetectionLabel)

# KITTI-360 directory names for dataset structure
DIR_ROOT = "root"
DIR_2D_RAW = "data_2d_raw"
DIR_2D_SMT = "data_2d_semantics"
DIR_3D_RAW = "data_3d_raw"
DIR_3D_SMT = "data_3d_semantics"
DIR_3D_BBOX = "data_3d_bboxes"
DIR_POSES = "data_poses"
DIR_CALIB = "calibration"
