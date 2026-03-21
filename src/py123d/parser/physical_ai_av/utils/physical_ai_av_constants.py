from typing import Dict, Set

from py123d.datatypes import CameraID
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry.pose import PoseSE3
from py123d.parser.registry import PhysicalAIAVBoxDetectionLabel

PHYSICAL_AI_AV_SPLITS: Set[str] = {"physical-ai-av_train", "physical-ai-av_val", "physical-ai-av_test"}

# PHYSICAL_AI_AV_CAMERAS: List[str] = [
#     "camera_front_wide_120fov",
#     "camera_front_tele_30fov",
#     "camera_cross_left_120fov",
#     "camera_cross_right_120fov",
#     "camera_rear_left_70fov",
#     "camera_rear_right_70fov",
#     "camera_rear_tele_30fov",
# ]

PHYSICAL_AI_AV_CAMERA_ID_MAPPING: Dict[str, CameraID] = {
    "camera_front_wide_120fov": CameraID.FTCAM_F0,
    "camera_front_tele_30fov": CameraID.FTCAM_TELE_F0,
    "camera_cross_left_120fov": CameraID.FTCAM_L0,
    "camera_cross_right_120fov": CameraID.FTCAM_R0,
    "camera_rear_left_70fov": CameraID.FTCAM_L1,
    "camera_rear_right_70fov": CameraID.FTCAM_R1,
    "camera_rear_tele_30fov": CameraID.FTCAM_TELE_B0,
}

# Vehicle dimensions from calibration/vehicle_dimensions (Hyperion 8 platform).
# center_to_imu_se3: rear_axle_to_bbox_center = 1.327 m longitudinal offset.
# rear_axle_to_imu_se3: identity (egomotion anchor frame is the reference).
PHYSICAL_AI_AV_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="nvidia_hyperion_8",
    width=2.121,
    length=4.872,
    height=1.473,
    wheel_base=2.850,
    center_to_imu_se3=PoseSE3(x=1.327, y=0.0, z=1.473 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)

PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(
    box_detection_label_class=PhysicalAIAVBoxDetectionLabel,
)

# Mapping from dataset label_class strings to PhysicalAIAVBoxDetectionLabel enum values.
PHYSICAL_AI_AV_LABEL_CLASS_MAPPING: Dict[str, PhysicalAIAVBoxDetectionLabel] = {
    "automobile": PhysicalAIAVBoxDetectionLabel.AUTOMOBILE,
    "person": PhysicalAIAVBoxDetectionLabel.PERSON,
    "bus": PhysicalAIAVBoxDetectionLabel.BUS,
    "heavy_truck": PhysicalAIAVBoxDetectionLabel.HEAVY_TRUCK,
    "other_vehicle": PhysicalAIAVBoxDetectionLabel.OTHER_VEHICLE,
    "protruding_object": PhysicalAIAVBoxDetectionLabel.PROTRUDING_OBJECT,
    "rider": PhysicalAIAVBoxDetectionLabel.RIDER,
    "stroller": PhysicalAIAVBoxDetectionLabel.STROLLER,
    "trailer": PhysicalAIAVBoxDetectionLabel.TRAILER,
    "animal": PhysicalAIAVBoxDetectionLabel.ANIMAL,
}
