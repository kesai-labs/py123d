from typing import Dict, Final, Set

from py123d.datatypes import CameraID, LaneType, RoadLineType
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry.pose import PoseSE3
from py123d.parser.registry import AV2SensorBoxDetectionLabel

AV2_SENSOR_SPLITS: Set[str] = {"av2-sensor_train", "av2-sensor_val", "av2-sensor_test"}

# Mapping from AV2 camera names to CameraID enums.
AV2_CAMERA_ID_MAPPING: Dict[str, CameraID] = {
    "ring_front_center": CameraID.PCAM_F0,
    "ring_front_left": CameraID.PCAM_L0,
    "ring_front_right": CameraID.PCAM_R0,
    "ring_side_left": CameraID.PCAM_L1,
    "ring_side_right": CameraID.PCAM_R1,
    "ring_rear_left": CameraID.PCAM_L2,
    "ring_rear_right": CameraID.PCAM_R2,
    "stereo_front_left": CameraID.PCAM_STEREO_L,
    "stereo_front_right": CameraID.PCAM_STEREO_R,
}

# Mapping from AV2 road line types to RoadLineType enums.
AV2_ROAD_LINE_TYPE_MAPPING: Dict[str, RoadLineType] = {
    "NONE": RoadLineType.NONE,
    "UNKNOWN": RoadLineType.UNKNOWN,
    "DASH_SOLID_YELLOW": RoadLineType.DASH_SOLID_YELLOW,
    "DASH_SOLID_WHITE": RoadLineType.DASH_SOLID_WHITE,
    "DASHED_WHITE": RoadLineType.DASHED_WHITE,
    "DASHED_YELLOW": RoadLineType.DASHED_YELLOW,
    "DOUBLE_SOLID_YELLOW": RoadLineType.DOUBLE_SOLID_YELLOW,
    "DOUBLE_SOLID_WHITE": RoadLineType.DOUBLE_SOLID_WHITE,
    "DOUBLE_DASH_YELLOW": RoadLineType.DOUBLE_DASH_YELLOW,
    "DOUBLE_DASH_WHITE": RoadLineType.DOUBLE_DASH_WHITE,
    "SOLID_YELLOW": RoadLineType.SOLID_YELLOW,
    "SOLID_WHITE": RoadLineType.SOLID_WHITE,
    "SOLID_DASH_WHITE": RoadLineType.SOLID_DASH_WHITE,
    "SOLID_DASH_YELLOW": RoadLineType.SOLID_DASH_YELLOW,
    "SOLID_BLUE": RoadLineType.SOLID_BLUE,
}


# Mapping from AV2 lane types to LaneType enums.
AV2_LANE_TYPE_MAPPING: Dict[str, LaneType] = {
    "VEHICLE": LaneType.SURFACE_STREET,
    "BIKE": LaneType.BIKE_LANE,
    "BUS": LaneType.BUS_LANE,
}

AV2_SENSOR_CAM_SHUTTER_INTERVAL_MS: Final[float] = 50.0
AV2_SENSOR_LIDAR_SWEEP_INTERVAL_W_BUFFER_NS: Final[float] = 102000000.0

# [1] https://en.wikipedia.org/wiki/Ford_Fusion_Hybrid#Second_generation
# https://github.com/argoverse/av2-api/blob/6b22766247eda941cb1953d6a58e8d5631c561da/tests/unit/map/test_map_api.py#L375
AV2_SENSOR_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="av2_ford_fusion_hybrid",
    width=1.852 + 0.275,  # 0.275 is the estimated width of the side mirrors
    length=4.869,
    height=1.476,
    wheel_base=2.850,
    center_to_imu_se3=PoseSE3(x=1.339, y=0.0, z=0.438, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)


AV2_SENSOR_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=AV2SensorBoxDetectionLabel)
