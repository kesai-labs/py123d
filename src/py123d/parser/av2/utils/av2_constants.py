from typing import Dict, Final, Set

from py123d.datatypes import LaneType, PinholeCameraID, RoadLineType

AV2_SENSOR_SPLITS: Set[str] = {"av2-sensor_train", "av2-sensor_val", "av2-sensor_test"}

# Mapping from AV2 camera names to PinholeCameraID enums.
AV2_CAMERA_ID_MAPPING: Dict[str, PinholeCameraID] = {
    "ring_front_center": PinholeCameraID.PCAM_F0,
    "ring_front_left": PinholeCameraID.PCAM_L0,
    "ring_front_right": PinholeCameraID.PCAM_R0,
    "ring_side_left": PinholeCameraID.PCAM_L1,
    "ring_side_right": PinholeCameraID.PCAM_R1,
    "ring_rear_left": PinholeCameraID.PCAM_L2,
    "ring_rear_right": PinholeCameraID.PCAM_R2,
    "stereo_front_left": PinholeCameraID.PCAM_STEREO_L,
    "stereo_front_right": PinholeCameraID.PCAM_STEREO_R,
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
