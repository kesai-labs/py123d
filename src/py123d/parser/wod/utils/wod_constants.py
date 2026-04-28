from typing import Dict, List

from py123d.datatypes.detections import TrafficLightStatus
from py123d.datatypes.map_objects import LaneType, RoadEdgeType, RoadLineType
from py123d.datatypes.sensors import CameraID, LidarID

# Map features:
# ----------------------------------------------------------------------------------------------------------------------

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L206
WAYMO_ROAD_LINE_TYPE_CONVERSION: Dict[int, RoadLineType] = {
    0: RoadLineType.UNKNOWN,  # UNKNOWN
    1: RoadLineType.DASHED_WHITE,  # BROKEN_SINGLE_WHITE
    2: RoadLineType.SOLID_WHITE,  # SOLID_SINGLE_WHITE
    3: RoadLineType.DOUBLE_SOLID_WHITE,  # SOLID_DOUBLE_WHITE
    4: RoadLineType.DASHED_YELLOW,  # BROKEN_SINGLE_YELLOW
    5: RoadLineType.DOUBLE_DASH_YELLOW,  # BROKEN_DOUBLE_YELLOW
    6: RoadLineType.SOLID_YELLOW,  # SOLID_SINGLE_YELLOW
    7: RoadLineType.DOUBLE_SOLID_YELLOW,  # SOLID_DOUBLE_YELLOW
    8: RoadLineType.DOUBLE_DASH_YELLOW,  # PASSING_DOUBLE_YELLOW
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L186
WAYMO_ROAD_EDGE_TYPE_CONVERSION: Dict[int, RoadEdgeType] = {
    0: RoadEdgeType.UNKNOWN,
    1: RoadEdgeType.ROAD_EDGE_BOUNDARY,
    2: RoadEdgeType.ROAD_EDGE_MEDIAN,
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L147
WAYMO_LANE_TYPE_CONVERSION: Dict[int, LaneType] = {
    0: LaneType.UNDEFINED,
    1: LaneType.FREEWAY,
    2: LaneType.SURFACE_STREET,
    3: LaneType.BIKE_LANE,
}


# Perception:
# ----------------------------------------------------------------------------------------------------------------------
# Single source of truth for WOD-Perception splits. Mirrors the motion pattern and is used by
# both the parser (local file discovery) and the downloader (GCS prefix construction).
WOD_PERCEPTION_SPLIT_TO_GCS_FOLDER: Dict[str, str] = {
    "wod-perception_train": "training",
    "wod-perception_val": "validation",
    "wod-perception_test": "testing",
}
WOD_PERCEPTION_AVAILABLE_SPLITS: List[str] = list(WOD_PERCEPTION_SPLIT_TO_GCS_FOLDER)

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L50
WOD_PERCEPTION_CAMERA_IDS: Dict[int, CameraID] = {
    1: CameraID.PCAM_F0,  # front_camera
    2: CameraID.PCAM_L0,  # front_left_camera
    3: CameraID.PCAM_R0,  # front_right_camera
    4: CameraID.PCAM_L1,  # left_camera
    5: CameraID.PCAM_R1,  # right_camera
}

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L66
WOD_PERCEPTION_LIDAR_IDS: Dict[int, LidarID] = {
    0: LidarID.LIDAR_UNKNOWN,  # UNKNOWN
    1: LidarID.LIDAR_TOP,  # TOP
    2: LidarID.LIDAR_FRONT,  # FRONT
    3: LidarID.LIDAR_SIDE_LEFT,  # SIDE_LEFT
    4: LidarID.LIDAR_SIDE_RIGHT,  # SIDE_RIGHT
    5: LidarID.LIDAR_BACK,  # REAR
}


# Motion:
# ----------------------------------------------------------------------------------------------------------------------

# Single source of truth for WOD-Motion splits. Only splits that actually exist on GCS are listed —
# training_20s has no val/test counterpart, and *_interactive has no train counterpart.
#
# On-disk layout of converted logs: ``logs_root/<split>/<scenario_id>/``, so each split here maps
# to its own folder (e.g. ``logs_root/wod-motion-interactive_val/...``).
WOD_MOTION_SPLIT_TO_GCS_FOLDER: Dict[str, str] = {
    "wod-motion_train": "training",
    "wod-motion_val": "validation",
    "wod-motion_test": "testing",
    "wod-motion-20s_train": "training_20s",
    "wod-motion-interactive_val": "validation_interactive",
    "wod-motion-interactive_test": "testing_interactive",
}
# Scenario flavor per split — surfaced via the aux custom modality so downstream code can branch
# without string-parsing the split name. Keys must match WOD_MOTION_SPLIT_TO_GCS_FOLDER.
WOD_MOTION_SPLIT_TO_VARIANT: Dict[str, str] = {
    "wod-motion_train": "default",
    "wod-motion_val": "default",
    "wod-motion_test": "default",
    "wod-motion-20s_train": "20s",
    "wod-motion-interactive_val": "interactive",
    "wod-motion-interactive_test": "interactive",
}
WOD_MOTION_AVAILABLE_SPLITS: List[str] = list(WOD_MOTION_SPLIT_TO_GCS_FOLDER)

# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L39
WOD_MOTION_TRAFFIC_LIGHT_MAPPING: Dict[int, TrafficLightStatus] = {
    0: TrafficLightStatus.UNKNOWN,  # LANE_STATE_UNKNOWN
    1: TrafficLightStatus.RED,  # LANE_STATE_ARROW_STOP
    2: TrafficLightStatus.YELLOW,  # LANE_STATE_ARROW_CAUTION
    3: TrafficLightStatus.GREEN,  # LANE_STATE_ARROW_GO
    4: TrafficLightStatus.RED,  # LANE_STATE_STOP
    5: TrafficLightStatus.YELLOW,  # LANE_STATE_CAUTION
    6: TrafficLightStatus.GREEN,  # LANE_STATE_GO
    7: TrafficLightStatus.RED,  # LANE_STATE_FLASHING_STOP
    8: TrafficLightStatus.YELLOW,  # LANE_STATE_FLASHING_CAUTION
}
