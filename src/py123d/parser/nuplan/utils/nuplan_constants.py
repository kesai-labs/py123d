from typing import Dict, Final, List, Set

from py123d.datatypes.detections import TrafficLightStatus
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.map_objects import RoadLineType
from py123d.datatypes.map_objects.map_layer_types import IntersectionType, LaneType, StopZoneType
from py123d.datatypes.sensors import LidarID
from py123d.datatypes.time import Timestamp
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry.pose import PoseSE3
from py123d.parser.registry import NuPlanBoxDetectionLabel

NUPLAN_DEFAULT_DT: Final[float] = 0.05

NUPLAN_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatus]] = {
    "green": TrafficLightStatus.GREEN,
    "yellow": TrafficLightStatus.YELLOW,
    "red": TrafficLightStatus.RED,
    "unknown": TrafficLightStatus.UNKNOWN,
}


NUPLAN_DETECTION_NAME_DICT = {
    "vehicle": NuPlanBoxDetectionLabel.VEHICLE,
    "bicycle": NuPlanBoxDetectionLabel.BICYCLE,
    "pedestrian": NuPlanBoxDetectionLabel.PEDESTRIAN,
    "traffic_cone": NuPlanBoxDetectionLabel.TRAFFIC_CONE,
    "barrier": NuPlanBoxDetectionLabel.BARRIER,
    "czone_sign": NuPlanBoxDetectionLabel.CZONE_SIGN,
    "generic_object": NuPlanBoxDetectionLabel.GENERIC_OBJECT,
}

# https://github.com/motional/nuplan-devkit/blob/e9241677997dd86bfc0bcd44817ab04fe631405b/nuplan/database/nuplan_db_orm/utils.py#L1129-L1135
# NOTE: The above description is not matching the actually loaded points clouds.
# Correct is the mapping; 1: LIDAR_SIDE_LEFT, 2: LIDAR_SIDE_RIGHT.
NUPLAN_LIDAR_DICT = {
    0: LidarID.LIDAR_TOP,
    1: LidarID.LIDAR_SIDE_LEFT,
    2: LidarID.LIDAR_SIDE_RIGHT,
    3: LidarID.LIDAR_BACK,
    4: LidarID.LIDAR_FRONT,
}

NUPLAN_DATA_SPLITS: Set[str] = {
    "nuplan_train",
    "nuplan_val",
    "nuplan_test",
    "nuplan-mini_train",
    "nuplan-mini_val",
    "nuplan-mini_test",
}

NUPLAN_MAP_LOCATIONS: List[str] = [
    "sg-one-north",
    "us-ma-boston",
    "us-nv-las-vegas-strip",
    "us-pa-pittsburgh-hazelwood",
]

NUPLAN_MAP_LOCATION_FILES: Dict[str, str] = {
    "sg-one-north": "sg-one-north/9.17.1964/map.gpkg",
    "us-ma-boston": "us-ma-boston/9.12.1817/map.gpkg",
    "us-nv-las-vegas-strip": "us-nv-las-vegas-strip/9.15.1915/map.gpkg",
    "us-pa-pittsburgh-hazelwood": "us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg",
}


NUPLAN_MAP_GPKG_LAYERS: Set[str] = {
    "baseline_paths",
    "carpark_areas",
    "generic_drivable_areas",
    "dubins_nodes",
    "lane_connectors",
    "intersections",
    "boundaries",
    "crosswalks",
    "lanes_polygons",
    "lane_group_connectors",
    "lane_groups_polygons",
    "road_segments",
    "stop_polygons",
    "traffic_lights",
    "walkways",
    "gen_lane_connectors_scaled_width_polygons",
}

NUPLAN_ROAD_LINE_CONVERSION = {
    0: RoadLineType.DASHED_WHITE,
    2: RoadLineType.SOLID_WHITE,
    3: RoadLineType.UNKNOWN,
}

# Manually checked the four nuPlan maps. Unique ids are 0 and 1, for vehicle and bike lanes, respectively.
NUPLAN_LANE_TYPE_CONVERSION = {
    0: LaneType.SURFACE_STREET,
    1: LaneType.BIKE_LANE,
}

# https://github.com/motional/nuplan-devkit/blob/master/nuplan/common/maps/maps_datatypes.py#L85
NUPLAN_INTERSECTION_TYPE_CONVERSION: Final[Dict[int, IntersectionType]] = {
    0: IntersectionType.DEFAULT,
    1: IntersectionType.TRAFFIC_LIGHT,
    2: IntersectionType.STOP_SIGN,
    3: IntersectionType.LANE_BRANCH,
    4: IntersectionType.LANE_MERGE,
    5: IntersectionType.PASS_THROUGH,
}

# https://github.com/motional/nuplan-devkit/blob/master/nuplan/common/maps/maps_datatypes.py#L61
NUPLAN_STOP_ZONE_TYPE_CONVERSION: Final[Dict[int, StopZoneType]] = {
    0: StopZoneType.PEDESTRIAN_CROSSING,
    1: StopZoneType.STOP_SIGN,
    2: StopZoneType.TRAFFIC_LIGHT,
    3: StopZoneType.TURN_STOP,
    4: StopZoneType.YIELD_SIGN,
    5: StopZoneType.UNKNOWN,
}

NUPLAN_ROLLING_SHUTTER_S: Final[Timestamp] = Timestamp.from_s(1 / 60)
NUPLAN_LIDAR_SWEEP_DURATION_US: Final[int] = 50_000  # 50ms at 20Hz

# NOTE: These parameters are mostly available in nuPlan, except for the rear_axle_to_center_vertical.
# The value is estimated based the Lidar point cloud.
# [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
NUPLAN_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="nuplan_chrysler_pacifica",
    width=2.297,
    length=5.176,
    height=1.777,
    wheel_base=3.089,
    center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=0.45, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)

NUPLAN_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=NuPlanBoxDetectionLabel)
