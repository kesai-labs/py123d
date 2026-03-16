from typing import Dict, Final, List, Set

from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.map_objects.map_layer_types import LaneType
from py123d.datatypes.sensors.pinhole_camera import CameraID
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import PoseSE3
from py123d.parser.registry import NuScenesBoxDetectionLabel

NUSCENES_MAP_LOCATIONS: Set[str] = {
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
}

NUSCENES_DATA_SPLITS: Final[List[str]] = [
    "nuscenes_train",
    "nuscenes_val",
    "nuscenes_test",
    "nuscenes-mini_train",
    "nuscenes-mini_val",
]

NUSCENES_INTERPOLATED_DATA_SPLITS: Final[List[str]] = [
    "nuscenes-interpolated_train",
    "nuscenes-interpolated_val",
    "nuscenes-interpolated_test",
    "nuscenes-interpolated-mini_train",
    "nuscenes-interpolated-mini_val",
]

TARGET_DT: Final[float] = 0.1
NUSCENES_DT: Final[float] = 0.5
NUSCENES_LIDAR_SWEEP_DURATION_US: Final[int] = 50_000  # 1/20s = 50ms, one full lidar rotation
NUSCENES_DETECTION_NAME_DICT = {
    # Vehicles (4+ wheels)
    "vehicle.car": NuScenesBoxDetectionLabel.VEHICLE_CAR,
    "vehicle.truck": NuScenesBoxDetectionLabel.VEHICLE_TRUCK,
    "vehicle.bus.bendy": NuScenesBoxDetectionLabel.VEHICLE_BUS_BENDY,
    "vehicle.bus.rigid": NuScenesBoxDetectionLabel.VEHICLE_BUS_RIGID,
    "vehicle.construction": NuScenesBoxDetectionLabel.VEHICLE_CONSTRUCTION,
    "vehicle.emergency.ambulance": NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_AMBULANCE,
    "vehicle.emergency.police": NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_POLICE,
    "vehicle.trailer": NuScenesBoxDetectionLabel.VEHICLE_TRAILER,
    # Bicycles / Motorcycles
    "vehicle.bicycle": NuScenesBoxDetectionLabel.VEHICLE_BICYCLE,
    "vehicle.motorcycle": NuScenesBoxDetectionLabel.VEHICLE_MOTORCYCLE,
    # Pedestrians (all subtypes)
    "human.pedestrian.adult": NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_ADULT,
    "human.pedestrian.child": NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_CHILD,
    "human.pedestrian.construction_worker": NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_CONSTRUCTION_WORKER,
    "human.pedestrian.personal_mobility": NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_PERSONAL_MOBILITY,
    "human.pedestrian.police_officer": NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_POLICE_OFFICER,
    "human.pedestrian.stroller": NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_STROLLER,
    "human.pedestrian.wheelchair": NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_WHEELCHAIR,
    # Traffic cone / barrier
    "movable_object.trafficcone": NuScenesBoxDetectionLabel.MOVABLE_OBJECT_TRAFFICCONE,
    "movable_object.barrier": NuScenesBoxDetectionLabel.MOVABLE_OBJECT_BARRIER,
    # Generic objects
    "movable_object.pushable_pullable": NuScenesBoxDetectionLabel.MOVABLE_OBJECT_PUSHABLE_PULLABLE,
    "movable_object.debris": NuScenesBoxDetectionLabel.MOVABLE_OBJECT_DEBRIS,
    "static_object.bicycle_rack": NuScenesBoxDetectionLabel.STATIC_OBJECT_BICYCLE_RACK,
    "animal": NuScenesBoxDetectionLabel.ANIMAL,
}


NUSCENES_DATABASE_VERSION_MAPPING: Dict[str, str] = {
    "nuscenes_train": "v1.0-trainval",
    "nuscenes_val": "v1.0-trainval",
    "nuscenes_test": "v1.0-test",
    "nuscenes-mini_train": "v1.0-mini",
    "nuscenes-mini_val": "v1.0-mini",
    "nuscenes-interpolated_train": "v1.0-trainval",
    "nuscenes-interpolated_val": "v1.0-trainval",
    "nuscenes-interpolated_test": "v1.0-test",
    "nuscenes-interpolated-mini_train": "v1.0-mini",
    "nuscenes-interpolated-mini_val": "v1.0-mini",
}

NUSCENES_LANE_TYPE_MAPPING: Dict[str, LaneType] = {
    "CAR": LaneType.SURFACE_STREET,
}

# NOTE: The parameters in nuScenes are estimates, and partially taken from the Renault Zoe model [1].
# The nuScenes ego_pose reference frame is at the midpoint of the rear vehicle axle at ground level.
# In py123d, this frame is called "IMU" by convention (see EgoStateSE3.from_imu), so
# rear_axle_to_imu_se3 = identity (ego_pose IS the rear axle frame) and center_to_imu_se3
# gives the offset from rear axle to the vehicle's geometric center.
# [1] https://en.wikipedia.org/wiki/Renault_Zoe
NUSCENES_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="nuscenes_renault_zoe",
    width=1.730,
    length=4.084,
    height=1.562,
    wheel_base=2.588,
    center_to_imu_se3=PoseSE3(x=1.385, y=0.0, z=1.562 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)

NUSCENES_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(
    box_detection_label_class=NuScenesBoxDetectionLabel,
)

NUSCENES_CAMERA_IDS = {
    CameraID.PCAM_F0: "CAM_FRONT",
    CameraID.PCAM_B0: "CAM_BACK",
    CameraID.PCAM_L0: "CAM_FRONT_LEFT",
    CameraID.PCAM_L1: "CAM_BACK_LEFT",
    CameraID.PCAM_R0: "CAM_FRONT_RIGHT",
    CameraID.PCAM_R1: "CAM_BACK_RIGHT",
}
