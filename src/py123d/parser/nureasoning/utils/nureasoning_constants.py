from typing import Dict, Final, List, Optional, Set, Tuple

from py123d.datatypes.detections import TrafficLightStatus
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.map_objects.map_layer_types import IntersectionType, LaneType, RoadLineType
from py123d.datatypes.sensors import LidarID
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.parser.registry import NureasoningBoxDetectionLabel


def _make_parts_list(num_parts: int) -> List[str]:
    return [f"part_{i}" for i in range(1, num_parts + 1)]


# Hugging Face source for the downloader (see ``nureasoning_download.py``). The dataset is gated:
# access must be requested on the dataset page, and downloads need a token with that access.
# https://huggingface.co/datasets/nureasoning/nuReasoning
NUREASONING_REPO_ID: Final[str] = "nureasoning/nuReasoning"
NUREASONING_REPO_TYPE: Final[str] = "dataset"
# Top-level folder inside the repo holding ``<split>/[<part>/]<clip>.tar``. Stripped when extracting
# locally so the on-disk layout matches what the parser expects under ``nureasoning_data_root``.
NUREASONING_REPO_DATA_DIR: Final[str] = "data"
# Split folders used by the upstream repo. ``train`` and ``validation`` are organized in ``part_<k>``
# folders, ``test`` holds its clips directly.
NUREASONING_HF_SPLITS: Final[Tuple[str, ...]] = ("train", "validation", "test")


# nuReasoning is captured at 10Hz.
NUREASONING_DEFAULT_DT: Final[float] = 0.1

# TODO: Verify the true lidar sweep duration. We assume a 20Hz spin (50ms) for now.
NUREASONING_LIDAR_SWEEP_DURATION_US: Final[int] = 50_000


# The merged lidar point cloud encodes its source sensor per point in the ``lidar_info`` field.
NUREASONING_LIDAR_DICT: Final[Dict[int, LidarID]] = {
    0: LidarID.LIDAR_TOP,
    1: LidarID.LIDAR_FRONT,
    2: LidarID.LIDAR_SIDE_LEFT,
    3: LidarID.LIDAR_SIDE_RIGHT,
    4: LidarID.LIDAR_BACK,
}


# NOTE@DanielDauner: Estimate value based on the lidar ground plane. May not be entirely accurate.
NUREASONING_REAR_AXLE_HEIGHT: Final[float] = 0.350


# Maps each py123d split to its upstream split folder and the part folders it covers (None = all parts).
# NOTE: ``nureasoning-mini_train`` covers parts 1-3 of ``train``. They hold the clips of the former
# mini release (``qixuewei/nuReasoning_mini``, 2,590 clips) and about 400 more.
NUREASONING_SPLIT_SOURCES: Final[Dict[str, Tuple[str, Optional[List[str]]]]] = {
    "nureasoning_train": ("train", None),
    "nureasoning_val": ("validation", None),
    "nureasoning_test": ("test", None),
    "nureasoning-mini_train": ("train", _make_parts_list(3)),
}

NUREASONING_DATA_SPLITS: Set[str] = set(NUREASONING_SPLIT_SOURCES.keys())

NUREASONING_DEFAULT_SPLITS: Final[Tuple[str, ...]] = ("nureasoning_train", "nureasoning_val", "nureasoning_test")


# Ego dimensions [m] used when ``metadata.json`` has no ``ego_dimensions`` block (the test split). These
# are the devkit's constants (``nureasoning/planning/benchmark.py``) and match the released train clips.
NUREASONING_DEFAULT_EGO_DIMENSIONS: Final[Dict[str, float]] = {
    "length": 4.64,
    "width": 2.176,
    "height": 1.763,
    "vehicle_rear_length": 0.79,
}


# Maps each py123d CameraID to its physical camera name, which is also the key used in
# the ``camera_calibrations`` block of ``metadata.json``.
NUREASONING_CAMERA_ID_MAPPING: Dict[CameraID, str] = {
    CameraID.PCAM_F0: "CAM_M_F",
    CameraID.PCAM_B0: "CAM_M_B",
    CameraID.PCAM_L0: "CAM_M_L0",
    CameraID.PCAM_L1: "CAM_M_L1",
    CameraID.PCAM_L2: "CAM_M_L2",
    CameraID.PCAM_R0: "CAM_M_R0",
    CameraID.PCAM_R1: "CAM_M_R1",
    CameraID.PCAM_R2: "CAM_M_R2",
}


# Maps each py123d CameraID to the per-frame sensor key used in ``frames[i].sensors.cameras``.
# This is distinct from the physical name above: image paths are keyed by these semantic
# keys (front, front_left, ...), while calibrations are keyed by the physical CAM_M_* name.
NUREASONING_CAMERA_KEY_MAPPING: Dict[CameraID, str] = {
    CameraID.PCAM_F0: "front",
    CameraID.PCAM_B0: "back",
    CameraID.PCAM_L0: "front_left",
    CameraID.PCAM_L1: "left",
    CameraID.PCAM_L2: "back_left",
    CameraID.PCAM_R0: "front_right",
    CameraID.PCAM_R1: "right",
    CameraID.PCAM_R2: "back_right",
}


# Object category strings to labels. The first block is observed in the released annotation pickles;
# the second block is listed by the devkit (``view_reasoning.py``, ``vqa/generate.py``) and was not
# seen in the sampled clips. Call sites fall back to OTHER_OTHER for unseen categories.
NUREASONING_DETECTION_NAME_DICT: Dict[str, NureasoningBoxDetectionLabel] = {
    # Observed in the released data.
    "vehicle.car": NureasoningBoxDetectionLabel.VEHICLE_CAR,
    "vehicle.personal_mobility.bicycle": NureasoningBoxDetectionLabel.VEHICLE_PERSONAL_MOBILITY_BYCICLE,
    "human": NureasoningBoxDetectionLabel.HUMAN,
    "other.trafficcone": NureasoningBoxDetectionLabel.OTHER_TRAFFICCONE,
    "other.temporary_trafficsign": NureasoningBoxDetectionLabel.OTHER_TEMPORARY_TRAFFICSIGN,
    "other.other": NureasoningBoxDetectionLabel.OTHER_OTHER,
    "vehicle.door": NureasoningBoxDetectionLabel.VEHICLE_DOOR,
    "construction.traffic_cone": NureasoningBoxDetectionLabel.CONSTRUCTION_TRAFFIC_CONE,
    "construction_zone_area": NureasoningBoxDetectionLabel.CONSTRUCTION_ZONE_AREA,
    "animal": NureasoningBoxDetectionLabel.ANIMAL,
    # Listed by the devkit.
    "vehicle.truck": NureasoningBoxDetectionLabel.VEHICLE_TRUCK,
    "vehicle.bus": NureasoningBoxDetectionLabel.VEHICLE_BUS,
    "vehicle.motorcycle": NureasoningBoxDetectionLabel.VEHICLE_MOTORCYCLE,
    "vehicle.bicycle": NureasoningBoxDetectionLabel.VEHICLE_BICYCLE,
    "human.pedestrian": NureasoningBoxDetectionLabel.HUMAN_PEDESTRIAN,
    "vehicle.construction": NureasoningBoxDetectionLabel.VEHICLE_CONSTRUCTION,
    "vehicle.emergency": NureasoningBoxDetectionLabel.VEHICLE_EMERGENCY,
    "vehicle.trailer": NureasoningBoxDetectionLabel.VEHICLE_TRAILER,
}


NUREASONING_BOX_DETECTIONS_SE3_METADATA: Final[BoxDetectionsSE3Metadata] = BoxDetectionsSE3Metadata(
    box_detection_label_class=NureasoningBoxDetectionLabel
)


# Traffic light state strings (annotations.traffic_light_states[].state) to py123d status.
# NOTE: unlike nuPlan, nuReasoning includes an explicit "off" state.
NUREASONING_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatus]] = {
    "green": TrafficLightStatus.GREEN,
    "yellow": TrafficLightStatus.YELLOW,
    "red": TrafficLightStatus.RED,
    "off": TrafficLightStatus.OFF,
    "unknown": TrafficLightStatus.UNKNOWN,
}


# ------------------------------------------------------------------------------------------------------------------
# Map element type conversions
# ------------------------------------------------------------------------------------------------------------------
# NOTE: nuReasoning is nuPlan-derived, so we ASSUME its integer map type codes match nuPlan's
# (see nuplan_constants.py). This has not been verified against a dataset taxonomy — treat these as a
# best-effort mapping. All call sites use ``.get(code, <default>)`` so unmapped codes fall back safely.

# Lane type code -> LaneType. Observed lane_type codes in the demo data: {0, 1}.
NUREASONING_LANE_TYPE_CONVERSION: Final[Dict[int, LaneType]] = {
    0: LaneType.SURFACE_STREET,
    1: LaneType.BIKE_LANE,
}

# Boundary type code -> RoadLineType. Observed boundary type codes in the demo data: {0, 1, 2, 3}.
# (Code 1 is unmapped here and falls back to RoadLineType.UNKNOWN at the call site.)
NUREASONING_ROAD_LINE_CONVERSION: Final[Dict[int, RoadLineType]] = {
    0: RoadLineType.DASHED_WHITE,
    2: RoadLineType.SOLID_WHITE,
    3: RoadLineType.UNKNOWN,
}

# Intersection type code -> IntersectionType. Observed intersection_type codes in the demo data: {0, 1, 2, 3, 5}.
NUREASONING_INTERSECTION_TYPE_CONVERSION: Final[Dict[int, IntersectionType]] = {
    0: IntersectionType.DEFAULT,
    1: IntersectionType.TRAFFIC_LIGHT,
    2: IntersectionType.STOP_SIGN,
    3: IntersectionType.LANE_BRANCH,
    4: IntersectionType.LANE_MERGE,
    5: IntersectionType.PASS_THROUGH,
}

# Maximum length [m] of a single road-edge segment derived from drivable-area outlines.
NUREASONING_MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0

# Half-width [m] used to synthesize left/right boundaries for lane connectors, which ship only a centerline.
NUREASONING_LANE_CONNECTOR_HALF_WIDTH: Final[float] = 2.0
