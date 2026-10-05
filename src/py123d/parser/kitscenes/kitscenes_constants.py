"""Constants for the KITScenes Multimodal dataset integration."""

from __future__ import annotations

from typing import Dict, Final, Tuple

from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import PoseSE3

DATASET_NAME: Final[str] = "kitscenes"

# Scenes live in ``<data_root>/data/<split>/<scene_uuid>/``, mirroring the HuggingFace repository layout.
DATA_SUBDIR: Final[str] = "data"
KITSCENES_SPLITS: Final[Tuple[str, ...]] = ("train", "val", "test", "test_e2e", "overlap_train_val")

# Per-scene file layout
CALIBRATION_FILE: Final[str] = "calibration/calib.json"
POSES_FILE: Final[str] = "poses.txt"
REFERENCE_TIMESTAMPS_FILE: Final[str] = "timestamp.reference.txt"
MAP_ORIGIN_FILE: Final[str] = "maps/origin.json"
FRAME_INDEX_WIDTH: Final[int] = 10  # sensor files are named ``{frame_index:010d}.{ext}``

# All camera images are rectified pinhole JPEGs, synchronized to the reference timeline.
CAMERA_ID_MAPPING: Final[Dict[str, CameraID]] = {
    "camera_ring_front": CameraID.PCAM_F0,
    "camera_ring_front_left": CameraID.PCAM_L0,
    "camera_ring_rear_left": CameraID.PCAM_L1,
    "camera_ring_front_right": CameraID.PCAM_R0,
    "camera_ring_rear_right": CameraID.PCAM_R1,
    "camera_ring_rear": CameraID.PCAM_B0,
    "camera_base_front_left_rect": CameraID.PCAM_STEREO_L,
    "camera_base_front_right_rect": CameraID.PCAM_STEREO_R,
    # High-resolution front camera (same field of view as the ring front camera, ~1.65x the angular resolution).
    "camera_base_front_center": CameraID.PCAM_F1,
}

# Approximate city centers (lat, lon), used to name a log's location from its map origin.
CITY_CENTERS: Final[Dict[str, Tuple[float, float]]] = {
    "karlsruhe": (49.0069, 8.4037),
    "frankfurt": (50.1109, 8.6821),
    "sindelfingen": (48.7133, 9.0028),
}

# NOTE: The KITScenes paper and devkit do not publish the vehicle model or its dimensions. The values below are
# estimates for a mid-size car and should be replaced once known. The ego reference frame ("base_frame" in
# ``calib.json``) coincides with the roof lidar ``lidar_top``, which sits ~2.0 m above the ground (measured from the
# point cloud). We assume the lidar is mounted above the vehicle center.
_LIDAR_TOP_HEIGHT_ABOVE_GROUND = 2.0
_VEHICLE_LENGTH = 5.0
_VEHICLE_WIDTH = 2.0
_VEHICLE_HEIGHT = 1.8
_WHEEL_BASE = 3.0
_WHEEL_RADIUS = 0.35

KITSCENES_EGO_STATE_SE3_METADATA: Final[EgoStateSE3Metadata] = EgoStateSE3Metadata(
    vehicle_name="kitscenes_unknown_vehicle",
    width=_VEHICLE_WIDTH,
    length=_VEHICLE_LENGTH,
    height=_VEHICLE_HEIGHT,
    wheel_base=_WHEEL_BASE,
    center_to_imu_se3=PoseSE3(
        x=0.0,
        y=0.0,
        z=_VEHICLE_HEIGHT / 2 - _LIDAR_TOP_HEIGHT_ABOVE_GROUND,
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
    ),
    rear_axle_to_imu_se3=PoseSE3(
        x=-_WHEEL_BASE / 2,
        y=0.0,
        z=_WHEEL_RADIUS - _LIDAR_TOP_HEIGHT_ABOVE_GROUND,
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
    ),
)
