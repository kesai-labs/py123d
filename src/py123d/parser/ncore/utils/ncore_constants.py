"""Constants for the NVIDIA NCore V4 dataset.

NCore (PhysicalAI-Autonomous-Vehicles-NCore) ships on the same Hyperion 8.1 sensor
platform and the same 10-class cuboid taxonomy as the PhysicalAI-AV dataset, so all
sensor IDs, label classes, vehicle metadata, and the registry enum are re-used from
``py123d.parser.physical_ai_av`` to avoid drift.
"""

from typing import Set

from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import (
    PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA as NCORE_BOX_DETECTIONS_SE3_METADATA,
)
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import (
    PHYSICAL_AI_AV_CAMERA_ID_MAPPING as NCORE_CAMERA_ID_MAPPING,
)
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import (
    PHYSICAL_AI_AV_EGO_STATE_SE3_METADATA as NCORE_EGO_STATE_SE3_METADATA,
)
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import (
    PHYSICAL_AI_AV_LABEL_CLASS_MAPPING as NCORE_LABEL_CLASS_MAPPING,
)
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import (
    resolve_physical_ai_av_label as resolve_ncore_label,
)

NCORE_SPLITS: Set[str] = {"ncore_train"}

NCORE_LIDAR_SENSOR_ID: str = "lidar_top_360fov"
NCORE_RIG_FRAME_ID: str = "rig"
NCORE_WORLD_FRAME_ID: str = "world"

__all__ = [
    "NCORE_SPLITS",
    "NCORE_LIDAR_SENSOR_ID",
    "NCORE_RIG_FRAME_ID",
    "NCORE_WORLD_FRAME_ID",
    "NCORE_CAMERA_ID_MAPPING",
    "NCORE_LABEL_CLASS_MAPPING",
    "NCORE_EGO_STATE_SE3_METADATA",
    "NCORE_BOX_DETECTIONS_SE3_METADATA",
    "resolve_ncore_label",
]
