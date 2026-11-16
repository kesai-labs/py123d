"""Constants of the Shifts vehicle motion prediction dataset (the SDC task of the Shifts benchmark).

References
----------
.. [1] https://github.com/Shifts-Project/shifts/tree/main/sdc
.. [2] https://arxiv.org/abs/2107.07455 (appendix E)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from py123d.datatypes import EgoStateSE3Metadata, TrafficLightStatus
from py123d.geometry import PoseSE3

SHIFTS_DATASET: str = "shifts"

# Scene timing: 25 past frames (index 24 is the prediction time T0) and 25 future frames at 5 Hz.
# The protobuf carries no absolute timestamps, so py123d timestamps are relative and start at 0 s.
SHIFTS_FRAME_INTERVAL_US: int = 200_000

# py123d split name → directory name below the data root (= the Shifts archive name without `.tar.gz`).
# Shifts calls its validation set "development" and its test set "evaluation" (paper, appendix E).
SHIFTS_SPLIT_TO_DIRECTORY: Dict[str, str] = {
    "shifts_train": "train",
    "shifts_val": "development",
    "shifts_test": "evaluation",
}
SHIFTS_AVAILABLE_SPLITS: List[str] = list(SHIFTS_SPLIT_TO_DIRECTORY.keys())

# Ego vehicle. The scenes report a constant 4.62 x 1.77 x 1.50 m ego box and nothing else about the
# vehicle, so the wheelbase and the axle position are placeholders for a mid-size sedan: IMU frame at the
# box centre, rear axle 1.35 m behind it at ground level (the box centre sits at half the height).
SHIFTS_EGO_LENGTH: float = 4.62
SHIFTS_EGO_WIDTH: float = 1.77
SHIFTS_EGO_HEIGHT: float = 1.50
SHIFTS_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="shifts_yandex_sdc",
    width=SHIFTS_EGO_WIDTH,
    length=SHIFTS_EGO_LENGTH,
    height=SHIFTS_EGO_HEIGHT,
    wheel_base=2.7,
    center_to_imu_se3=PoseSE3.identity(),
    rear_axle_to_imu_se3=PoseSE3(x=-1.35, y=0.0, z=-SHIFTS_EGO_HEIGHT / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
)

SHIFTS_AUX_MODALITY_ID: str = "aux"

# Traffic light section states, see map.proto `TrafficLightState`. A lane references its sections by id;
# the main section is the regular light, the left/right sections are arrow lights.
# https://github.com/Shifts-Project/shifts/blob/main/sdc/ysdc_dataset_api/proto/map.proto
SHIFTS_TRAFFIC_LIGHT_MAPPING: Dict[int, TrafficLightStatus] = {
    0: TrafficLightStatus.UNKNOWN,  # STATE_UNKNOWN
    1: TrafficLightStatus.GREEN,  # STATE_GREEN
    2: TrafficLightStatus.GREEN,  # STATE_BLINKING_GREEN
    3: TrafficLightStatus.YELLOW,  # STATE_YELLOW
    4: TrafficLightStatus.RED,  # STATE_RED
    5: TrafficLightStatus.YELLOW,  # STATE_RED_YELLOW
    6: TrafficLightStatus.OFF,  # STATE_NOT_WORKING
    7: TrafficLightStatus.GREEN,  # STATE_ENABLED (arrow section)
    8: TrafficLightStatus.GREEN,  # STATE_BLINKING_ENABLED (arrow section)
    9: TrafficLightStatus.OFF,  # STATE_DISABLED (a main section that is switched off)
    10: TrafficLightStatus.UNKNOWN,  # STATE_INVISIBLE
    11: TrafficLightStatus.RED,  # STATE_BLINKING_RED
}
# For an arrow section, "disabled" means the arrow is not lit, i.e. the turn is not allowed. This follows
# `get_lane_availability` in the official `ysdc_dataset_api.utils.map`.
SHIFTS_ARROW_SECTION_MAPPING: Dict[int, TrafficLightStatus] = {
    9: TrafficLightStatus.RED,  # STATE_DISABLED
}

# Hosting on Yandex Object Storage: plain tar files that wrap one gzipped tar per split (plus pre-rendered
# feature maps that py123d does not use). The storage supports HTTP range requests, so a split is streamed
# straight out of the outer tar. Offsets and sizes were read from the live archives on 2026-10-11 and are
# re-checked against the member's tar header before every download.
SHIFTS_BASE_URL: str = "https://storage.yandexcloud.net/yandex-research/shifts/sdc/"


@dataclass(frozen=True)
class ShiftsArchiveMember:
    """One gzipped scene archive inside an outer Shifts tar file."""

    outer_tar: str
    """File name of the outer tar on the storage, e.g. ``canonical-trn-dev-data.tar``."""

    member_name: str
    """Member name inside the outer tar, e.g. ``data/train.tar.gz``."""

    data_offset: int
    """Byte offset of the member's data in the outer tar (its 512-byte header sits right before it)."""

    size: int
    """Size of the member in bytes."""


SHIFTS_SPLIT_TO_ARCHIVE: Dict[str, ShiftsArchiveMember] = {
    "shifts_train": ShiftsArchiveMember(
        outer_tar="canonical-trn-dev-data.tar",
        member_name="data/train.tar.gz",
        data_offset=62997669376,
        size=61362422713,
    ),
    "shifts_val": ShiftsArchiveMember(
        outer_tar="canonical-trn-dev-data.tar",
        member_name="data/development.tar.gz",
        data_offset=1024,
        size=5581099848,
    ),
    "shifts_test": ShiftsArchiveMember(
        outer_tar="canonical-eval-data.tar",
        member_name="canonical-eval-data/evaluation.tar.gz",
        data_offset=5325999616,
        size=5964272279,
    ),
}
