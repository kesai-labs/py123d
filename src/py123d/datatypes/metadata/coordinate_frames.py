from enum import Enum

import numpy as np


class GlobalFrame(Enum):
    """Global/map coordinate frame convention.

    ENU: East-North-Up. X=East, Y=North, Z=Up. Heading 0=East, positive CCW.
    NED: North-East-Down. X=North, Y=East, Z=Down. Heading 0=North, positive CW.
    """

    ENU = "ENU"
    NED = "NED"


class EgoFrame(Enum):
    """Ego-vehicle coordinate frame convention.

    FLU: Forward-Left-Up. X=Forward, Y=Left, Z=Up. Heading 0=Forward, positive CCW.
    FRD: Forward-Right-Down. X=Forward, Y=Right, Z=Down. Heading 0=Forward, positive CW.
    RFU: Right-Forward-Up. X=Right, Y=Forward, Z=Up. (e.g. nuScenes, nuPlan)
    """

    FLU = "FLU"
    FRD = "FRD"
    RFU = "RFU"


# Rotation matrices for frame conversions.
# Both ENU<->NED and FLU<->FRD are self-inverse (applying twice = identity).

_GLOBAL_FRAME_ROTATIONS = {
    (GlobalFrame.ENU, GlobalFrame.NED): np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
    ),
    (GlobalFrame.NED, GlobalFrame.ENU): np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
    ),
}

_EGO_FRAME_ROTATIONS = {
    (EgoFrame.FLU, EgoFrame.FRD): np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    (EgoFrame.FRD, EgoFrame.FLU): np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    # RFU (X=Right, Y=Forward, Z=Up) <-> FLU (X=Forward, Y=Left, Z=Up): 90° CCW around Z
    (EgoFrame.RFU, EgoFrame.FLU): np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    (EgoFrame.FLU, EgoFrame.RFU): np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    # RFU <-> FRD: swap X/Y + flip Z
    (EgoFrame.RFU, EgoFrame.FRD): np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    (EgoFrame.FRD, EgoFrame.RFU): np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
}


def get_global_frame_rotation(from_frame, to_frame):
    """Return 3x3 rotation matrix to convert coordinates from one global frame to another."""
    if isinstance(from_frame, str):
        from_frame = GlobalFrame(from_frame)
    if isinstance(to_frame, str):
        to_frame = GlobalFrame(to_frame)
    if from_frame == to_frame:
        return np.eye(3, dtype=np.float64)
    return _GLOBAL_FRAME_ROTATIONS[(from_frame, to_frame)]


def get_ego_frame_rotation(from_frame, to_frame):
    """Return 3x3 rotation matrix to convert coordinates from one ego frame to another."""
    if isinstance(from_frame, str):
        from_frame = EgoFrame(from_frame)
    if isinstance(to_frame, str):
        to_frame = EgoFrame(to_frame)
    if from_frame == to_frame:
        return np.eye(3, dtype=np.float64)
    return _EGO_FRAME_ROTATIONS[(from_frame, to_frame)]
