import numpy as np
import numpy.typing as npt

from py123d.geometry.pose import PoseSE3


def quat_scalar_last_to_pose_se3(qx: float, qy: float, qz: float, qw: float, x: float, y: float, z: float) -> PoseSE3:
    """Convert a scalar-last quaternion (qx, qy, qz, qw) + translation to PoseSE3.

    Physical AI AV uses scalar-last convention; py123d PoseSE3 uses (x, y, z, qw, qx, qy, qz).

    :param qx: Quaternion x component.
    :param qy: Quaternion y component.
    :param qz: Quaternion z component.
    :param qw: Quaternion w (scalar) component.
    :param x: Translation x.
    :param y: Translation y.
    :param z: Translation z.
    :return: A :class:`PoseSE3` instance.
    """
    return PoseSE3(x=x, y=y, z=z, qw=qw, qx=qx, qy=qy, qz=qz)


def find_closest_index(timestamps: npt.NDArray[np.int64], target: int) -> int:
    """Find the index of the closest timestamp to the target using binary search.

    :param timestamps: Sorted array of timestamps.
    :param target: Target timestamp to match.
    :return: Index of the closest timestamp.
    """
    idx = np.searchsorted(timestamps, target)
    if idx == 0:
        result = 0
    elif idx == len(timestamps):
        result = len(timestamps) - 1
    elif abs(timestamps[idx] - target) < abs(timestamps[idx - 1] - target):
        result = int(idx)
    else:
        result = int(idx - 1)
    return result
