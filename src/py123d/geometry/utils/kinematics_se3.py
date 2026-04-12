import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


def linear_velocity_global(
    xyz_a: npt.NDArray[np.float64],
    xyz_b: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """Compute a global-frame linear velocity via a first-order finite difference.

    :param xyz_a: Translation at the earlier time, shape (3,).
    :param xyz_b: Translation at the later time, shape (3,).
    :param dt: Time delta ``t_b - t_a`` in seconds. Must be strictly positive.
    :return: Linear velocity ``(xyz_b - xyz_a) / dt``, shape (3,).
    """
    return (xyz_b - xyz_a) / dt


def linear_acceleration_global(
    xyz_prev: npt.NDArray[np.float64],
    xyz_curr: npt.NDArray[np.float64],
    xyz_next: npt.NDArray[np.float64],
    dt_prev: float,
    dt_next: float,
) -> npt.NDArray[np.float64]:
    """Compute a global-frame linear acceleration via a three-point finite difference.

    Uses the non-uniform second-order scheme that reduces to
    ``(next - 2 * curr + prev) / dt**2`` when ``dt_prev == dt_next``.

    :param xyz_prev: Translation at the earlier neighbor, shape (3,).
    :param xyz_curr: Translation at the frame whose acceleration is requested, shape (3,).
    :param xyz_next: Translation at the later neighbor, shape (3,).
    :param dt_prev: ``t_curr - t_prev`` in seconds, must be strictly positive.
    :param dt_next: ``t_next - t_curr`` in seconds, must be strictly positive.
    :return: Linear acceleration vector, shape (3,).
    """
    denom = dt_prev * dt_next * (dt_prev + dt_next)
    return 2.0 * (dt_prev * (xyz_next - xyz_curr) - dt_next * (xyz_curr - xyz_prev)) / denom


def angular_velocity_body(
    rotation_a: npt.NDArray[np.float64],
    rotation_b: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """Compute an angular velocity expressed in the body frame of ``rotation_a``.

    Uses the axis-angle log of the relative rotation
    ``R_rel = rotation_a.T @ rotation_b``, which lives in the body frame of
    ``rotation_a``.

    :param rotation_a: 3x3 rotation matrix at the earlier time.
    :param rotation_b: 3x3 rotation matrix at the later time.
    :param dt: Positive time delta in seconds.
    :return: Angular velocity vector, shape (3,).
    """
    relative_rotation = rotation_a.T @ rotation_b
    rotvec = Rotation.from_matrix(relative_rotation).as_rotvec()
    return rotvec / dt


def rotate_to_body(
    vec_global: npt.NDArray[np.float64],
    rotation_body_to_world: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Express a global-frame 3-vector in the body frame via ``R.T @ vec``.

    :param vec_global: Global-frame 3-vector, shape (3,).
    :param rotation_body_to_world: 3x3 rotation matrix mapping body to world.
    :return: The same vector expressed in the body frame, shape (3,).
    """
    return rotation_body_to_world.T @ vec_global
