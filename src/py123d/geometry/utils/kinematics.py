import numpy as np
import numpy.typing as npt


def phase_unwrap(yaws: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Returns an array of heading angles equal mod 2 pi to the input heading angles, and such that the difference \
        between successive output angles is less than or equal to pi radians in absolute value

    :param yaws: An array of yaws (radians)
    :return: The phase-unwrapped equivalent yaws.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = yaws[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= yaws[j+1] - yaws[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (yaws[j+1] - yaws[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((yaws[j+1] - yaws[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(yaws)
    adjustments[1:] = np.cumsum(np.round(np.diff(yaws) / two_pi))
    unwrapped = yaws - two_pi * adjustments
    return unwrapped


def extract_linear_velocity_from_se2(poses_se2: npt.NDArray[np.float64], delta_t: float) -> npt.NDArray[np.float64]:
    """Extracts scalar linear velocities from a sequence of SE2 poses using finite differences.

    :param poses_se2: Array of SE2 poses of shape (N, 3), indexed by :class:`~py123d.geometry.PoseSE2Index`.
    :param delta_t: Time step between successive poses in seconds.
    :return: Array of scalar linear velocities of shape (N-1,).
    """
    xys = poses_se2[..., :2]
    delta_xys = np.diff(xys, axis=0)
    distances = np.linalg.norm(delta_xys, axis=-1) / delta_t
    velocities = distances
    return velocities


def extract_linear_acceleration_from_se2(poses_se2: npt.NDArray[np.float64], delta_t: float) -> npt.NDArray[np.float64]:
    """Extracts scalar linear accelerations from a sequence of SE2 poses using second-order finite differences.

    :param poses_se2: Array of SE2 poses of shape (N, 3), indexed by :class:`~py123d.geometry.PoseSE2Index`.
    :param delta_t: Time step between successive poses in seconds.
    :return: Array of scalar linear accelerations of shape (N-2,).
    """
    velocities = extract_linear_velocity_from_se2(poses_se2, delta_t)
    delta_velocities = np.diff(velocities)
    accelerations = delta_velocities / delta_t
    return accelerations


def extract_yaw_rate_from_se2(poses_se2: npt.NDArray[np.float64], delta_t: float) -> npt.NDArray[np.float64]:
    """Extracts yaw rates from a sequence of SE2 poses using phase-unwrapped finite differences.

    :param poses_se2: Array of SE2 poses of shape (N, 3), indexed by :class:`~py123d.geometry.PoseSE2Index`.
    :param delta_t: Time step between successive poses in seconds.
    :return: Array of yaw rates in radians per second of shape (N-1,).
    """
    yaws = poses_se2[..., 2]
    unwrapped_yaws = phase_unwrap(yaws)
    delta_yaws = np.diff(unwrapped_yaws, axis=0)
    yaw_rates = delta_yaws / delta_t
    return yaw_rates
