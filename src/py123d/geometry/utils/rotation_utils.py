from typing import Union

import numpy as np
import numpy.typing as npt

from py123d.geometry.geometry_index import EulerAnglesIndex, QuaternionIndex


def batch_matmul(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Batch matrix multiplication for arrays of matrices.
    # TODO: move somewhere else

    :param A: Array of shape (..., M, N)
    :param B: Array of shape (..., N, P)
    :return: Array of shape (..., M, P) resulting from batch matrix multiplication of A and B.
    """
    assert A.ndim >= 2 and B.ndim >= 2
    assert A.shape[-1] == B.shape[-2], (
        f"Inner dimensions must match for matrix multiplication, got {A.shape} and {B.shape}"
    )
    return np.einsum("...ij,...jk->...ik", A, B)


def normalize_angle(angle: Union[float, npt.NDArray[np.float64]]) -> Union[float, npt.NDArray[np.float64]]:
    """Normalizes an angle or array of angles to the range [-pi, pi].

    :param angle: Angle or array of angles in radians to normalize.
    :return: Normalized angle or array of angles.
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def get_rotation_matrices_from_euler_array(euler_angles_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Euler angles to rotation matrices using Tait-Bryan ZYX convention (yaw-pitch-roll).

    :param euler_angles_array: Array of Euler angles of shape (..., 3), \
        indexed by :class:`~py123d.geometry.EulerAnglesIndex`
    :return: Array of rotation matrices of shape (..., 3, 3)
    """
    assert euler_angles_array.ndim >= 1 and euler_angles_array.shape[-1] == len(EulerAnglesIndex)

    # Store original shape for reshaping later
    original_shape = euler_angles_array.shape[:-1]

    # Flatten to 2D if needed
    if euler_angles_array.ndim > 2:
        euler_angles_array_ = euler_angles_array.reshape(-1, len(EulerAnglesIndex))
    else:
        euler_angles_array_ = euler_angles_array

    # Extract roll, pitch, yaw for all samples at once
    roll = euler_angles_array_[:, EulerAnglesIndex.ROLL]
    pitch = euler_angles_array_[:, EulerAnglesIndex.PITCH]
    yaw = euler_angles_array_[:, EulerAnglesIndex.YAW]

    # Compute sin/cos for all angles at once
    # NOTE: (c/s = cos/sin, r/p/y = roll/pitch/yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Build rotation matrices for entire batch
    batch_size = euler_angles_array_.shape[0]
    rotation_matrices = np.zeros((batch_size, 3, 3), dtype=np.float64)

    # Formula for ZYX Tait-Bryan rotation matrix:
    # R = | cy*cp   cy*sp*sr - sy*cr   cy*sp*cr + sy*sr |
    #     | sy*cp   sy*sp*sr + cy*cr   sy*sp*cr - cy*sr |
    #     | -sp     cp*sr              cp*cr            |

    # ZYX Tait-Bryan rotation matrix elements
    rotation_matrices[:, 0, 0] = cy * cp
    rotation_matrices[:, 1, 0] = sy * cp
    rotation_matrices[:, 2, 0] = -sp

    rotation_matrices[:, 0, 1] = cy * sp * sr - sy * cr
    rotation_matrices[:, 1, 1] = sy * sp * sr + cy * cr
    rotation_matrices[:, 2, 1] = cp * sr

    rotation_matrices[:, 0, 2] = cy * sp * cr + sy * sr
    rotation_matrices[:, 1, 2] = sy * sp * cr - cy * sr
    rotation_matrices[:, 2, 2] = cp * cr

    # Reshape back to original batch dimensions + (3, 3)
    if len(original_shape) > 1:
        rotation_matrices = rotation_matrices.reshape(original_shape + (3, 3))

    return rotation_matrices


def get_euler_array_from_rotation_matrices(rotation_matrices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert rotation matrices to Euler angles using Tait-Bryan ZYX convention (yaw-pitch-roll).

    :param rotation_matrices: Rotation matrices of shape (..., 3, 3)
    :return: Euler angles of shape (..., 3), indexed by EulerAnglesIndex
    """
    assert rotation_matrices.ndim >= 2 and rotation_matrices.shape[-2:] == (3, 3)

    original_shape = rotation_matrices.shape[:-2]

    # Flatten to 3D if needed, i.e. (N, 3, 3)
    if rotation_matrices.ndim > 3:
        R = rotation_matrices.reshape(-1, 3, 3)
    else:
        R = rotation_matrices

    batch_size = R.shape[0]
    euler_angles = np.zeros((batch_size, len(EulerAnglesIndex)), dtype=np.float64)

    # Calculate yaw (rotation around Z-axis)
    euler_angles[:, EulerAnglesIndex.YAW] = np.arctan2(R[:, 1, 0], R[:, 0, 0])

    # Calculate pitch (rotation around Y-axis)
    # NOTE: Clip to avoid numerical issues with arcsin
    sin_pitch = np.clip(-R[:, 2, 0], -1.0, 1.0)
    euler_angles[:, EulerAnglesIndex.PITCH] = np.arcsin(sin_pitch)

    # Calculate roll (rotation around X-axis)
    euler_angles[:, EulerAnglesIndex.ROLL] = np.arctan2(R[:, 2, 1], R[:, 2, 2])

    # Reshape back to original batch dimensions + (3,)
    if len(original_shape) > 1:
        euler_angles = euler_angles.reshape(original_shape + (len(EulerAnglesIndex),))

    return euler_angles


def get_euler_array_from_rotation_matrix(rotation_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a rotation matrix to Euler angles using Tait-Bryan ZYX convention (yaw-pitch-roll).

    :param rotation_matrix: Rotation matrix of shape (3, 3)
    :return: Euler angles of shape (3,), indexed by :class:`~py123d.geometry.EulerAnglesIndex`
    """
    assert rotation_matrix.ndim == 2 and rotation_matrix.shape == (3, 3)
    return get_euler_array_from_rotation_matrices(rotation_matrix[None, ...])[0]


def get_quaternion_array_from_rotation_matrices(rotation_matrices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert rotation matrices to quaternions.

    :param rotation_matrices: Rotation matrices of shape (..., 3, 3)
    :return: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    """
    assert rotation_matrices.ndim >= 2
    assert rotation_matrices.shape[-1] == rotation_matrices.shape[-2] == 3
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    # TODO: Update with:
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

    original_shape = rotation_matrices.shape[:-2]

    # Extract rotation matrix elements for vectorized operations
    if rotation_matrices.ndim > 3:
        R = rotation_matrices.reshape(-1, 3, 3)
    else:
        R = rotation_matrices

    N = R.shape[0]
    quaternions = np.zeros((N, 4), dtype=np.float64)

    # Compute trace for each matrix
    trace = np.trace(R, axis1=1, axis2=2)

    # Case 1: trace > 0 (most common case)
    mask1 = trace > 0
    s1 = np.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
    quaternions[mask1, QuaternionIndex.QW] = 0.25 * s1
    quaternions[mask1, QuaternionIndex.QX] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    quaternions[mask1, QuaternionIndex.QY] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    quaternions[mask1, QuaternionIndex.QZ] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = np.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s = 4 * qx
    quaternions[mask2, QuaternionIndex.QW] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    quaternions[mask2, QuaternionIndex.QX] = 0.25 * s2  # x
    quaternions[mask2, QuaternionIndex.QY] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    quaternions[mask2, QuaternionIndex.QZ] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = np.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s = 4 * qy
    quaternions[mask3, QuaternionIndex.QW] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    quaternions[mask3, QuaternionIndex.QX] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    quaternions[mask3, QuaternionIndex.QY] = 0.25 * s3  # y
    quaternions[mask3, QuaternionIndex.QZ] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] is largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = np.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s = 4 * qz
    quaternions[mask4, QuaternionIndex.QW] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    quaternions[mask4, QuaternionIndex.QX] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    quaternions[mask4, QuaternionIndex.QY] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    quaternions[mask4, QuaternionIndex.QZ] = 0.25 * s4  # z

    assert np.all(mask1 | mask2 | mask3 | mask4), "All matrices should fall into one of the four cases."

    quaternions = normalize_quaternion_array(quaternions)

    # Reshape back to original batch dimensions + (4,)
    if len(original_shape) > 1:
        quaternions = quaternions.reshape(original_shape + (len(QuaternionIndex),))

    return quaternions


def get_quaternion_array_from_rotation_matrix(rotation_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a rotation matrix to a quaternion.

    :param rotation_matrix: Rotation matrix of shape (3, 3)
    :return: Quaternion of shape (4,), indexed by :class:`~py123d.geometry.QuaternionIndex`.
    """
    assert rotation_matrix.ndim == 2 and rotation_matrix.shape == (3, 3)
    return get_quaternion_array_from_rotation_matrices(rotation_matrix[None, ...])[0]


def get_quaternion_array_from_euler_array(euler_angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts array of euler angles to array of quaternions.

    :param euler_angles: Euler angles of shape (..., 3), indexed by :class:`~py123d.geometry.EulerAnglesIndex`
    :return: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    """
    assert euler_angles.ndim >= 1 and euler_angles.shape[-1] == len(EulerAnglesIndex)

    # Store original shape for reshaping later
    original_shape = euler_angles.shape[:-1]

    # Flatten to 2D if needed
    if euler_angles.ndim > 2:
        euler_angles_ = euler_angles.reshape(-1, len(EulerAnglesIndex))
    else:
        euler_angles_ = euler_angles

    # Extract roll, pitch, yaw
    roll = euler_angles_[..., EulerAnglesIndex.ROLL]
    pitch = euler_angles_[..., EulerAnglesIndex.PITCH]
    yaw = euler_angles_[..., EulerAnglesIndex.YAW]

    # Half angles
    roll_half = roll / 2.0
    pitch_half = pitch / 2.0
    yaw_half = yaw / 2.0

    # Compute sin/cos for half angles
    cos_roll_half = np.cos(roll_half)
    sin_roll_half = np.sin(roll_half)
    cos_pitch_half = np.cos(pitch_half)
    sin_pitch_half = np.sin(pitch_half)
    cos_yaw_half = np.cos(yaw_half)
    sin_yaw_half = np.sin(yaw_half)

    # Compute quaternion components (ZYX intrinsic rotation order)
    qw = cos_roll_half * cos_pitch_half * cos_yaw_half + sin_roll_half * sin_pitch_half * sin_yaw_half
    qx = sin_roll_half * cos_pitch_half * cos_yaw_half - cos_roll_half * sin_pitch_half * sin_yaw_half
    qy = cos_roll_half * sin_pitch_half * cos_yaw_half + sin_roll_half * cos_pitch_half * sin_yaw_half
    qz = cos_roll_half * cos_pitch_half * sin_yaw_half - sin_roll_half * sin_pitch_half * cos_yaw_half

    # Stack into quaternion array
    quaternions = np.stack([qw, qx, qy, qz], axis=-1)

    # Reshape back to original batch dimensions + (4,)
    if len(original_shape) > 1:
        quaternions = quaternions.reshape(original_shape + (len(QuaternionIndex),))

    return normalize_quaternion_array(quaternions)  # type: ignore


def get_rotation_matrix_from_euler_array(euler_angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Euler angles to rotation matrix using Tait-Bryan ZYX convention (yaw-pitch-roll).

    :param euler_angles: Euler angles of shape (3,), indexed by :class:`~py123d.geometry.EulerAnglesIndex`
    :return: Rotation matrix of shape (3, 3)
    """
    assert euler_angles.ndim == 1 and euler_angles.shape[0] == len(EulerAnglesIndex)
    return get_rotation_matrices_from_euler_array(euler_angles[None, ...])[0]


def get_rotation_matrices_from_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert array of quaternions to array of rotation matrices.

    :param quaternion_array: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    :return: Rotation matrices of shape (..., 3, 3)
    """
    assert quaternion_array.ndim >= 1 and quaternion_array.shape[-1] == len(QuaternionIndex)

    q = normalize_quaternion_array(quaternion_array)
    qw = q[..., QuaternionIndex.QW]
    qx = q[..., QuaternionIndex.QX]
    qy = q[..., QuaternionIndex.QY]
    qz = q[..., QuaternionIndex.QZ]

    # Precompute repeated products
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    # Build rotation matrices using the direct algebraic formula
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)

    return R


def get_rotation_matrix_from_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a quaternion to a rotation matrix.

    :param quaternion_array: Quaternion of shape (4,), indexed by :class:`~py123d.geometry.QuaternionIndex`
    :return: Rotation matrix of shape (3, 3)
    """
    assert quaternion_array.ndim == 1 and quaternion_array.shape[0] == len(QuaternionIndex)

    # Fast path for single quaternion: use scalar math to avoid array overhead
    inv_norm = 1.0 / np.linalg.norm(quaternion_array)
    w = float(quaternion_array[QuaternionIndex.QW]) * inv_norm
    x = float(quaternion_array[QuaternionIndex.QX]) * inv_norm
    y = float(quaternion_array[QuaternionIndex.QY]) * inv_norm
    z = float(quaternion_array[QuaternionIndex.QZ]) * inv_norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def get_euler_array_from_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts array of quaternions to array of euler angles.

    :param quaternion_array: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    :return: Euler angles of shape (..., 3), indexed by :class:`~py123d.geometry.EulerAnglesIndex`
    """
    assert quaternion_array.ndim >= 1 and quaternion_array.shape[-1] == len(QuaternionIndex)
    norm_quaternion = normalize_quaternion_array(quaternion_array)
    QW, QX, QY, QZ = (
        norm_quaternion[..., QuaternionIndex.QW],
        norm_quaternion[..., QuaternionIndex.QX],
        norm_quaternion[..., QuaternionIndex.QY],
        norm_quaternion[..., QuaternionIndex.QZ],
    )

    euler_angles = np.zeros_like(quaternion_array[..., :3])
    euler_angles[..., EulerAnglesIndex.YAW] = np.arctan2(
        2 * (QW * QZ - QX * QY),
        1 - 2 * (QY**2 + QZ**2),
    )
    euler_angles[..., EulerAnglesIndex.PITCH] = np.arcsin(
        np.clip(2 * (QW * QY + QZ * QX), -1.0, 1.0),
    )
    euler_angles[..., EulerAnglesIndex.ROLL] = np.arctan2(
        2 * (QW * QX - QY * QZ),
        1 - 2 * (QX**2 + QY**2),
    )

    return euler_angles


def conjugate_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the conjugate of an array of quaternions, i.e. negating the vector part.

    :param quaternion_array: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    :return: Conjugated quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    """

    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)
    conjugated = quaternion_array.copy()
    conjugated[..., QuaternionIndex.VECTOR] *= -1
    return conjugated


def invert_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the inverse of an array of quaternions, i.e. conjugate divided by norm squared.

    :param quaternion_array: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    :return: Inverted quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)
    norm_squared = np.sum(quaternion_array**2, axis=-1, keepdims=True)
    assert np.all(norm_squared > 0), "Cannot invert a quaternion with zero norm."
    conjugated_quaternions = conjugate_quaternion_array(quaternion_array)
    inverted_quaternions = conjugated_quaternions / norm_squared
    return inverted_quaternions


def normalize_quaternion_array(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalizes an array of quaternions to unit length.

    :param quaternion_array: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    :return: Normalized quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)
    norm = np.linalg.norm(quaternion_array, axis=-1, keepdims=True)
    assert np.all(norm > 0), "Cannot normalize a quaternion with zero norm."
    normalized_quaternions = quaternion_array / norm
    return normalized_quaternions


def multiply_quaternion_arrays(q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Multiplies two arrays of quaternions.

    :param q1: First array of quaternions, indexed by :class:`~py123d.geometry.QuaternionIndex` in the last dim.
    :param q2: Second array of quaternions, indexed by :class:`~py123d.geometry.QuaternionIndex` in the last dim.
    :return: Array of resulting quaternions after multiplication, \
        indexed by :class:`~py123d.geometry.QuaternionIndex` in the last dim.
    """
    assert q1.ndim >= 1
    assert q2.ndim >= 1
    assert q1.shape[-1] == q2.shape[-1] == len(QuaternionIndex)

    # Vectorized quaternion multiplication
    qw1, qx1, qy1, qz1 = (
        q1[..., QuaternionIndex.QW],
        q1[..., QuaternionIndex.QX],
        q1[..., QuaternionIndex.QY],
        q1[..., QuaternionIndex.QZ],
    )
    qw2, qx2, qy2, qz2 = (
        q2[..., QuaternionIndex.QW],
        q2[..., QuaternionIndex.QX],
        q2[..., QuaternionIndex.QY],
        q2[..., QuaternionIndex.QZ],
    )

    result = np.empty(np.broadcast_shapes(q1.shape, q2.shape), dtype=np.float64)
    result[..., QuaternionIndex.QW] = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
    result[..., QuaternionIndex.QX] = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
    result[..., QuaternionIndex.QY] = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
    result[..., QuaternionIndex.QZ] = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
    return result


def slerp_quaternion_arrays(
    q1: npt.NDArray[np.float64],
    q2: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Spherical linear interpolation (SLERP) between two arrays of quaternions.

    Interpolates along the shortest path on the unit quaternion hypersphere with constant angular velocity.

    :param q1: Start quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`.
    :param q2: End quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`.
    :param t: Interpolation parameter(s) in [0, 1], shape (...).
    :return: Interpolated quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`.
    """
    assert q1.shape[-1] == q2.shape[-1] == len(QuaternionIndex)

    dot = np.sum(q1 * q2, axis=-1, keepdims=True)

    # Ensure shortest path by flipping q2 where dot product is negative
    q2 = np.where(dot < 0, -q2, q2)
    dot = np.abs(dot)

    t_expanded = t[..., np.newaxis]
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)

    # SLERP weights (suppress expected division-by-zero for near-identical quaternions)
    near = sin_theta < 1e-6
    safe_sin_theta = np.where(near, 1.0, sin_theta)
    w1 = np.sin((1.0 - t_expanded) * theta) / safe_sin_theta
    w2 = np.sin(t_expanded * theta) / safe_sin_theta

    # Fall back to NLERP for nearly-identical quaternions
    w1 = np.where(near, 1.0 - t_expanded, w1)
    w2 = np.where(near, t_expanded, w2)

    return normalize_quaternion_array(w1 * q1 + w2 * q2)


def nlerp_quaternion_arrays(
    q1: npt.NDArray[np.float64],
    q2: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Normalized linear interpolation (NLERP) between two arrays of quaternions.

    Faster than SLERP but does not maintain constant angular velocity.

    :param q1: Start quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`.
    :param q2: End quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`.
    :param t: Interpolation parameter(s) in [0, 1], shape (...).
    :return: Interpolated quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`.
    """
    assert q1.shape[-1] == q2.shape[-1] == len(QuaternionIndex)

    dot = np.sum(q1 * q2, axis=-1, keepdims=True)

    # Ensure shortest path
    q2 = np.where(dot < 0, -q2, q2)

    t_expanded = t[..., np.newaxis]
    return normalize_quaternion_array((1.0 - t_expanded) * q1 + t_expanded * q2)


def get_q_matrices(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the Q matrices for an array of quaternions.

    :param quaternion_array: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    :return: Array of Q matrices of shape (..., 4, 4)
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)

    qw = quaternion_array[..., QuaternionIndex.QW]
    qx = quaternion_array[..., QuaternionIndex.QX]
    qy = quaternion_array[..., QuaternionIndex.QY]
    qz = quaternion_array[..., QuaternionIndex.QZ]

    batch_shape = quaternion_array.shape[:-1]
    Q_matrices = np.zeros(batch_shape + (4, 4), dtype=np.float64)

    Q_matrices[..., 0, 0] = qw
    Q_matrices[..., 0, 1] = -qx
    Q_matrices[..., 0, 2] = -qy
    Q_matrices[..., 0, 3] = -qz

    Q_matrices[..., 1, 0] = qx
    Q_matrices[..., 1, 1] = qw
    Q_matrices[..., 1, 2] = -qz
    Q_matrices[..., 1, 3] = qy

    Q_matrices[..., 2, 0] = qy
    Q_matrices[..., 2, 1] = qz
    Q_matrices[..., 2, 2] = qw
    Q_matrices[..., 2, 3] = -qx

    Q_matrices[..., 3, 0] = qz
    Q_matrices[..., 3, 1] = -qy
    Q_matrices[..., 3, 2] = qx
    Q_matrices[..., 3, 3] = qw

    return Q_matrices


def get_q_bar_matrices(quaternion_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the Q-bar matrices for an array of quaternions.

    :param quaternion_array: Quaternions of shape (..., 4), indexed by :class:`~py123d.geometry.QuaternionIndex`
    :return: Array of Q-bar matrices of shape (..., 4, 4)
    """
    assert quaternion_array.ndim >= 1
    assert quaternion_array.shape[-1] == len(QuaternionIndex)

    qw = quaternion_array[..., QuaternionIndex.QW]
    qx = quaternion_array[..., QuaternionIndex.QX]
    qy = quaternion_array[..., QuaternionIndex.QY]
    qz = quaternion_array[..., QuaternionIndex.QZ]

    batch_shape = quaternion_array.shape[:-1]
    Q_bar_matrices = np.zeros(batch_shape + (4, 4), dtype=np.float64)

    Q_bar_matrices[..., 0, 0] = qw
    Q_bar_matrices[..., 0, 1] = -qx
    Q_bar_matrices[..., 0, 2] = -qy
    Q_bar_matrices[..., 0, 3] = -qz

    Q_bar_matrices[..., 1, 0] = qx
    Q_bar_matrices[..., 1, 1] = qw
    Q_bar_matrices[..., 1, 2] = qz
    Q_bar_matrices[..., 1, 3] = -qy

    Q_bar_matrices[..., 2, 0] = qy
    Q_bar_matrices[..., 2, 1] = -qz
    Q_bar_matrices[..., 2, 2] = qw
    Q_bar_matrices[..., 2, 3] = qx

    Q_bar_matrices[..., 3, 0] = qz
    Q_bar_matrices[..., 3, 1] = qy
    Q_bar_matrices[..., 3, 2] = -qx
    Q_bar_matrices[..., 3, 3] = qw

    return Q_bar_matrices
