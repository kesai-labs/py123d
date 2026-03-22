from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyquaternion

from py123d.common.utils.mixin import ArrayMixin, indexed_array_repr
from py123d.geometry.geometry_index import EulerAnglesIndex, QuaternionIndex
from py123d.geometry.utils.rotation_utils import (
    get_euler_array_from_quaternion_array,
    get_euler_array_from_rotation_matrix,
    get_quaternion_array_from_euler_array,
    get_quaternion_array_from_rotation_matrix,
    get_rotation_matrix_from_euler_array,
    get_rotation_matrix_from_quaternion_array,
)


class EulerAngles(ArrayMixin):
    """Class to represent 3D rotation using Euler angles (roll, pitch, yaw) in radians.

    Examples
    --------
    >>> import numpy as np
    >>> from py123d.geometry import EulerAngles
    >>> euler_angles = EulerAngles(roll=0.0, pitch=0.0, yaw=np.pi)
    >>> euler_angles.roll
    0.0
    >>> euler_angles.yaw
    3.141592653589793
    >>> euler_angles.array
    array([0.0, 0.0, 3.14159265])
    >>> EulerAngles.from_rotation_matrix(euler_angles.rotation_matrix).yaw
    3.141592653589793

    Notes
    -----
    The rotation order is intrinsic Z-Y'-X'' (yaw-pitch-roll) [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Euler_angles

    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, roll: float, pitch: float, yaw: float):
        """Initialize EulerAngles with roll, pitch, yaw angles in radians.

        :param roll: The roll (x-axis rotation) angle in radians.
        :param pitch: The pitch (y-axis rotation) angle in radians.
        :param yaw: The yaw (z-axis rotation) angle in radians.
        """
        array = np.zeros(len(EulerAnglesIndex), dtype=np.float64)
        array[EulerAnglesIndex.ROLL] = roll
        array[EulerAnglesIndex.PITCH] = pitch
        array[EulerAnglesIndex.YAW] = yaw
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> EulerAngles:
        """Constructs a :class:`EulerAngles` from a numpy array of shape (3,) representing, indexed by \
            :class:`~py123d.geometry.EulerAnglesIndex`.

        :param array: Array of shape (3,) representing the euler angles [roll, pitch, yaw].
        :param copy: Whether to copy the input array. Defaults to True.
        :return: A :class:`EulerAngles` instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(EulerAnglesIndex)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix: npt.NDArray[np.float64]) -> EulerAngles:
        """Constructs a :class:`EulerAngles` from a 3x3 rotation matrix.

        :param rotation_matrix: A 3x3 numpy array representing the rotation matrix.
        :return: A :class:`EulerAngles` instance.
        """
        assert rotation_matrix.ndim == 2
        assert rotation_matrix.shape == (3, 3)
        return EulerAngles.from_array(get_euler_array_from_rotation_matrix(rotation_matrix), copy=False)

    @classmethod
    def identity(cls) -> EulerAngles:
        """Returns the identity Euler angles representing no rotation."""
        return EulerAngles.from_array(np.zeros(len(EulerAnglesIndex), dtype=np.float64), copy=False)

    @property
    def roll(self) -> float:
        """The roll (x-axis rotation) angle in radians."""
        return self._array[EulerAnglesIndex.ROLL]

    @property
    def pitch(self) -> float:
        """The pitch (y-axis rotation) angle in radians."""
        return self._array[EulerAnglesIndex.PITCH]

    @property
    def yaw(self) -> float:
        """The yaw (z-axis rotation) angle in radians."""
        return self._array[EulerAnglesIndex.YAW]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Converts the EulerAngles instance to a numpy array of shape (3,),\
            indexed by :class:`~py123d.geometry.EulerAnglesIndex`.
        """
        return self._array

    @property
    def quaternion(self) -> Quaternion:
        """The :class:`Quaternion` representation of the Euler angles."""
        return Quaternion.from_euler_angles(self)

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 rotation matrix representation of the Euler angles."""
        return get_rotation_matrix_from_euler_array(self.array)

    def __repr__(self) -> str:
        """String representation of :class:`EulerAngles`."""
        return indexed_array_repr(self, EulerAnglesIndex)


class Quaternion(ArrayMixin):
    """
    Represents a quaternion for 3D rotations.

    Examples
    --------
    >>> import numpy as np
    >>> from py123d.geometry import Quaternion
    >>> quat = Quaternion(1.0, 0.0, 0.0, 0.0)
    >>> quat.qw
    1.0
    >>> quat.qx
    0.0
    >>> quat.array
    array([1.0, 0.0, 0.0, 0.0])
    >>> quat.rotation_matrix
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, qw: float, qx: float, qy: float, qz: float):
        """Initialize Quaternion with components.

        :param qw: The scalar component of the quaternion.
        :param qx: The x component of the quaternion.
        :param qy: The y component of the quaternion.
        :param qz: The z component of the quaternion.
        """
        array = np.zeros(len(QuaternionIndex), dtype=np.float64)
        array[QuaternionIndex.QW] = qw
        array[QuaternionIndex.QX] = qx
        array[QuaternionIndex.QY] = qy
        array[QuaternionIndex.QZ] = qz
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> Quaternion:
        """Constructs a Quaternion from a numpy array.

        :param array: A 1D numpy array of shape (4,) containing the quaternion components [qw, qx, qy, qz].
        :param copy: Whether to copy the array data, defaults to True.
        :return: A Quaternion instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(QuaternionIndex)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix: npt.NDArray[np.float64]) -> Quaternion:
        """Constructs a Quaternion from a 3x3 rotation matrix.

        :param rotation_matrix: A 3x3 numpy array representing the rotation matrix.
        :return: A Quaternion instance.
        """
        assert rotation_matrix.ndim == 2
        assert rotation_matrix.shape == (3, 3)
        return Quaternion.from_array(get_quaternion_array_from_rotation_matrix(rotation_matrix), copy=False)

    @classmethod
    def from_euler_angles(cls, euler_angles: EulerAngles) -> Quaternion:
        """Constructs a Quaternion from Euler angles.
        NOTE: The rotation order is intrinsic Z-Y'-X'' (yaw-pitch-roll).

        :param euler_angles: An EulerAngles instance representing the Euler angles.
        :return: A Quaternion instance.
        """
        return Quaternion.from_array(get_quaternion_array_from_euler_array(euler_angles.array), copy=False)

    @classmethod
    def identity(cls) -> Quaternion:
        """Returns the identity quaternion representing no rotation."""
        quat = np.zeros(len(QuaternionIndex), dtype=np.float64)
        quat[QuaternionIndex.QW] = 1.0
        return Quaternion.from_array(quat, copy=False)

    @property
    def qw(self) -> float:
        """The scalar component of the quaternion."""
        return self._array[QuaternionIndex.QW]

    @property
    def qx(self) -> float:
        """The x component of the quaternion."""
        return self._array[QuaternionIndex.QX]

    @property
    def qy(self) -> float:
        """The y component of the quaternion."""
        return self._array[QuaternionIndex.QY]

    @property
    def qz(self) -> float:
        """The z component of the quaternion."""
        return self._array[QuaternionIndex.QZ]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The numpy array of shape (4,) containing the quaternion [qw, qx, qy, qz], indexed by \
            :class:`~py123d.geometry.QuaternionIndex`.
        """
        return self._array

    @property
    def pyquaternion(self) -> pyquaternion.Quaternion:
        """The pyquaternion.Quaternion representation of the quaternion."""
        return pyquaternion.Quaternion(array=self.array)

    @property
    def euler_angles(self) -> EulerAngles:
        """The :class:`EulerAngles` representation of the quaternion."""
        return EulerAngles.from_array(get_euler_array_from_quaternion_array(self.array), copy=False)

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Returns the 3x3 rotation matrix representation of the quaternion."""
        return get_rotation_matrix_from_quaternion_array(self.array)

    def __repr__(self) -> str:
        """String representation of :class:`Quaternion`."""
        return indexed_array_repr(self, QuaternionIndex)
