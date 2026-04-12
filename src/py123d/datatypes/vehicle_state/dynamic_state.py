from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import classproperty
from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry import Vector2D, Vector3D


class DynamicStateSE3Index(IntEnum):
    """The indices for the dynamic state in SE3."""

    VELOCITY_X = 0
    """Velocity in the X direction (forward)."""

    VELOCITY_Y = 1
    """Velocity in the Y direction (left)."""

    VELOCITY_Z = 2
    """Velocity in the Z direction (up)."""

    ACCELERATION_X = 3
    """Acceleration in the X direction (forward)."""

    ACCELERATION_Y = 4
    """Acceleration in the Y direction (left)."""

    ACCELERATION_Z = 5
    """Acceleration in the Z direction (up)."""

    ANGULAR_VELOCITY_X = 6
    """Angular velocity around the X axis (roll)."""

    ANGULAR_VELOCITY_Y = 7
    """Angular velocity around the Y axis (pitch)."""

    ANGULAR_VELOCITY_Z = 8
    """Angular velocity around the Z axis (yaw)."""

    @classproperty
    def VELOCITY_3D(cls) -> slice:
        """Slice for the 3D velocity components (x,y,z)."""
        return slice(cls.VELOCITY_X, cls.VELOCITY_Z + 1)

    @classproperty
    def VELOCITY_2D(cls) -> slice:
        """Slice for the 2D velocity components (x,y)."""
        return slice(cls.VELOCITY_X, cls.VELOCITY_Y + 1)

    @classproperty
    def ACCELERATION_3D(cls) -> slice:
        """Slice for the 3D acceleration components (x,y,z)."""
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Z + 1)

    @classproperty
    def ACCELERATION_2D(cls) -> slice:
        """Slice for the 2D acceleration components (x,y)."""
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Y + 1)

    @classproperty
    def ANGULAR_VELOCITY_3D(cls) -> slice:
        """Slice for the 3D angular velocity components (x,y,z)."""
        return slice(cls.ANGULAR_VELOCITY_X, cls.ANGULAR_VELOCITY_Z + 1)


class DynamicStateSE3(ArrayMixin):
    """The dynamic state of a vehicle in SE3 (3D space). All components are in the ego frame."""

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(
        self,
        velocity: Vector3D,
        acceleration: Vector3D,
        angular_velocity: Vector3D,
    ):
        """Initialize a :class:`DynamicStateSE3` instance.

        :param velocity: 3D velocity vector in the ego frame.
        :param acceleration: 3D acceleration vector in the ego frame.
        :param angular_velocity: 3D angular velocity vector in the ego frame.
        """
        array = np.zeros(len(DynamicStateSE3Index), dtype=np.float64)
        array[DynamicStateSE3Index.VELOCITY_3D] = velocity.array
        array[DynamicStateSE3Index.ACCELERATION_3D] = acceleration.array
        array[DynamicStateSE3Index.ANGULAR_VELOCITY_3D] = angular_velocity.array
        self._array = array

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> DynamicStateSE3:
        """Create a :class:`DynamicStateSE3` from NumPy array of shape (9,), indexed by :class:`DynamicStateSE3Index`.

        :param array: The array containing the dynamic state information.
        :param copy: Whether to copy the array data.
        :return: A :class:`DynamicStateSE3` instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(DynamicStateSE3Index)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @property
    def velocity_3d(self) -> Vector3D:
        """3D velocity vector in the ego frame."""
        return Vector3D.from_array(self._array[DynamicStateSE3Index.VELOCITY_3D], copy=False)

    @property
    def velocity_2d(self) -> Vector2D:
        """2D velocity vector in the ego frame."""
        return Vector2D.from_array(self._array[DynamicStateSE3Index.VELOCITY_2D], copy=False)

    @property
    def acceleration_3d(self) -> Vector3D:
        """3D acceleration vector in the ego frame."""
        return Vector3D.from_array(self._array[DynamicStateSE3Index.ACCELERATION_3D], copy=False)

    @property
    def acceleration_2d(self) -> Vector2D:
        """2D acceleration vector in the ego frame."""
        return Vector2D.from_array(self._array[DynamicStateSE3Index.ACCELERATION_2D], copy=False)

    @property
    def angular_velocity(self) -> Vector3D:
        """3D angular velocity vector in the ego frame."""
        return Vector3D.from_array(self._array[DynamicStateSE3Index.ANGULAR_VELOCITY_3D], copy=False)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """NumPy array representation of shape (9,), indexed by :class:`DynamicStateSE3Index`."""
        return self._array

    @property
    def dynamic_state_se2(self) -> DynamicStateSE2:
        """The :class:`DynamicStateSE2` projection of this SE3 dynamic state."""
        _array = np.zeros(len(DynamicStateSE2Index), dtype=np.float64)
        _array[DynamicStateSE2Index.VELOCITY_2D] = self._array[DynamicStateSE3Index.VELOCITY_2D]
        _array[DynamicStateSE2Index.ACCELERATION_2D] = self._array[DynamicStateSE3Index.ACCELERATION_2D]
        _array[DynamicStateSE2Index.ANGULAR_VELOCITY_Z] = self._array[DynamicStateSE3Index.ANGULAR_VELOCITY_Z]
        return DynamicStateSE2.from_array(_array, copy=False)


class DynamicStateSE2Index(IntEnum):
    """The indices for the dynamic state in SE2."""

    VELOCITY_X = 0
    """Velocity in the X direction (forward)."""

    VELOCITY_Y = 1
    """Velocity in the Y direction (left)."""

    ACCELERATION_X = 2
    """Acceleration in the X direction (forward)."""

    ACCELERATION_Y = 3
    """Acceleration in the Y direction (left)."""

    ANGULAR_VELOCITY_Z = 4
    """Angular velocity around the Z axis (yaw)."""

    @classproperty
    def VELOCITY_2D(cls) -> slice:
        """Slice for the 2D velocity components (x,y)."""
        return slice(cls.VELOCITY_X, cls.VELOCITY_Y + 1)

    @classproperty
    def ACCELERATION_2D(cls) -> slice:
        """Slice for the 2D acceleration components (x,y)."""
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Y + 1)

    @classproperty
    def ANGULAR_VELOCITY(cls) -> int:
        """Index for the angular velocity component (yaw)."""
        return cls.ANGULAR_VELOCITY_Z


@dataclass
class DynamicStateSE2(ArrayMixin):
    """The dynamic state of a vehicle in SE2 (2D plane)."""

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(
        self,
        velocity: Vector2D,
        acceleration: Vector2D,
        angular_velocity: float,
    ):
        """Initialize a :class:`DynamicStateSE2` instance.

        :param velocity: 2D velocity vector in the ego frame.
        :param acceleration: 2D acceleration vector in the ego frame.
        :param angular_velocity: Angular velocity around the Z axis (yaw).
        """
        array = np.zeros(len(DynamicStateSE2Index), dtype=np.float64)
        array[DynamicStateSE2Index.VELOCITY_2D] = velocity.array
        array[DynamicStateSE2Index.ACCELERATION_2D] = acceleration.array
        array[DynamicStateSE2Index.ANGULAR_VELOCITY_Z] = angular_velocity
        self._array = array

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> DynamicStateSE2:
        """Create a :class:`DynamicStateSE2` from NumPy array of shape (5,), indexed by :class:`DynamicStateSE2Index`.

        :param array: The array containing the dynamic state information.
        :param copy: Whether to copy the array data.
        :return: A :class:`DynamicStateSE2` instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(DynamicStateSE2Index)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @property
    def velocity_2d(self) -> Vector2D:
        """2D velocity vector in the ego frame."""
        return Vector2D.from_array(self._array[DynamicStateSE2Index.VELOCITY_2D], copy=False)

    @property
    def acceleration_2d(self) -> Vector2D:
        """2D acceleration vector in the ego frame."""
        return Vector2D.from_array(self._array[DynamicStateSE2Index.ACCELERATION_2D], copy=False)

    @property
    def angular_velocity(self) -> float:
        """Angular velocity around the Z axis (yaw)."""
        return self._array[DynamicStateSE2Index.ANGULAR_VELOCITY_Z]

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """NumPy array representation of shape (5,), indexed by :class:`DynamicStateSE2Index`."""
        return self._array
