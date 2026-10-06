from __future__ import annotations

from enum import IntEnum, IntFlag

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import classproperty
from py123d.common.utils.mixin import ArrayMixin
from py123d.geometry import Vector3D


class EgoUncertaintySE3Index(IntEnum):
    """The indices for the ego state uncertainty in SE3."""

    POSITION_SIGMA_X = 0
    """One-sigma position uncertainty along the global X axis (east), meters."""

    POSITION_SIGMA_Y = 1
    """One-sigma position uncertainty along the global Y axis (north), meters."""

    POSITION_SIGMA_Z = 2
    """One-sigma position uncertainty along the global Z axis (up), meters."""

    VELOCITY_SIGMA_X = 3
    """One-sigma velocity uncertainty in the ego frame X direction (forward), m/s."""

    VELOCITY_SIGMA_Y = 4
    """One-sigma velocity uncertainty in the ego frame Y direction (left), m/s."""

    VELOCITY_SIGMA_Z = 5
    """One-sigma velocity uncertainty in the ego frame Z direction (up), m/s."""

    ORIENTATION_SIGMA_ROLL = 6
    """One-sigma roll uncertainty, radians."""

    ORIENTATION_SIGMA_PITCH = 7
    """One-sigma pitch uncertainty, radians."""

    ORIENTATION_SIGMA_YAW = 8
    """One-sigma yaw uncertainty, radians."""

    @classproperty
    def POSITION_SIGMA_3D(cls) -> slice:
        """Slice for the 3D position uncertainty components (x,y,z)."""
        return slice(cls.POSITION_SIGMA_X, cls.POSITION_SIGMA_Z + 1)

    @classproperty
    def VELOCITY_SIGMA_3D(cls) -> slice:
        """Slice for the 3D velocity uncertainty components (x,y,z)."""
        return slice(cls.VELOCITY_SIGMA_X, cls.VELOCITY_SIGMA_Z + 1)

    @classproperty
    def ORIENTATION_SIGMA_3D(cls) -> slice:
        """Slice for the 3D orientation uncertainty components (roll,pitch,yaw)."""
        return slice(cls.ORIENTATION_SIGMA_ROLL, cls.ORIENTATION_SIGMA_YAW + 1)


class EgoQualityFlag(IntFlag):
    """What was supporting the ego state at one instant.

    A pose estimator is corrected by different sources at different rates, and its covariance
    alone does not say which. These flags do, so a consumer can select the stretches a given
    source was actually present for -- or reject the ones where nothing but dead reckoning was.
    """

    NONE = 0
    """Nothing corrected the state here: it is dead reckoning."""

    GNSS_POSITION = 1 << 0
    """A GNSS position was applied recently."""

    GNSS_VELOCITY = 1 << 1
    """A GNSS velocity solution was applied recently."""

    RADAR_VELOCITY = 1 << 2
    """A radar Doppler velocity was applied recently."""

    ZERO_VELOCITY = 1 << 3
    """The vehicle was detected to be stationary and a zero-velocity update was applied."""

    NON_HOLONOMIC = 1 << 4
    """The vehicle-motion (no side slip) constraint was applied."""

    BAROMETER = 1 << 5
    """A barometric height measurement was applied recently."""

    OUTLIER_GATED = 1 << 6
    """A measurement disagreed with the state and was down-weighted. Treat the epoch as suspect."""


class EgoUncertaintySE3(ArrayMixin):
    """The uncertainty of an ego state in SE3, as one sigma per component.

    Position uncertainty is in the global frame and velocity uncertainty in the ego frame, each
    matching the frame its quantity is expressed in on :class:`~py123d.datatypes.vehicle_state.
    ego_state.EgoStateSE3`. Only the marginal standard deviations are kept, not the full
    covariance: they are what a consumer selecting or weighting frames actually uses, and they
    cost nine numbers a frame instead of a hundred and thirty-six.
    """

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(
        self,
        position_sigma: Vector3D,
        velocity_sigma: Vector3D,
        orientation_sigma: Vector3D,
    ):
        """Initialize an :class:`EgoUncertaintySE3` instance.

        :param position_sigma: One-sigma position uncertainty in the global frame, meters.
        :param velocity_sigma: One-sigma velocity uncertainty in the ego frame, m/s.
        :param orientation_sigma: One-sigma roll/pitch/yaw uncertainty, radians.
        """
        array = np.zeros(len(EgoUncertaintySE3Index), dtype=np.float64)
        array[EgoUncertaintySE3Index.POSITION_SIGMA_3D] = position_sigma.array
        array[EgoUncertaintySE3Index.VELOCITY_SIGMA_3D] = velocity_sigma.array
        array[EgoUncertaintySE3Index.ORIENTATION_SIGMA_3D] = orientation_sigma.array
        self._array = array

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> EgoUncertaintySE3:
        """Create an :class:`EgoUncertaintySE3` from a NumPy array of shape (9,).

        :param array: The array, indexed by :class:`EgoUncertaintySE3Index`.
        :param copy: Whether to copy the array data.
        :return: An :class:`EgoUncertaintySE3` instance.
        """
        assert array.ndim == 1
        assert array.shape[0] == len(EgoUncertaintySE3Index)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @property
    def position_sigma(self) -> Vector3D:
        """One-sigma position uncertainty in the global frame, meters."""
        return Vector3D.from_array(self._array[EgoUncertaintySE3Index.POSITION_SIGMA_3D], copy=False)

    @property
    def velocity_sigma(self) -> Vector3D:
        """One-sigma velocity uncertainty in the ego frame, m/s."""
        return Vector3D.from_array(self._array[EgoUncertaintySE3Index.VELOCITY_SIGMA_3D], copy=False)

    @property
    def orientation_sigma(self) -> Vector3D:
        """One-sigma roll/pitch/yaw uncertainty, radians."""
        return Vector3D.from_array(self._array[EgoUncertaintySE3Index.ORIENTATION_SIGMA_3D], copy=False)

    @property
    def horizontal_position_sigma(self) -> float:
        """One-sigma horizontal position uncertainty, meters."""
        return float(
            np.hypot(*self._array[EgoUncertaintySE3Index.POSITION_SIGMA_X : EgoUncertaintySE3Index.POSITION_SIGMA_Z])
        )

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """NumPy array representation of shape (9,), indexed by :class:`EgoUncertaintySE3Index`."""
        return self._array

    def __repr__(self) -> str:
        return (
            f"EgoUncertaintySE3(position_sigma={self.position_sigma}, "
            f"velocity_sigma={self.velocity_sigma}, orientation_sigma={self.orientation_sigma})"
        )
