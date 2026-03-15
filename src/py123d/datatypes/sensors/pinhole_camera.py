from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum, classproperty
from py123d.common.utils.mixin import ArrayMixin, indexed_array_repr
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, CameraModel, register_camera_metadata
from py123d.geometry import PoseSE3


class PinholeCameraID(SerialIntEnum):
    """Enumeration of pinhole camera types."""

    PCAM_F0 = 0
    """Front camera."""

    PCAM_B0 = 1
    """Back camera."""

    PCAM_L0 = 2
    """Left camera, first from front to back."""

    PCAM_L1 = 3
    """Left camera, second from front to back."""

    PCAM_L2 = 4
    """Left camera, third from front to back."""

    PCAM_R0 = 5
    """Right camera, first from front to back."""

    PCAM_R1 = 6
    """Right camera, second from front to back."""

    PCAM_R2 = 7
    """Right camera, third from front to back."""

    PCAM_STEREO_L = 8
    """Left stereo camera."""

    PCAM_STEREO_R = 9
    """Right stereo camera."""


class PinholeIntrinsicsIndex(IntEnum):
    """Enumeration of pinhole camera intrinsic parameters."""

    FX = 0
    """Focal length in x direction."""

    FY = 1
    """Focal length in y direction."""

    CX = 2
    """Optical center x coordinate."""

    CY = 3
    """Optical center y coordinate."""

    SKEW = 4
    """Skew coefficient. Not used in most cases."""

    @classproperty
    def FX_MATRIX(cls) -> tuple[int, int]:
        """The index of the focal length in x direction (fx) in the 3x3 camera intrinsic matrix."""
        return (0, 0)

    @classproperty
    def FY_MATRIX(cls) -> tuple[int, int]:
        """The index of the focal length in y direction (fy) in the 3x3 camera intrinsic matrix."""
        return (1, 1)

    @classproperty
    def CX_MATRIX(cls) -> tuple[int, int]:
        """The index of the optical center x coordinate (cx) in the 3x3 camera intrinsic matrix."""
        return (0, 2)

    @classproperty
    def CY_MATRIX(cls) -> tuple[int, int]:
        """The index of the optical center y coordinate (cy) in the 3x3 camera intrinsic matrix."""
        return (1, 2)

    @classproperty
    def SKEW_MATRIX(cls) -> tuple[int, int]:
        """The index of the skew coefficient in the 3x3 camera intrinsic matrix. Not used in most cases."""
        return (0, 1)


class PinholeIntrinsics(ArrayMixin):
    """Pinhole camera intrinsics representation."""

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, fx: float, fy: float, cx: float, cy: float, skew: float = 0.0) -> None:
        """Initialize PinholeIntrinsics.

        :param fx: Focal length in x direction.
        :param fy: Focal length in y direction.
        :param cx: Optical center x coordinate.
        :param cy: Optical center y coordinate.
        :param skew: Skew coefficient. Not used in most cases, defaults to 0.0
        """
        array = np.zeros(len(PinholeIntrinsicsIndex), dtype=np.float64)
        array[PinholeIntrinsicsIndex.FX] = fx
        array[PinholeIntrinsicsIndex.FY] = fy
        array[PinholeIntrinsicsIndex.CX] = cx
        array[PinholeIntrinsicsIndex.CY] = cy
        array[PinholeIntrinsicsIndex.SKEW] = skew
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PinholeIntrinsics:
        """Creates a PinholeIntrinsics from a numpy array, indexed by :class:`PinholeIntrinsicsIndex`.

        :param array: A 1D numpy array containing the intrinsic parameters.
        :param copy: Whether to copy the array, defaults to True
        :return: A :class:`PinholeIntrinsics` instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(PinholeIntrinsicsIndex)
        instance = object.__new__(cls)
        setattr(instance, "_array", array.copy() if copy else array)
        return instance

    @classmethod
    def from_camera_matrix(cls, intrinsic: npt.NDArray[np.float64]) -> PinholeIntrinsics:
        """Create a PinholeIntrinsics from a 3x3 intrinsic matrix.

        :param intrinsic: A 3x3 numpy array representing the intrinsic matrix.
        :return: A :class:`PinholeIntrinsics` instance.
        """
        assert intrinsic.shape == (3, 3)
        fx = intrinsic[PinholeIntrinsicsIndex.FX_MATRIX]
        fy = intrinsic[PinholeIntrinsicsIndex.FY_MATRIX]
        cx = intrinsic[PinholeIntrinsicsIndex.CX_MATRIX]
        cy = intrinsic[PinholeIntrinsicsIndex.CY_MATRIX]
        skew = intrinsic[PinholeIntrinsicsIndex.SKEW_MATRIX]  # Not used in most cases.
        array = np.array([fx, fy, cx, cy, skew], dtype=np.float64)
        return cls.from_array(array, copy=False)

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """A numpy array representation of the pinhole intrinsics, indexed by :class:`PinholeIntrinsicsIndex`."""
        return self._array

    @property
    def fx(self) -> float:
        """Focal length in x direction."""
        return self._array[PinholeIntrinsicsIndex.FX]

    @property
    def fy(self) -> float:
        """Focal length in y direction."""
        return self._array[PinholeIntrinsicsIndex.FY]

    @property
    def cx(self) -> float:
        """Optical center x coordinate."""
        return self._array[PinholeIntrinsicsIndex.CX]

    @property
    def cy(self) -> float:
        """Optical center y coordinate."""
        return self._array[PinholeIntrinsicsIndex.CY]

    @property
    def skew(self) -> float:
        """Skew coefficient. Not used in most cases."""
        return self._array[PinholeIntrinsicsIndex.SKEW]

    @property
    def camera_matrix(self) -> npt.NDArray[np.float64]:
        """The 3x3 camera intrinsic matrix K."""
        K = np.eye(3, dtype=np.float64)
        K[PinholeIntrinsicsIndex.FX_MATRIX] = self.fx
        K[PinholeIntrinsicsIndex.FY_MATRIX] = self.fy
        K[PinholeIntrinsicsIndex.CX_MATRIX] = self.cx
        K[PinholeIntrinsicsIndex.CY_MATRIX] = self.cy
        K[PinholeIntrinsicsIndex.SKEW_MATRIX] = self.skew
        return K

    def __repr__(self) -> str:
        """String representation of :class:`PinholeIntrinsics`."""
        return indexed_array_repr(self, PinholeIntrinsicsIndex)


class PinholeDistortionIndex(IntEnum):
    """Enumeration of pinhole camera distortion parameters."""

    K1 = 0
    """Radial distortion coefficient k1."""

    K2 = 1
    """Radial distortion coefficient k2."""

    P1 = 2
    """Tangential distortion coefficient p1."""

    P2 = 3
    """Tangential distortion coefficient p2."""

    K3 = 4
    """Radial distortion coefficient k3."""


class PinholeDistortion(ArrayMixin):
    """Pinhole camera distortion representation."""

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, k1: float, k2: float, p1: float, p2: float, k3: float) -> None:
        """Initialize :class:`:PinholeDistortion`.

        :param k1: Radial distortion coefficient k1.
        :param k2: Radial distortion coefficient k2.
        :param p1: Tangential distortion coefficient p1.
        :param p2: Tangential distortion coefficient p2.
        :param k3: Radial distortion coefficient k3.
        """
        array = np.zeros(len(PinholeDistortionIndex), dtype=np.float64)
        array[PinholeDistortionIndex.K1] = k1
        array[PinholeDistortionIndex.K2] = k2
        array[PinholeDistortionIndex.P1] = p1
        array[PinholeDistortionIndex.P2] = p2
        array[PinholeDistortionIndex.K3] = k3
        setattr(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> PinholeDistortion:
        """Creates a PinholeDistortion from a numpy array, indexed by :class:`PinholeDistortionIndex`.

        :param array: A 1D numpy array containing the distortion parameters.
        :param copy: Whether to copy the array, defaults to True
        :return: A :class:`PinholeDistortion` instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(PinholeDistortionIndex)
        instance = object.__new__(cls)
        setattr(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """A numpy array representation of the pinhole distortion, indexed by :class:`PinholeDistortionIndex`."""
        return self._array

    @property
    def k1(self) -> float:
        """Radial distortion coefficient k1."""
        return self._array[PinholeDistortionIndex.K1]

    @property
    def k2(self) -> float:
        """Radial distortion coefficient k2."""
        return self._array[PinholeDistortionIndex.K2]

    @property
    def p1(self) -> float:
        """Tangential distortion coefficient p1."""
        return self._array[PinholeDistortionIndex.P1]

    @property
    def p2(self) -> float:
        """Tangential distortion coefficient p2."""
        return self._array[PinholeDistortionIndex.P2]

    @property
    def k3(self) -> float:
        """Radial distortion coefficient k3."""
        return self._array[PinholeDistortionIndex.K3]

    def __repr__(self) -> str:
        """String representation of :class:`PinholeDistortion`."""
        return indexed_array_repr(self, PinholeDistortionIndex)


@register_camera_metadata(CameraModel.PINHOLE)
class PinholeCameraMetadata(BaseCameraMetadata):
    """Static metadata for a pinhole camera, stored in a log."""

    __slots__ = (
        "_camera_name",
        "_camera_id",
        "_intrinsics",
        "_distortion",
        "_width",
        "_height",
        "_camera_to_imu_se3",
        "_is_undistorted",
    )

    def __init__(
        self,
        camera_name: str,
        camera_id: PinholeCameraID,
        intrinsics: Optional[PinholeIntrinsics],
        distortion: Optional[PinholeDistortion],
        width: int,
        height: int,
        camera_to_imu_se3: PoseSE3,
        is_undistorted: bool = False,
    ) -> None:
        """Initialize a :class:`PinholeCameraMetadata` instance.

        :param camera_name: The name of the pinhole camera, according to the dataset naming convention.
        :param camera_id: The :class:`PinholeCameraID` of the pinhole camera.
        :param intrinsics: The :class:`PinholeIntrinsics` of the pinhole camera.
        :param distortion: The :class:`PinholeDistortion` of the pinhole camera.
        :param width: The image width in pixels.
        :param height: The image height in pixels.
        :param camera_to_imu_se3: The camera-to-IMU extrinsic :class:`~py123d.geometry.PoseSE3` of the pinhole camera.
        :param is_undistorted: Whether the camera images are already undistorted, defaults to False.
        """
        self._camera_name = camera_name
        self._camera_id = camera_id
        self._intrinsics = intrinsics
        self._distortion = distortion
        self._width = width
        self._height = height
        self._camera_to_imu_se3 = camera_to_imu_se3
        self._is_undistorted = is_undistorted

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> PinholeCameraMetadata:
        """Create a :class:`PinholeCameraMetadata` from a dictionary.

        :param data_dict: A dictionary containing the metadata.
        :return: A PinholeCameraMetadata instance.
        """
        _intrinsics = (
            PinholeIntrinsics.from_list(data_dict["intrinsics"]) if data_dict["intrinsics"] is not None else None
        )
        _distortion = (
            PinholeDistortion.from_list(data_dict["distortion"]) if data_dict["distortion"] is not None else None
        )
        return PinholeCameraMetadata(
            camera_name=data_dict["camera_name"],
            camera_id=PinholeCameraID(data_dict["camera_id"]),
            intrinsics=_intrinsics,
            distortion=_distortion,
            width=data_dict["width"],
            height=data_dict["height"],
            camera_to_imu_se3=PoseSE3.from_list(data_dict["camera_to_imu_se3"]),
            is_undistorted=data_dict["is_undistorted"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the :class:`PinholeCameraMetadata` to a dictionary.

        :return: A dictionary representation of the PinholeCameraMetadata instance, with default Python types.
        """
        data_dict = {}
        data_dict["camera_model"] = self.camera_model.serialize()
        data_dict["camera_name"] = self.camera_name
        data_dict["camera_id"] = int(self.camera_id)
        data_dict["intrinsics"] = self.intrinsics.tolist() if self.intrinsics is not None else None
        data_dict["distortion"] = self.distortion.tolist() if self.distortion is not None else None
        data_dict["width"] = self.width
        data_dict["height"] = self.height
        data_dict["camera_to_imu_se3"] = self.camera_to_imu_se3.tolist()
        data_dict["is_undistorted"] = self.is_undistorted
        return data_dict

    @property
    def camera_model(self) -> CameraModel:
        """The projection model of this camera."""
        return CameraModel.PINHOLE

    @property
    def camera_name(self) -> str:
        """The name of the pinhole camera, according to the dataset naming convention."""
        return self._camera_name

    @property
    def camera_id(self) -> PinholeCameraID:
        """The :class:`PinholeCameraID` of the pinhole camera."""
        return self._camera_id

    @property
    def intrinsics(self) -> Optional[PinholeIntrinsics]:
        """The :class:`PinholeIntrinsics` of the pinhole camera."""
        return self._intrinsics

    @property
    def distortion(self) -> Optional[PinholeDistortion]:
        """The :class:`PinholeDistortion` of the pinhole camera."""
        return self._distortion

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self._height

    @property
    def camera_to_imu_se3(self) -> PoseSE3:
        """The camera-to-IMU extrinsic :class:`~py123d.geometry.PoseSE3` of the pinhole camera."""
        return self._camera_to_imu_se3

    @property
    def is_undistorted(self) -> bool:
        """Whether the camera images are already undistorted."""
        return self._is_undistorted

    @property
    def is_distorted(self) -> bool:
        """Whether the camera images are distorted."""
        return not self._is_undistorted

    @property
    def fov_x(self) -> Optional[float]:
        """The horizontal field of view (FOV) of the pinhole camera in radians, if available."""
        if self.intrinsics is not None:
            return 2 * np.arctan(self.width / (2 * self.intrinsics.fx))
        return None

    @property
    def fov_y(self) -> Optional[float]:
        """The vertical field of view (FOV) of the pinhole camera in radians, if available."""
        if self.intrinsics is not None:
            return 2 * np.arctan(self.height / (2 * self.intrinsics.fy))
        return None
