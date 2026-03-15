from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from py123d.common.utils.mixin import ArrayMixin, indexed_array_repr
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, CameraID, CameraModel, register_camera_metadata
from py123d.geometry import PoseSE3
from py123d.geometry.geometry_index import Point3DIndex


class FisheyeMEIDistortionIndex(IntEnum):
    """Indexing for fisheye MEI distortion parameters."""

    K1 = 0
    """Radial distortion coefficient k1."""

    K2 = 1
    """Radial distortion coefficient k2."""

    P1 = 2
    """Tangential distortion coefficient p1."""

    P2 = 3
    """Tangential distortion coefficient p2."""


class FisheyeMEIDistortion(ArrayMixin):
    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, k1: float, k2: float, p1: float, p2: float) -> None:
        """Initialize the fisheye MEI distortion parameters.

        :param k1: Radial distortion coefficient k1.
        :param k2: Radial distortion coefficient k2.
        :param p1: Tangential distortion coefficient p1.
        :param p2: Tangential distortion coefficient p2.
        """
        array = np.zeros(len(FisheyeMEIDistortionIndex), dtype=np.float64)
        array[FisheyeMEIDistortionIndex.K1] = k1
        array[FisheyeMEIDistortionIndex.K2] = k2
        array[FisheyeMEIDistortionIndex.P1] = p1
        array[FisheyeMEIDistortionIndex.P2] = p2
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> FisheyeMEIDistortion:
        """Creates a :class:`FisheyeMEIDistortion` instance from a NumPy array,
            indexing according to :class:`FisheyeMEIDistortionIndex`.

        :param array: Input array containing distortion parameters.
        :param copy: Whether to copy the array data, defaults to True.
        :return: A new instance of :class:`FisheyeMEIDistortion`.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(FisheyeMEIDistortionIndex)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Underlying NumPy array of distortion parameters, indexed by :class:`FisheyeMEIDistortionIndex`."""
        return self._array

    @property
    def k1(self) -> float:
        """Radial distortion coefficient k1."""
        return self._array[FisheyeMEIDistortionIndex.K1]

    @property
    def k2(self) -> float:
        """Radial distortion coefficient k2."""
        return self._array[FisheyeMEIDistortionIndex.K2]

    @property
    def p1(self) -> float:
        """Tangential distortion coefficient p1."""
        return self._array[FisheyeMEIDistortionIndex.P1]

    @property
    def p2(self) -> float:
        """Tangential distortion coefficient p2."""
        return self._array[FisheyeMEIDistortionIndex.P2]

    def __repr__(self) -> str:
        """String representation of :class:`FisheyeMEIDistortion`."""
        return indexed_array_repr(self, FisheyeMEIDistortionIndex)


class FisheyeMEIProjectionIndex(IntEnum):
    """Indexing for fisheye MEI projection parameters."""

    GAMMA1 = 0
    """Generalized focal length gamma1."""

    GAMMA2 = 1
    """Generalized focal length gamma2."""

    U0 = 2
    """Principal point x-coordinate."""

    V0 = 3
    """Principal point y-coordinate."""


class FisheyeMEIProjection(ArrayMixin):
    """Fisheye MEI projection parameters."""

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(self, gamma1: float, gamma2: float, u0: float, v0: float) -> None:
        """Initialize the fisheye MEI projection parameters.

        :param gamma1: Generalized focal length gamma1.
        :param gamma2: Generalized focal length gamma2.
        :param u0: Principal point x-coordinate.
        :param v0: Principal point y-coordinate.
        """
        array = np.zeros(len(FisheyeMEIProjectionIndex), dtype=np.float64)
        array[FisheyeMEIProjectionIndex.GAMMA1] = gamma1
        array[FisheyeMEIProjectionIndex.GAMMA2] = gamma2
        array[FisheyeMEIProjectionIndex.U0] = u0
        array[FisheyeMEIProjectionIndex.V0] = v0
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> FisheyeMEIProjection:
        """Intializes a :class:`FisheyeMEIProjection` from a NumPy array,
            indexing according to :class:`FisheyeMEIProjectionIndex`.

        :param array: Input array containing projection parameters.
        :param copy: Whether to copy the array data, defaults to True.
        :return: A new instance of :class:`FisheyeMEIProjection`.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(FisheyeMEIProjectionIndex)
        instance = object.__new__(cls)
        setattr(instance, "_array", array.copy() if copy else array)
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Underlying NumPy array of projection parameters, indexed by :class:`FisheyeMEIProjectionIndex`."""
        return self._array

    @property
    def gamma1(self) -> float:
        """Generalized focal length gamma1."""
        return self._array[FisheyeMEIProjectionIndex.GAMMA1]

    @property
    def gamma2(self) -> float:
        """Generalized focal length gamma2."""
        return self._array[FisheyeMEIProjectionIndex.GAMMA2]

    @property
    def u0(self) -> float:
        """Principal point x-coordinate."""
        return self._array[FisheyeMEIProjectionIndex.U0]

    @property
    def v0(self) -> float:
        """Principal point y-coordinate."""
        return self._array[FisheyeMEIProjectionIndex.V0]

    def __repr__(self) -> str:
        """String representation of :class:`FisheyeMEIProjection`."""
        return indexed_array_repr(self, FisheyeMEIProjectionIndex)


@register_camera_metadata(CameraModel.FISHEYE_MEI)
class FisheyeMEICameraMetadata(BaseCameraMetadata):
    """Metadata for a fisheye MEI camera."""

    __slots__ = (
        "_camera_name",
        "_camera_id",
        "_mirror_parameter",
        "_distortion",
        "_projection",
        "_width",
        "_height",
        "_camera_to_imu_se3",
    )

    def __init__(
        self,
        camera_name: str,
        camera_id: CameraID,
        mirror_parameter: Optional[float],
        distortion: Optional[FisheyeMEIDistortion],
        projection: Optional[FisheyeMEIProjection],
        width: int,
        height: int,
        camera_to_imu_se3: PoseSE3,
    ) -> None:
        """Initialize the fisheye MEI camera metadata.

        :param camera_name: Name of the fisheye MEI camera, according to the dataset naming convention.
        :param camera_id: ID of the fisheye MEI camera.
        :param mirror_parameter: Mirror parameter of the camera model.
        :param distortion: Distortion parameters of the camera.
        :param projection: Projection parameters of the camera.
        :param width: Width of the camera image in pixels.
        :param height: Height of the camera image in pixels.
        :param camera_to_imu_se3: Static extrinsic pose of the fisheye MEI camera.
        """
        self._camera_name = camera_name
        self._camera_id = camera_id
        self._mirror_parameter = mirror_parameter
        self._distortion = distortion
        self._projection = projection
        self._width = width
        self._height = height
        self._camera_to_imu_se3 = camera_to_imu_se3

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> FisheyeMEICameraMetadata:
        """Create a :class:`FisheyeMEICameraMetadata` instance from a dictionary.

        :param data_dict: Dictionary containing camera metadata.
        :return: A new instance of :class:`FisheyeMEICameraMetadata`.
        """
        _distortion = (
            FisheyeMEIDistortion.from_list(data_dict["distortion"]) if data_dict["distortion"] is not None else None
        )
        _projection = (
            FisheyeMEIProjection.from_list(data_dict["projection"]) if data_dict["projection"] is not None else None
        )
        return FisheyeMEICameraMetadata(
            camera_name=data_dict["camera_name"],
            camera_id=CameraID(data_dict["camera_id"]),
            mirror_parameter=data_dict["mirror_parameter"],
            distortion=_distortion,
            projection=_projection,
            width=data_dict["width"],
            height=data_dict["height"],
            camera_to_imu_se3=PoseSE3.from_list(data_dict["camera_to_imu_se3"]),
        )

    @property
    def camera_model(self) -> CameraModel:
        """The projection model of this camera."""
        return CameraModel.FISHEYE_MEI

    @property
    def camera_name(self) -> str:
        """The name of the fisheye MEI camera, according to the dataset naming convention."""
        return self._camera_name

    @property
    def camera_id(self) -> CameraID:
        """The ID of the fisheye MEI camera."""
        return self._camera_id

    @property
    def mirror_parameter(self) -> Optional[float]:
        """The mirror parameter of the fisheye MEI camera."""
        return self._mirror_parameter

    @property
    def distortion(self) -> Optional[FisheyeMEIDistortion]:
        """The distortion parameters of the fisheye MEI camera, if available."""
        return self._distortion

    @property
    def projection(self) -> Optional[FisheyeMEIProjection]:
        """The projection parameters of the fisheye MEI camera, if available."""
        return self._projection

    @property
    def width(self) -> int:
        """The width of the fisheye MEI camera image in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """The height of the fisheye MEI camera image in pixels."""
        return self._height

    @property
    def camera_to_imu_se3(self) -> PoseSE3:
        """The static extrinsic pose of the fisheye MEI camera."""
        return self._camera_to_imu_se3

    def to_dict(self) -> Dict[str, Any]:
        """Converts the :class:`FisheyeMEICameraMetadata` instance to a Python dictionary.

        :return: A dictionary representation of the camera metadata.
        """
        data_dict: Dict[str, Any] = {}
        data_dict["camera_model"] = self.camera_model.serialize()
        data_dict["camera_name"] = self._camera_name
        data_dict["camera_id"] = int(self.camera_id)
        data_dict["mirror_parameter"] = self._mirror_parameter
        data_dict["distortion"] = self._distortion.array.tolist() if self._distortion is not None else None
        data_dict["projection"] = self._projection.array.tolist() if self._projection is not None else None
        data_dict["width"] = self._width
        data_dict["height"] = self._height
        data_dict["camera_to_imu_se3"] = self._camera_to_imu_se3.to_list()
        return data_dict

    def cam2image(self, points_3d: npt.NDArray[np.float64]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """camera coordinate to image plane"""
        assert points_3d.ndim == 2
        assert points_3d.shape[1] == len(Point3DIndex)
        assert self.mirror_parameter is not None

        norm = np.linalg.norm(points_3d, axis=1)

        x = points_3d[:, 0] / norm
        y = points_3d[:, 1] / norm
        z = points_3d[:, 2] / norm

        x /= z + self.mirror_parameter
        y /= z + self.mirror_parameter

        if self.distortion is not None:
            k1 = self.distortion.k1
            k2 = self.distortion.k2
        else:
            k1 = k2 = 0.0

        if self.projection is not None:
            gamma1 = self.projection.gamma1
            gamma2 = self.projection.gamma2
            u0 = self.projection.u0
            v0 = self.projection.v0
        else:
            gamma1 = gamma2 = 1.0
            u0 = v0 = 0.0

        ro2 = x * x + y * y
        x *= 1 + k1 * ro2 + k2 * ro2 * ro2
        y *= 1 + k1 * ro2 + k2 * ro2 * ro2

        x = gamma1 * x + u0
        y = gamma2 * y + v0

        return x, y, norm * points_3d[:, 2] / np.abs(points_3d[:, 2])
