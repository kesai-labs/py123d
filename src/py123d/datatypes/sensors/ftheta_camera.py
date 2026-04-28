from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import classproperty
from py123d.common.utils.mixin import ArrayMixin, indexed_array_repr
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, CameraID, CameraModel, register_camera_metadata
from py123d.geometry import PoseSE3


class FThetaIntrinsicsIndex(IntEnum):
    """Indexing for f-theta camera intrinsic parameters."""

    CX = 0
    """Principal point x-coordinate."""

    CY = 1
    """Principal point y-coordinate."""

    FW_POLY_0 = 2
    """Forward polynomial coefficient 0."""

    FW_POLY_1 = 3
    """Forward polynomial coefficient 1."""

    FW_POLY_2 = 4
    """Forward polynomial coefficient 2."""

    FW_POLY_3 = 5
    """Forward polynomial coefficient 3."""

    FW_POLY_4 = 6
    """Forward polynomial coefficient 4."""

    FW_POLY_5 = 7
    """Forward polynomial coefficient 5."""

    BW_POLY_0 = 8
    """Backward polynomial coefficient 0."""

    BW_POLY_1 = 9
    """Backward polynomial coefficient 1."""

    BW_POLY_2 = 10
    """Backward polynomial coefficient 2."""

    BW_POLY_3 = 11
    """Backward polynomial coefficient 3."""

    BW_POLY_4 = 12
    """Backward polynomial coefficient 4."""

    BW_POLY_5 = 13
    """Backward polynomial coefficient 5."""

    @classproperty
    def FW_POLY(cls) -> slice:
        """Slice for the forward polynomial coefficients."""
        return slice(cls.FW_POLY_0, cls.FW_POLY_5 + 1)

    @classproperty
    def BW_POLY(cls) -> slice:
        """Slice for the backward polynomial coefficients."""
        return slice(cls.BW_POLY_0, cls.BW_POLY_5 + 1)


class FThetaIntrinsics(ArrayMixin):
    """F-theta camera intrinsics with forward and backward polynomial distortion coefficients."""

    __slots__ = ("_array",)
    _array: npt.NDArray[np.float64]

    def __init__(
        self,
        cx: float,
        cy: float,
        fw_poly: npt.NDArray[np.float64],
        bw_poly: npt.NDArray[np.float64],
    ) -> None:
        """Initialize FThetaIntrinsics.

        :param cx: Principal point x-coordinate.
        :param cy: Principal point y-coordinate.
        :param fw_poly: Forward polynomial coefficients (6 values).
        :param bw_poly: Backward polynomial coefficients (6 values).
        """
        assert len(fw_poly) == 6, f"Expected 6 forward polynomial coefficients, got {len(fw_poly)}"
        assert len(bw_poly) == 6, f"Expected 6 backward polynomial coefficients, got {len(bw_poly)}"
        array = np.zeros(len(FThetaIntrinsicsIndex), dtype=np.float64)
        array[FThetaIntrinsicsIndex.CX] = cx
        array[FThetaIntrinsicsIndex.CY] = cy
        array[FThetaIntrinsicsIndex.FW_POLY] = fw_poly
        array[FThetaIntrinsicsIndex.BW_POLY] = bw_poly
        object.__setattr__(self, "_array", array)

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> FThetaIntrinsics:
        """Creates a FThetaIntrinsics from a numpy array, indexed by :class:`FThetaIntrinsicsIndex`.

        :param array: A 1D numpy array containing the intrinsic parameters.
        :param copy: Whether to copy the array, defaults to True
        :return: A :class:`FThetaIntrinsics` instance.
        """
        assert array.ndim == 1
        assert array.shape[-1] == len(FThetaIntrinsicsIndex)
        instance = object.__new__(cls)
        instance._array = array.copy() if copy else array
        return instance

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """A numpy array representation of the f-theta intrinsics, indexed by :class:`FThetaIntrinsicsIndex`."""
        return self._array

    @property
    def cx(self) -> float:
        """Principal point x-coordinate."""
        return self._array[FThetaIntrinsicsIndex.CX]

    @property
    def cy(self) -> float:
        """Principal point y-coordinate."""
        return self._array[FThetaIntrinsicsIndex.CY]

    @property
    def fw_poly(self) -> npt.NDArray[np.float64]:
        """Forward polynomial coefficients (6 values)."""
        return self._array[FThetaIntrinsicsIndex.FW_POLY]

    @property
    def bw_poly(self) -> npt.NDArray[np.float64]:
        """Backward polynomial coefficients (6 values)."""
        return self._array[FThetaIntrinsicsIndex.BW_POLY]

    def __repr__(self) -> str:
        """String representation of :class:`FThetaIntrinsics`."""
        return indexed_array_repr(self, FThetaIntrinsicsIndex)


@register_camera_metadata(CameraModel.FTHETA)
class FThetaCameraMetadata(BaseCameraMetadata):
    """Metadata for an f-theta polynomial camera."""

    __slots__ = (
        "_camera_name",
        "_camera_id",
        "_intrinsics",
        "_width",
        "_height",
        "_camera_to_imu_se3",
    )

    def __init__(
        self,
        camera_name: str,
        camera_id: CameraID,
        intrinsics: Optional[FThetaIntrinsics],
        width: int,
        height: int,
        camera_to_imu_se3: PoseSE3,
    ) -> None:
        """Initialize the f-theta camera metadata.

        :param camera_name: Name of the camera, according to the dataset naming convention.
        :param camera_id: ID of the camera.
        :param intrinsics: The :class:`FThetaIntrinsics` of the camera.
        :param width: Width of the camera image in pixels.
        :param height: Height of the camera image in pixels.
        :param camera_to_imu_se3: Static extrinsic pose of the camera relative to the IMU frame.
        """
        self._camera_name = camera_name
        self._camera_id = camera_id
        self._intrinsics = intrinsics
        self._width = width
        self._height = height
        self._camera_to_imu_se3 = camera_to_imu_se3

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> FThetaCameraMetadata:
        """Create a :class:`FThetaCameraMetadata` instance from a dictionary.

        :param data_dict: Dictionary containing camera metadata.
        :return: A new instance of :class:`FThetaCameraMetadata`.
        """
        _intrinsics = (
            FThetaIntrinsics.from_list(data_dict["intrinsics"]) if data_dict["intrinsics"] is not None else None
        )
        return FThetaCameraMetadata(
            camera_name=data_dict["camera_name"],
            camera_id=CameraID(data_dict["camera_id"]),
            intrinsics=_intrinsics,
            width=data_dict["width"],
            height=data_dict["height"],
            camera_to_imu_se3=PoseSE3.from_list(data_dict["camera_to_imu_se3"]),
        )

    @property
    def camera_model(self) -> CameraModel:
        """The projection model of this camera."""
        return CameraModel.FTHETA

    @property
    def camera_name(self) -> str:
        """The name of the f-theta camera, according to the dataset naming convention."""
        return self._camera_name

    @property
    def camera_id(self) -> CameraID:
        """The ID of the f-theta camera."""
        return self._camera_id

    @property
    def intrinsics(self) -> Optional[FThetaIntrinsics]:
        """The :class:`FThetaIntrinsics` of the f-theta camera, if available."""
        return self._intrinsics

    @property
    def width(self) -> int:
        """The width of the f-theta camera image in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """The height of the f-theta camera image in pixels."""
        return self._height

    @property
    def camera_to_imu_se3(self) -> PoseSE3:
        """The static extrinsic pose of the f-theta camera."""
        return self._camera_to_imu_se3

    def project_to_image(
        self,
        points_cam: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Project 3D points in camera frame to image pixel coordinates using the f-theta model.

        Uses the forward polynomial to map ray angle to pixel distance from principal point.

        :param points_cam: (N, 3) array of 3D points in the camera coordinate frame.
        :return: A tuple of (pixel_coords (N,2), in_fov_mask (N,), depth (N,)).
        :raises ValueError: If intrinsics are not set.
        """
        if self._intrinsics is None:
            raise ValueError("Cannot project: f-theta intrinsics not set.")

        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        depth = z.copy()

        r_xy = np.sqrt(x * x + y * y)
        theta = np.arctan2(r_xy, z)

        fw_poly = self._intrinsics.fw_poly
        powers = np.column_stack([theta**i for i in range(len(fw_poly))])
        r_px = powers @ fw_poly

        phi = np.arctan2(y, x)
        u = r_px * np.cos(phi) + self._intrinsics.cx
        v = r_px * np.sin(phi) + self._intrinsics.cy

        pixel_coords = np.column_stack([u, v])
        in_fov_mask = self._compute_in_fov_mask(pixel_coords, depth)
        result = (pixel_coords, in_fov_mask, depth)
        return result

    def _evaluate_bw_poly(self, r: float) -> float:
        """Evaluate the backward polynomial at pixel radius *r*.

        Uses the convention ``theta(r) = bw_0 + bw_1*r + bw_2*r^2 + bw_3*r^3 + bw_4*r^4``
        (bw_0 is typically 0, so theta(0) = 0).

        :param r: Pixel distance from the principal point. Must be non-negative.
        :raises ValueError: If *r* is negative.
        :raises ValueError: If the polynomial evaluates to a non-positive angle, which
            indicates an invalid calibration or out-of-range extrapolation.
        :return: The angle in radians corresponding to pixel radius *r*.
        """
        if r < 0.0:
            raise ValueError(f"Pixel radius must be non-negative, got {r}.")

        assert self._intrinsics is not None
        bw = self._intrinsics.bw_poly
        # Polynomial: theta(r) = bw_0 + bw_1*r + bw_2*r^2 + bw_3*r^3 + bw_4*r^4
        # Powers start at r^0, not r^1.
        powers = np.array([r**i for i in range(len(bw))])
        theta = float(np.dot(bw, powers))

        if theta <= 0.0:
            raise ValueError(
                f"Backward polynomial evaluated to a non-positive angle ({theta:.6f} rad) "
                f"at pixel radius r={r:.4f}. This indicates an invalid calibration or "
                f"out-of-range extrapolation beyond the calibrated field of view."
            )

        return theta

    @property
    def fov_x(self) -> Optional[float]:
        """The horizontal field of view (FOV) in radians, if available.

        Computed by evaluating the backward polynomial at the pixel distances from the
        principal point to the left and right image edges (at pixel-centre coordinates),
        then summing the two half-angles.

        The right-edge distance uses ``width - 1 - cx`` to stay within the last valid
        pixel centre rather than past the sensor boundary.
        """
        if self._intrinsics is None:
            return None
        r_left = self._intrinsics.cx
        r_right = (self._width - 1) - self._intrinsics.cx
        return self._evaluate_bw_poly(r_left) + self._evaluate_bw_poly(r_right)

    @property
    def fov_y(self) -> Optional[float]:
        """The vertical field of view (FOV) in radians, if available.

        Computed by evaluating the backward polynomial at the pixel distances from the
        principal point to the top and bottom image edges (at pixel-centre coordinates),
        then summing the two half-angles.

        The bottom-edge distance uses ``height - 1 - cy`` to stay within the last valid
        pixel centre rather than past the sensor boundary.
        """
        if self._intrinsics is None:
            return None
        r_top = self._intrinsics.cy
        r_bottom = (self._height - 1) - self._intrinsics.cy
        return self._evaluate_bw_poly(r_top) + self._evaluate_bw_poly(r_bottom)

    @property
    def angular_aspect_ratio(self) -> Optional[float]:
        """The angular aspect ratio (fov_x / fov_y), if available.

        For f-theta cameras, the angular aspect ratio differs from the pixel aspect ratio
        (width / height) because the polynomial projection is non-linear. Viewers that
        assume pinhole geometry should use this ratio instead of :attr:`aspect_ratio`.
        """
        fov_x = self.fov_x
        fov_y = self.fov_y
        if fov_x is None or fov_y is None or fov_y == 0.0:
            return None
        return fov_x / fov_y

    def to_dict(self) -> Dict[str, Any]:
        """Converts the :class:`FThetaCameraMetadata` instance to a Python dictionary.

        :return: A dictionary representation of the camera metadata.
        """
        data_dict: Dict[str, Any] = {}
        data_dict["camera_model"] = self.camera_model.serialize()
        data_dict["camera_name"] = self._camera_name
        data_dict["camera_id"] = int(self.camera_id)
        data_dict["intrinsics"] = self._intrinsics.array.tolist() if self._intrinsics is not None else None
        data_dict["width"] = self._width
        data_dict["height"] = self._height
        data_dict["camera_to_imu_se3"] = self._camera_to_imu_se3.to_list()
        return data_dict
