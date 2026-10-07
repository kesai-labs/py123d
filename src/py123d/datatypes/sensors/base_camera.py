from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

_T = TypeVar("_T")

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.sensors.camera_segmentation_label import (
    colorize_instance_label_map,
    colorize_semantic_label_map,
)
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3
from py123d.geometry.transform import abs_to_rel_points_3d_array


class CameraChannelType(SerialIntEnum):
    """Enumeration of camera channel types."""

    RGB = 0
    GRAYSCALE = 1
    SEMANTIC = 2
    """A single-channel, per-pixel semantic class-id label map (lossless integer image)."""
    INSTANCE = 3
    """A single-channel, per-pixel panoptic/instance label map (lossless integer image). For datasets
    that pack semantics into the instance id (e.g. KITTI-360 ``semanticId * 1000 + instanceId``) the
    raw packed value is stored verbatim; recover ``semantic = value // 1000``, ``instance = value % 1000``."""
    DEPTH = 4
    """A single-channel, per-pixel metric depth map, quantized to a lossless integer image (uint8 or
    uint16). The continuous depth in metres is recovered via
    :meth:`~py123d.datatypes.DepthCameraMetadata.decode_depth`, whose quantization
    contract (far plane and bit depth) is carried by the accompanying ``DepthCameraMetadata``."""


class CameraModel(SerialIntEnum):
    """Enumeration of camera projection models."""

    PINHOLE = 0
    """Standard pinhole camera model."""

    FISHEYE_MEI = 1
    """Fisheye camera using the MEI (mirror) model."""

    FTHETA = 2
    """F-theta polynomial camera model."""


class CameraID(SerialIntEnum):
    """Enumeration of camera IDs. These are unique within a sensor rig and can be used as modality IDs for camera metadata."""

    # Pinhole cameras
    # ------------------------------------------------------------------------------------------------------------------

    PCAM_F0 = 0
    """Front pinhole camera."""

    PCAM_F1 = 23
    """Second front pinhole camera, e.g. a high-resolution camera next to a surround-view ring."""

    PCAM_B0 = 1
    """Back pinhole camera."""

    PCAM_L0 = 2
    """Left pinhole camera, first from front to back."""

    PCAM_L1 = 3
    """Left pinhole camera, second from front to back."""

    PCAM_L2 = 4
    """Left pinhole camera, third from front to back."""

    PCAM_R0 = 5
    """Right pinhole camera, first from front to back."""

    PCAM_R1 = 6
    """Right pinhole camera, second from front to back."""

    PCAM_R2 = 7
    """Right pinhole camera, third from front to back."""

    PCAM_STEREO_L = 8
    """Left pinhole stereo camera."""

    PCAM_STEREO_R = 9
    """Right pinhole stereo camera."""

    PCAM_D0 = 19
    """Nadir (downward-looking) pinhole camera, e.g. a UAV's bottom camera."""

    # Fisheye MEI cameras
    # ------------------------------------------------------------------------------------------------------------------

    FMCAM_L = 10
    """Left-facing fisheye MEI camera."""

    FMCAM_R = 11
    """Right-facing fisheye MEI camera."""

    # F-theta cameras
    # ------------------------------------------------------------------------------------------------------------------

    FTCAM_F0 = 12
    """Front F-theta camera."""

    FTCAM_B0 = 20
    """Back F-theta camera."""

    FTCAM_TELE_F0 = 13
    """Front telephoto F-theta camera."""

    FTCAM_TELE_B0 = 18
    """Back telephoto F-theta camera."""

    FTCAM_L0 = 14
    """Left F-theta camera, first from front to back."""

    FTCAM_L1 = 15
    """Left F-theta camera, second from front to back."""

    FTCAM_R0 = 16
    """Right F-theta camera, first from front to back."""

    FTCAM_R1 = 17
    """Right F-theta camera, second from front to back."""

    FTCAM_L2 = 21
    """Left F-theta camera, third from front to back."""

    FTCAM_R2 = 22
    """Right F-theta camera, third from front to back."""


ALL_PINHOLE_CAMERA_IDS = [
    CameraID.PCAM_F0,
    CameraID.PCAM_F1,
    CameraID.PCAM_B0,
    CameraID.PCAM_L0,
    CameraID.PCAM_L1,
    CameraID.PCAM_L2,
    CameraID.PCAM_R0,
    CameraID.PCAM_R1,
    CameraID.PCAM_R2,
    CameraID.PCAM_STEREO_L,
    CameraID.PCAM_STEREO_R,
    CameraID.PCAM_D0,
]

ALL_FISHEYE_MEI_CAMERA_IDS = [
    CameraID.FMCAM_L,
    CameraID.FMCAM_R,
]

ALL_FTHETA_CAMERA_IDS = [
    CameraID.FTCAM_F0,
    CameraID.FTCAM_B0,
    CameraID.FTCAM_TELE_F0,
    CameraID.FTCAM_TELE_B0,
    CameraID.FTCAM_L0,
    CameraID.FTCAM_L1,
    CameraID.FTCAM_L2,
    CameraID.FTCAM_R0,
    CameraID.FTCAM_R1,
    CameraID.FTCAM_R2,
]

# ---------------------------------------------------------------------------
# Camera metadata registry (populated by subclasses via register_camera_metadata)
# ---------------------------------------------------------------------------

_CAMERA_METADATA_REGISTRY: Dict[CameraModel, Type[BaseCameraMetadata]] = {}


def register_camera_metadata(camera_model: CameraModel):
    """Class decorator that registers a BaseCameraMetadata subclass for a given CameraModel."""

    def decorator(cls: _T) -> _T:
        _CAMERA_METADATA_REGISTRY[camera_model] = cls  # type: ignore[assignment]
        return cls

    return decorator


def camera_metadata_from_dict(data_dict: Dict[str, Any]) -> BaseCameraMetadata:
    """Factory function: deserialize a camera metadata dict into the correct subclass.

    Reads the ``"camera_model"`` discriminator field and dispatches to the registered subclass.

    :param data_dict: A dictionary containing the camera metadata with a ``"camera_model"`` field.
    :return: A :class:`BaseCameraMetadata` subclass instance.
    """
    camera_model = CameraModel.from_arbitrary(data_dict["camera_model"])
    if camera_model not in _CAMERA_METADATA_REGISTRY:
        raise ValueError(f"No camera metadata class registered for camera model '{camera_model}'")
    metadata = _CAMERA_METADATA_REGISTRY[camera_model].from_dict(data_dict)
    assert isinstance(metadata, BaseCameraMetadata)
    return metadata


class BaseCameraMetadata(BaseModalityMetadata, abc.ABC):
    """Base class for camera metadata. Provides the shared interface for all camera models."""

    __slots__ = ()

    @property
    @abc.abstractmethod
    def camera_model(self) -> CameraModel:
        """The projection model of the camera."""

    @property
    @abc.abstractmethod
    def camera_id(self) -> CameraID:
        """The camera ID, unique within a sensor rig."""

    @property
    @abc.abstractmethod
    def camera_name(self) -> str:
        """The camera name, according to the dataset naming convention."""

    @property
    @abc.abstractmethod
    def camera_to_imu_se3(self) -> PoseSE3:
        """The static extrinsic pose of the camera relative to the IMU frame."""

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """The width of the camera image in pixels."""

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """The height of the camera image in pixels."""

    @property
    def channel_type(self) -> CameraChannelType:
        """The channel type of the camera image. Defaults to RGB."""
        return CameraChannelType.RGB

    @property
    def modality_type(self) -> ModalityType:
        """Returns the type of the modality that this metadata describes.

        The channel type drives the modality: a :attr:`CameraChannelType.SEMANTIC` camera is a
        per-pixel semantic segmentation stream (:attr:`ModalityType.CAMERA_SEMANTIC`), a
        :attr:`CameraChannelType.INSTANCE` camera a per-pixel panoptic/instance stream
        (:attr:`ModalityType.CAMERA_INSTANCE`), and a :attr:`CameraChannelType.DEPTH` camera a
        per-pixel metric depth stream (:attr:`ModalityType.CAMERA_DEPTH`), each written to its own
        Arrow file so it sits alongside — and never collides with — the RGB camera that shares its
        :attr:`camera_id`. All other channel types are regular :attr:`ModalityType.CAMERA`.
        """
        if self.channel_type == CameraChannelType.SEMANTIC:
            modality_type = ModalityType.CAMERA_SEMANTIC
        elif self.channel_type == CameraChannelType.INSTANCE:
            modality_type = ModalityType.CAMERA_INSTANCE
        elif self.channel_type == CameraChannelType.DEPTH:
            modality_type = ModalityType.CAMERA_DEPTH
        else:
            modality_type = ModalityType.CAMERA
        return modality_type

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """Returns the camera ID as the modality ID."""
        return self.camera_id

    @property
    def aspect_ratio(self) -> float:
        """The aspect ratio (width / height) of the camera."""
        return self.width / self.height

    def _compute_in_fov_mask(
        self,
        pixel_coords: npt.NDArray[np.float64],
        depth: npt.NDArray[np.float64],
        eps: float = 1e-6,
    ) -> npt.NDArray[np.bool_]:
        """Compute a boolean mask for points in front of the camera and within image bounds.

        :param pixel_coords: (N, 2) array of (u, v) pixel coordinates.
        :param depth: (N,) array of depths (positive = in front of camera).
        :param eps: minimum depth threshold, defaults to 1e-6.
        :return: (N,) boolean mask.
        """
        mask = depth > eps
        mask = mask & (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < self.width)
        mask = mask & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < self.height)
        return mask

    @abc.abstractmethod
    def project_to_image(
        self,
        points_cam: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Project 3D points in camera frame to image pixel coordinates.

        :param points_cam: (N, 3) array of 3D points in the camera coordinate frame.
        :return: A tuple of (pixel_coords (N,2), in_fov_mask (N,), depth (N,)).
        """


class Camera(BaseModality):
    """A camera observation: image, extrinsic pose, timestamp, and model-specific metadata."""

    __slots__ = ("_metadata", "_image", "_camera_to_global_se3", "_timestamp", "_exposure_factor", "_arrival_timestamp")

    def __init__(
        self,
        metadata: BaseCameraMetadata,
        image: npt.NDArray[np.uint8],
        camera_to_global_se3: PoseSE3,
        timestamp: Timestamp,
        exposure_factor: Optional[float] = None,
        arrival_timestamp: Optional[Timestamp] = None,
    ) -> None:
        """Initialize a Camera instance.

        :param metadata: The camera metadata (determines the camera model).
        :param image: The image captured by the camera.
        :param camera_to_global_se3: The extrinsic pose of the camera in global coordinates.
        :param timestamp: The timestamp of the image capture.
        :param exposure_factor: Per-frame exposure normalization gain applied upstream of
            the stored image, if known (see :attr:`exposure_factor`).
        :param arrival_timestamp: Optional time the recording system received the measurement
            (see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`).
        """
        self._metadata = metadata
        self._image = image
        self._camera_to_global_se3 = camera_to_global_se3
        self._timestamp = timestamp
        self._exposure_factor = exposure_factor
        self._arrival_timestamp = arrival_timestamp

    @property
    def timestamp(self) -> Timestamp:
        """The :class:`~py123d.datatypes.Timestamp` of the image capture."""
        return self._timestamp

    @property
    def exposure_factor(self) -> Optional[float]:
        """Per-frame exposure normalization gain applied upstream of the stored image, if known.

        Recording pipelines with auto-exposure normalization (e.g. Kesai) scale the linear
        sensor signal by this gain before encoding; dividing the linearized image by it
        recovers the absolute radiance scale. None when the dataset does not provide it.
        """
        return self._exposure_factor

    @property
    def metadata(self) -> BaseCameraMetadata:
        """The :class:`BaseCameraMetadata` associated with the camera."""
        return self._metadata

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this measurement, if recorded."""
        return self._arrival_timestamp

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """The image captured by the camera, as a numpy array."""
        return self._image

    @property
    def rgb_image(self) -> Optional[npt.NDArray[np.uint8]]:
        """A 3-channel ``(H, W, 3)`` uint8 RGB view of the image, regardless of channel type.

        ``RGB`` is returned as-is and ``GRAYSCALE`` is broadcast to three channels. A ``SEMANTIC`` label map is
        colorized with the canonical Cityscapes palette: each raw class id is mapped to its
        :class:`~py123d.datatypes.camera_segmentation_label.DefaultCameraSegmentationLabel` and then to a
        color. An ``INSTANCE`` label map is colorized by cycling a Tableau-20 palette over the raw ids
        (id ``0`` rendered black). A ``DEPTH`` raster is colorized with a TURBO ramp over the full
        ``[0, max_depth]`` range, so each modality can be displayed through the regular RGB camera path.
        """
        channel_type = self._metadata.channel_type
        if channel_type == CameraChannelType.RGB:
            result = self._image
        elif channel_type == CameraChannelType.GRAYSCALE:
            result = np.repeat(self._image[:, :, None], 3, axis=2)
        elif channel_type == CameraChannelType.SEMANTIC:
            # Only SegmentationCameraMetadata carries the taxonomy; access it duck-typed to avoid importing it
            # here (it imports this module, so a direct import would be circular).
            label_class = getattr(self._metadata, "segmentation_label_class", None)
            if label_class is None:
                raise ValueError("SEMANTIC camera is missing its segmentation_label_class; cannot colorize.")
            result = colorize_semantic_label_map(self._image, label_class)
        elif channel_type == CameraChannelType.INSTANCE:
            result = colorize_instance_label_map(self._image)
        elif channel_type == CameraChannelType.DEPTH:
            # Deferred import: depth_camera imports this module, so a top-level import would be circular.
            from py123d.datatypes.sensors.depth_camera import colorize_depth_map

            result = colorize_depth_map(self._image, max_raw=getattr(self._metadata, "max_raw", None))
            if getattr(self._metadata, "has_invalid", False):
                result[np.asarray(self._image) == 0] = 0
        else:
            raise ValueError(f"Unsupported channel type {channel_type} for RGB conversion.")
        return result

    @property
    def camera_to_global_se3(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the camera in global coordinates."""
        return self._camera_to_global_se3

    def project_points_global(
        self,
        points_global: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Project 3D points in global frame to image pixel coordinates.

        Convenience method that transforms points from global to camera frame,
        then delegates to :meth:`BaseCameraMetadata.project_to_image`.

        :param points_global: (N, 3) array of 3D points in global coordinates.
        :return: A tuple of:
            - pixel_coords: (N, 2) array of (u, v) pixel coordinates.
            - in_fov_mask: (N,) boolean mask.
            - depth: (N,) array of signed depths.
        """
        points_cam = abs_to_rel_points_3d_array(self._camera_to_global_se3, points_global)
        result = self._metadata.project_to_image(points_cam)
        return result
