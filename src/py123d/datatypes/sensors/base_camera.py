from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Type, TypeVar, Union

_T = TypeVar("_T")

import numpy as np
import numpy.typing as npt

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3


class CameraModel(SerialIntEnum):
    """Enumeration of camera projection models."""

    PINHOLE = 0
    """Standard pinhole camera model."""

    FISHEYE_MEI = 1
    """Fisheye camera using the MEI (mirror) model."""


class CameraChannelType(SerialIntEnum):
    """Enumeration of camera channel types."""

    RGB = 0
    GRAYSCALE = 1


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
    def camera_id(self) -> SerialIntEnum:
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
        """Returns the type of the modality that this metadata describes."""
        return ModalityType.CAMERA

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """Returns the camera ID as the modality ID."""
        return self.camera_id

    @property
    def aspect_ratio(self) -> float:
        """The aspect ratio (width / height) of the camera."""
        return self.width / self.height


class Camera(BaseModality):
    """A camera observation: image, extrinsic pose, timestamp, and model-specific metadata."""

    __slots__ = ("_metadata", "_image", "_extrinsic", "_timestamp")

    def __init__(
        self,
        metadata: BaseCameraMetadata,
        image: npt.NDArray[np.uint8],
        extrinsic: PoseSE3,
        timestamp: Timestamp,
    ) -> None:
        """Initialize a Camera instance.

        :param metadata: The camera metadata (determines the camera model).
        :param image: The image captured by the camera.
        :param extrinsic: The extrinsic pose of the camera.
        :param timestamp: The timestamp of the image capture.
        """
        self._metadata = metadata
        self._image = image
        self._extrinsic = extrinsic
        self._timestamp = timestamp

    @property
    def timestamp(self) -> Timestamp:
        """The :class:`~py123d.datatypes.time.Timestamp` of the image capture."""
        return self._timestamp

    @property
    def metadata(self) -> BaseCameraMetadata:
        """The :class:`BaseCameraMetadata` associated with the camera."""
        return self._metadata

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """The image captured by the camera, as a numpy array."""
        return self._image

    @property
    def extrinsic(self) -> PoseSE3:
        """The extrinsic :class:`~py123d.geometry.PoseSE3` of the camera, relative to the ego vehicle frame."""
        return self._extrinsic
