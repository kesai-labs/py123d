import abc
from abc import abstractmethod
from typing import Optional, Union

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.time.timestamp import Timestamp


class ModalityType(SerialIntEnum):
    """Enum for modality types."""

    CUSTOM = 0
    EGO_STATE_SE3 = 1
    BOX_DETECTIONS_SE3 = 2
    TRAFFIC_LIGHT_DETECTIONS = 3
    LIDAR = 4
    CAMERA = 5

    CAMERA_SEMANTIC = 6
    """Per-pixel semantic segmentation image, pixel-aligned to a sibling :attr:`CAMERA` stream."""

    CAMERA_INSTANCE = 7
    """Per-pixel panoptic/instance segmentation image, pixel-aligned to a sibling :attr:`CAMERA` stream."""

    RADAR = 8
    """Radar point cloud, potentially with additional attributes like velocity, radar cross section, etc."""

    CAMERA_DEPTH = 9
    """Per-pixel metric depth image, pixel-aligned to a sibling :attr:`CAMERA` stream."""

    IMU = 10
    """Inertial measurement unit stream: gyroscope, accelerometer, optional fused orientation."""

    GNSS = 11
    """Global navigation satellite system fixes: geodetic position and fix quality."""

    BAROMETER = 12
    """Barometric pressure stream, with optional derived altitude and environmental readings."""

    MAGNETOMETER = 13
    """Magnetic field stream."""

    ROUTE_POSITION = 14
    """Ego's arc-length position on the log's route polyline (``route.arrow``), derived
    by the log writer from ego odometry or a provided route — no parser emits it."""


class BaseModalityMetadata(BaseMetadata):
    """Base class for modality metadata."""

    __slots__ = ()

    @property
    @abstractmethod
    def modality_type(self) -> ModalityType:
        """Returns the type of the modality that this metadata describes."""

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """Optional identifier for the modality, e.g. sensor ID for sensor modalities. Can be a string or a SerialIntEnum."""
        return None

    @property
    def modality_key(self) -> str:
        """Returns a unique key for this modality, combining type and id if applicable."""
        return get_modality_key(self.modality_type, self.modality_id)

    @property
    def has_arrival_time(self) -> bool:
        """Whether the modality's Arrow file stores a per-row arrival time (see
        :attr:`BaseModality.arrival_timestamp`). False unless a sensor metadata class sets it."""
        return False


class BaseModality(abc.ABC):
    """Abstract base class for modality data."""

    __slots__ = ()

    @property
    @abstractmethod
    def timestamp(self) -> Timestamp:
        """Returns the timestamp associated with this modality data, if available."""

    @property
    @abstractmethod
    def metadata(self) -> BaseModalityMetadata:
        """Returns the metadata associated with this modality data."""

    @property
    def arrival_timestamp(self) -> Optional[Timestamp]:
        """The time the recording system received this data, on the recorder's clock, if known.

        :attr:`timestamp` is the measurement time. The arrival time is later by the sensor's
        transport and processing latency, and a replay of a log "as live" needs it. It is
        stored for that purpose only: synchronization, timestamp lookups and scene
        construction always use :attr:`timestamp`. None for modalities and datasets that do
        not record it.
        """
        return None

    @property
    def modality_type(self) -> ModalityType:
        """Convenience property to access the modality type directly from the modality data."""
        return self.metadata.modality_type

    @property
    def modality_id(self) -> Optional[Union[str, SerialIntEnum]]:
        """Convenience property to access the modality id directly from the modality data."""
        return self.metadata.modality_id

    @property
    def modality_key(self) -> str:
        """Convenience property to access the modality key directly from the modality data."""
        return self.metadata.modality_key


def get_modality_type_from_key(modality_key: str) -> ModalityType:
    modality_type_str = modality_key.split(".", maxsplit=1)[0] if "." in modality_key else modality_key
    return ModalityType.deserialize(modality_type_str)


def get_modality_id_from_key(
    modality_key: str, id_enum: Optional[SerialIntEnum] = None
) -> Optional[Union[str, SerialIntEnum]]:
    _modality_id: Optional[Union[str, SerialIntEnum]] = None
    if "." in modality_key:
        id_str = modality_key.split(".")[1]
        if id_enum is not None:
            try:
                _modality_id = id_enum.from_arbitrary(id_str)
            except ValueError:
                _modality_id = None
    return _modality_id


def get_modality_key(
    modality_type: Union[str, ModalityType],
    modality_id: Optional[Union[str, SerialIntEnum]] = None,
) -> str:
    _modality_type_string = modality_type.serialize() if isinstance(modality_type, ModalityType) else modality_type
    _modality_id = modality_id.serialize() if isinstance(modality_id, SerialIntEnum) else modality_id
    if _modality_id is None:
        _modality_key = _modality_type_string
    else:
        _modality_key = f"{_modality_type_string}.{_modality_id}"
    return _modality_key
