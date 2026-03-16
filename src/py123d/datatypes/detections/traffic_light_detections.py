from __future__ import annotations

from typing import Any, Dict, List, Optional

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.time.timestamp import Timestamp


class TrafficLightStatus(SerialIntEnum):
    """
    Enum for that represents the status of a traffic light.
    """

    GREEN = 0
    """Green light is on."""

    YELLOW = 1
    """Yellow light is on."""

    RED = 2
    """Red light is on."""

    OFF = 3
    """Traffic light is off."""

    UNKNOWN = 4
    """Traffic light status is unknown."""


class TrafficLightDetection:
    """
    Single traffic light detection of a lane, that includes the lane id and status (green, yellow, red, off, unknown).
    """

    __slots__ = ("_lane_id", "_status")

    def __init__(self, lane_id: int, status: TrafficLightStatus) -> None:
        """Initialize a TrafficLightDetection instance.

        :param lane_id: The lane id associated with the traffic light detection.
        :param status: The status of the traffic light (green, yellow, red, off, unknown).
        """

        self._lane_id = lane_id
        self._status = status

    @property
    def lane_id(self) -> int:
        """The lane id associated with the traffic light detection."""
        return self._lane_id

    @property
    def status(self) -> TrafficLightStatus:
        """The :class:`TrafficLightStatus` of the traffic light detection."""
        return self._status


class TrafficLightDetectionsMetadata(BaseModalityMetadata):
    @property
    def modality_type(self) -> ModalityType:
        """The modality name for this metadata, which is 'traffic_light_detections'."""
        return ModalityType.TRAFFIC_LIGHT_DETECTIONS

    def to_dict(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> TrafficLightDetectionsMetadata:
        return cls()

    def __repr__(self) -> str:
        return f"TrafficLightDetectionsMetadata(modality_type={self.modality_type})"


class TrafficLightDetections(BaseModality):
    """The TrafficLightDetections is a container for multiple traffic light detections.
    It provides methods to access individual detections as well as to retrieve a detection by lane id.
    The wrapper is used to read and write traffic light detections from/to logs.
    """

    __slots__ = ("_detections", "_timestamp", "_metadata")

    def __init__(
        self,
        detections: List[TrafficLightDetection],
        timestamp: Timestamp,
        metadata: TrafficLightDetectionsMetadata = TrafficLightDetectionsMetadata(),
    ) -> None:
        """Initialize a TrafficLightDetections instance.

        :param detections: List of :class:`TrafficLightDetection`.
        :param timestamp: The :class:`~py123d.datatypes.time.Timestamp` of the traffic light detections.
        :param metadata: The metadata for the traffic light detections.
        """
        self._detections = detections
        self._timestamp = timestamp
        self._metadata = metadata

    @property
    def detections(self) -> List[TrafficLightDetection]:
        """List of individual :class:`TrafficLightDetection`."""
        return self._detections

    @property
    def timestamp(self) -> Timestamp:
        """The :class:`~py123d.datatypes.time.Timestamp` of the traffic light detections."""
        return self._timestamp

    @property
    def metadata(self) -> TrafficLightDetectionsMetadata:
        """The metadata for the traffic light detections."""
        return self._metadata

    def __getitem__(self, index: int) -> TrafficLightDetection:
        """Retrieve a traffic light detection by its index.

        :param index: The index of the traffic light detection.
        :return: :class:`TrafficLightDetection` at the given index.
        """
        return self.detections[index]

    def __len__(self) -> int:
        """The number of traffic light detections in the wrapper."""
        return len(self.detections)

    def __iter__(self):
        """Iterator over the traffic light detections in the wrapper."""
        return iter(self.detections)

    def get_by_lane_id(self, lane_id: int) -> Optional[TrafficLightDetection]:
        """Retrieve a traffic light detection by its lane id.

        :param lane_id: The lane id to search for.
        :return: The traffic light detection for the given lane id, or None if not found.
        """
        for detection in self.detections:
            if int(detection.lane_id) == int(lane_id):
                return detection
        return None
