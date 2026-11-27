"""2D boxes in the pixels of one camera's stored images: one set of boxes per camera image.

The boxes belong to a camera stream, ``camera.<camera_id>``, and are stored in their own stream,
``box_detections_2d.<camera_id>``, with one row per image of the camera that was processed or annotated (also for
such images without boxes), at the image's time stamp. Uses: 2D object detections or annotations, or the regions of
the images that a processing step changed, e.g. the faces and license plates that an anonymization step obscured.

Each box has an object box (float pixels, ``x1, y1, x2, y2``), a label, a score and an origin; optionally a track
token and the pixel region that a processing step applied to (integers, ``x2`` and ``y2`` exclusive). The label is a
member of the stream's :class:`~py123d.datatypes.detections.box_detection_label.BoxDetectionLabel` enum, as for 3D
boxes. The origin is an index into the origin names of the stream's metadata, which also holds free information
about the producer of the boxes (``info``, e.g. a software version and its settings).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

from py123d.datatypes.detections.box_detection_label import (
    BoxDetectionLabel,
    DefaultBoxDetectionLabel,
    resolve_box_detection_label_class,
)
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType, get_modality_key
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.time.timestamp import Timestamp


class BoxDetections2DMetadata(BaseModalityMetadata):
    """Metadata of a 2D box stream: the camera it belongs to, the label class, the origin names, and producer
    information."""

    __slots__ = ("_camera_id", "_box_detection_label_class", "_origin_names", "_info")

    def __init__(
        self,
        camera_id: Union[CameraID, str],
        box_detection_label_class: Type[BoxDetectionLabel],
        origin_names: Sequence[str] = ("detection",),
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param camera_id: The id of the camera stream, a :class:`~py123d.datatypes.CameraID` or its name (the part
            after ``camera.`` in the modality key of the camera).
        :param box_detection_label_class: The :class:`BoxDetectionLabel` enum class of the labels of the boxes.
        :param origin_names: The names of the ways a box can be made (e.g. a detection, or a box filled in by a
            tracker); a box's origin is an index into them.
        :param info: Free information about the producer of the boxes, JSON-serializable (e.g. a software version,
            its settings and the checksums of its models).
        """
        self._camera_id = CameraID.from_arbitrary(camera_id)
        self._box_detection_label_class = box_detection_label_class
        self._origin_names: Tuple[str, ...] = tuple(str(name) for name in origin_names)
        self._info: Dict[str, Any] = dict(info) if info is not None else {}

    @property
    def modality_type(self) -> ModalityType:
        """The modality type, :attr:`ModalityType.BOX_DETECTIONS_2D`."""
        return ModalityType.BOX_DETECTIONS_2D

    @property
    def modality_id(self) -> CameraID:
        """The id of the camera the boxes belong to."""
        return self._camera_id

    @property
    def camera_id(self) -> CameraID:
        """The id of the camera the boxes belong to."""
        return self._camera_id

    @property
    def camera_key(self) -> str:
        """The modality key of the camera the boxes belong to, ``camera.<camera_id>``."""
        return get_modality_key(ModalityType.CAMERA, self._camera_id)

    @property
    def box_detection_label_class(self) -> Type[BoxDetectionLabel]:
        """The :class:`BoxDetectionLabel` enum class of the labels of the boxes."""
        return self._box_detection_label_class

    @property
    def origin_names(self) -> Tuple[str, ...]:
        """The origin names; a box's origin is an index into them."""
        return self._origin_names

    @property
    def info(self) -> Dict[str, Any]:
        """Free information about the producer of the boxes."""
        return self._info

    def to_dict(self) -> Dict[str, Any]:
        label_class = self._box_detection_label_class
        return {
            "camera_id": self._camera_id.serialize(),
            "box_detection_label_class": f"{label_class.__module__}.{label_class.__qualname__}",
            "origin_names": list(self._origin_names),
            "info": self._info,
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> BoxDetections2DMetadata:
        return cls(
            camera_id=data_dict["camera_id"],
            box_detection_label_class=resolve_box_detection_label_class(data_dict["box_detection_label_class"]),
            origin_names=data_dict.get("origin_names", ["detection"]),
            info=data_dict.get("info", {}),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoxDetections2DMetadata) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((self._camera_id, self._box_detection_label_class, self._origin_names))

    def __repr__(self) -> str:
        return (
            f"BoxDetections2DMetadata(camera_id={self._camera_id.serialize()!r}, "
            f"box_detection_label_class={self._box_detection_label_class.__name__}, "
            f"origin_names={list(self._origin_names)})"
        )


class BoxDetection2D:
    """One 2D box in the pixels of a camera image."""

    __slots__ = ("_box", "_label", "_score", "_origin", "_region", "_track_token")

    def __init__(
        self,
        box: Sequence[float],
        label: BoxDetectionLabel,
        score: float = 1.0,
        origin: int = 0,
        region: Optional[Sequence[int]] = None,
        track_token: Optional[str] = None,
    ) -> None:
        """
        :param box: The object box ``x1, y1, x2, y2`` in pixels of the stored image (float).
        :param label: The label, a member of the stream's label class.
        :param score: The score of the box.
        :param origin: How the box was made, an index into the stream's origin names.
        :param region: Optional pixel region ``x1, y1, x2, y2`` (integers, ``x2`` and ``y2`` exclusive) that a
            processing step applied to, e.g. the obscured pixels of an anonymization step.
        :param track_token: Optional track token of the box, the same for an object over the images of the camera.
        """
        if len(box) != 4:
            raise ValueError(f"A 2D box has 4 values (x1, y1, x2, y2), got {len(box)}.")
        self._box: Tuple[float, float, float, float] = tuple(float(v) for v in box)  # type: ignore[assignment]
        self._label = label
        self._score = float(score)
        self._origin = int(origin)
        if region is not None and len(region) != 4:
            raise ValueError(f"A region has 4 values (x1, y1, x2, y2), got {len(region)}.")
        self._region: Optional[Tuple[int, int, int, int]] = (
            None if region is None else tuple(int(v) for v in region)  # type: ignore[assignment]
        )
        self._track_token = track_token

    @property
    def box(self) -> Tuple[float, float, float, float]:
        """The object box ``x1, y1, x2, y2`` in pixels of the stored image."""
        return self._box

    @property
    def label(self) -> BoxDetectionLabel:
        """The :class:`BoxDetectionLabel`, a member of the stream's label class."""
        return self._label

    @property
    def default_label(self) -> DefaultBoxDetectionLabel:
        """The unified :class:`DefaultBoxDetectionLabel` corresponding to the box's label."""
        return self._label.to_default()

    @property
    def score(self) -> float:
        """The score of the box."""
        return self._score

    @property
    def origin(self) -> int:
        """How the box was made, an index into the stream's origin names."""
        return self._origin

    @property
    def region(self) -> Optional[Tuple[int, int, int, int]]:
        """The pixel region ``x1, y1, x2, y2`` (``x2``, ``y2`` exclusive) that a processing step applied to, if any."""
        return self._region

    @property
    def track_token(self) -> Optional[str]:
        """The track token of the box, the same for an object over the images of the camera, if any."""
        return self._track_token

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoxDetection2D) and (
            self._box,
            self._label,
            self._score,
            self._origin,
            self._region,
            self._track_token,
        ) == (other._box, other._label, other._score, other._origin, other._region, other._track_token)

    def __hash__(self) -> int:
        return hash((self._box, self._label, self._score, self._origin, self._region, self._track_token))

    def __repr__(self) -> str:
        return (
            f"BoxDetection2D(box={self._box}, label={self._label.name}, score={self._score}, origin={self._origin}, "
            f"region={self._region}, track_token={self._track_token!r})"
        )


class BoxDetections2D(BaseModality):
    """The 2D boxes of one camera image."""

    __slots__ = ("_box_detections", "_timestamp", "_metadata")

    def __init__(
        self,
        box_detections: List[BoxDetection2D],
        timestamp: Timestamp,
        metadata: BoxDetections2DMetadata,
    ) -> None:
        """
        :param box_detections: The boxes of the image (may be empty).
        :param timestamp: The time stamp of the camera image.
        :param metadata: The metadata of the stream.
        """
        self._box_detections = list(box_detections)
        self._timestamp = timestamp
        self._metadata = metadata

    @property
    def box_detections(self) -> List[BoxDetection2D]:
        """The boxes of the image."""
        return self._box_detections

    @property
    def timestamp(self) -> Timestamp:
        """The time stamp of the camera image."""
        return self._timestamp

    @property
    def metadata(self) -> BoxDetections2DMetadata:
        """The metadata of the stream."""
        return self._metadata

    def origin_name(self, box_detection: BoxDetection2D) -> str:
        """The origin name of a box of this set."""
        return self._metadata.origin_names[box_detection.origin]

    def __getitem__(self, index: int) -> BoxDetection2D:
        return self._box_detections[index]

    def __len__(self) -> int:
        return len(self._box_detections)

    def __iter__(self) -> Iterator[BoxDetection2D]:
        return iter(self._box_detections)

    def __repr__(self) -> str:
        return (
            f"BoxDetections2D(camera_id={self._metadata.camera_id.serialize()!r}, timestamp={self._timestamp}, "
            f"boxes={len(self)})"
        )
