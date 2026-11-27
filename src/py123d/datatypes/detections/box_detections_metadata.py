from __future__ import annotations

from typing import Any, Dict, Type

from py123d.datatypes.detections.box_detection_label import BoxDetectionLabel, resolve_box_detection_label_class
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata, ModalityType


class BoxDetectionsSE3Metadata(BaseModalityMetadata):
    __slots__ = ("_box_detection_label_class",)

    def __init__(self, box_detection_label_class: Type[BoxDetectionLabel]) -> None:
        self._box_detection_label_class = box_detection_label_class

    @property
    def box_detection_label_class(self) -> Type[BoxDetectionLabel]:
        """The dataset-specific :class:`~py123d.parser.registry.BoxDetectionLabel` enum class."""
        return self._box_detection_label_class

    @property
    def modality_type(self) -> ModalityType:
        """The modality name for this metadata, which is 'box_detections'."""
        return ModalityType.BOX_DETECTIONS_SE3

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> BoxDetectionsSE3Metadata:
        label_class = resolve_box_detection_label_class(data_dict["box_detection_label_class"])
        return cls(box_detection_label_class=label_class)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata to a dictionary.

        :return: Dictionary with the fully qualified label class path.
        """
        cls = self._box_detection_label_class
        return {"box_detection_label_class": f"{cls.__module__}.{cls.__qualname__}"}

    def __repr__(self) -> str:
        return f"BoxDetectionsSE3Metadata(box_detection_label_class={self._box_detection_label_class.__name__})"
