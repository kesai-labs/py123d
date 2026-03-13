from __future__ import annotations

import importlib
from typing import Any, Dict, Type

from py123d.datatypes.detections.box_detection_label import BOX_DETECTION_LABEL_REGISTRY, BoxDetectionLabel
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
        qualified_name = data_dict["box_detection_label_class"]

        # Backward compat: plain class name -> registry lookup
        if qualified_name in BOX_DETECTION_LABEL_REGISTRY:
            label_class = BOX_DETECTION_LABEL_REGISTRY[qualified_name]
        elif "." in qualified_name:
            # Fully qualified path: dynamically import the module
            module_path, class_name = qualified_name.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                label_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Cannot import box detection label class: {qualified_name}") from e
        else:
            raise ValueError(f"Unknown box detection label class: {qualified_name}")

        return cls(box_detection_label_class=label_class)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata to a dictionary.

        :return: Dictionary with the fully qualified label class path.
        """
        cls = self._box_detection_label_class
        return {"box_detection_label_class": f"{cls.__module__}.{cls.__qualname__}"}

    def __repr__(self) -> str:
        return f"BoxDetectionsSE3Metadata(box_detection_label_class={self._box_detection_label_class.__name__})"
