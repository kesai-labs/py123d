from __future__ import annotations

import importlib
from typing import Any, Dict, Type

from py123d.datatypes.detections.box_detection_label import BOX_DETECTION_LABEL_REGISTRY, BoxDetectionLabel
from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata


class BoxDetectionMetadata(AbstractMetadata):
    __slots__ = ("_box_detection_label_class",)

    def __init__(self, box_detection_label_class: Type[BoxDetectionLabel]) -> None:
        """Initialize a BoxDetectionLabelMetadata instance.

        :param box_detection_label_class: The dataset-specific BoxDetectionLabel enum class.
        """
        self._box_detection_label_class = box_detection_label_class

    @property
    def box_detection_label_class(self) -> Type[BoxDetectionLabel]:
        """The dataset-specific :class:`~py123d.conversion.registry.BoxDetectionLabel` enum class."""
        return self._box_detection_label_class

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> BoxDetectionMetadata:
        """Create a BoxDetectionMetadata instance from a dictionary.

        :param data_dict: Dictionary containing the metadata fields.
        :raises ValueError: If the label class name is not found in the registry and cannot be imported.
        :return: A BoxDetectionMetadata instance.
        """
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
        return {
            "box_detection_label_class": f"{cls.__module__}.{cls.__qualname__}",
        }

    def __repr__(self) -> str:
        return f"BoxDetectionLabelMetadata(box_detection_label_class={self._box_detection_label_class.__name__})"
