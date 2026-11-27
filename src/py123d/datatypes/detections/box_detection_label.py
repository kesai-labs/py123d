from __future__ import annotations

import abc
import importlib
from typing import Type

from py123d.common.utils.enums import SerialIntEnum

BOX_DETECTION_LABEL_REGISTRY = {}


def register_box_detection_label(enum_class):
    """Decorator to register a BoxDetectionLabel enum class."""
    BOX_DETECTION_LABEL_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


def resolve_box_detection_label_class(qualified_name: str) -> Type["BoxDetectionLabel"]:
    """Resolve a :class:`BoxDetectionLabel` subclass from a registry name or fully qualified path."""
    label_class: Type[BoxDetectionLabel]
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
    return label_class


class BoxDetectionLabel(SerialIntEnum):
    """Base class for all box detection label enums."""

    @abc.abstractmethod
    def to_default(self) -> DefaultBoxDetectionLabel:
        """Convert to the default box detection label."""


@register_box_detection_label
class DefaultBoxDetectionLabel(BoxDetectionLabel):
    """Default box detection labels used in 123D. Common labels across datasets."""

    # Vehicles
    EGO = 0
    VEHICLE = 1
    TRAIN = 2

    # Vulnerable Road Users
    TWO_WHEELER = 3
    PERSON = 4
    ANIMAL = 5

    # Traffic Control
    TRAFFIC_SIGN = 6
    TRAFFIC_CONE = 7
    TRAFFIC_LIGHT = 8

    # Other Objects
    BARRIER = 9
    GENERIC_OBJECT = 10

    # Miscellaneous, that may not fit into the above categories
    OTHER = 11

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        return self


@register_box_detection_label
class AnonymizationBoxDetectionLabel(BoxDetectionLabel):
    """Labels of the image regions that an anonymization step obscures, stored as 2D boxes."""

    FACE = 0
    LICENSE_PLATE = 1

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass. A face or a license plate is a part of an object, not a person or a vehicle."""
        return DefaultBoxDetectionLabel.OTHER
