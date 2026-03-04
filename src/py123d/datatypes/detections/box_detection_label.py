from __future__ import annotations

import abc

from py123d.common.utils.enums import SerialIntEnum

BOX_DETECTION_LABEL_REGISTRY = {}


def register_box_detection_label(enum_class):
    """Decorator to register a BoxDetectionLabel enum class."""
    BOX_DETECTION_LABEL_REGISTRY[enum_class.__name__] = enum_class
    return enum_class


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
    BICYCLE = 3
    PERSON = 4
    ANIMAL = 5

    # Traffic Control
    TRAFFIC_SIGN = 6
    TRAFFIC_CONE = 7
    TRAFFIC_LIGHT = 8

    # Other Obstacles
    BARRIER = 9
    GENERIC_OBJECT = 10

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        return self
