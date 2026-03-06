from __future__ import annotations

from py123d.common.utils.enums import SerialIntEnum

# TODO @DanielDauner:
# - Consider adding types for intersections or other layers.


class MapLayer(SerialIntEnum):
    """Enum for different map layers (i.e. object types) in a map."""

    LANE = 0
    """Lanes (surface)."""

    LANE_GROUP = 1
    """Lane groups (surface)."""

    INTERSECTION = 2
    """Intersections (surface)."""

    CROSSWALK = 3
    """Crosswalks (surface)."""

    WALKWAY = 4
    """Walkways (surface)."""

    CARPARK = 5
    """Carparks (surface)."""

    GENERIC_DRIVABLE = 6
    """Generic drivable (surface)."""

    STOP_ZONE = 7
    """Stop zones (surface)."""

    ROAD_EDGE = 8
    """Road edges (lines)."""

    ROAD_LINE = 9
    """Road lines (lines)."""


class LaneType(SerialIntEnum):
    """Enum for different lane types.

    Notes
    -----
    The lane types follow the Waymo specification [1]_.

    References
    ----------
    .. [1] https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L147

    """

    UNDEFINED = 0
    """Undefined lane type."""

    FREEWAY = 1
    """Freeway lane type."""

    SURFACE_STREET = 2
    """Surface street lane type, i.e. regular lanes for vehicles."""

    BIKE_LANE = 3
    """Bike lane type."""

    BUS_LANE = 4
    """Bus lane type."""


class IntersectionType(SerialIntEnum):
    """Enum for different intersection types.

    Notes
    -----
    The intersection types follow the nuPlan specification [1]_.

    References
    ----------
    .. [1] https://github.com/motional/nuplan-devkit/blob/master/nuplan/common/maps/maps_datatypes.py#L85

    """

    DEFAULT = 0
    """Default intersection type with no specific features."""

    TRAFFIC_LIGHT = 1
    """Intersection controlled by a traffic light."""

    STOP_SIGN = 2
    """Intersection controlled by a stop sign."""

    LANE_BRANCH = 3
    """Intersection where lanes branch off."""

    LANE_MERGE = 4
    """Intersection where lanes merge."""

    PASS_THROUGH = 5
    """Intersection where lanes pass through without branching or merging."""

    UNKNOWN = 6
    """Unknown intersection type."""


class StopZoneType(SerialIntEnum):
    """Enum for different stop zone types."""

    UNKNOWN = 0
    """Unknown stop zone type."""

    TRAFFIC_LIGHT = 1
    """Stop zone controlled by a traffic light."""

    STOP_SIGN = 2
    """Stop zone controlled by a stop sign."""

    YIELD_SIGN = 3
    """Stop zone controlled by a yield sign."""

    PEDESTRIAN_CROSSING = 4
    """Stop zone controlled by a pedestrian crossing."""

    TURN_STOP = 5
    """Stop zone for turning vehicles."""


class RoadEdgeType(SerialIntEnum):
    """Enum for different road edge types.

    Notes
    -----
    The road edge types follow the Waymo specification [1]_.

    References
    ----------
    .. [1] https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto#L188
    """

    UNKNOWN = 0
    """Unknown road edge type."""

    ROAD_EDGE_BOUNDARY = 1
    """Physical road boundary that doesn't have traffic on the other side."""

    ROAD_EDGE_MEDIAN = 2
    """Physical road boundary that separates the car from other traffic."""


class RoadLineType(SerialIntEnum):
    """Enum for different road line types.

    Notes
    -----
    The road line types follow the Argoverse 2 specification [1]_.

    References
    ----------
    .. [1] https://github.com/argoverse/av2-api/blob/6b22766247eda941cb1953d6a58e8d5631c561da/src/av2/map/lane_segment.py#L33
    """

    NONE = 0
    """No painted line is present."""

    UNKNOWN = 1
    """Unknown or unclassified painted line type."""

    DASH_SOLID_YELLOW = 2
    """Yellow line with dashed marking on one side and solid on the other."""

    DASH_SOLID_WHITE = 3
    """White line with dashed marking on one side and solid on the other."""

    DASHED_WHITE = 4
    """White dashed line marking."""

    DASHED_YELLOW = 5
    """Yellow dashed line marking."""

    DOUBLE_SOLID_YELLOW = 6
    """Double yellow solid line marking."""

    DOUBLE_SOLID_WHITE = 7
    """Double white solid line marking."""

    DOUBLE_DASH_YELLOW = 8
    """Double yellow dashed line marking."""

    DOUBLE_DASH_WHITE = 9
    """Double white dashed line marking."""

    SOLID_YELLOW = 10
    """Single solid yellow line marking."""

    SOLID_WHITE = 11
    """Single solid white line marking."""

    SOLID_DASH_WHITE = 12
    """Single solid white line with dashed marking on one side."""

    SOLID_DASH_YELLOW = 13
    """Single solid yellow line with dashed marking on one side."""

    SOLID_BLUE = 14
    """Single solid blue line marking."""
