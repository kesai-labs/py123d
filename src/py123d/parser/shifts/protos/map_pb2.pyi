from . import geom_pb2 as _geom_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MovementType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MOVEMENT_FORBIDDEN: _ClassVar[MovementType]
    MOVEMENT_ALLOWED_NO_PRIORITY: _ClassVar[MovementType]
    MOVEMENT_ALLOWED_WITH_PRIORITY: _ClassVar[MovementType]

class TrafficLightState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATE_UNKNOWN: _ClassVar[TrafficLightState]
    STATE_GREEN: _ClassVar[TrafficLightState]
    STATE_BLINKING_GREEN: _ClassVar[TrafficLightState]
    STATE_YELLOW: _ClassVar[TrafficLightState]
    STATE_RED: _ClassVar[TrafficLightState]
    STATE_RED_YELLOW: _ClassVar[TrafficLightState]
    STATE_NOT_WORKING: _ClassVar[TrafficLightState]
    STATE_ENABLED: _ClassVar[TrafficLightState]
    STATE_BLINKING_ENABLED: _ClassVar[TrafficLightState]
    STATE_DISABLED: _ClassVar[TrafficLightState]
    STATE_INVISIBLE: _ClassVar[TrafficLightState]
    STATE_BLINKING_RED: _ClassVar[TrafficLightState]
MOVEMENT_FORBIDDEN: MovementType
MOVEMENT_ALLOWED_NO_PRIORITY: MovementType
MOVEMENT_ALLOWED_WITH_PRIORITY: MovementType
STATE_UNKNOWN: TrafficLightState
STATE_GREEN: TrafficLightState
STATE_BLINKING_GREEN: TrafficLightState
STATE_YELLOW: TrafficLightState
STATE_RED: TrafficLightState
STATE_RED_YELLOW: TrafficLightState
STATE_NOT_WORKING: TrafficLightState
STATE_ENABLED: TrafficLightState
STATE_BLINKING_ENABLED: TrafficLightState
STATE_DISABLED: TrafficLightState
STATE_INVISIBLE: TrafficLightState
STATE_BLINKING_RED: TrafficLightState

class TrafficLightRule(_message.Message):
    __slots__ = ("sections", "movement_type")
    SECTIONS_FIELD_NUMBER: _ClassVar[int]
    MOVEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    sections: _containers.RepeatedCompositeFieldContainer[TrafficLightSection]
    movement_type: int
    def __init__(self, sections: _Optional[_Iterable[_Union[TrafficLightSection, _Mapping]]] = ..., movement_type: _Optional[int] = ...) -> None: ...

class Crosswalk(_message.Message):
    __slots__ = ("geometry", "control_rules")
    GEOMETRY_FIELD_NUMBER: _ClassVar[int]
    CONTROL_RULES_FIELD_NUMBER: _ClassVar[int]
    geometry: _geom_pb2.Polygon
    control_rules: _containers.RepeatedCompositeFieldContainer[TrafficLightRule]
    def __init__(self, geometry: _Optional[_Union[_geom_pb2.Polygon, _Mapping]] = ..., control_rules: _Optional[_Iterable[_Union[TrafficLightRule, _Mapping]]] = ...) -> None: ...

class RoadPolygon(_message.Message):
    __slots__ = ("geometry",)
    GEOMETRY_FIELD_NUMBER: _ClassVar[int]
    geometry: _geom_pb2.Polygon
    def __init__(self, geometry: _Optional[_Union[_geom_pb2.Polygon, _Mapping]] = ...) -> None: ...

class Lane(_message.Message):
    __slots__ = ("centers", "max_velocity", "gives_way_to_some_lane", "traffic_light_section_ids")
    CENTERS_FIELD_NUMBER: _ClassVar[int]
    MAX_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    GIVES_WAY_TO_SOME_LANE_FIELD_NUMBER: _ClassVar[int]
    TRAFFIC_LIGHT_SECTION_IDS_FIELD_NUMBER: _ClassVar[int]
    centers: _containers.RepeatedCompositeFieldContainer[_geom_pb2.Vector3]
    max_velocity: float
    gives_way_to_some_lane: bool
    traffic_light_section_ids: TrafficLightSectionIds
    def __init__(self, centers: _Optional[_Iterable[_Union[_geom_pb2.Vector3, _Mapping]]] = ..., max_velocity: _Optional[float] = ..., gives_way_to_some_lane: bool = ..., traffic_light_section_ids: _Optional[_Union[TrafficLightSectionIds, _Mapping]] = ...) -> None: ...

class PathGraph(_message.Message):
    __slots__ = ("lanes", "crosswalks", "road_polygons")
    LANES_FIELD_NUMBER: _ClassVar[int]
    CROSSWALKS_FIELD_NUMBER: _ClassVar[int]
    ROAD_POLYGONS_FIELD_NUMBER: _ClassVar[int]
    lanes: _containers.RepeatedCompositeFieldContainer[Lane]
    crosswalks: _containers.RepeatedCompositeFieldContainer[Crosswalk]
    road_polygons: _containers.RepeatedCompositeFieldContainer[RoadPolygon]
    def __init__(self, lanes: _Optional[_Iterable[_Union[Lane, _Mapping]]] = ..., crosswalks: _Optional[_Iterable[_Union[Crosswalk, _Mapping]]] = ..., road_polygons: _Optional[_Iterable[_Union[RoadPolygon, _Mapping]]] = ...) -> None: ...

class TrafficLightSection(_message.Message):
    __slots__ = ("id", "state")
    ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    id: int
    state: TrafficLightState
    def __init__(self, id: _Optional[int] = ..., state: _Optional[_Union[TrafficLightState, str]] = ...) -> None: ...

class TrafficLightSections(_message.Message):
    __slots__ = ("sections",)
    SECTIONS_FIELD_NUMBER: _ClassVar[int]
    sections: _containers.RepeatedCompositeFieldContainer[TrafficLightSection]
    def __init__(self, sections: _Optional[_Iterable[_Union[TrafficLightSection, _Mapping]]] = ...) -> None: ...

class TrafficLightSectionIds(_message.Message):
    __slots__ = ("left_section_id", "main_section_id", "right_section_id")
    LEFT_SECTION_ID_FIELD_NUMBER: _ClassVar[int]
    MAIN_SECTION_ID_FIELD_NUMBER: _ClassVar[int]
    RIGHT_SECTION_ID_FIELD_NUMBER: _ClassVar[int]
    left_section_id: int
    main_section_id: int
    right_section_id: int
    def __init__(self, left_section_id: _Optional[int] = ..., main_section_id: _Optional[int] = ..., right_section_id: _Optional[int] = ...) -> None: ...
