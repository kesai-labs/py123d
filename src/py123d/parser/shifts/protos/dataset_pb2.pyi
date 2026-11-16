from . import geom_pb2 as _geom_pb2
from . import map_pb2 as _map_pb2
from . import tags_pb2 as _tags_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActorType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    kVehicle: _ClassVar[ActorType]
    kPedestrian: _ClassVar[ActorType]
kVehicle: ActorType
kPedestrian: ActorType

class PedestrianTrack(_message.Message):
    __slots__ = ("track_id", "position", "dimensions", "linear_velocity")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    LINEAR_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    track_id: int
    position: _geom_pb2.Vector3
    dimensions: _geom_pb2.Vector3
    linear_velocity: _geom_pb2.Vector3
    def __init__(self, track_id: _Optional[int] = ..., position: _Optional[_Union[_geom_pb2.Vector3, _Mapping]] = ..., dimensions: _Optional[_Union[_geom_pb2.Vector3, _Mapping]] = ..., linear_velocity: _Optional[_Union[_geom_pb2.Vector3, _Mapping]] = ...) -> None: ...

class VehicleTrack(_message.Message):
    __slots__ = ("track_id", "position", "dimensions", "linear_velocity", "linear_acceleration", "yaw")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    LINEAR_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    LINEAR_ACCELERATION_FIELD_NUMBER: _ClassVar[int]
    YAW_FIELD_NUMBER: _ClassVar[int]
    track_id: int
    position: _geom_pb2.Vector3
    dimensions: _geom_pb2.Vector3
    linear_velocity: _geom_pb2.Vector3
    linear_acceleration: _geom_pb2.Vector3
    yaw: float
    def __init__(self, track_id: _Optional[int] = ..., position: _Optional[_Union[_geom_pb2.Vector3, _Mapping]] = ..., dimensions: _Optional[_Union[_geom_pb2.Vector3, _Mapping]] = ..., linear_velocity: _Optional[_Union[_geom_pb2.Vector3, _Mapping]] = ..., linear_acceleration: _Optional[_Union[_geom_pb2.Vector3, _Mapping]] = ..., yaw: _Optional[float] = ...) -> None: ...

class VehicleTracks(_message.Message):
    __slots__ = ("tracks",)
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    tracks: _containers.RepeatedCompositeFieldContainer[VehicleTrack]
    def __init__(self, tracks: _Optional[_Iterable[_Union[VehicleTrack, _Mapping]]] = ...) -> None: ...

class PedestrianTracks(_message.Message):
    __slots__ = ("tracks",)
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    tracks: _containers.RepeatedCompositeFieldContainer[PedestrianTrack]
    def __init__(self, tracks: _Optional[_Iterable[_Union[PedestrianTrack, _Mapping]]] = ...) -> None: ...

class PredictionRequest(_message.Message):
    __slots__ = ("track_id", "trajectory_tags")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_TAGS_FIELD_NUMBER: _ClassVar[int]
    track_id: int
    trajectory_tags: _containers.RepeatedScalarFieldContainer[_tags_pb2.TrajectoryTag]
    def __init__(self, track_id: _Optional[int] = ..., trajectory_tags: _Optional[_Iterable[_Union[_tags_pb2.TrajectoryTag, str]]] = ...) -> None: ...

class Scene(_message.Message):
    __slots__ = ("id", "past_vehicle_tracks", "past_pedestrian_tracks", "past_ego_track", "future_vehicle_tracks", "future_pedestrian_tracks", "future_ego_track", "prediction_requests", "path_graph", "traffic_lights", "scene_tags")
    ID_FIELD_NUMBER: _ClassVar[int]
    PAST_VEHICLE_TRACKS_FIELD_NUMBER: _ClassVar[int]
    PAST_PEDESTRIAN_TRACKS_FIELD_NUMBER: _ClassVar[int]
    PAST_EGO_TRACK_FIELD_NUMBER: _ClassVar[int]
    FUTURE_VEHICLE_TRACKS_FIELD_NUMBER: _ClassVar[int]
    FUTURE_PEDESTRIAN_TRACKS_FIELD_NUMBER: _ClassVar[int]
    FUTURE_EGO_TRACK_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    PATH_GRAPH_FIELD_NUMBER: _ClassVar[int]
    TRAFFIC_LIGHTS_FIELD_NUMBER: _ClassVar[int]
    SCENE_TAGS_FIELD_NUMBER: _ClassVar[int]
    id: str
    past_vehicle_tracks: _containers.RepeatedCompositeFieldContainer[VehicleTracks]
    past_pedestrian_tracks: _containers.RepeatedCompositeFieldContainer[PedestrianTracks]
    past_ego_track: _containers.RepeatedCompositeFieldContainer[VehicleTrack]
    future_vehicle_tracks: _containers.RepeatedCompositeFieldContainer[VehicleTracks]
    future_pedestrian_tracks: _containers.RepeatedCompositeFieldContainer[PedestrianTracks]
    future_ego_track: _containers.RepeatedCompositeFieldContainer[VehicleTrack]
    prediction_requests: _containers.RepeatedCompositeFieldContainer[PredictionRequest]
    path_graph: _map_pb2.PathGraph
    traffic_lights: _containers.RepeatedCompositeFieldContainer[_map_pb2.TrafficLightSections]
    scene_tags: _tags_pb2.SceneTags
    def __init__(self, id: _Optional[str] = ..., past_vehicle_tracks: _Optional[_Iterable[_Union[VehicleTracks, _Mapping]]] = ..., past_pedestrian_tracks: _Optional[_Iterable[_Union[PedestrianTracks, _Mapping]]] = ..., past_ego_track: _Optional[_Iterable[_Union[VehicleTrack, _Mapping]]] = ..., future_vehicle_tracks: _Optional[_Iterable[_Union[VehicleTracks, _Mapping]]] = ..., future_pedestrian_tracks: _Optional[_Iterable[_Union[PedestrianTracks, _Mapping]]] = ..., future_ego_track: _Optional[_Iterable[_Union[VehicleTrack, _Mapping]]] = ..., prediction_requests: _Optional[_Iterable[_Union[PredictionRequest, _Mapping]]] = ..., path_graph: _Optional[_Union[_map_pb2.PathGraph, _Mapping]] = ..., traffic_lights: _Optional[_Iterable[_Union[_map_pb2.TrafficLightSections, _Mapping]]] = ..., scene_tags: _Optional[_Union[_tags_pb2.SceneTags, _Mapping]] = ...) -> None: ...
