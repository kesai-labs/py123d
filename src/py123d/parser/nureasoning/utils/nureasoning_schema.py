import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---- Static map ------------------------------------------------------------


@dataclass
class Lane:
    id: int
    lane_type: int
    polygon: List[List[float]]
    lane_group: Optional[int] = None
    lane_index: Optional[int] = None
    speed_limit: Optional[float] = None
    centerline: Optional[List[List[float]]] = None


@dataclass
class MapBoundary:
    id: List[int]
    type: str
    geometry: List[List[float]]  # [[x, y, z], ...]


@dataclass
class Crosswalk:
    id: int
    intersection_id: List[int]
    geometry: List[List[float]]  # [[x, y, z], ...]


@dataclass
class Intersection:
    id: int
    intersection_type: int
    boundary_id: List[int]
    geometry: List[List[float]]  # [[x, y, z], ...]


@dataclass
class StopPolygon:
    id: int
    geometry: List[List[float]]  # [[x, y, z], ...]


@dataclass
class RoadBlock:
    id: int
    lane_group_ids: List[int]
    geometry: List[List[float]]  # [[x, y, z], ...]


@dataclass
class TrafficLight:
    id: int
    geometry: List[List[float]]  # [[x, y, z], ...]
    light_face_control_type: Optional[str] = None


@dataclass
class LaneConnector:
    id: int
    geometry: List[List[float]]  # [[x, y, z], ...]
    lane_group_connector_id: Optional[int] = None
    intersection_id: Optional[int] = None
    speed_limit: Optional[float] = None
    lane_connector_index: Optional[int] = None


@dataclass
class BaselinePath:
    id: Optional[int]
    geometry: List[List[float]]  # [[x, y, z], ...]
    lane_id: Optional[int] = None
    lane_group_id: Optional[int] = None
    road_segment_id: Optional[int] = None
    lane_connector_id: Optional[int] = None


@dataclass
class nuReasoningStaticMap:
    lanes: List[Lane] = field(default_factory=list)
    baseline_paths: List[BaselinePath] = field(default_factory=list)
    boundaries: List[MapBoundary] = field(default_factory=list)
    crosswalks: List[Crosswalk] = field(default_factory=list)
    intersections: List[Intersection] = field(default_factory=list)
    stop_polygons: List[StopPolygon] = field(default_factory=list)
    road_blocks: List[RoadBlock] = field(default_factory=list)
    traffic_lights: List[TrafficLight] = field(default_factory=list)
    lane_connectors: List[LaneConnector] = field(default_factory=list)


# ---- Frame-level -----------------------------------------------------------


@dataclass
class TrafficLightState:
    id: int
    state: str  # "red" | "green" | "yellow" | "off" | "unknown"
    raw_color: Optional[str] = None
    source: Optional[str] = None  # "lane_connector" | "roadblock"
    lane_connector_id: Optional[int] = None
    roadblock_id: Optional[int] = None


@dataclass
class ObjectAnnotation:
    track_token: str
    category: str
    pose: Dict[str, float]  # {"x":..., "y":..., "z":..., "yaw":...}
    velocity: Dict[str, float]  # {"vx":..., "vy":..., "vz":...}
    dimensions: Dict[str, float]  # {"l":..., "w":..., "h":...}


@dataclass
class EgoState:
    pose: Dict[str, float]  # {"x","y","z", "yaw", "qw","qx","qy","qz"}
    velocity: Dict[str, float]  # ego frame
    acceleration: Dict[str, float]
    dimensions: Optional[Dict[str, float]] = None
    trajectory_history: Optional[List[List[float]]] = None
    trajectory_future: Optional[List[List[float]]] = None


@dataclass
class CameraCalibration:
    intrinsic: List[List[float]]
    sensor2lidar_translation: List[float]
    sensor2lidar_rotation: List[float]
    width: int
    height: int


@dataclass
class CameraPaths:
    front: str
    front_left: str
    front_right: str
    left: str
    right: str
    back: str
    back_left: str
    back_right: str


@dataclass
class LidarData:
    point_cloud_path: str


@dataclass
class Sensors:
    cameras: CameraPaths
    lidar: LidarData


@dataclass
class MissionGoal:
    command: str  # "LANE_FOLLOW", "LEFT_LANE_CHANGE", "TURN_RIGHT", ...
    route_path: List[List[float]]  # anchor path points as [[x, y], ...] in map coords


@dataclass
class Annotations:
    objects: List[ObjectAnnotation] = field(default_factory=list)
    traffic_light_states: List[TrafficLightState] = field(default_factory=list)


# ---- Frame-level container -------------------------------------------------


@dataclass
class nuReasoningFrame:
    token: str
    frame_index: int  # index in frames array; can be derived if needed
    timestamp_us: int
    relative_time_s: float

    ego_state: str  # Path to ego_state/{timestamp}.pkl
    sensors: Sensors
    annotations: str  # Path to annotations/{timestamp}.pkl

    reasoning: str  # Path to reasoning/{timestamp}.json
    mission_goal: Optional[MissionGoal] = None


# ---- Clip-level ------------------------------------------------------------


@dataclass
class nuReasoningClip:
    clip_token: str
    clip_location: str
    log_name: str
    scenario_type: str
    start_timestamp_us: int
    end_timestamp_us: int
    frame_rate_hz: float
    total_frames: int

    map_annotation: str  # Path to map file
    frames: List[nuReasoningFrame] = field(default_factory=list)
    camera_calibrations: Dict[str, CameraCalibration] = field(default_factory=dict)


# ---- Pickle loading --------------------------------------------------------

# The nuReasoning pickles were serialized by an upstream pipeline whose schema lived in a
# top-level module named ``data_schema`` (the devkit also aliases ``data_schema_v0`` and ships
# the classes as ``nureasoning.common.schema``). The classes are vendored here, so we resolve
# these module names to this module at unpickle time instead of requiring them to be importable.
_SCHEMA_MODULES = {"data_schema", "data_schema_v0", "nureasoning.common.schema", __name__}

_SCHEMA_CLASSES: Dict[str, type] = {
    schema_class.__name__: schema_class
    for schema_class in (
        Lane,
        MapBoundary,
        Crosswalk,
        Intersection,
        StopPolygon,
        RoadBlock,
        TrafficLight,
        LaneConnector,
        BaselinePath,
        nuReasoningStaticMap,
        TrafficLightState,
        ObjectAnnotation,
        EgoState,
        CameraCalibration,
        CameraPaths,
        LidarData,
        Sensors,
        MissionGoal,
        Annotations,
        nuReasoningFrame,
        nuReasoningClip,
    )
}


class _NuReasoningUnpickler(pickle.Unpickler):
    """Resolves the schema classes of a nuReasoning pickle, and nothing else.

    The pickles are downloaded data. Unpickling can import and call arbitrary objects, so only the
    vendored schema dataclasses are allowed (the released pickles reference no other globals).
    """

    def find_class(self, module: str, name: str) -> Any:
        if module in _SCHEMA_MODULES and name in _SCHEMA_CLASSES:
            return _SCHEMA_CLASSES[name]
        raise pickle.UnpicklingError(f"Unexpected global in nuReasoning pickle: {module}.{name}")


def load_schema_pickle(path: Union[str, Path]) -> Any:
    """Load a nuReasoning schema pickle, resolving its classes to the ones vendored in this module."""
    with open(Path(path), "rb") as f:
        result = _NuReasoningUnpickler(f).load()
    return result
