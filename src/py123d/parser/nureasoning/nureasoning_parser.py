from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    DynamicStateSE3,
    EgoStateSE3,
    LidarID,
    LogMetadata,
    MapMetadata,
    PinholeCameraMetadata,
    PinholeIntrinsics,
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
)
from py123d.datatypes.custom.custom_modality import CustomModality, CustomModalityMetadata
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, LidarMetadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, EulerAngles, PoseSE3, Vector3D
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.nureasoning.nureasoning_map_parser import NureasoningMapParser
from py123d.parser.nureasoning.utils.nureasoning_constants import (
    NUREASONING_BOX_DETECTIONS_SE3_METADATA,
    NUREASONING_CAMERA_ID_MAPPING,
    NUREASONING_CAMERA_KEY_MAPPING,
    NUREASONING_DATA_SPLITS,
    NUREASONING_DEFAULT_DT,
    NUREASONING_DEFAULT_EGO_DIMENSIONS,
    NUREASONING_DETECTION_NAME_DICT,
    NUREASONING_LIDAR_DICT,
    NUREASONING_LIDAR_SWEEP_DURATION_US,
    NUREASONING_REAR_AXLE_HEIGHT,
    NUREASONING_SPLIT_SOURCES,
    NUREASONING_TRAFFIC_STATUS_DICT,
)
from py123d.parser.nureasoning.utils.nureasoning_schema import Annotations, load_schema_pickle
from py123d.parser.registry import NureasoningBoxDetectionLabel

if TYPE_CHECKING:
    from py123d.parser.nureasoning.nureasoning_download import NureasoningDownloader

logger = logging.getLogger(__name__)

_METADATA_FILE = "metadata.json"
_MAP_FILE = "map.pkl"
_REASONING_QUESTIONS_FILE = "reasoning_questions.json"

# Annotation categories without a label, to warn about each of them only once per process.
_UNKNOWN_CATEGORIES: Set[str] = set()


class NureasoningParser(BaseDatasetParser):
    """Dataset parser for the nuReasoning dataset."""

    def __init__(
        self,
        splits: List[str],
        nureasoning_data_root: Optional[Union[Path, str]] = None,
        log_names: Optional[List[str]] = None,
        downloader: Optional["NureasoningDownloader"] = None,
    ) -> None:
        """Initializes the :class:`NureasoningParser`.

        :param splits: Splits to convert. Available: ``"nureasoning_train"``,
            ``"nureasoning_val"``, ``"nureasoning_test"`` and ``"nureasoning-mini_train"``
            (parts 1-3 of train).
        :param nureasoning_data_root: Root of an already-extracted dataset
            (``<root>/<split>/[<part>/]<clip>/...``). Required when ``downloader`` is
            ``None``; ignored otherwise.
        :param log_names: Clip names (``<log_name>_<keyframe_token>``) to convert.
            ``None`` converts every clip of the selected splits.
        :param downloader: Optional
            :class:`~py123d.parser.nureasoning.nureasoning_download.NureasoningDownloader`
            for streaming mode. When provided, the selected clips are materialized once
            (into a :class:`tempfile.TemporaryDirectory` that is deleted when this parser
            is garbage-collected, unless the downloader has its own ``output_dir``), and
            both log and map parsers read from it just like local mode. The downloader
            fetches the splits of this parser; the remaining clip selection
            (parts/log_names/num_logs) is driven by the downloader.
            ``nureasoning_data_root`` is not required in this mode.
        """
        for split in splits:
            assert split in NUREASONING_DATA_SPLITS, (
                f"Split {split} is not available. Available splits: {NUREASONING_DATA_SPLITS}"
            )

        self._splits = splits
        self._log_names = log_names
        self._downloader = downloader
        # Handle for the streaming temp dir (kept alive for the parser's lifetime). Set early so
        # __del__ is safe even if materialization below raises.
        self._stream_temp_dir_handle: Optional[tempfile.TemporaryDirectory] = None

        if downloader is not None:
            self._nureasoning_data_root = self._materialize_streaming_root(downloader)
        else:
            assert nureasoning_data_root is not None, (
                "`nureasoning_data_root` must be provided when `downloader` is None."
            )
            self._nureasoning_data_root = Path(nureasoning_data_root)

        self._split_log_path_pairs: List[Tuple[str, Path]] = self._collect_split_log_path_pairs()

    def _materialize_streaming_root(self, downloader: "NureasoningDownloader") -> Path:
        """Download the selected clips into a managed temp dir and return it as the data root.

        Mirrors the nuScenes streaming model: because nuReasoning maps are per-log
        (``map.pkl`` lives inside each clip), the cleanest way to feed both the log and
        map parsers is to materialize the selected subset once into a directory that
        mirrors the on-disk layout, then read from it exactly like local mode. A temp
        dir (and the extracted clips) is removed in :meth:`__del__`.
        """
        # The downloader fetches the splits this parser converts.
        downloader.splits = list(self._splits)
        # The BaseDownloader contract lets the parser assign output_dir when it is None.
        if downloader.output_dir is None:
            self._stream_temp_dir_handle = tempfile.TemporaryDirectory(prefix="py123d-nureasoning-")
            downloader.output_dir = Path(self._stream_temp_dir_handle.name)
        logger.info("nuReasoning streaming: materializing selected clips into %s", downloader.output_dir)
        downloader.download()
        return Path(downloader.output_dir)

    def __del__(self) -> None:
        handle = getattr(self, "_stream_temp_dir_handle", None)
        if handle is not None:
            handle.cleanup()

    def _collect_split_log_path_pairs(self) -> List[Tuple[str, Path]]:
        """Collects the (split, log_path) pairs for the specified splits."""
        split_log_path_pairs: List[Tuple[str, Path]] = []

        for split in self._splits:
            hf_split, split_parts = NUREASONING_SPLIT_SOURCES[split]
            nureasoning_split_folder = self._nureasoning_data_root / hf_split
            if not nureasoning_split_folder.is_dir():
                logger.warning("nuReasoning split %s has no folder at %s; skipping.", split, nureasoning_split_folder)
                continue

            # A few clips are released in two parts of the same split. They share one log name, so
            # only the first copy is converted.
            log_folders: Dict[str, Path] = {}
            for log_folder in _find_nureasoning_log_folders(nureasoning_split_folder, split_parts):
                if self._log_names is None or log_folder.name in self._log_names:
                    log_folders.setdefault(log_folder.name, log_folder)

            for log_folder in log_folders.values():
                split_log_path_pairs.append((split, log_folder))

        return split_log_path_pairs

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        # nuReasoning maps are per-log: one ``map.pkl`` per clip, so one map parser per log.
        # Clips of the test split ship without a map.
        return [
            NureasoningMapParser(
                split=split,
                log_name=source_log_path.name,
                source_log_path=source_log_path,
            )
            for split, source_log_path in self._split_log_path_pairs
            if (source_log_path / _MAP_FILE).is_file()
        ]

    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        return [
            NureasoningLogParser(
                split=split,
                source_log_path=source_log_path,
                nureasoning_data_root=self._nureasoning_data_root,
            )
            for split, source_log_path in self._split_log_path_pairs
        ]


class NureasoningLogParser(BaseLogParser):
    """Lightweight, picklable handle to one nuReasoning log."""

    def __init__(
        self,
        split: str,
        source_log_path: Path,
        nureasoning_data_root: Path,
    ) -> None:
        self._split = split
        self._source_log_path = source_log_path
        self._nureasoning_data_root = nureasoning_data_root

    def _get_nureasoning_metadata_json(self) -> Dict[str, Any]:
        """Helper function to load the nuReasoning metadata JSON for this log."""
        metadata_json_path = self._source_log_path / _METADATA_FILE
        if not metadata_json_path.exists() or not metadata_json_path.is_file():
            raise FileNotFoundError(
                f"Metadata JSON file not found for log {self._source_log_path}: {metadata_json_path}"
            )

        with open(metadata_json_path, "r", encoding="utf-8") as f:
            metadata_json = json.load(f)

        return metadata_json

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        metadata_json = self._get_nureasoning_metadata_json()
        # NOTE: Use the full folder name (``<log_name>_<clip_token>``) as the unique log id. The
        # folder name contains dots, so ``Path.stem`` would truncate it; ``metadata["log_name"]``
        # omits the clip token and is not guaranteed unique across clips.
        log_name = self._source_log_path.name
        location = metadata_json.get("clip_location", None)
        location = location.replace(" ", "-") if location else None

        # Each clip ships a per-log ``map.pkl`` (converted by NureasoningMapParser), except in the test split.
        # The map is stored in 2D (all map geometry has z == 0), and routed by (split, log_name) at read time.
        map_metadata: Optional[MapMetadata] = None
        if (self._source_log_path / _MAP_FILE).is_file():
            map_metadata = MapMetadata(
                dataset="nureasoning",
                split=self._split,
                log_name=log_name,
                location=location,
                map_has_z=False,
                map_is_per_log=True,
            )
        return LogMetadata(
            dataset="nureasoning",
            split=self._split,
            log_name=log_name,
            location=location,
            map_metadata=map_metadata,
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        metadata_json = self._get_nureasoning_metadata_json()

        ego_state_se3_metadata = _get_nureasoning_ego_state_se3_metadata(metadata_json)
        camera_metadatas = _get_nureasoning_camera_metadata(self._source_log_path, metadata_json)
        lidar_merged_metadata = _get_nureasoning_lidar_merged_metadata()
        box_detections_se3_metadata = NUREASONING_BOX_DETECTIONS_SE3_METADATA
        scenario_type = metadata_json.get("scenario_type", None)

        frames = _deduplicate_nureasoning_frames(metadata_json["frames"])
        key_frame_index = _get_nureasoning_key_frame_index(metadata_json, frames)
        reasoning_questions = _load_nureasoning_reasoning_questions(self._source_log_path)
        reasoning_questions_frame = _find_nureasoning_reasoning_questions_frame(frames, reasoning_questions)

        # Image path last emitted per camera, to skip the images a frame repeats from its predecessor.
        emitted_camera_paths: Dict[CameraID, str] = {}

        for frame in frames:
            timestamp = Timestamp.from_us(frame["timestamp_us"])

            # 1. Ego State
            ego_state_se3 = _extract_nureasoning_ego_state(self._source_log_path, frame, ego_state_se3_metadata)
            ego_trajectory = _extract_nureasoning_ego_trajectory(self._source_log_path, frame, timestamp)
            modalities: List[BaseModality] = [ego_state_se3]

            # 2. Annotations (not available in the test split). Emitted for every annotated frame, even if empty.
            relative_annotations_path = frame.get("annotations", None)
            if relative_annotations_path:
                annotations = load_schema_pickle(self._source_log_path / relative_annotations_path)
                assert isinstance(annotations, Annotations), f"Expected Annotations object, got {type(annotations)}"
                modalities.append(
                    _extract_nureasoning_box_detections(annotations, timestamp, box_detections_se3_metadata)
                )
                modalities.append(_extract_nureasoning_traffic_lights(annotations, timestamp))
            modalities.append(ego_trajectory)

            # 3. Sensors
            parsed_cameras = _extract_nureasoning_cameras(
                source_log_path=self._source_log_path,
                nureasoning_data_root=self._nureasoning_data_root,
                frame=frame,
                ego_state_se3=ego_state_se3,
                camera_metadatas=camera_metadatas,
                emitted_camera_paths=emitted_camera_paths,
            )
            modalities.extend(parsed_cameras)

            parsed_lidar = _extract_nureasoning_lidar_data(
                self._source_log_path, self._nureasoning_data_root, frame, lidar_merged_metadata
            )
            if parsed_lidar is not None:
                modalities.append(parsed_lidar)

            reasoning = _extract_nureasoning_reasoning(self._source_log_path, frame, timestamp)
            if reasoning is not None:
                modalities.append(reasoning)

            if reasoning_questions is not None and frame is reasoning_questions_frame:
                modalities.append(
                    CustomModality(
                        data=reasoning_questions,
                        metadata=CustomModalityMetadata(modality_id="reasoning_questions"),
                        timestamp=timestamp,
                    )
                )

            modalities.append(_extract_nureasoning_scenario(frame, scenario_type, key_frame_index, timestamp))

            yield ModalitiesSync(timestamp=timestamp, modalities=modalities)


# ------------------------------------------------------------------------------------------------------------------
# Clip / frame helpers
# ------------------------------------------------------------------------------------------------------------------


def _find_nureasoning_log_folders(split_folder: Path, split_parts: Optional[List[str]]) -> List[Path]:
    """Collects the clip folders of a split folder, i.e. the folders holding a ``metadata.json``.

    Clips sit either directly in the split folder (test) or one level down in ``part_<k>`` folders
    (train, validation). ``split_parts`` restricts the result to clips of these part folders.
    """
    log_folders: List[Path] = []
    for folder in sorted(split_folder.iterdir()):
        if (folder / _METADATA_FILE).is_file():
            if split_parts is None:
                log_folders.append(folder)
        elif folder.is_dir() and folder.name.startswith("part_"):
            if split_parts is None or folder.name in split_parts:
                log_folders.extend(
                    log_folder for log_folder in sorted(folder.iterdir()) if (log_folder / _METADATA_FILE).is_file()
                )
    return log_folders


def _deduplicate_nureasoning_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges consecutive frame records that share a timestamp.

    Most clips list their key frame twice: two records with the same token and timestamp, of which only
    the second one carries the reasoning path, and the ``frame_index`` that ``key_frame_index`` refers to.
    The merged record keeps the second record's values wherever they are set.
    """
    unique_frames: List[Dict[str, Any]] = []
    for frame in frames:
        if unique_frames and frame["timestamp_us"] == unique_frames[-1]["timestamp_us"]:
            unique_frames[-1] = {
                **unique_frames[-1],
                **{key: value for key, value in frame.items() if value not in ("", None)},
            }
        else:
            unique_frames.append(frame)
    return unique_frames


def _get_nureasoning_key_frame_index(metadata_json: Dict[str, Any], frames: List[Dict[str, Any]]) -> Optional[int]:
    """Returns the upstream ``frame_index`` of the clip's key frame, or None if the clip has none.

    The test split names it in ``metadata.json``. Elsewhere it is the frame whose token is the clip token,
    which a few clips do not contain.
    """
    key_frame_index: Optional[int] = metadata_json.get("key_frame_index", None)
    if key_frame_index is None:
        clip_token = metadata_json.get("clip_token", None)
        for frame in frames:
            if clip_token is not None and frame.get("token", None) == clip_token:
                key_frame_index = frame.get("frame_index", None)
    return key_frame_index


def _load_nureasoning_reasoning_questions(source_log_path: Path) -> Optional[Dict[str, Any]]:
    """Loads the clip's ``reasoning_questions.json`` (test split only), if present."""
    reasoning_questions: Optional[Dict[str, Any]] = None
    reasoning_questions_path = source_log_path / _REASONING_QUESTIONS_FILE
    if reasoning_questions_path.is_file():
        with open(reasoning_questions_path, "r", encoding="utf-8") as f:
            reasoning_questions_json = json.load(f)
        # CustomModality expects a dict with string keys; wrap non-dict payloads.
        reasoning_questions = (
            reasoning_questions_json
            if isinstance(reasoning_questions_json, dict)
            else {"questions": reasoning_questions_json}
        )
    return reasoning_questions


def _find_nureasoning_reasoning_questions_frame(
    frames: List[Dict[str, Any]], reasoning_questions: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Returns the frame the reasoning questions refer to (their ``frame_timestamp``), else the last frame."""
    questions_frame: Optional[Dict[str, Any]] = None
    if reasoning_questions is not None and frames:
        questions_frame = frames[-1]
        for frame in frames:
            if frame["timestamp_us"] == reasoning_questions.get("frame_timestamp", None):
                questions_frame = frame
    return questions_frame


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_nureasoning_ego_state_se3_metadata(metadata_json: Dict[str, Any]) -> EgoStateSE3Metadata:
    """Extracts the nuReasoning ego state SE3 metadata for a given log."""
    # NOTE @DanielDauner: Assuming Hyundai Ioniq 5 vehicle model.
    # https://en.wikipedia.org/wiki/Hyundai_Ioniq_5
    # NOTE: The test split has no ``ego_dimensions`` block. Its ego-state pickles carry a ``dimensions``
    # field instead, but with the nuPlan vehicle's values, so the devkit's constants are used.
    ego_dimensions = metadata_json.get("ego_dimensions", None) or NUREASONING_DEFAULT_EGO_DIMENSIONS

    _length = ego_dimensions["length"]
    _height = ego_dimensions["height"]

    # NOTE: Assuming distance from rear-axle to vehicle rear. Needs verification.
    vehicle_rear_length = ego_dimensions["vehicle_rear_length"]

    # TODO @DanielDauner: Verify these values, specifically once lidar available.
    half_length = _length / 2.0
    rear_axle_to_center_longitudinal = half_length - vehicle_rear_length
    rear_axle_to_center_vertical = (_height / 2.0) - NUREASONING_REAR_AXLE_HEIGHT

    center_to_imu_se3 = PoseSE3.from_R_t(
        rotation=np.zeros((3,), dtype=np.float64),
        translation=np.array([rear_axle_to_center_longitudinal, 0.0, rear_axle_to_center_vertical], dtype=np.float64),
    )

    return EgoStateSE3Metadata(
        vehicle_name="TODO",  # TODO @DanielDauner: Set proper vehicle name.
        width=ego_dimensions["width"],
        length=ego_dimensions["length"],
        height=ego_dimensions["height"],
        wheel_base=3.000,  # [m] NOTE @DanielDauner: Value from Wikipedia, needs verification.
        center_to_imu_se3=center_to_imu_se3,
        # NOTE: Assuming rear axle and IMU are co-located. Should be verified for nuReasoning.
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def _get_nureasoning_camera_metadata(
    source_log_path: Path, metadata_json: Dict[str, Any]
) -> Dict[CameraID, PinholeCameraMetadata]:
    """Extracts the nuReasoning camera metadata for a given log."""
    camera_metadatas: Dict[CameraID, PinholeCameraMetadata] = {}
    camera_calibrations_json: Dict[str, Any] = metadata_json.get("camera_calibrations", {})

    for camera_id, camera_name in NUREASONING_CAMERA_ID_MAPPING.items():
        has_valid_path = (source_log_path / "cameras" / camera_name).exists()
        has_metadata = camera_name in camera_calibrations_json.keys()

        if has_valid_path and has_metadata:
            camera_calibration_json = camera_calibrations_json[camera_name]

            _width, _height = camera_calibration_json["width"], camera_calibration_json["height"]

            # NOTE: The extrinsic is camera->lidar. We treat it as camera->IMU, which is correct only if
            # the lidar, IMU, and ego-pose origin coincide. Needs verification.
            extrinsic = PoseSE3.from_R_t(
                rotation=np.array(camera_calibration_json["sensor2lidar_rotation"], dtype=np.float64),
                translation=np.array(camera_calibration_json["sensor2lidar_translation"], dtype=np.float64),
            )

            camera_matrix = np.array(camera_calibration_json["intrinsic"], dtype=np.float64).reshape(3, 3)
            intrinsic = PinholeIntrinsics.from_camera_matrix(camera_matrix)

            camera_metadatas[camera_id] = PinholeCameraMetadata(
                camera_name=camera_name,
                camera_id=camera_id,
                width=_width,
                height=_height,
                intrinsics=intrinsic,
                distortion=None,  # TODO @DanielDauner: Verify if images are rectified.
                camera_to_imu_se3=extrinsic,
                is_undistorted=True,  # TODO @DanielDauner: Verify if correct.
            )

    return camera_metadatas


# ------------------------------------------------------------------------------------------------------------------
# Modality extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _extract_nureasoning_ego_state(
    source_log_path: Path,
    frame: Dict[str, Any],
    ego_state_se3_metadata: EgoStateSE3Metadata,
    # timestamp: Timestamp,
) -> EgoStateSE3:
    """Extracts the nuReasoning ego state from the per-frame ego_state pickle."""

    file_string = frame["ego_state"]
    timestamp = Timestamp.from_us(int(file_string.split("/")[-1].removesuffix(".pkl")))
    ego_state = load_schema_pickle(source_log_path / file_string)
    pose = ego_state.pose
    velocity = ego_state.velocity
    acceleration = ego_state.acceleration

    imu_pose = PoseSE3(
        x=pose["x"], y=pose["y"], z=pose["z"], qw=pose["qw"], qx=pose["qx"], qy=pose["qy"], qz=pose["qz"]
    )
    dynamic_state_se3 = DynamicStateSE3(
        velocity=Vector3D(x=velocity["vx"], y=velocity["vy"], z=velocity["vz"]),
        acceleration=Vector3D(x=acceleration["ax"], y=acceleration["ay"], z=acceleration["az"]),
        # NOTE: Angular velocity is not provided by nuReasoning.
        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_pose,
        metadata=ego_state_se3_metadata,
        dynamic_state_se3=dynamic_state_se3,
        timestamp=timestamp,
    )


def _extract_nureasoning_ego_trajectory(
    source_log_path: Path, frame: Dict[str, Any], timestamp: Timestamp
) -> CustomModality:
    """Extracts the ego history/future trajectory from the per-frame ego_state pickle as a custom modality.

    Both polylines are ``[x, y, yaw]`` samples in the global frame at the dataset frame rate (history grows
    up to 3 s, future up to 5 s). Either may be empty at the log boundaries, in which case it is stored as a
    ``(0, 3)`` array. The trajectory is in the same global frame as ``EgoStateSE3.imu_se3``.
    """
    ego_state = load_schema_pickle(source_log_path / frame["ego_state"])
    history_global = np.asarray(ego_state.trajectory_history or [], dtype=np.float64).reshape(-1, 3)
    future_global = np.asarray(ego_state.trajectory_future or [], dtype=np.float64).reshape(-1, 3)

    return CustomModality(
        data={"history_global": history_global, "future_global": future_global},
        metadata=CustomModalityMetadata(
            modality_id="ego_trajectory",
            metadata={"frame": "global", "columns": ["x", "y", "yaw"], "dt_s": NUREASONING_DEFAULT_DT},
        ),
        timestamp=timestamp,
    )


def _extract_nureasoning_box_detections(
    annotations: Annotations, timestamp: Timestamp, box_detections_se3_metadata: BoxDetectionsSE3Metadata
) -> BoxDetectionsSE3:
    """Extracts the nuReasoning box detections from the per-frame annotations pickle."""

    box_detections: List[BoxDetectionSE3] = []
    for obj in annotations.objects:
        if obj.category not in NUREASONING_DETECTION_NAME_DICT and obj.category not in _UNKNOWN_CATEGORIES:
            _UNKNOWN_CATEGORIES.add(obj.category)
            logger.warning("Unknown nuReasoning object category %r, stored as OTHER_OTHER.", obj.category)

        pose, velocity, dimensions = obj.pose, obj.velocity, obj.dimensions
        quaternion = EulerAngles(roll=DEFAULT_ROLL, pitch=DEFAULT_PITCH, yaw=pose["yaw"]).quaternion
        bounding_box = BoundingBoxSE3(
            center_se3=PoseSE3(
                x=pose["x"],
                y=pose["y"],
                z=pose["z"],
                qw=quaternion.qw,
                qx=quaternion.qx,
                qy=quaternion.qy,
                qz=quaternion.qz,
            ),
            length=dimensions["l"],
            width=dimensions["w"],
            height=dimensions["h"],
        )
        box_detections.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(
                    # NOTE: Fall back to OTHER_OTHER for categories not in the mapping.
                    label=NUREASONING_DETECTION_NAME_DICT.get(obj.category, NureasoningBoxDetectionLabel.OTHER_OTHER),
                    track_token=obj.track_token,
                ),
                bounding_box_se3=bounding_box,
                # NOTE: We assume object velocity is in the global frame, as expected by
                # BoxDetectionSE3.velocity_3d. Needs verification.
                velocity_3d=Vector3D(x=velocity["vx"], y=velocity["vy"], z=velocity["vz"]),
            )
        )

    return BoxDetectionsSE3(
        box_detections=box_detections,
        timestamp=timestamp,
        metadata=box_detections_se3_metadata,
    )


def _extract_nureasoning_traffic_lights(annotations: Annotations, timestamp: Timestamp) -> TrafficLightDetections:
    """Extracts the nuReasoning traffic light detections from the per-frame annotations pickle."""

    detections: List[TrafficLightDetection] = []
    for traffic_light in annotations.traffic_light_states:
        # Prefer the lane connector as the lane reference; fall back to the roadblock id.
        # Skip if neither is available, since we cannot anchor it without a map.
        lane_id = traffic_light.lane_connector_id
        if lane_id is None:
            lane_id = traffic_light.roadblock_id

        if lane_id is not None:
            detections.append(
                TrafficLightDetection(
                    lane_id=int(lane_id),
                    status=NUREASONING_TRAFFIC_STATUS_DICT[traffic_light.state],
                )
            )

    return TrafficLightDetections(detections=detections, timestamp=timestamp)


def _extract_nureasoning_cameras(
    source_log_path: Path,
    nureasoning_data_root: Path,
    frame: Dict[str, Any],
    ego_state_se3: EgoStateSE3,
    camera_metadatas: Dict[CameraID, PinholeCameraMetadata],
    emitted_camera_paths: Dict[CameraID, str],
) -> List[ParsedCamera]:
    """Extracts the nuReasoning camera data for all cameras with a new image in this frame.

    The camera-to-global pose is composed from the ego (IMU) pose and the static camera-to-IMU
    extrinsic. Image bytes are not loaded here; the log writer reads them at write time.

    A frame can list the image of its predecessor again (seen on the last frame of a clip, which then
    follows after 50 ms). ``emitted_camera_paths`` tracks the last image per camera across calls, so
    that such an image is stored once, with the pose of the frame it was first listed in.
    """
    camera_paths: Dict[str, str] = frame.get("sensors", {}).get("cameras", {})
    camera_data_list: List[ParsedCamera] = []

    for camera_id, camera_metadata in camera_metadatas.items():
        frame_key = NUREASONING_CAMERA_KEY_MAPPING[camera_id]
        relative_camera_path = camera_paths.get(frame_key, None)
        if not relative_camera_path or relative_camera_path == emitted_camera_paths.get(camera_id, None):
            continue

        full_image_path = source_log_path / relative_camera_path
        if not (full_image_path.exists() and full_image_path.is_file()):
            continue

        camera_to_global_se3 = rel_to_abs_se3(
            origin=ego_state_se3.imu_se3,
            pose_se3=camera_metadata.camera_to_imu_se3,
        )

        timestamp = Timestamp.from_us(int(relative_camera_path.split("/")[-1].removesuffix(".jpg").split("_")[-1]))

        emitted_camera_paths[camera_id] = relative_camera_path
        camera_data_list.append(
            ParsedCamera(
                metadata=camera_metadata,
                timestamp=timestamp,
                camera_to_global_se3=camera_to_global_se3,
                dataset_root=nureasoning_data_root,
                relative_path=full_image_path.relative_to(nureasoning_data_root),
            )
        )

    return camera_data_list


def _get_nureasoning_lidar_merged_metadata() -> LidarMergedMetadata:
    """Builds the merged-lidar metadata for nuReasoning.

    The point cloud merges multiple lidar sensors (see ``NUREASONING_LIDAR_DICT``). The points are
    already in the common ego/lidar frame, so per-sensor extrinsics are the identity.
    """
    # NOTE: lidar == IMU is assumed, so the extrinsics are the identity.
    metadata: Dict[LidarID, LidarMetadata] = {
        lidar_id: LidarMetadata(
            lidar_name=lidar_id.serialize(),
            lidar_id=lidar_id,
            lidar_to_imu_se3=PoseSE3.identity(),
        )
        for lidar_id in NUREASONING_LIDAR_DICT.values()
    }
    return LidarMergedMetadata(metadata)


def _extract_nureasoning_lidar_data(
    source_log_path: Path,
    nureasoning_data_root: Path,
    frame: Dict[str, Any],
    lidar_merged_metadata: LidarMergedMetadata,
) -> Optional[ParsedLidar]:
    """Extracts the nuReasoning lidar data from the per-frame lidar path, if present.

    Only the path is stored (see the ``lidar_store_option: "path"`` conversion config); the point
    cloud is decoded at read time by ``nureasoning_sensor_io``. Lidar is only present in some clips,
    and never in the test split.
    """
    parsed_lidar: Optional[ParsedLidar] = None

    relative_lidar_path = frame.get("sensors", {}).get("lidar", {}).get("point_cloud_path", None)
    if relative_lidar_path:
        full_lidar_path = source_log_path / relative_lidar_path

        if full_lidar_path.exists() and full_lidar_path.is_file():
            timestamp = Timestamp.from_us(int(relative_lidar_path.split("/")[-1].removesuffix(".pcd").split("_")[-1]))
            parsed_lidar = ParsedLidar(
                metadata=lidar_merged_metadata,
                start_timestamp=timestamp,
                end_timestamp=Timestamp.from_us(timestamp.time_us + NUREASONING_LIDAR_SWEEP_DURATION_US),
                dataset_root=nureasoning_data_root,
                relative_path=full_lidar_path.relative_to(nureasoning_data_root),
            )
        else:
            logger.debug(f"Lidar file not found: {full_lidar_path}")

    return parsed_lidar


def _extract_nureasoning_reasoning(
    source_log_path: Path, frame: Dict[str, Any], timestamp: Timestamp
) -> Optional[CustomModality]:
    """Extracts the nuReasoning reasoning annotations (raw passthrough) when present for this frame."""
    custom_modality: Optional[CustomModality] = None

    relative_reasoning_path = frame.get("reasoning", "")
    if relative_reasoning_path:
        with open(source_log_path / relative_reasoning_path, "r", encoding="utf-8") as f:
            reasoning_json = json.load(f)

        # CustomModality expects a dict with string keys; wrap non-dict payloads.
        data = reasoning_json if isinstance(reasoning_json, dict) else {"reasoning": reasoning_json}
        custom_modality = CustomModality(
            data=data,
            metadata=CustomModalityMetadata(modality_id="reasoning"),
            timestamp=timestamp,
        )

    return custom_modality


def _extract_nureasoning_scenario(
    frame: Dict[str, Any], scenario_type: Optional[str], key_frame_index: Optional[int], timestamp: Timestamp
) -> CustomModality:
    """Extracts the nuReasoning mission-goal / scenario metadata as a custom modality.

    ``frame_index`` and ``frame_token`` are the upstream identifiers of the frame. The index is what the
    reasoning annotations and the challenge refer to, and can run ahead of the position in the converted
    log (see :func:`_deduplicate_nureasoning_frames`).
    """
    mission_goal = frame.get("mission_goal", None) or {}
    frame_index = frame.get("frame_index", None)
    data: Dict[str, Any] = {
        "command": mission_goal.get("command", None),
        "route_path": mission_goal.get("route_path", []),
        "scenario_type": scenario_type,
        "frame_index": frame_index,
        "frame_token": frame.get("token", None),
        "is_key_frame": frame_index is not None and frame_index == key_frame_index,
    }
    return CustomModality(
        data=data,
        metadata=CustomModalityMetadata(modality_id="scenario"),
        timestamp=timestamp,
    )
