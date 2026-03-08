from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import override

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    DynamicStateSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.detections.box_detection_label_metadata import BoxDetectionMetadata
from py123d.datatypes.metadata.sensor_metadata import FisheyeMEICameraMetadatas, LidarMetadatas, PinholeCameraMetadatas
from py123d.datatypes.vehicle_state.ego_metadata import EgoMetadata
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D
from py123d.parser.abstract_dataset_parser import (
    CameraData,
    DatasetParser,
    FrameData,
    LidarData,
    LogParser,
)
from py123d.parser.nuscenes.nuscenes_map_parser import NuScenesMapParser
from py123d.parser.nuscenes.utils.nuscenes_constants import (
    NUSCENES_CAMERA_IDS,
    NUSCENES_DATA_SPLITS,
    NUSCENES_DATABASE_VERSION_MAPPING,
    NUSCENES_DETECTION_NAME_DICT,
    NUSCENES_DT,
    NUSCENES_LIDAR_SWEEP_DURATION_US,
    NUSCENES_MAP_LOCATIONS,
)
from py123d.parser.registry import NuScenesBoxDetectionLabel

check_dependencies(["nuscenes"], "nuscenes")
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes


class NuScenesParser(DatasetParser):
    """Dataset parser for the nuScenes dataset."""

    def __init__(
        self,
        splits: List[str],
        nuscenes_data_root: Union[Path, str],
        nuscenes_map_root: Union[Path, str],
    ) -> None:
        """Initializes the NuScenesParser.

        :param splits: List of dataset splits, e.g. ["nuscenes_train", "nuscenes_val"].
        :param nuscenes_data_root: Root directory of the nuScenes data.
        :param nuscenes_map_root: Root directory of the nuScenes maps.
        """
        assert nuscenes_data_root is not None, "The variable `nuscenes_data_root` must be provided."
        assert nuscenes_map_root is not None, "The variable `nuscenes_map_root` must be provided."
        for split in splits:
            assert split in NUSCENES_DATA_SPLITS, (
                f"Split {split} is not available. Available splits: {NUSCENES_DATA_SPLITS}"
            )

        self._splits: List[str] = splits
        self._nuscenes_data_root: Path = Path(nuscenes_data_root)
        self._nuscenes_map_root: Path = Path(nuscenes_map_root)

        self._nuscenes_dbs: Dict[str, NuScenes] = {}
        self._scene_tokens_per_split: Dict[str, List[str]] = self._collect_scene_tokens()

    def _collect_scene_tokens(self) -> Dict[str, List[str]]:
        """Collects scene tokens for the specified splits."""
        scene_tokens_per_split: Dict[str, List[str]] = {}
        nuscenes_split_name_mapping = {
            "nuscenes_train": "train",
            "nuscenes_val": "val",
            "nuscenes_test": "test",
            "nuscenes-mini_train": "mini_train",
            "nuscenes-mini_val": "mini_val",
        }

        scene_splits = create_splits_scenes()

        for split in self._splits:
            database_version = NUSCENES_DATABASE_VERSION_MAPPING[split]
            nusc = self._nuscenes_dbs.get(database_version)
            if nusc is None:
                nusc = NuScenes(
                    version=database_version,
                    dataroot=str(self._nuscenes_data_root),
                    verbose=False,
                )
                self._nuscenes_dbs[database_version] = nusc

            available_scenes = [scene for scene in nusc.scene]
            nuscenes_split = nuscenes_split_name_mapping[split]
            scene_names = scene_splits.get(nuscenes_split, [])

            scene_tokens = [scene["token"] for scene in available_scenes if scene["name"] in scene_names]
            scene_tokens_per_split[split] = scene_tokens
        return scene_tokens_per_split

    def get_log_parsers(self) -> List[NuScenesLogParser]:  # type: ignore
        """Inherited, see superclass."""
        log_parsers: List[NuScenesLogParser] = []
        for split, scene_tokens in self._scene_tokens_per_split.items():
            database_version = NUSCENES_DATABASE_VERSION_MAPPING[split]
            nusc = self._nuscenes_dbs[database_version]
            for scene_token in scene_tokens:
                scene = nusc.get("scene", scene_token)
                log_record = nusc.get("log", scene["log_token"])
                log_parsers.append(
                    NuScenesLogParser(
                        split=split,
                        scene_token=scene_token,
                        scene_name=scene["name"],
                        location=log_record["location"],
                        nuscenes_data_root=self._nuscenes_data_root,
                        database_version=database_version,
                    )
                )
        return log_parsers

    def get_map_parsers(self) -> List[NuScenesMapParser]:  # type: ignore
        """Inherited, see superclass."""
        return [
            NuScenesMapParser(nuscenes_maps_root=self._nuscenes_map_root, location=location)
            for location in NUSCENES_MAP_LOCATIONS
        ]


class NuScenesLogParser(LogParser):
    """Lightweight, picklable handle to one nuScenes scene/log."""

    def __init__(
        self,
        split: str,
        scene_token: str,
        scene_name: str,
        location: str,
        nuscenes_data_root: Path,
        database_version: str,
    ) -> None:
        self._split = split
        self._scene_token = scene_token
        self._scene_name = scene_name
        self._location = location
        self._nuscenes_data_root = nuscenes_data_root
        self._database_version = database_version

    def _load_nusc(self) -> NuScenes:
        return NuScenes(
            version=self._database_version,
            dataroot=str(self._nuscenes_data_root),
            verbose=False,
        )

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset="nuscenes",
            split=self._split,
            log_name=self._scene_name,
            location=self._location,
            timestep_seconds=NUSCENES_DT,
        )

    @override
    def get_ego_metadata(self) -> Optional[EgoMetadata]:
        """Inherited, see superclass."""
        # NOTE: The parameters in nuScenes are estimates, and partially taken from the Renault Zoe model [1].
        # [1] https://en.wikipedia.org/wiki/Renault_Zoe
        return EgoMetadata(
            vehicle_name="nuscenes_renault_zoe",
            width=1.730,
            length=4.084,
            height=1.562,
            wheel_base=2.588,
            center_to_imu_se3=PoseSE3(x=1.385, y=0.0, z=1.562 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )

    @override
    def get_box_detection_metadata(self) -> Optional[BoxDetectionMetadata]:
        """Inherited, see superclass."""
        return BoxDetectionMetadata(box_detection_label_class=NuScenesBoxDetectionLabel)

    @override
    def get_pinhole_camera_metadatas(self) -> Optional[PinholeCameraMetadatas]:
        """Inherited, see superclass."""
        nusc = self._load_nusc()
        try:
            scene = nusc.get("scene", self._scene_token)
            metadata = _get_nuscenes_pinhole_camera_metadata(nusc, scene)
            if metadata:
                return PinholeCameraMetadatas(metadata)
            return None
        finally:
            del nusc
            gc.collect()

    @override
    def get_fisheye_mei_camera_metadatas(self) -> Optional[FisheyeMEICameraMetadatas]:
        """Inherited, see superclass."""
        return None  # nuScenes does not have fisheye MEI cameras

    @override
    def get_lidar_metadatas(self) -> Optional[LidarMetadatas]:
        """Inherited, see superclass."""
        nusc = self._load_nusc()
        try:
            scene = nusc.get("scene", self._scene_token)
            metadata = _get_nuscenes_lidar_metadata(nusc, scene)
            if metadata:
                return LidarMetadatas(metadata)
            return None
        finally:
            del nusc
            gc.collect()

    def iter_frames(self) -> Iterator[FrameData]:
        """Inherited, see superclass."""
        nusc = self._load_nusc()
        ego_metadata = self.get_ego_metadata()
        assert ego_metadata is not None

        try:
            can_bus = NuScenesCanBus(dataroot=str(self._nuscenes_data_root))
            scene = nusc.get("scene", self._scene_token)

            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                timestamp = Timestamp.from_us(sample["timestamp"])

                lidar_data = _extract_nuscenes_lidar(
                    nusc=nusc,
                    sample=sample,
                    nuscenes_data_root=self._nuscenes_data_root,
                )

                yield FrameData(
                    timestamp=timestamp,
                    ego_state_se3=_extract_nuscenes_ego_state(nusc, sample, can_bus, ego_metadata),
                    box_detections_se3=_extract_nuscenes_box_detections(nusc, sample),
                    pinhole_cameras=_extract_nuscenes_cameras(
                        nusc=nusc,
                        sample=sample,
                        nuscenes_data_root=self._nuscenes_data_root,
                    ),
                    lidar=lidar_data,
                )

                sample_token = sample["next"]
        finally:
            del nusc
            gc.collect()

    # ------------------------------------------------------------------------------------------------------------------
    # Per-modality iterators (async / native-rate)
    # ------------------------------------------------------------------------------------------------------------------

    @override
    def iter_ego_states_se3(self) -> Iterator[EgoStateSE3]:
        """Yields ego states at full lidar sweep rate (~20Hz).

        Each lidar sample_data record has its own ego_pose_token, providing real
        (non-interpolated) ego poses at the native lidar rate.
        """
        nusc = self._load_nusc()
        ego_metadata = self.get_ego_metadata()
        assert ego_metadata is not None

        try:
            can_bus = NuScenesCanBus(dataroot=str(self._nuscenes_data_root))
            scene = nusc.get("scene", self._scene_token)
            lidar_timeline = _collect_lidar_sweep_timeline(nusc, scene)

            for sweep in lidar_timeline:
                yield _extract_ego_state_from_sample_data(nusc, sweep, can_bus, self._scene_name, ego_metadata)
        finally:
            del nusc
            gc.collect()

    @override
    def iter_pinhole_cameras(self) -> Iterator[CameraData]:
        """Yields all pinhole camera observations at native rate (~12Hz per camera).

        Camera records are yielded individually (one per camera per trigger), sorted
        by timestamp across all channels.
        """
        nusc = self._load_nusc()

        try:
            scene = nusc.get("scene", self._scene_token)
            camera_timelines = _collect_camera_timelines(nusc, scene)

            all_records: List[Tuple[int, PinholeCameraID, str, Dict[str, Any]]] = []
            for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
                timeline = camera_timelines.get(camera_channel, [])
                for cam_sd in timeline:
                    all_records.append((cam_sd["timestamp"], camera_type, camera_channel, cam_sd))

            all_records.sort(key=lambda r: r[0])

            for _, camera_type, camera_channel, cam_data in all_records:
                calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
                translation_array = np.array(calib["translation"], dtype=np.float64)
                rotation_array = np.array(calib["rotation"], dtype=np.float64)
                extrinsic = PoseSE3.from_R_t(rotation=rotation_array, translation=translation_array)

                cam_path = self._nuscenes_data_root / str(cam_data["filename"])
                if cam_path.exists() and cam_path.is_file():
                    yield CameraData(
                        camera_name=camera_channel,
                        camera_id=camera_type,
                        extrinsic=extrinsic,
                        relative_path=cam_path.relative_to(self._nuscenes_data_root),
                        dataset_root=self._nuscenes_data_root,
                        timestamp=Timestamp.from_us(cam_data["timestamp"]),
                    )
        finally:
            del nusc
            gc.collect()

    @override
    def iter_lidars(self) -> Iterator[LidarData]:
        """Yields all lidar sweeps at native rate (~20Hz).

        The nuScenes lidar timestamp marks the end of a sweep. For each sweep,
        ``start_timestamp`` is set to ``end_timestamp - 50ms`` (one full rotation).
        """
        nusc = self._load_nusc()

        try:
            scene = nusc.get("scene", self._scene_token)
            lidar_timeline = _collect_lidar_sweep_timeline(nusc, scene)

            for sweep in lidar_timeline:
                absolute_lidar_path = self._nuscenes_data_root / sweep["filename"]
                if absolute_lidar_path.exists() and absolute_lidar_path.is_file():
                    # The nuScenes lidar timestamp marks the end of the sweep (full rotation).
                    # The sweep covers the 1/20s (50ms) period before that timestamp.
                    yield LidarData(
                        lidar_name="LIDAR_TOP",
                        lidar_type=LidarID.LIDAR_TOP,
                        start_timestamp=Timestamp.from_us(sweep["timestamp"] - NUSCENES_LIDAR_SWEEP_DURATION_US),
                        end_timestamp=Timestamp.from_us(sweep["timestamp"]),
                        relative_path=absolute_lidar_path.relative_to(self._nuscenes_data_root),
                        dataset_root=self._nuscenes_data_root,
                    )
        finally:
            del nusc
            gc.collect()


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_nuscenes_pinhole_camera_metadata(
    nusc: NuScenes,
    scene: Dict[str, Any],
) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Extracts the pinhole camera metadata from a nuScenes scene."""
    camera_metadata: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)
    for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
        cam_token = first_sample["data"][camera_channel]
        cam_data = nusc.get("sample_data", cam_token)
        calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

        intrinsic_matrix = np.array(calib["camera_intrinsic"])
        intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsic_matrix)
        distortion = PinholeDistortion.from_array(np.zeros(5), copy=False)

        translation_array = np.array(calib["translation"], dtype=np.float64)
        rotation_array = np.array(calib["rotation"], dtype=np.float64)
        camera_to_imu_se3 = PoseSE3.from_R_t(rotation=rotation_array, translation=translation_array)

        camera_metadata[camera_type] = PinholeCameraMetadata(
            camera_name=camera_channel,
            camera_id=camera_type,
            width=cam_data["width"],
            height=cam_data["height"],
            intrinsics=intrinsic,
            distortion=distortion,
            camera_to_imu_se3=camera_to_imu_se3,
            is_undistorted=True,
        )

    return camera_metadata


def _get_nuscenes_lidar_metadata(
    nusc: NuScenes,
    scene: Dict[str, Any],
) -> Dict[LidarID, LidarMetadata]:
    """Extracts the Lidar metadata from a nuScenes scene."""
    metadata: Dict[LidarID, LidarMetadata] = {}
    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)
    lidar_token = first_sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    lidar_to_imu_se3 = PoseSE3.from_R_t(
        rotation=np.array(calib["rotation"], dtype=np.float64),
        translation=np.array(calib["translation"], dtype=np.float64),
    )
    metadata[LidarID.LIDAR_TOP] = LidarMetadata(
        lidar_name="LIDAR_TOP",
        lidar_id=LidarID.LIDAR_TOP,
        lidar_to_imu_se3=lidar_to_imu_se3,
    )
    return metadata


# ------------------------------------------------------------------------------------------------------------------
# Sensor extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _extract_nuscenes_ego_state(
    nusc: NuScenes, sample: Dict[str, Any], can_bus: NuScenesCanBus, ego_metadata: EgoMetadata
) -> EgoStateSE3:
    """Extracts the ego state from a nuScenes sample."""
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    imu_pose = PoseSE3.from_R_t(
        rotation=np.array(ego_pose["rotation"], dtype=np.float64),
        translation=np.array(ego_pose["translation"], dtype=np.float64),
    )

    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    try:
        pose_msgs = can_bus.get_messages(scene_name, "pose")
    except Exception:
        pose_msgs = []
    if pose_msgs:
        closest_msg = None
        min_time_diff = float("inf")
        for msg in pose_msgs:
            time_diff = abs(msg["utime"] - sample["timestamp"])
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_msg = msg

        if closest_msg and min_time_diff < 500000:
            velocity = [*closest_msg["vel"]]
            acceleration = [*closest_msg["accel"]]
            angular_velocity = [*closest_msg["rotation_rate"]]
        else:
            velocity = acceleration = angular_velocity = [0.0, 0.0, 0.0]
    else:
        velocity = acceleration = angular_velocity = [0.0, 0.0, 0.0]

    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(*velocity),
        acceleration=Vector3D(*acceleration),
        angular_velocity=Vector3D(*angular_velocity),
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_pose,
        dynamic_state_se3=dynamic_state,
        ego_metadata=ego_metadata,
        timestamp=Timestamp.from_us(sample["timestamp"]),
    )


def _extract_nuscenes_box_detections(nusc: NuScenes, sample: Dict[str, Any]) -> BoxDetectionsSE3:
    """Extracts the box detections from a nuScenes sample."""
    box_detections: List[BoxDetectionSE3] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        center_se3 = PoseSE3.from_R_t(
            rotation=np.array(ann["rotation"], dtype=np.float64),
            translation=np.array(ann["translation"], dtype=np.float64),
        )
        width, length, height = ann["size"]
        bounding_box = BoundingBoxSE3(
            center_se3=center_se3,
            length=length,
            width=width,
            height=height,
        )
        category = ann["category_name"]
        label = NUSCENES_DETECTION_NAME_DICT[category]

        velocity = nusc.box_velocity(ann_token)
        velocity_3d = Vector3D(x=velocity[0], y=velocity[1], z=velocity[2] if len(velocity) > 2 else 0.0)

        metadata = BoxDetectionAttributes(
            label=label,
            track_token=ann["instance_token"],
            num_lidar_points=ann.get("num_lidar_pts", 0),
        )
        box_detection = BoxDetectionSE3(
            metadata=metadata,
            bounding_box_se3=bounding_box,
            velocity_3d=velocity_3d,
        )
        box_detections.append(box_detection)
    return BoxDetectionsSE3(box_detections=box_detections, timestamp=Timestamp.from_us(sample["timestamp"]))  # type: ignore


def _extract_nuscenes_cameras(
    nusc: NuScenes,
    sample: Dict[str, Any],
    nuscenes_data_root: Path,
) -> List[CameraData]:
    """Extracts the pinhole camera data from a nuScenes sample."""
    camera_data_list: List[CameraData] = []
    for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
        cam_token = sample["data"][camera_channel]
        cam_data = nusc.get("sample_data", cam_token)

        # Check timestamp synchronization (within 100ms)
        if abs(cam_data["timestamp"] - sample["timestamp"]) > 100000:
            continue

        calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        translation_array = np.array(calib["translation"], dtype=np.float64)
        rotation_array = np.array(calib["rotation"], dtype=np.float64)
        extrinsic = PoseSE3.from_R_t(rotation=rotation_array, translation=translation_array)
        cam_path = nuscenes_data_root / str(cam_data["filename"])
        if cam_path.exists() and cam_path.is_file():
            camera_data_list.append(
                CameraData(
                    camera_name=camera_channel,
                    camera_id=camera_type,
                    extrinsic=extrinsic,
                    relative_path=cam_path.relative_to(nuscenes_data_root),
                    dataset_root=nuscenes_data_root,
                    timestamp=Timestamp.from_us(cam_data["timestamp"]),
                )
            )

    return camera_data_list


def _extract_nuscenes_lidar(
    nusc: NuScenes,
    sample: Dict[str, Any],
    nuscenes_data_root: Path,
) -> Optional[LidarData]:
    """Extracts the Lidar data from a nuScenes sample."""
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    absolute_lidar_path = nuscenes_data_root / lidar_data["filename"]
    if absolute_lidar_path.exists() and absolute_lidar_path.is_file():
        # The nuScenes lidar timestamp marks the end of the sweep (full rotation).
        # The sweep covers the 1/20s (50ms) period before that timestamp.
        end_timestamp = Timestamp.from_us(sample["timestamp"])
        start_timestamp = Timestamp.from_us(sample["timestamp"] - NUSCENES_LIDAR_SWEEP_DURATION_US)
        return LidarData(
            lidar_name="LIDAR_TOP",
            lidar_type=LidarID.LIDAR_TOP,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            relative_path=absolute_lidar_path.relative_to(nuscenes_data_root),
            dataset_root=nuscenes_data_root,
            iteration=lidar_data.get("iteration"),
        )
    return None


# ------------------------------------------------------------------------------------------------------------------
# Lidar sweep timeline helpers (for native-rate async iteration)
# ------------------------------------------------------------------------------------------------------------------


def _collect_lidar_sweep_timeline(nusc: NuScenes, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collects all LIDAR_TOP sample_data records for a scene (keyframes + sweeps).

    Walks the sample_data linked list starting from the first keyframe's lidar token
    forward through the entire scene.

    :param nusc: The NuScenes database instance.
    :param scene: The scene record.
    :return: Chronologically ordered list of lidar sweep dicts with keys:
        token, timestamp, ego_pose_token, filename, is_key_frame, sample_token.
    """
    first_sample = nusc.get("sample", scene["first_sample_token"])
    last_sample = nusc.get("sample", scene["last_sample_token"])
    last_kf_timestamp = last_sample["timestamp"]

    lidar_sd_token = first_sample["data"]["LIDAR_TOP"]
    timeline: List[Dict[str, Any]] = []

    current = nusc.get("sample_data", lidar_sd_token)
    while current:
        if current["timestamp"] > last_kf_timestamp and not current["is_key_frame"]:
            break

        timeline.append(
            {
                "token": current["token"],
                "timestamp": current["timestamp"],
                "ego_pose_token": current["ego_pose_token"],
                "filename": current["filename"],
                "is_key_frame": current["is_key_frame"],
                "sample_token": current.get("sample_token", ""),
            }
        )

        if current["next"]:
            current = nusc.get("sample_data", current["next"])
        else:
            break

    return timeline


def _collect_camera_timelines(
    nusc: NuScenes,
    scene: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Collects all sample_data records for each camera channel in a scene.

    :param nusc: The NuScenes database instance.
    :param scene: The scene record.
    :return: Dict mapping camera channel name to its chronological list of sample_data records.
    """
    first_sample = nusc.get("sample", scene["first_sample_token"])
    last_sample = nusc.get("sample", scene["last_sample_token"])
    last_kf_timestamp = last_sample["timestamp"]

    timelines: Dict[str, List[Dict[str, Any]]] = {}

    for _camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
        timeline: List[Dict[str, Any]] = []
        sd_token = first_sample["data"][camera_channel]
        current = nusc.get("sample_data", sd_token)

        while current:
            if current["timestamp"] > last_kf_timestamp and not current["is_key_frame"]:
                break
            timeline.append(current)
            if current["next"]:
                current = nusc.get("sample_data", current["next"])
            else:
                break

        timelines[camera_channel] = timeline

    return timelines


def _extract_ego_state_from_sample_data(
    nusc: NuScenes,
    sweep: Dict[str, Any],
    can_bus: NuScenesCanBus,
    scene_name: str,
    ego_metadata: EgoMetadata,
) -> EgoStateSE3:
    """Extracts the ego state from a lidar sample_data record (keyframe or non-keyframe).

    Uses the real ego pose from the sample_data's ego_pose_token and matches
    CAN bus data for dynamic state (velocity, acceleration, angular velocity).

    :param nusc: The NuScenes database instance.
    :param sweep: A lidar sweep dict from the timeline.
    :param can_bus: The NuScenes CAN bus API.
    :param scene_name: The scene name for CAN bus lookup.
    :param ego_metadata: Vehicle parameters for ego state construction.
    :return: The ego state.
    """
    ego_pose = nusc.get("ego_pose", sweep["ego_pose_token"])

    imu_pose = PoseSE3.from_R_t(
        rotation=np.array(ego_pose["rotation"], dtype=np.float64),
        translation=np.array(ego_pose["translation"], dtype=np.float64),
    )

    try:
        pose_msgs = can_bus.get_messages(scene_name, "pose")
    except Exception:
        pose_msgs = []

    if pose_msgs:
        closest_msg = None
        min_time_diff = float("inf")
        for msg in pose_msgs:
            time_diff = abs(msg["utime"] - sweep["timestamp"])
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_msg = msg

        if closest_msg and min_time_diff < 500_000:
            velocity = [*closest_msg["vel"]]
            acceleration = [*closest_msg["accel"]]
            angular_velocity = [*closest_msg["rotation_rate"]]
        else:
            velocity = acceleration = angular_velocity = [0.0, 0.0, 0.0]
    else:
        velocity = acceleration = angular_velocity = [0.0, 0.0, 0.0]

    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(*velocity),
        acceleration=Vector3D(*acceleration),
        angular_velocity=Vector3D(*angular_velocity),
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_pose,
        dynamic_state_se3=dynamic_state,
        ego_metadata=ego_metadata,
        timestamp=Timestamp.from_us(sweep["timestamp"]),
    )
