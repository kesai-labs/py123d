import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pyquaternion import Quaternion

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.api.scene.abstract_log_writer import AbstractLogWriter, CameraData, LidarData
from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.nuscenes.nuscenes_map_conversion import NUSCENES_MAPS, write_nuscenes_map
from py123d.conversion.datasets.nuscenes.utils.nuscenes_constants import (
    NUSCENES_CAMERA_IDS,
    NUSCENES_DATA_SPLITS,
    NUSCENES_DATABASE_VERSION_MAPPING,
    NUSCENES_DETECTION_NAME_DICT,
    NUSCENES_DT,
)
from py123d.conversion.registry.box_detection_label_registry import NuScenesBoxDetectionLabel
from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    DynamicStateSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    MapMetadata,
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.vehicle_state.ego_metadata import get_nuscenes_renault_zoe_parameters
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D

check_dependencies(["nuscenes"], "nuscenes")
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes


class NuScenesConverter(AbstractDatasetConverter):
    """Dataset converter for the nuScenes dataset."""

    def __init__(
        self,
        splits: List[str],
        nuscenes_data_root: Union[Path, str],
        nuscenes_map_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
        nuscenes_dbs: Optional[Dict[str, NuScenes]] = None,
    ) -> None:
        """Initializes the :class:`NuScenesConverter`.

        :param splits: List of splits to include in the conversion, e.g., ["nuscenes_train", "nuscenes_val"]
        :param nuscenes_data_root: Path to the root directory of the nuScenes dataset
        :param nuscenes_map_root: Path to the root directory of the nuScenes map data
        :param dataset_converter_config: Configuration for the dataset converter
        """
        super().__init__(dataset_converter_config)

        assert nuscenes_data_root is not None, "The variable `nuscenes_data_root` must be provided."
        assert nuscenes_map_root is not None, "The variable `nuscenes_map_root` must be provided."
        for split in splits:
            assert split in NUSCENES_DATA_SPLITS, (
                f"Split {split} is not available. Available splits: {NUSCENES_DATA_SPLITS}"
            )

        self._splits: List[str] = splits

        self._nuscenes_data_root: Path = Path(nuscenes_data_root)
        self._nuscenes_map_root: Path = Path(nuscenes_map_root)

        self._nuscenes_dbs: Dict[str, NuScenes] = nuscenes_dbs if nuscenes_dbs is not None else {}
        self._scene_tokens_per_split: Dict[str, List[str]] = self._collect_scene_tokens()

    def __reduce__(self):
        return (
            self.__class__,
            (
                self._splits,
                self._nuscenes_data_root,
                self._nuscenes_map_root,
                self.dataset_converter_config,
                self._nuscenes_dbs,
            ),
        )

    def _collect_scene_tokens(self) -> Dict[str, List[str]]:
        """Collects scene tokens for the specified splits."""

        scene_tokens_per_split: Dict[str, List[str]] = {}
        # Conversion from nuScenes internal split names to our split names
        nuscenes_split_name_mapping = {
            "nuscenes_train": "train",
            "nuscenes_val": "val",
            "nuscenes_test": "test",
            "nuscenes-mini_train": "mini_train",
            "nuscenes-mini_val": "mini_val",
        }

        # Loads the mapping from split names to scene names in nuScenes
        scene_splits = create_splits_scenes()

        # Iterate over split names,
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

            # get token
            scene_tokens = [scene["token"] for scene in available_scenes if scene["name"] in scene_names]
            scene_tokens_per_split[split] = scene_tokens
        return scene_tokens_per_split

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(NUSCENES_MAPS)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return sum(len(scene_tokens) for scene_tokens in self._scene_tokens_per_split.values())

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        map_name = NUSCENES_MAPS[map_index]

        map_metadata = _get_nuscenes_map_metadata(map_name)
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)

        if map_needs_writing:
            write_nuscenes_map(nuscenes_maps_root=self._nuscenes_map_root, location=map_name, map_writer=map_writer)

        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""
        # Find the scene token for the given log index
        all_scene_tokens = []
        for split, scene_tokens in self._scene_tokens_per_split.items():
            all_scene_tokens.extend([(split, token) for token in scene_tokens])

        if log_index >= len(all_scene_tokens):
            raise ValueError(f"Log index {log_index} is out of range. Total logs: {len(all_scene_tokens)}")

        split, scene_token = all_scene_tokens[log_index]

        database_version = NUSCENES_DATABASE_VERSION_MAPPING[split]
        nusc = self._nuscenes_dbs[database_version]
        scene = nusc.get("scene", scene_token)
        log_record = nusc.get("log", scene["log_token"])

        # 1. Initialize log metadata
        log_metadata = LogMetadata(
            dataset="nuscenes",
            split=split,
            log_name=scene["name"],
            location=log_record["location"],
            timestep_seconds=NUSCENES_DT,
            vehicle_parameters=get_nuscenes_renault_zoe_parameters(),
            box_detection_label_class=NuScenesBoxDetectionLabel,
            pinhole_camera_metadata=_get_nuscenes_pinhole_camera_metadata(nusc, scene, self.dataset_converter_config),
            lidar_metadata=_get_nuscenes_lidar_metadata(nusc, scene, self.dataset_converter_config),
            map_metadata=_get_nuscenes_map_metadata(log_record["location"]),
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        if log_needs_writing:
            can_bus = NuScenesCanBus(dataroot=str(self._nuscenes_data_root))

            step_interval = max(1, int(NUSCENES_DT / NUSCENES_DT))
            sample_count = 0

            # Traverse all samples in the scene
            sample_token = scene["first_sample_token"]
            while sample_token:
                if sample_count % step_interval == 0:
                    sample = nusc.get("sample", sample_token)

                    log_writer.write(
                        timestamp=Timestamp.from_us(sample["timestamp"]),
                        ego_state=_extract_nuscenes_ego_state(nusc, sample, can_bus),
                        box_detections=_extract_nuscenes_box_detections(nusc, sample),
                        pinhole_cameras=_extract_nuscenes_cameras(
                            nusc=nusc,
                            sample=sample,
                            nuscenes_data_root=self._nuscenes_data_root,
                            dataset_converter_config=self.dataset_converter_config,
                        ),
                        lidars=[_l]
                        if (
                            _l := _extract_nuscenes_lidar(
                                nusc=nusc,
                                sample=sample,
                                nuscenes_data_root=self._nuscenes_data_root,
                                dataset_converter_config=self.dataset_converter_config,
                            )
                        )
                        is not None
                        else None,
                    )

                sample_token = sample["next"]
                sample_count += 1

        log_writer.close()
        del nusc
        gc.collect()


def _get_nuscenes_pinhole_camera_metadata(
    nusc: NuScenes,
    scene: Dict[str, Any],
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Extracts the pinhole camera metadata from a nuScenes scene."""
    camera_metadata: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    if dataset_converter_config.include_pinhole_cameras:
        first_sample_token = scene["first_sample_token"]
        first_sample = nusc.get("sample", first_sample_token)
        for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
            cam_token = first_sample["data"][camera_channel]
            cam_data = nusc.get("sample_data", cam_token)
            calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

            # Intrinsic & distortion parameters
            intrinsic_matrix = np.array(calib["camera_intrinsic"])
            intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsic_matrix)
            distortion = PinholeDistortion.from_array(np.zeros(5), copy=False)

            # Extrinsic parameters
            translation_array = np.array(calib["translation"], dtype=np.float64)  # array of shape (3,)
            rotation_array = np.array(calib["rotation"], dtype=np.float64)  # array of shape (4,)
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
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LidarID, LidarMetadata]:
    """Extracts the Lidar metadata from a nuScenes scene."""
    metadata: Dict[LidarID, LidarMetadata] = {}
    if dataset_converter_config.include_lidars:
        first_sample_token = scene["first_sample_token"]
        first_sample = nusc.get("sample", first_sample_token)
        lidar_token = first_sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)
        calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        translation = np.array(calib["translation"])
        rotation = Quaternion(calib["rotation"]).rotation_matrix
        lidar_to_imu_se3 = np.eye(4)
        lidar_to_imu_se3[:3, :3] = rotation
        lidar_to_imu_se3[:3, 3] = translation
        lidar_to_imu_se3 = PoseSE3.from_transformation_matrix(lidar_to_imu_se3)
        metadata[LidarID.LIDAR_TOP] = LidarMetadata(
            lidar_name="LIDAR_TOP",
            lidar_id=LidarID.LIDAR_TOP,
            lidar_to_imu_se3=lidar_to_imu_se3,
        )
    return metadata


def _get_nuscenes_map_metadata(location):
    """Creates nuScenes map metadata for a given location."""
    return MapMetadata(
        dataset="nuscenes",
        split=None,
        log_name=None,
        location=location,
        map_has_z=False,
        map_is_local=False,
    )


def _extract_nuscenes_ego_state(nusc, sample, can_bus) -> EgoStateSE3:
    """Extracts the ego state from a nuScenes sample."""
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    quat = Quaternion(ego_pose["rotation"])
    vehicle_parameters = get_nuscenes_renault_zoe_parameters()
    imu_pose = PoseSE3(
        x=ego_pose["translation"][0],
        y=ego_pose["translation"][1],
        z=ego_pose["translation"][2],
        qw=quat.w,
        qx=quat.x,
        qy=quat.y,
        qz=quat.z,
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
        vehicle_parameters=vehicle_parameters,
    )


def _extract_nuscenes_box_detections(nusc: NuScenes, sample: Dict[str, Any]) -> BoxDetectionsSE3:
    """Extracts the box detections from a nuScenes sample."""
    box_detections: List[BoxDetectionSE3] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))

        # Create PoseSE3 for box center and orientation
        center_quat = box.orientation
        center = PoseSE3(
            box.center[0],
            box.center[1],
            box.center[2],
            center_quat.w,
            center_quat.x,
            center_quat.y,
            center_quat.z,
        )
        bounding_box = BoundingBoxSE3(
            center_se3=center,
            length=box.wlh[1],
            width=box.wlh[0],
            height=box.wlh[2],
        )
        # Get detection type
        category = ann["category_name"]
        label = NUSCENES_DETECTION_NAME_DICT[category]

        # Get velocity if available
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
    dataset_converter_config: DatasetConverterConfig,
) -> List[CameraData]:
    """Extracts the pinhole camera metadata from a nuScenes scene."""
    camera_data_list: List[CameraData] = []
    if dataset_converter_config.include_pinhole_cameras:
        for camera_type, camera_channel in NUSCENES_CAMERA_IDS.items():
            cam_token = sample["data"][camera_channel]
            cam_data = nusc.get("sample_data", cam_token)

            # Check timestamp synchronization (within 100ms)
            if abs(cam_data["timestamp"] - sample["timestamp"]) > 100000:
                continue

            calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            translation_array = np.array(calib["translation"], dtype=np.float64)  # array of shape (3,)
            rotation_array = np.array(calib["rotation"], dtype=np.float64)  # array of shape (4,)
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
    dataset_converter_config: DatasetConverterConfig,
) -> Optional[LidarData]:
    """Extracts the Lidar data from a nuScenes sample."""
    lidar: Optional[LidarData] = None
    if dataset_converter_config.include_lidars:
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)
        absolute_lidar_path = nuscenes_data_root / lidar_data["filename"]
        if absolute_lidar_path.exists() and absolute_lidar_path.is_file():
            lidar = LidarData(
                lidar_name="LIDAR_TOP",
                lidar_type=LidarID.LIDAR_TOP,
                relative_path=absolute_lidar_path.relative_to(nuscenes_data_root),
                dataset_root=nuscenes_data_root,
                iteration=lidar_data.get("iteration"),
            )
    return lidar
