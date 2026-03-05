from __future__ import annotations

import logging
import pickle
import typing
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional, Tuple, Union

import numpy as np
import yaml

import py123d.parser.nuplan.utils as nuplan_utils
from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    BoxDetectionMetadata,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    DynamicStateSE3,
    EgoMetadata,
    EgoStateSE3,
    FisheyeMEICameraMetadatas,
    LidarID,
    LidarMetadata,
    LidarMetadatas,
    LogMetadata,
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeCameraMetadatas,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
)
from py123d.geometry import PoseSE3, Vector3D
from py123d.geometry.transform import reframe_se3_array
from py123d.parser.abstract_dataset_parser import (
    CameraData,
    DatasetParser,
    FrameData,
    LidarData,
    LogParser,
)
from py123d.parser.nuplan.nuplan_map_parser import NuplanMapParser
from py123d.parser.nuplan.utils.nuplan_constants import (
    NUPLAN_DATA_SPLITS,
    NUPLAN_DEFAULT_DT,
    NUPLAN_LIDAR_DICT,
    NUPLAN_MAP_LOCATIONS,
    NUPLAN_ROLLING_SHUTTER_S,
    NUPLAN_TRAFFIC_STATUS_DICT,
)
from py123d.parser.nuplan.utils.nuplan_sql_helper import (
    get_box_detections_for_lidarpc_token_from_db,
    get_nearest_ego_pose_for_timestamp_from_db,
)
from py123d.parser.registry import NuPlanBoxDetectionLabel

check_dependencies(["nuplan"], "nuplan")
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_cameras, get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.simulation.observation.observation_type import CameraChannel

# NOTE: Leaving this constant here, to avoid having a nuplan dependency in nuplan_constants.py
NUPLAN_CAMERA_MAPPING = {
    PinholeCameraID.PCAM_F0: CameraChannel.CAM_F0,
    PinholeCameraID.PCAM_B0: CameraChannel.CAM_B0,
    PinholeCameraID.PCAM_L0: CameraChannel.CAM_L0,
    PinholeCameraID.PCAM_L1: CameraChannel.CAM_L1,
    PinholeCameraID.PCAM_L2: CameraChannel.CAM_L2,
    PinholeCameraID.PCAM_R0: CameraChannel.CAM_R0,
    PinholeCameraID.PCAM_R1: CameraChannel.CAM_R1,
    PinholeCameraID.PCAM_R2: CameraChannel.CAM_R2,
}

TARGET_DT: Final[float] = 0.1  # TODO: make configurable

logger = logging.getLogger(__name__)


def _create_splits_logs() -> Dict[str, List[str]]:
    """Load the nuPlan log split assignments from the bundled YAML file."""
    yaml_filepath = Path(nuplan_utils.__path__[0]) / "log_splits.yaml"
    with open(yaml_filepath, "r", encoding="utf-8") as stream:
        splits = yaml.safe_load(stream)
    return splits["log_splits"]


class NuplanParser(DatasetParser):
    """Dataset parser for the nuPlan dataset."""

    def __init__(
        self,
        splits: List[str],
        nuplan_data_root: Union[Path, str],
        nuplan_maps_root: Union[Path, str],
        nuplan_sensor_root: Union[Path, str],
    ) -> None:
        """Initializes the NuplanParser.

        :param splits: List of splits to convert, e.g. ["nuplan_train", "nuplan_val"].
        :param nuplan_data_root: Root directory of the nuPlan data.
        :param nuplan_maps_root: Root directory of the nuPlan maps.
        :param nuplan_sensor_root: Root directory of the nuPlan sensor data.
        """
        assert nuplan_data_root is not None, "The variable `nuplan_data_root` must be provided."
        assert nuplan_maps_root is not None, "The variable `nuplan_maps_root` must be provided."
        assert nuplan_sensor_root is not None, "The variable `nuplan_sensor_root` must be provided."

        for split in splits:
            assert split in NUPLAN_DATA_SPLITS, (
                f"Split {split} is not available. Available splits: {NUPLAN_DATA_SPLITS}"
            )

        self._splits = splits
        self._nuplan_data_root = Path(nuplan_data_root)
        self._nuplan_maps_root = Path(nuplan_maps_root)
        self._nuplan_sensor_root = Path(nuplan_sensor_root)
        self._split_log_path_pairs: List[Tuple[str, Path]] = self._collect_split_log_path_pairs()

    def _collect_split_log_path_pairs(self) -> List[Tuple[str, Path]]:
        """Collects the (split, log_path) pairs for the specified splits."""
        split_log_path_pairs: List[Tuple[str, Path]] = []
        log_names_per_split = _create_splits_logs()

        for split in self._splits:
            split_type = split.split("_")[-1]
            assert split_type in {"train", "val", "test"}

            if split in {"nuplan_train", "nuplan_val"}:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "trainval"
            elif split in {"nuplan_test"}:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "test"
            elif split in {"nuplan-mini_train", "nuplan-mini_val", "nuplan-mini_test"}:
                nuplan_split_folder = self._nuplan_data_root / "nuplan-v1.1" / "splits" / "mini"
            else:
                raise ValueError(f"Unknown nuPlan split: {split}")

            all_log_files_in_path = list(nuplan_split_folder.glob("*.db"))
            all_log_names = {str(log_file.stem) for log_file in all_log_files_in_path}
            log_names_in_split = set(log_names_per_split[split_type])
            valid_log_names = list(all_log_names & log_names_in_split)

            for log_name in valid_log_names:
                log_path = nuplan_split_folder / f"{log_name}.db"
                split_log_path_pairs.append((split, log_path))

        return split_log_path_pairs

    def get_log_parsers(self) -> List[NuplanLogParser]:  # type: ignore
        """Inherited, see superclass."""
        return [
            NuplanLogParser(
                split=split,
                source_log_path=source_log_path,
                nuplan_data_root=self._nuplan_data_root,
                nuplan_sensor_root=self._nuplan_sensor_root,
            )
            for split, source_log_path in self._split_log_path_pairs
        ]

    def get_map_parsers(self) -> List[NuplanMapParser]:  # type: ignore
        """Inherited, see superclass."""
        return [
            NuplanMapParser(nuplan_maps_root=self._nuplan_maps_root, location=location)
            for location in NUPLAN_MAP_LOCATIONS
        ]


class NuplanLogParser(LogParser):
    """Lightweight, picklable handle to one nuPlan log."""

    def __init__(
        self,
        split: str,
        source_log_path: Path,
        nuplan_data_root: Path,
        nuplan_sensor_root: Path,
    ) -> None:
        self._split = split
        self._source_log_path = source_log_path
        self._nuplan_data_root = nuplan_data_root
        self._nuplan_sensor_root = nuplan_sensor_root

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        nuplan_log_db = NuPlanDB(str(self._nuplan_data_root), str(self._source_log_path), None)
        log_name = nuplan_log_db.log_name
        location = nuplan_log_db.log.map_version

        metadata = LogMetadata(
            dataset="nuplan",
            split=self._split,
            log_name=log_name,
            location=location,
            timestep_seconds=TARGET_DT,
        )

        nuplan_log_db.detach_tables()
        nuplan_log_db.remove_ref()
        del nuplan_log_db

        return metadata

    @typing.override
    def get_ego_metadata(self) -> Optional[EgoMetadata]:
        """Inherited, see superclass."""
        # NOTE: These parameters are mostly available in nuPlan, except for the rear_axle_to_center_vertical.
        # The value is estimated based the Lidar point cloud.
        # [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
        return EgoMetadata(
            vehicle_name="nuplan_chrysler_pacifica",
            width=2.297,
            length=5.176,
            height=1.777,
            wheel_base=3.089,
            center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=0.45, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )

    @typing.override
    def get_box_detection_metadata(self) -> Optional[BoxDetectionMetadata]:
        """Inherited, see superclass."""
        return BoxDetectionMetadata(box_detection_label_class=NuPlanBoxDetectionLabel)

    def get_pinhole_camera_metadatas(self) -> Optional[PinholeCameraMetadatas]:
        """Inherited, see superclass."""
        camera_metadata = _get_nuplan_camera_metadata(
            self._source_log_path,
            self._nuplan_sensor_root,
        )
        if camera_metadata:
            return PinholeCameraMetadatas(camera_metadata)
        return None

    @typing.override
    def get_fisheye_mei_camera_metadatas(self) -> Optional[FisheyeMEICameraMetadatas]:
        """Inherited, see superclass."""
        return None

    def get_lidar_metadatas(self) -> Optional[LidarMetadatas]:
        """Inherited, see superclass."""
        log_name = self._source_log_path.stem
        lidar_metadata = _get_nuplan_lidar_metadata(
            self._nuplan_sensor_root,
            log_name,
        )
        if lidar_metadata:
            return LidarMetadatas(lidar_metadata)
        return None

    def iter_frames(self) -> Iterator[FrameData]:
        """Inherited, see superclass."""
        nuplan_log_db = NuPlanDB(str(self._nuplan_data_root), str(self._source_log_path), None)
        ego_metadata = self.get_ego_metadata()
        assert ego_metadata is not None

        try:
            step_interval: int = int(TARGET_DT / NUPLAN_DEFAULT_DT)
            offset = _get_ideal_lidar_pc_offset(self._source_log_path, nuplan_log_db)
            num_steps = len(nuplan_log_db.lidar_pc)

            for lidar_pc_index in range(offset, num_steps, step_interval):
                nuplan_lidar_pc = nuplan_log_db.lidar_pc[lidar_pc_index]
                lidar_pc_token: str = nuplan_lidar_pc.token
                timestamp = Timestamp.from_us(nuplan_lidar_pc.timestamp)

                yield FrameData(
                    timestamp=timestamp,
                    ego_state_se3=_extract_nuplan_ego_state(nuplan_lidar_pc, ego_metadata),
                    box_detections_se3=_extract_nuplan_box_detections(
                        nuplan_lidar_pc,
                        self._source_log_path,
                        timestamp,
                    ),
                    traffic_lights=_extract_nuplan_traffic_lights(nuplan_log_db, lidar_pc_token, timestamp),
                    pinhole_cameras=_extract_nuplan_cameras(
                        nuplan_log_db=nuplan_log_db,
                        nuplan_lidar_pc=nuplan_lidar_pc,
                        source_log_path=self._source_log_path,
                        nuplan_sensor_root=self._nuplan_sensor_root,
                    ),
                    lidar=_extract_nuplan_lidar_data(
                        nuplan_lidar_pc=nuplan_lidar_pc,
                        nuplan_sensor_root=self._nuplan_sensor_root,
                    ),
                )
                del nuplan_lidar_pc
        finally:
            # NOTE: The nuPlanDB class has several internal references, which makes memory management tricky.
            # We need to ensure all references are released properly.
            nuplan_log_db.detach_tables()
            nuplan_log_db.remove_ref()
            del nuplan_log_db


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_nuplan_camera_metadata(
    source_log_path: Path,
    nuplan_sensor_root: Path,
) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Extracts the nuPlan camera metadata for a given log."""

    def _get_camera_metadata(camera_id: PinholeCameraID) -> PinholeCameraMetadata:
        cam = list(get_cameras(str(source_log_path), [str(NUPLAN_CAMERA_MAPPING[camera_id].value)]))[0]

        # Load intrinsics
        intrinsics_camera_matrix = np.array(pickle.loads(cam.intrinsic), dtype=np.float64)  # type: ignore
        intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsics_camera_matrix)

        # Load distortion
        distortion_array = np.array(pickle.loads(cam.distortion), dtype=np.float64)  # type: ignore
        distortion = PinholeDistortion.from_array(distortion_array, copy=False)

        # Load static extrinsic
        translation_array = np.array(pickle.loads(cam.translation), dtype=np.float64)  # type: ignore
        rotation_array = np.array(pickle.loads(cam.rotation), dtype=np.float64)  # type: ignore
        extrinsic = PoseSE3.from_R_t(rotation=rotation_array, translation=translation_array)

        return PinholeCameraMetadata(
            camera_name=str(NUPLAN_CAMERA_MAPPING[camera_id].value),
            camera_id=camera_id,
            width=cam.width,  # type: ignore
            height=cam.height,  # type: ignore
            intrinsics=intrinsic,
            distortion=distortion,
            camera_to_imu_se3=extrinsic,
        )

    camera_metadata: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    log_name = source_log_path.stem
    for camera_id, nuplan_camera_type in NUPLAN_CAMERA_MAPPING.items():
        camera_folder = nuplan_sensor_root / log_name / f"{nuplan_camera_type.value}"
        if camera_folder.exists() and camera_folder.is_dir():
            camera_metadata[camera_id] = _get_camera_metadata(camera_id)

    return camera_metadata


def _get_nuplan_lidar_metadata(
    nuplan_sensor_root: Path,
    log_name: str,
) -> Dict[LidarID, LidarMetadata]:
    """Extracts the nuPlan Lidar metadata for a given log."""
    metadata: Dict[LidarID, LidarMetadata] = {}
    log_lidar_folder = nuplan_sensor_root / log_name / "MergedPointCloud"
    # NOTE: We first need to check if the Lidar folder exists, as not all logs have Lidar data
    if log_lidar_folder.exists() and log_lidar_folder.is_dir():
        for lidar_type in NUPLAN_LIDAR_DICT.values():
            metadata[lidar_type] = LidarMetadata(
                lidar_name=lidar_type.serialize(),  # NOTE: nuPlan does not have specific names for the Lidars
                lidar_id=lidar_type,
                lidar_to_imu_se3=PoseSE3.identity(),  # NOTE: Lidar extrinsic are unknown
            )
    return metadata


# ------------------------------------------------------------------------------------------------------------------
# Sensor extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _extract_nuplan_ego_state(nuplan_lidar_pc: LidarPc, ego_metadata: EgoMetadata) -> EgoStateSE3:
    """Extracts the nuPlan ego state from a given LidarPc database object."""
    imu_pose = PoseSE3(
        x=nuplan_lidar_pc.ego_pose.x,
        y=nuplan_lidar_pc.ego_pose.y,
        z=nuplan_lidar_pc.ego_pose.z,
        qw=nuplan_lidar_pc.ego_pose.qw,
        qx=nuplan_lidar_pc.ego_pose.qx,
        qy=nuplan_lidar_pc.ego_pose.qy,
        qz=nuplan_lidar_pc.ego_pose.qz,
    )
    dynamic_state_se3 = DynamicStateSE3(
        velocity=Vector3D(
            x=nuplan_lidar_pc.ego_pose.vx,
            y=nuplan_lidar_pc.ego_pose.vy,
            z=nuplan_lidar_pc.ego_pose.vz,
        ),
        acceleration=Vector3D(
            x=nuplan_lidar_pc.ego_pose.acceleration_x,
            y=nuplan_lidar_pc.ego_pose.acceleration_y,
            z=nuplan_lidar_pc.ego_pose.acceleration_z,
        ),
        angular_velocity=Vector3D(
            x=nuplan_lidar_pc.ego_pose.angular_rate_x,
            y=nuplan_lidar_pc.ego_pose.angular_rate_y,
            z=nuplan_lidar_pc.ego_pose.angular_rate_z,
        ),
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_pose,
        vehicle_parameters=ego_metadata,
        dynamic_state_se3=dynamic_state_se3,
        timestamp=Timestamp.from_us(nuplan_lidar_pc.ego_pose.timestamp),
    )


def _extract_nuplan_box_detections(lidar_pc: LidarPc, source_log_path: Path, timestamp: Timestamp) -> BoxDetectionsSE3:
    """Extracts the nuPlan box detections from a given LidarPc database object."""
    box_detections: List[BoxDetectionSE3] = get_box_detections_for_lidarpc_token_from_db(
        str(source_log_path), lidar_pc.token
    )
    return BoxDetectionsSE3(box_detections=box_detections, timestamp=timestamp)  # type: ignore


def _extract_nuplan_traffic_lights(
    log_db: NuPlanDB, lidar_pc_token: str, timestamp: Timestamp
) -> TrafficLightDetections:
    """Extracts the nuPlan traffic light detections from a given LidarPc database object."""
    detections: List[TrafficLightDetection] = [
        TrafficLightDetection(
            lane_id=int(traffic_light.lane_connector_id),
            status=NUPLAN_TRAFFIC_STATUS_DICT[traffic_light.status],
        )
        for traffic_light in log_db.traffic_light_status.select_many(lidar_pc_token=lidar_pc_token)
    ]
    return TrafficLightDetections(detections=detections, timestamp=timestamp)


def _extract_nuplan_cameras(
    nuplan_log_db: NuPlanDB,
    nuplan_lidar_pc: LidarPc,
    source_log_path: Path,
    nuplan_sensor_root: Path,
) -> List[CameraData]:
    """Extracts the nuPlan camera data from a given LidarPc database object."""
    camera_data_list: List[CameraData] = []
    current_ego_pose = PoseSE3.from_transformation_matrix(nuplan_lidar_pc.ego_pose.trans_matrix)

    log_cam_infos = {camera.token: camera for camera in nuplan_log_db.log.cameras}
    for camera_type, camera_channel in NUPLAN_CAMERA_MAPPING.items():
        image_class = list(
            get_images_from_lidar_tokens(
                log_file=str(source_log_path), tokens=[nuplan_lidar_pc.token], channels=[str(camera_channel.value)]
            )
        )

        if len(image_class) != 0:
            image = image_class[0]
            filename_jpg = nuplan_sensor_root / image.filename_jpg  # type: ignore
            if filename_jpg.exists() and filename_jpg.is_file():
                # Query nearest ego pose for the image timestamp
                timestamp = image.timestamp + NUPLAN_ROLLING_SHUTTER_S.time_us  # type: ignore
                nearest_ego_poses, timestamp_poses = get_nearest_ego_pose_for_timestamp_from_db(
                    str(source_log_path),
                    timestamp,
                    [nuplan_lidar_pc.token],
                )
                nearest_ego_pose = nearest_ego_poses[np.argmin(timestamp_poses)]
                extrinsic_static = PoseSE3.from_transformation_matrix(log_cam_infos[image.camera_token].trans_matrix)  # type: ignore
                extrinsic_compensated_array = reframe_se3_array(
                    from_origin=nearest_ego_pose, to_origin=current_ego_pose, pose_se3_array=extrinsic_static.array
                )
                extrinsic = PoseSE3.from_array(extrinsic_compensated_array)

                camera_data_list.append(
                    CameraData(
                        camera_name=str(camera_channel.value),
                        camera_id=camera_type,
                        extrinsic=extrinsic,
                        dataset_root=nuplan_sensor_root,
                        relative_path=filename_jpg.relative_to(nuplan_sensor_root),
                        timestamp=Timestamp.from_us(image.timestamp),  # type: ignore
                    )
                )
    return camera_data_list


def _extract_nuplan_lidar_data(
    nuplan_lidar_pc: LidarPc,
    nuplan_sensor_root: Path,
) -> Optional[LidarData]:
    """Extracts the nuPlan Lidar data from a given LidarPc database object."""

    lidar_full_path: Path = nuplan_sensor_root / nuplan_lidar_pc.filename
    if lidar_full_path.exists() and lidar_full_path.is_file():
        return LidarData(
            lidar_name=LidarID.LIDAR_MERGED.serialize(),
            lidar_type=LidarID.LIDAR_MERGED,
            start_timestamp=Timestamp.from_us(nuplan_lidar_pc.timestamp),
            end_timestamp=Timestamp.from_us(nuplan_lidar_pc.timestamp),
            dataset_root=nuplan_sensor_root,
            relative_path=nuplan_lidar_pc.filename,
        )
    else:
        logger.debug(f"Lidar file not found: {lidar_full_path}")
    return None


def _get_ideal_lidar_pc_offset(source_log_path: Path, nuplan_log_db: NuPlanDB) -> int:
    """Helper function to get the ideal initial step offset of a log.

    NOTE: In nuPlan, lidars are captured at 20Hz (every 50ms), whereas cameras are captured at 10Hz (every 100ms).
    However, cameras are triggered with the sweeping lidar motion, thus within a time-frame of [-25ms, 25ms] of every
    second sweep. We need to find out which alternating sweep provides a better camera matching.

    :param source_log_path: Path to the source log .db file.
    :param nuplan_log_db: The nuPlan database object.
    :return: Either 0 or 1, as integer.
    """
    QUERY_START: int = 10
    average_offsets = np.zeros((2,), dtype=np.float64)
    for offset in [0, 1]:
        lidar_pc = nuplan_log_db.lidar_pc[QUERY_START + offset]
        lidar_pc_timestamp_us = lidar_pc.timestamp
        camera_channels = [str(channel.value) for channel in NUPLAN_CAMERA_MAPPING.values()]
        images = list(
            get_images_from_lidar_tokens(
                log_file=str(source_log_path),
                tokens=[lidar_pc.token],
                channels=camera_channels,
            )
        )
        absolute_time_offset_ms = []
        for image in images:
            image_timestamp_us = image.timestamp
            absolute_time_offset_ms.append(abs(image_timestamp_us - lidar_pc_timestamp_us) / 1e3)
        average_offsets[offset] = np.mean(absolute_time_offset_ms)

    return int(np.argmin(average_offsets))
