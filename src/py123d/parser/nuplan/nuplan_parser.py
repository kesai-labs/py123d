from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional, Tuple, Union

import numpy as np
import yaml

import py123d.parser.nuplan.utils as nuplan_utils
from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    DynamicStateSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
    TrafficLightDetection,
    TrafficLightDetections,
)
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.lidar import LidarMergedMetadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, EulerAngles, PoseSE3, Vector3D
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.nuplan.nuplan_map_parser import NuplanMapParser
from py123d.parser.nuplan.utils.nuplan_constants import (
    NUPLAN_BOX_DETECTIONS_SE3_METADATA,
    NUPLAN_DATA_SPLITS,
    NUPLAN_DEFAULT_DT,
    NUPLAN_DETECTION_NAME_DICT,
    NUPLAN_EGO_STATE_SE3_METADATA,
    NUPLAN_LIDAR_DICT,
    NUPLAN_LIDAR_SWEEP_DURATION_US,
    NUPLAN_MAP_LOCATIONS,
    NUPLAN_ROLLING_SHUTTER_S,
    NUPLAN_TRAFFIC_STATUS_DICT,
)
from py123d.parser.nuplan.utils.nuplan_sql_helper import (
    get_box_detections_for_lidarpc_token_from_db,
    get_interpolated_ego_pose_from_db,
    iter_all_box_detections_from_db,
    iter_all_ego_poses_from_db,
    iter_all_images_from_db,
    iter_all_lidar_pc_from_db,
    iter_all_traffic_lights_from_db,
)

check_dependencies(["nuplan"], "nuplan")
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_cameras, get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.planning.simulation.observation.observation_type import CameraChannel

# NOTE: Leaving this constant here, to avoid having a nuplan dependency in nuplan_constants.py
NUPLAN_CAMERA_MAPPING = {
    CameraID.PCAM_F0: CameraChannel.CAM_F0,
    CameraID.PCAM_B0: CameraChannel.CAM_B0,
    CameraID.PCAM_L0: CameraChannel.CAM_L0,
    CameraID.PCAM_L1: CameraChannel.CAM_L1,
    CameraID.PCAM_L2: CameraChannel.CAM_L2,
    CameraID.PCAM_R0: CameraChannel.CAM_R0,
    CameraID.PCAM_R1: CameraChannel.CAM_R1,
    CameraID.PCAM_R2: CameraChannel.CAM_R2,
}

TARGET_DT: Final[float] = 0.1  # TODO: make configurable

logger = logging.getLogger(__name__)


def _create_splits_logs() -> Dict[str, List[str]]:
    """Load the nuPlan log split assignments from the bundled YAML file."""
    yaml_filepath = Path(nuplan_utils.__path__[0]) / "log_splits.yaml"
    with open(yaml_filepath, "r", encoding="utf-8") as stream:
        splits = yaml.safe_load(stream)
    return splits["log_splits"]


class NuplanParser(BaseDatasetParser):
    """Dataset parser for the nuPlan dataset."""

    def __init__(
        self,
        splits: List[str],
        nuplan_data_root: Union[Path, str],
        nuplan_maps_root: Union[Path, str],
        nuplan_sensor_root: Union[Path, str],
        log_names: Optional[List[str]] = None,
    ) -> None:
        """Initializes the NuplanParser.

        :param splits: List of splits to convert, e.g. ["nuplan_train", "nuplan_val"].
        :param log_names: Optional list of log names to convert. If None, all logs in the specified splits will be converted.
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
        self._log_names = log_names
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

            if self._log_names is not None:
                valid_log_names = [log_name for log_name in valid_log_names if log_name in self._log_names]

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


class NuplanLogParser(BaseLogParser):
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

        nuplan_log_db.detach_tables()
        nuplan_log_db.remove_ref()
        del nuplan_log_db

        return LogMetadata(
            dataset="nuplan",
            split=self._split,
            log_name=log_name,
            location=location,
            map_metadata=MapMetadata(
                dataset="nuplan",
                location=location,
                map_has_z=False,
                map_is_per_log=False,
            ),
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        ego_state_se3_metadata = NUPLAN_EGO_STATE_SE3_METADATA
        box_detections_se3_metadata = NUPLAN_BOX_DETECTIONS_SE3_METADATA
        pinhole_camera_metadatas = _get_nuplan_camera_metadata(self._source_log_path, self._nuplan_sensor_root)
        lidar_merged_metadata = _get_nuplan_lidar_merged_metadata(self._nuplan_sensor_root, self._source_log_path.stem)

        nuplan_log_db = NuPlanDB(str(self._nuplan_data_root), str(self._source_log_path), None)

        try:
            step_interval: int = int(TARGET_DT / NUPLAN_DEFAULT_DT)
            offset = _get_ideal_lidar_pc_offset(self._source_log_path, nuplan_log_db)
            num_steps = len(nuplan_log_db.lidar_pc)

            for lidar_pc_index in range(offset, num_steps, step_interval):
                nuplan_lidar_pc = nuplan_log_db.lidar_pc[lidar_pc_index]
                lidar_pc_token: str = nuplan_lidar_pc.token
                timestamp = Timestamp.from_us(nuplan_lidar_pc.timestamp)

                ego_state_se3 = _extract_nuplan_ego_state(nuplan_lidar_pc, ego_state_se3_metadata)
                box_detections_se3 = _extract_nuplan_box_detections(
                    nuplan_lidar_pc, self._source_log_path, timestamp, box_detections_se3_metadata
                )
                traffic_lights = _extract_nuplan_traffic_lights(nuplan_log_db, lidar_pc_token, timestamp)
                parsed_pinhole_cameras = _extract_nuplan_cameras(
                    lidar_pc_token=lidar_pc_token,
                    source_log_path=self._source_log_path,
                    nuplan_sensor_root=self._nuplan_sensor_root,
                    metadatas=pinhole_camera_metadatas,
                )
                parsed_lidar = _extract_nuplan_lidar_data(
                    nuplan_lidar_pc=nuplan_lidar_pc,
                    nuplan_sensor_root=self._nuplan_sensor_root,
                    metadata=lidar_merged_metadata,
                )

                modalities: List[BaseModality] = [ego_state_se3, box_detections_se3, traffic_lights]
                modalities.extend(parsed_pinhole_cameras)
                if parsed_lidar is not None:
                    modalities.append(parsed_lidar)

                yield ModalitiesSync(timestamp=timestamp, modalities=modalities)
                del nuplan_lidar_pc
        finally:
            # NOTE: The nuPlanDB class has several internal references, which makes memory management tricky.
            # We need to ensure all references are released properly.
            nuplan_log_db.detach_tables()
            nuplan_log_db.remove_ref()
            del nuplan_log_db

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        ego_state_se3_metadata = NUPLAN_EGO_STATE_SE3_METADATA
        box_detections_se3_metadata = NUPLAN_BOX_DETECTIONS_SE3_METADATA
        lidar_merged_metadata = _get_nuplan_lidar_merged_metadata(self._nuplan_sensor_root, self._source_log_path.stem)

        yield from self._iter_ego_states_se3(ego_state_se3_metadata)
        yield from self._iter_box_detections_se3(box_detections_se3_metadata)
        yield from self._iter_traffic_lights()
        yield from self._iter_lidars(lidar_merged_metadata)
        yield from self._iter_pinhole_cameras()

    # ------------------------------------------------------------------------------------------------------------------
    # Per-modality iterators (async / native-rate)
    # ------------------------------------------------------------------------------------------------------------------

    def _iter_ego_states_se3(self, modality_metadata: EgoStateSE3Metadata) -> Iterator[EgoStateSE3]:
        """Yields all ego state observations at native rate from the ego_pose table."""
        for row in iter_all_ego_poses_from_db(str(self._source_log_path)):
            imu_pose = PoseSE3(
                x=row["x"],
                y=row["y"],
                z=row["z"],
                qw=row["qw"],
                qx=row["qx"],
                qy=row["qy"],
                qz=row["qz"],
            )
            dynamic_state = DynamicStateSE3(
                velocity=Vector3D(x=row["vx"], y=row["vy"], z=row["vz"]),
                acceleration=Vector3D(
                    x=row["acceleration_x"],
                    y=row["acceleration_y"],
                    z=row["acceleration_z"],
                ),
                angular_velocity=Vector3D(
                    x=row["angular_rate_x"],
                    y=row["angular_rate_y"],
                    z=row["angular_rate_z"],
                ),
            )
            yield EgoStateSE3.from_imu(
                imu_se3=imu_pose,
                metadata=modality_metadata,
                dynamic_state_se3=dynamic_state,
                timestamp=Timestamp.from_us(row["timestamp"]),
            )

    def _iter_box_detections_se3(self, modality_metadata: BoxDetectionsSE3Metadata) -> Iterator[BoxDetectionsSE3]:
        """Yields all box detection frames at native lidar rate.

        Iterates over all lidar_pc timestamps so that timestamps with no box detections still produce an
        empty ``BoxDetectionsSE3`` entry. This prevents temporal gaps in the reference modality when used
        with async conversion.
        """
        # Build a mapping: lidar_pc timestamp -> list of box detection rows.
        boxes_by_timestamp: Dict[int, list] = {}
        for row in iter_all_box_detections_from_db(str(self._source_log_path)):
            boxes_by_timestamp.setdefault(row["timestamp"], []).append(row)

        # Iterate over ALL lidar_pc timestamps (sorted), yielding empty detections where no boxes exist.
        for lidar_row in iter_all_lidar_pc_from_db(str(self._source_log_path)):
            timestamp = lidar_row["timestamp"]
            detection_rows = boxes_by_timestamp.get(timestamp, [])

            box_detections: List[BoxDetectionSE3] = []
            for row in detection_rows:
                quaternion = EulerAngles(roll=DEFAULT_ROLL, pitch=DEFAULT_PITCH, yaw=row["yaw"]).quaternion
                bounding_box = BoundingBoxSE3(
                    center_se3=PoseSE3(
                        x=row["x"],
                        y=row["y"],
                        z=row["z"],
                        qw=quaternion.qw,
                        qx=quaternion.qx,
                        qy=quaternion.qy,
                        qz=quaternion.qz,
                    ),
                    length=row["length"],
                    width=row["width"],
                    height=row["height"],
                )
                box_detections.append(
                    BoxDetectionSE3(
                        attributes=BoxDetectionAttributes(
                            label=NUPLAN_DETECTION_NAME_DICT[row["category_name"]],
                            track_token=row["track_token"].hex(),
                        ),
                        bounding_box_se3=bounding_box,
                        velocity_3d=Vector3D(x=row["vx"], y=row["vy"], z=row["vz"]),
                    )
                )
            yield BoxDetectionsSE3(
                box_detections=box_detections,
                timestamp=Timestamp.from_us(timestamp),
                metadata=modality_metadata,
            )

    def _iter_traffic_lights(self) -> Iterator[TrafficLightDetections]:
        """Yields all traffic light detection frames at native lidar rate.

        Iterates over all lidar_pc timestamps so that timestamps with no traffic light detections still
        produce an empty ``TrafficLightDetections`` entry. This prevents temporal gaps in the sync table
        during async conversion.
        """
        # Build a mapping: lidar_pc timestamp -> list of traffic light rows.
        tl_by_timestamp: Dict[int, list] = {}
        for row in iter_all_traffic_lights_from_db(str(self._source_log_path)):
            tl_by_timestamp.setdefault(row["timestamp"], []).append(row)

        # Iterate over ALL lidar_pc timestamps, yielding empty detections where none exist.
        for lidar_row in iter_all_lidar_pc_from_db(str(self._source_log_path)):
            timestamp = lidar_row["timestamp"]
            detection_rows = tl_by_timestamp.get(timestamp, [])

            detections: List[TrafficLightDetection] = [
                TrafficLightDetection(
                    lane_id=int(row["lane_connector_id"]),
                    status=NUPLAN_TRAFFIC_STATUS_DICT[row["status"]],
                )
                for row in detection_rows
            ]
            yield TrafficLightDetections(detections=detections, timestamp=Timestamp.from_us(timestamp))

    def _iter_pinhole_cameras(self) -> Iterator[ParsedCamera]:
        """Yields pinhole camera observations for all cameras at native rate (~10Hz).

        Iterates the image table once and dispatches each image to its corresponding camera metadata.
        For each image, the ego pose is interpolated at the rolling-shutter-compensated camera timestamp
        and composed with the static camera-to-IMU extrinsic to produce a camera-to-global pose.
        """
        pinhole_camera_metadatas = _get_nuplan_camera_metadata(self._source_log_path, self._nuplan_sensor_root)
        channel_to_camera_id = {str(v.value): k for k, v in NUPLAN_CAMERA_MAPPING.items()}
        log_file = str(self._source_log_path)

        for row in iter_all_images_from_db(log_file):
            channel = row["channel"]
            if channel not in channel_to_camera_id:
                continue

            camera_id = channel_to_camera_id[channel]
            if camera_id not in pinhole_camera_metadatas:
                continue

            filename_jpg = row["filename_jpg"]
            full_path = self._nuplan_sensor_root / filename_jpg
            if not full_path.exists():
                continue

            camera_metadata = pinhole_camera_metadatas[camera_id]

            # Interpolate ego pose at the rolling-shutter-compensated camera timestamp
            compensated_timestamp_us = row["timestamp"] + NUPLAN_ROLLING_SHUTTER_S.time_us
            ego_pose = get_interpolated_ego_pose_from_db(log_file, compensated_timestamp_us)
            camera_to_global_se3 = rel_to_abs_se3(
                origin=ego_pose,
                pose_se3=camera_metadata.camera_to_imu_se3,
            )

            yield ParsedCamera(
                metadata=camera_metadata,
                timestamp=Timestamp.from_us(row["timestamp"]),
                camera_to_global_se3=camera_to_global_se3,
                dataset_root=self._nuplan_sensor_root,
                relative_path=filename_jpg,
            )

    def _iter_lidars(self, modality_metadata: LidarMergedMetadata) -> Iterator[ParsedLidar]:
        """Yields all lidar sweeps at native rate (20Hz)."""
        for row in iter_all_lidar_pc_from_db(str(self._source_log_path)):
            lidar_full_path = self._nuplan_sensor_root / row["filename"]
            if lidar_full_path.exists() and lidar_full_path.is_file():
                yield ParsedLidar(
                    metadata=modality_metadata,
                    start_timestamp=Timestamp.from_us(row["timestamp"]),
                    end_timestamp=Timestamp.from_us(row["timestamp"] + NUPLAN_LIDAR_SWEEP_DURATION_US),
                    dataset_root=self._nuplan_sensor_root,
                    relative_path=row["filename"],
                )


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_nuplan_camera_metadata(
    source_log_path: Path,
    nuplan_sensor_root: Path,
) -> Dict[CameraID, PinholeCameraMetadata]:
    """Extracts the nuPlan camera metadata for a given log."""

    def _get_camera_metadata(camera_id: CameraID) -> PinholeCameraMetadata:
        cam = list(get_cameras(str(source_log_path), [str(NUPLAN_CAMERA_MAPPING[camera_id].value)]))[0]

        # Load intrinsics
        intrinsics_camera_matrix = np.array(pickle.loads(cam.intrinsic), dtype=np.float64)  # type: ignore
        intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsics_camera_matrix)

        # Load distortion
        distortion_array = np.array(pickle.loads(cam.distortion), dtype=np.float64)  # type: ignore
        distortion = PinholeDistortion.from_array(distortion_array, copy=False)

        # Load static extrinsic (camera-to-ego transform)
        # NOTE: nuPlan stores rotation as a scalar-first quaternion [w, x, y, z], matching QuaternionIndex.
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

    camera_metadata: Dict[CameraID, PinholeCameraMetadata] = {}
    log_name = source_log_path.stem
    for camera_id, nuplan_camera_type in NUPLAN_CAMERA_MAPPING.items():
        camera_folder = nuplan_sensor_root / log_name / f"{nuplan_camera_type.value}"
        if camera_folder.exists() and camera_folder.is_dir():
            camera_metadata[camera_id] = _get_camera_metadata(camera_id)

    return camera_metadata


def _get_nuplan_lidar_merged_metadata(
    nuplan_sensor_root: Path,
    log_name: str,
) -> LidarMergedMetadata:
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
    return LidarMergedMetadata(metadata)


# ------------------------------------------------------------------------------------------------------------------
# Sensor extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _extract_nuplan_ego_state(
    nuplan_lidar_pc: LidarPc,
    ego_state_se3_metadata: EgoStateSE3Metadata,
) -> EgoStateSE3:
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
        metadata=ego_state_se3_metadata,
        dynamic_state_se3=dynamic_state_se3,
        timestamp=Timestamp.from_us(nuplan_lidar_pc.ego_pose.timestamp),
    )


def _extract_nuplan_box_detections(
    lidar_pc: LidarPc,
    source_log_path: Path,
    timestamp: Timestamp,
    box_detections_se3_metadata: BoxDetectionsSE3Metadata,
) -> BoxDetectionsSE3:
    """Extracts the nuPlan box detections from a given LidarPc database object."""
    box_detections: List[BoxDetectionSE3] = get_box_detections_for_lidarpc_token_from_db(
        str(source_log_path), lidar_pc.token
    )
    return BoxDetectionsSE3(
        box_detections=box_detections,
        timestamp=timestamp,
        metadata=box_detections_se3_metadata,
    )


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
    lidar_pc_token: str,
    source_log_path: Path,
    nuplan_sensor_root: Path,
    metadatas: Dict[CameraID, PinholeCameraMetadata],
) -> List[ParsedCamera]:
    """Extracts the nuPlan camera data for cameras associated with a given lidar sweep.

    For each camera image matched to the lidar sweep, the ego pose is interpolated at the
    rolling-shutter-compensated camera timestamp and composed with the static camera-to-IMU
    extrinsic to produce a camera-to-global pose.

    :param lidar_pc_token: Token of the lidar point cloud sweep.
    :param source_log_path: Path to the source log ``.db`` file.
    :param nuplan_sensor_root: Root directory of the nuPlan sensor data.
    :param metadatas: Camera metadata dict keyed by :class:`CameraID`.
    :return: List of :class:`ParsedCamera` for all matched camera images.
    """
    camera_data_list: List[ParsedCamera] = []
    log_file = str(source_log_path)

    for camera_type, camera_channel in NUPLAN_CAMERA_MAPPING.items():
        if camera_type not in metadatas:
            continue

        images = list(
            get_images_from_lidar_tokens(
                log_file=log_file, tokens=[lidar_pc_token], channels=[str(camera_channel.value)]
            )
        )
        if len(images) == 0:
            continue

        image = images[0]
        image_path = nuplan_sensor_root / image.filename_jpg  # type: ignore
        if not (image_path.exists() and image_path.is_file()):
            continue

        # Interpolate ego pose at the rolling-shutter-compensated camera timestamp
        compensated_timestamp_us = image.timestamp + NUPLAN_ROLLING_SHUTTER_S.time_us  # type: ignore
        ego_pose = get_interpolated_ego_pose_from_db(log_file, compensated_timestamp_us)
        camera_to_global_se3 = rel_to_abs_se3(
            origin=ego_pose,
            pose_se3=metadatas[camera_type].camera_to_imu_se3,
        )

        # NOTE @DanielDauner: We pass the original image timestamp here (without rolling shutter compensation).
        # This could be changed in the future, but documentation here remains scarce, and compensation is a heuristic.
        camera_data_list.append(
            ParsedCamera(
                metadata=metadatas[camera_type],
                timestamp=Timestamp.from_us(image.timestamp),  # type: ignore
                camera_to_global_se3=camera_to_global_se3,
                dataset_root=nuplan_sensor_root,
                relative_path=image_path.relative_to(nuplan_sensor_root),
            )
        )

    return camera_data_list


def _extract_nuplan_lidar_data(
    nuplan_lidar_pc: LidarPc,
    nuplan_sensor_root: Path,
    metadata: LidarMergedMetadata,
) -> Optional[ParsedLidar]:
    """Extracts the nuPlan Lidar data from a given LidarPc database object.

    :param nuplan_lidar_pc: The nuPlan LidarPc database object.
    :param nuplan_sensor_root: Root directory of the nuPlan sensor data.
    :param metadata: Lidar merged metadata.
    :return: A :class:`ParsedLidar` if the file exists, otherwise ``None``.
    """
    parsed_lidar: Optional[ParsedLidar] = None
    lidar_full_path: Path = nuplan_sensor_root / nuplan_lidar_pc.filename

    if lidar_full_path.exists() and lidar_full_path.is_file():
        parsed_lidar = ParsedLidar(
            metadata=metadata,
            start_timestamp=Timestamp.from_us(nuplan_lidar_pc.timestamp),
            end_timestamp=Timestamp.from_us(nuplan_lidar_pc.timestamp + NUPLAN_LIDAR_SWEEP_DURATION_US),
            dataset_root=nuplan_sensor_root,
            relative_path=nuplan_lidar_pc.filename,
        )
    else:
        logger.debug(f"Lidar file not found: {lidar_full_path}")

    return parsed_lidar


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
    average_offsets = np.full((2,), fill_value=np.inf, dtype=np.float64)
    camera_channels = [str(channel.value) for channel in NUPLAN_CAMERA_MAPPING.values()]

    for offset in [0, 1]:
        lidar_pc = nuplan_log_db.lidar_pc[QUERY_START + offset]
        lidar_pc_timestamp_us = lidar_pc.timestamp
        images = list(
            get_images_from_lidar_tokens(
                log_file=str(source_log_path),
                tokens=[lidar_pc.token],
                channels=camera_channels,
            )
        )
        if len(images) > 0:
            absolute_time_offset_ms = [abs(image.timestamp - lidar_pc_timestamp_us) / 1e3 for image in images]
            average_offsets[offset] = np.mean(absolute_time_offset_ms)

    return int(np.argmin(average_offsets))
