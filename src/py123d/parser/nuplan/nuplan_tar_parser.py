"""nuPlan parser that reads camera images and lidar sweeps from the sensor archives, without extracting them.

The train, val and test sensor shards of nuPlan hold about 150 million files. This parser reads them from
the downloaded archives through the index of :mod:`py123d.parser.nuplan.nuplan_tar_index`. The log
databases and the maps are read from their extracted locations, as in
:class:`~py123d.parser.nuplan.nuplan_parser.NuplanParser`.

The log writer cannot store a path to a file inside an archive, so the conversion has to store the
sensor data itself (``camera_store_option: jpeg_binary``, ``lidar_store_option: binary``), see the
``nuplan-tar`` dataset config. The mini sensor shards are compressed zip archives and are not read.

Everything but the sensor access follows :class:`~py123d.parser.nuplan.nuplan_parser.NuplanLogParser`.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union

import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    CameraID,
    LidarID,
    LidarMetadata,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.lidar import Lidar, LidarMergedMetadata
from py123d.geometry import PoseSE3
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import ModalitiesSync, ParsedCamera
from py123d.parser.nuplan.nuplan_parser import (
    NUPLAN_CAMERA_MAPPING,
    TARGET_DT,
    NuplanLogParser,
    NuplanParser,
    _extract_nuplan_box_detections,
    _extract_nuplan_ego_state,
    _extract_nuplan_scenario_data,
    _extract_nuplan_traffic_lights,
    _get_ideal_lidar_pc_offset,
)
from py123d.parser.nuplan.nuplan_sensor_io import load_nuplan_point_cloud_data_from_bytes
from py123d.parser.nuplan.nuplan_tar_index import NuplanTarIndex, TarMember, TarMemberReader, load_log_members
from py123d.parser.nuplan.utils.nuplan_constants import (
    NUPLAN_BOX_DETECTIONS_SE3_METADATA,
    NUPLAN_DEFAULT_DT,
    NUPLAN_EGO_STATE_SE3_METADATA,
    NUPLAN_LIDAR_DICT,
    NUPLAN_LIDAR_SWEEP_DURATION_US,
    NUPLAN_ROLLING_SHUTTER_S,
)
from py123d.parser.nuplan.utils.nuplan_sql_helper import (
    get_interpolated_ego_pose_from_db,
    iter_all_images_from_db,
    iter_all_lidar_pc_from_db,
)

check_dependencies(["nuplan"], "nuplan")
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_cameras, get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB

logger = logging.getLogger(__name__)


class NuplanTarParser(NuplanParser):
    """Dataset parser for the nuPlan dataset with the sensor data in its downloaded archives."""

    def __init__(
        self,
        splits: List[str],
        nuplan_data_root: Union[Path, str],
        nuplan_maps_root: Union[Path, str],
        nuplan_archive_root: Union[Path, str],
        log_names: Optional[List[str]] = None,
    ) -> None:
        """Initializes the NuplanTarParser.

        :param splits: List of splits to convert, e.g. ["nuplan_train", "nuplan_val"].
        :param nuplan_data_root: Root directory of the nuPlan data, with the extracted log databases.
        :param nuplan_maps_root: Root directory of the nuPlan maps.
        :param nuplan_archive_root: Directory with the sensor archives and the index files built by
            :mod:`py123d.parser.nuplan.nuplan_tar_index`.
        :param log_names: Optional list of log names to convert. If None, all logs in the specified splits will be converted.
        """
        assert nuplan_archive_root is not None, "`nuplan_archive_root` must be provided."
        super().__init__(
            splits=splits,
            nuplan_data_root=nuplan_data_root,
            nuplan_maps_root=nuplan_maps_root,
            nuplan_sensor_root=nuplan_archive_root,
            log_names=log_names,
        )
        self._tar_index = NuplanTarIndex(Path(nuplan_archive_root))

    def get_log_parsers(self) -> List[NuplanTarLogParser]:  # type: ignore
        """Inherited, see superclass."""
        log_parsers: List[NuplanTarLogParser] = []
        for split, source_log_path in self._split_log_path_pairs:
            log_parsers.append(
                NuplanTarLogParser(
                    split=split,
                    source_log_path=source_log_path,
                    nuplan_data_root=self._nuplan_data_root,
                    nuplan_archive_root=self._nuplan_sensor_root,
                    archive_paths=self._tar_index.archives_for_log(source_log_path.stem),
                )
            )
        num_without_sensors = sum(len(log_parser._archive_paths) == 0 for log_parser in log_parsers)
        if num_without_sensors > 0:
            logger.info(
                "%d of %d logs have no files in the indexed sensor archives and are converted without sensors.",
                num_without_sensors,
                len(log_parsers),
            )
        return log_parsers


class NuplanTarLogParser(NuplanLogParser):
    """Lightweight, picklable handle to one nuPlan log with its sensor data in archives."""

    def __init__(
        self,
        split: str,
        source_log_path: Path,
        nuplan_data_root: Path,
        nuplan_archive_root: Path,
        archive_paths: List[Path],
    ) -> None:
        super().__init__(
            split=split,
            source_log_path=source_log_path,
            nuplan_data_root=nuplan_data_root,
            nuplan_sensor_root=nuplan_archive_root,
        )
        self._archive_paths = archive_paths
        # Loaded by the process that converts the log, and not pickled.
        self._members: Optional[Dict[str, TarMember]] = None
        self._reader: Optional[TarMemberReader] = None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_members"] = None
        state["_reader"] = None
        return state

    # ------------------------------------------------------------------------------------------------------------------
    # Archive access
    # ------------------------------------------------------------------------------------------------------------------

    def _get_members(self) -> Dict[str, TarMember]:
        """Where the sensor files of the log are stored, by the file names the log database uses."""
        if self._members is None:
            self._members = load_log_members(self._source_log_path.stem, self._archive_paths)
        return self._members

    def _get_sensor_folders(self) -> Set[str]:
        """Names of the sensor folders the log has files in, e.g. ``CAM_F0`` or ``MergedPointCloud``."""
        return {relative_path.split("/")[1] for relative_path in self._get_members()}

    def _read_file(self, member: TarMember) -> bytes:
        if self._reader is None:
            self._reader = TarMemberReader()
        return self._reader.read(*member)

    # ------------------------------------------------------------------------------------------------------------------
    # Synchronized iteration
    # ------------------------------------------------------------------------------------------------------------------

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        ego_state_se3_metadata = NUPLAN_EGO_STATE_SE3_METADATA
        box_detections_se3_metadata = NUPLAN_BOX_DETECTIONS_SE3_METADATA
        pinhole_camera_metadatas = self._get_camera_metadatas()
        lidar_merged_metadata = self._get_lidar_merged_metadata()

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
                parsed_pinhole_cameras = self._extract_cameras(lidar_pc_token, pinhole_camera_metadatas)
                lidar = self._extract_lidar(nuplan_lidar_pc, lidar_merged_metadata)

                custom_modality = _extract_nuplan_scenario_data(nuplan_lidar_pc)

                modalities: List[BaseModality] = [ego_state_se3, box_detections_se3, traffic_lights, custom_modality]
                modalities.extend(parsed_pinhole_cameras)
                if lidar is not None:
                    modalities.append(lidar)

                yield ModalitiesSync(timestamp=timestamp, modalities=modalities)
                del nuplan_lidar_pc
        finally:
            # NOTE: The nuPlanDB class has several internal references, which makes memory management tricky.
            # We need to ensure all references are released properly.
            nuplan_log_db.detach_tables()
            nuplan_log_db.remove_ref()
            del nuplan_log_db

    def _extract_cameras(
        self, lidar_pc_token: str, metadatas: Dict[CameraID, PinholeCameraMetadata]
    ) -> List[ParsedCamera]:
        """Extracts the camera images associated with a lidar sweep, see ``_extract_nuplan_cameras``."""
        camera_data_list: List[ParsedCamera] = []
        log_file = str(self._source_log_path)
        members = self._get_members()

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
            member = members.get(str(image.filename_jpg))
            if member is None:
                continue

            # Interpolate ego pose at the rolling-shutter-compensated camera timestamp
            compensated_timestamp_us = image.timestamp + NUPLAN_ROLLING_SHUTTER_S.time_us  # type: ignore
            ego_pose = get_interpolated_ego_pose_from_db(log_file, compensated_timestamp_us)
            camera_to_global_se3 = rel_to_abs_se3(
                origin=ego_pose,
                pose_se3=metadatas[camera_type].camera_to_imu_se3,
            )

            camera_data_list.append(
                ParsedCamera(
                    metadata=metadatas[camera_type],
                    timestamp=Timestamp.from_us(image.timestamp),  # type: ignore
                    camera_to_global_se3=camera_to_global_se3,
                    byte_string=self._read_file(member),
                )
            )

        return camera_data_list

    def _extract_lidar(self, nuplan_lidar_pc: LidarPc, metadata: LidarMergedMetadata) -> Optional[Lidar]:
        """Extracts the lidar sweep of a LidarPc database object, or None if the archives do not have it."""
        return self._load_lidar(str(nuplan_lidar_pc.filename), int(nuplan_lidar_pc.timestamp), metadata)

    # ------------------------------------------------------------------------------------------------------------------
    # Asynchronous iteration (native-rate)
    # ------------------------------------------------------------------------------------------------------------------

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        yield from self._iter_ego_states_se3(NUPLAN_EGO_STATE_SE3_METADATA)
        yield from self._iter_box_detections_se3(NUPLAN_BOX_DETECTIONS_SE3_METADATA)
        yield from self._iter_traffic_lights()
        yield from self._iter_lidars(self._get_lidar_merged_metadata())
        yield from self._iter_pinhole_cameras()

    def _iter_pinhole_cameras(self) -> Iterator[ParsedCamera]:
        """Yields pinhole camera observations for all cameras at native rate (~10Hz)."""
        pinhole_camera_metadatas = self._get_camera_metadatas()
        channel_to_camera_id = {str(v.value): k for k, v in NUPLAN_CAMERA_MAPPING.items()}
        log_file = str(self._source_log_path)
        members = self._get_members()

        for row in iter_all_images_from_db(log_file):
            channel = row["channel"]
            if channel not in channel_to_camera_id:
                continue

            camera_id = channel_to_camera_id[channel]
            if camera_id not in pinhole_camera_metadatas:
                continue

            member = members.get(str(row["filename_jpg"]))
            if member is None:
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
                byte_string=self._read_file(member),
            )

    def _iter_lidars(self, modality_metadata: LidarMergedMetadata) -> Iterator[Lidar]:  # type: ignore[override]
        """Yields all lidar sweeps at native rate (20Hz)."""
        for row in iter_all_lidar_pc_from_db(str(self._source_log_path)):
            lidar = self._load_lidar(str(row["filename"]), int(row["timestamp"]), modality_metadata)
            if lidar is not None:
                yield lidar

    # ------------------------------------------------------------------------------------------------------------------
    # Sensor helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _load_lidar(self, filename: str, timestamp_us: int, metadata: LidarMergedMetadata) -> Optional[Lidar]:
        """Loads a lidar sweep from the archives, or returns None if they do not have the file."""
        member = self._get_members().get(filename)
        if member is None:
            logger.debug(f"Lidar file not found in the archives: {filename}")
            return None
        point_cloud_3d, point_cloud_features = load_nuplan_point_cloud_data_from_bytes(self._read_file(member))
        return Lidar(
            timestamp=Timestamp.from_us(timestamp_us),
            timestamp_end=Timestamp.from_us(timestamp_us + NUPLAN_LIDAR_SWEEP_DURATION_US),
            metadata=metadata,
            point_cloud_3d=point_cloud_3d,
            point_cloud_features=point_cloud_features,
        )

    def _get_camera_metadatas(self) -> Dict[CameraID, PinholeCameraMetadata]:
        """Extracts the metadata of the cameras the log has images of, see ``_get_nuplan_camera_metadata``."""
        sensor_folders = self._get_sensor_folders()
        camera_metadata: Dict[CameraID, PinholeCameraMetadata] = {}
        for camera_id, nuplan_camera_type in NUPLAN_CAMERA_MAPPING.items():
            if str(nuplan_camera_type.value) not in sensor_folders:
                continue
            cam = list(get_cameras(str(self._source_log_path), [str(nuplan_camera_type.value)]))[0]

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

            camera_metadata[camera_id] = PinholeCameraMetadata(
                camera_name=str(nuplan_camera_type.value),
                camera_id=camera_id,
                width=cam.width,  # type: ignore
                height=cam.height,  # type: ignore
                intrinsics=intrinsic,
                distortion=distortion,
                camera_to_imu_se3=extrinsic,
            )
        return camera_metadata

    def _get_lidar_merged_metadata(self) -> LidarMergedMetadata:
        """Extracts the lidar metadata of the log, see ``_get_nuplan_lidar_merged_metadata``."""
        metadata: Dict[LidarID, LidarMetadata] = {}
        # NOTE: We first need to check if the Lidar folder exists, as not all logs have Lidar data
        if "MergedPointCloud" in self._get_sensor_folders():
            for lidar_type in NUPLAN_LIDAR_DICT.values():
                metadata[lidar_type] = LidarMetadata(
                    lidar_name=lidar_type.serialize(),  # NOTE: nuPlan does not have specific names for the Lidars
                    lidar_id=lidar_type,
                    lidar_to_imu_se3=PoseSE3.identity(),  # NOTE: Lidar extrinsic are unknown
                )
        return LidarMergedMetadata(metadata)
