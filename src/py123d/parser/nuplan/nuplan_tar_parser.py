"""nuPlan parser that reads the sensor files directly from the downloaded sensor archives (tar archives).

An alternative to :class:`py123d.parser.nuplan.nuplan_parser.NuplanParser` for file systems on which the ~150
million extracted sensor files cost too many inodes or metadata operations. The log databases and maps are read
from their extracted locations (few files); each camera and lidar file is read with one ``pread`` at the offset
from the index files of :mod:`py123d.parser.nuplan.nuplan_tar_index`.

The sensor data are stored in the 123D logs (camera store option ``jpeg_binary``, ``mp4`` or ``av1``; lidar store
option ``binary``); the store option ``path`` would reference an extracted tree that does not exist.

The annotations (database queries, poses, boxes, traffic lights, sync loop) come from the
:class:`~py123d.parser.nuplan.nuplan_parser.NuplanLogParser`; only its file system accesses are replaced by index
lookups and archive reads.

Two in-memory tables replace per-image database queries of the camera path: :class:`_EgoPoseInterpolator` (ego
pose at the camera time) and :class:`_CameraImageIndex` (image of a lidar scan). The devkit helpers open a new
connection per query, and the ego_pose bracket query has no usable index (full scan and sort, ~100 ms per image on
network storage), which made these queries the largest part of the conversion time and read about 50 times more
data from the shared storage.

The kedec conversion: ``py123d-conversion dataset=nuplan-tar codec=kedec`` (docs/notes/kedec.rst). The cameras
keep all their images (10 per second, one per 10 Hz sync row), the lidar is stored at 10 Hz.
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    CameraID,
    LidarMetadata,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarMergedMetadata
from py123d.geometry import PoseSE3
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.geometry.utils.rotation_utils import slerp_quaternion_arrays
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    ModalitiesSync,
    ParsedCamera,
)
from py123d.parser.nuplan.nuplan_map_parser import NuplanMapParser
from py123d.parser.nuplan.nuplan_parser import (
    NUPLAN_CAMERA_MAPPING,
    TARGET_DT,
    NuplanLogParser,
    _create_splits_logs,
    _extract_nuplan_box_detections,
    _extract_nuplan_ego_state,
    _extract_nuplan_scenario_data,
    _extract_nuplan_traffic_lights,
    _get_ideal_lidar_pc_offset,
)
from py123d.parser.nuplan.nuplan_sensor_io import _load_pcd_from_bytes
from py123d.parser.nuplan.utils.nuplan_constants import (
    NUPLAN_BOX_DETECTIONS_SE3_METADATA,
    NUPLAN_DEFAULT_DT,
    NUPLAN_EGO_STATE_SE3_METADATA,
    NUPLAN_LIDAR_DICT,
    NUPLAN_LIDAR_SWEEP_DURATION_US,
    NUPLAN_MAP_LOCATIONS,
    NUPLAN_ROLLING_SHUTTER_S,
)
from py123d.parser.nuplan.utils.nuplan_sql_helper import (
    iter_all_images_from_db,
    iter_all_lidar_pc_from_db,
)

check_dependencies(["nuplan"], "nuplan")
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_cameras  # noqa: E402
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc  # noqa: E402
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB  # noqa: E402

from py123d.parser.nuplan.nuplan_tar_index import (
    NuplanTarIndex,
    TarMember,
    TarMemberReader,
    load_log_members,
)

logger = logging.getLogger(__name__)


def _pcd_bytes_to_point_cloud(raw: bytes) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Decode a nuPlan MergedPointCloud .pcd payload into (xyz, features).

    Mirrors :func:`py123d.parser.nuplan.nuplan_sensor_io.load_nuplan_point_cloud_data_from_path`
    minus the filesystem read.
    """
    pcd = _load_pcd_from_bytes(raw)

    lidar_ids = np.zeros(len(pcd), dtype=np.uint8)
    lidar_info = pcd["lidar_info"]
    for nuplan_lidar_id, lidar_id in NUPLAN_LIDAR_DICT.items():
        lidar_ids[lidar_info == nuplan_lidar_id] = int(lidar_id)

    point_cloud_3d = np.column_stack((pcd["x"], pcd["y"], pcd["z"])).astype(np.float32)
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): pcd["intensity"].astype(np.uint8),
        LidarFeature.CHANNEL.serialize(): pcd["ring"].astype(np.uint8),
        LidarFeature.IDS.serialize(): lidar_ids,
    }
    return point_cloud_3d, point_cloud_features


# The devkit's get_images_from_lidar_tokens matches images to a lidar_pc in a
# +-50 ms window (images are 10 Hz, lidar_pcs 20 Hz, so the window between
# neighboring lidar_pcs contains at most one image per channel).
_IMAGE_MATCH_WINDOW_US = 50_000

# Steps (of 100 ms) whose sensor files are announced to the kernel ahead of reading: about 9 files of ~1.5 MB per
# step, so ~270 MB of page cache per log in flight.
_PREFETCH_STEPS = 20


def _open_log_db_readonly(log_file: Path) -> sqlite3.Connection:
    # immutable=1 skips POSIX locking entirely: the nuPlan .db files are static,
    # and on NFS-like storage every lock acquisition invalidates the page cache.
    return sqlite3.connect(f"file:{log_file}?mode=ro&immutable=1", uri=True)


class _EgoPoseInterpolator:
    """One-pass in-memory replacement for ``get_interpolated_ego_pose_from_db``.

    The upstream helper opens two fresh sqlite connections per call and the
    bracket query has no index on ``ego_pose.timestamp`` (full scan + sort);
    called twice per camera image, it dominated conversion wall time. Loading
    the table once turns each lookup into a ~1 us searchsorted.
    """

    def __init__(self, timestamps_us: np.ndarray, xyz: np.ndarray, quaternions_wxyz: np.ndarray) -> None:
        self._timestamps_us = timestamps_us
        self._xyz = xyz
        self._quaternions_wxyz = quaternions_wxyz

    @classmethod
    def from_log_db(cls, log_file: Path) -> "_EgoPoseInterpolator":
        with closing(_open_log_db_readonly(log_file)) as connection:
            rows = connection.execute(
                "SELECT timestamp, x, y, z, qw, qx, qy, qz FROM ego_pose ORDER BY timestamp"
            ).fetchall()
        if not rows:
            raise ValueError(f"No ego poses found in log file {log_file}")
        data = np.asarray(rows, dtype=np.float64)
        return cls(data[:, 0].astype(np.int64), data[:, 1:4], data[:, 4:8])

    def _pose_at(self, index: int) -> PoseSE3:
        x, y, z = self._xyz[index]
        qw, qx, qy, qz = self._quaternions_wxyz[index]
        return PoseSE3(x=x, y=y, z=z, qw=qw, qx=qx, qy=qy, qz=qz)

    def interpolate(self, timestamp_us: int) -> PoseSE3:
        """Mirrors the upstream helper: boundary clamp, exact match, or lerp + slerp."""
        after = int(np.searchsorted(self._timestamps_us, timestamp_us, side="right"))
        before = after - 1
        if before < 0:
            return self._pose_at(0)
        if after == len(self._timestamps_us) or self._timestamps_us[before] == timestamp_us:
            return self._pose_at(before)

        t0 = int(self._timestamps_us[before])
        t1 = int(self._timestamps_us[after])
        alpha = float(timestamp_us - t0) / float(t1 - t0)

        position = (1.0 - alpha) * self._xyz[before] + alpha * self._xyz[after]
        quaternion = slerp_quaternion_arrays(
            self._quaternions_wxyz[before], self._quaternions_wxyz[after], np.array(alpha)
        )
        return PoseSE3.from_R_t(rotation=quaternion, translation=position)


class _CameraImageIndex:
    """One-pass in-memory replacement for the devkit's ``get_images_from_lidar_tokens``.

    The devkit helper opens a fresh sqlite connection per (lidar_pc, channel)
    pair. This index reads the image table once and answers lookups with a
    searchsorted over per-channel timestamps.
    """

    def __init__(self, by_channel: Dict[str, Tuple[np.ndarray, List[str]]]) -> None:
        self._by_channel = by_channel

    @classmethod
    def from_log_db(cls, log_file: Path) -> "_CameraImageIndex":
        with closing(_open_log_db_readonly(log_file)) as connection:
            rows = connection.execute(
                """
                SELECT c.channel, i.timestamp, i.filename_jpg
                FROM image AS i
                INNER JOIN camera AS c ON c.token = i.camera_token
                ORDER BY i.timestamp
                """
            ).fetchall()
        grouped: Dict[str, Tuple[List[int], List[str]]] = {}
        for channel, timestamp_us, filename_jpg in rows:
            timestamps, filenames = grouped.setdefault(channel, ([], []))
            timestamps.append(timestamp_us)
            filenames.append(filename_jpg)
        return cls(
            {
                channel: (np.asarray(timestamps, dtype=np.int64), filenames)
                for channel, (timestamps, filenames) in grouped.items()
            }
        )

    def find(self, channel: str, timestamp_us: int) -> Optional[Tuple[int, str]]:
        """Returns (timestamp_us, filename_jpg) of the image closest to ``timestamp_us``
        within the +-50 ms devkit match window, or None. Ties break to the earlier image."""
        entry = self._by_channel.get(channel)
        if entry is None:
            return None
        timestamps, filenames = entry
        lo = int(np.searchsorted(timestamps, timestamp_us - _IMAGE_MATCH_WINDOW_US, side="left"))
        hi = int(np.searchsorted(timestamps, timestamp_us + _IMAGE_MATCH_WINDOW_US, side="right"))
        if lo == hi:
            return None
        best = lo + int(np.argmin(np.abs(timestamps[lo:hi] - timestamp_us)))
        return int(timestamps[best]), filenames[best]


class NuplanTarLogParser(NuplanLogParser):
    """Picklable handle to one nuPlan log whose sensor blobs live in tar archives."""

    def __init__(
        self,
        split: str,
        source_log_path: Path,
        nuplan_data_root: Path,
        archive_paths: List[Path],
        jpeg_quality: Optional[int] = None,
    ) -> None:
        # The sensor root is never touched; every filesystem access of the base
        # class is overridden below to go through the tar index instead.
        super().__init__(
            split=split,
            source_log_path=source_log_path,
            nuplan_data_root=nuplan_data_root,
            nuplan_sensor_root=Path("/nonexistent-nuplan-sensor-root"),
        )
        self._archive_paths = [Path(p) for p in archive_paths]
        self._jpeg_quality = int(jpeg_quality) if jpeg_quality is not None else None
        # Lazy, per-worker state -- excluded from pickling.
        self._members: Optional[Dict[str, TarMember]] = None
        self._sensor_dirs: Optional[Set[str]] = None
        self._reader: Optional[TarMemberReader] = None
        self._ego_interpolator: Optional[_EgoPoseInterpolator] = None
        self._image_index: Optional[_CameraImageIndex] = None

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_members"] = None
        state["_sensor_dirs"] = None
        state["_reader"] = None
        state["_ego_interpolator"] = None
        state["_image_index"] = None
        return state

    # ------------------------------------------------------------------------------------------------------------------
    # Tar-index plumbing
    # ------------------------------------------------------------------------------------------------------------------

    def _ensure_members(self) -> Tuple[Dict[str, TarMember], Set[str]]:
        if self._members is None:
            log_name = self._source_log_path.stem
            self._members = load_log_members(log_name, self._archive_paths)
            # relative_path is "<log>/<sensor_dir>/<file>"; collect the sensor dirs
            # present so metadata helpers can tell which sensors this log has.
            self._sensor_dirs = {
                relative_path.split("/")[1] for relative_path in self._members if relative_path.count("/") >= 2
            }
            logger.debug(
                "Log %s: %d tar members, sensor dirs: %s", log_name, len(self._members), sorted(self._sensor_dirs)
            )
        assert self._sensor_dirs is not None
        return self._members, self._sensor_dirs

    def _read_member(self, member: TarMember) -> bytes:
        if self._reader is None:
            self._reader = TarMemberReader()
        return self._reader.read(*member)

    def _ensure_ego_interpolator(self) -> _EgoPoseInterpolator:
        if self._ego_interpolator is None:
            self._ego_interpolator = _EgoPoseInterpolator.from_log_db(self._source_log_path)
        return self._ego_interpolator

    def _ensure_image_index(self) -> _CameraImageIndex:
        if self._image_index is None:
            self._image_index = _CameraImageIndex.from_log_db(self._source_log_path)
        return self._image_index

    def _read_camera_jpeg(self, member: TarMember) -> bytes:
        """Camera payload, re-encoded at ``jpeg_quality`` (None = original bytes).

        A payload that does not decode is passed through unchanged rather than
        dropped -- QA's sample-decode check reports it against the source data.
        """
        raw = self._read_member(member)
        if self._jpeg_quality is None:
            return raw
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Undecodable camera jpeg in %s; storing original bytes.", self._source_log_path.stem)
            return raw
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        return bytes(encoded) if ok else raw

    # ------------------------------------------------------------------------------------------------------------------
    # Sensor metadata (upstream versions probe directories under the sensor root)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_camera_metadatas(self, sensor_dirs: Set[str]) -> Dict[CameraID, PinholeCameraMetadata]:
        camera_metadata: Dict[CameraID, PinholeCameraMetadata] = {}
        for camera_id, nuplan_camera_type in NUPLAN_CAMERA_MAPPING.items():
            if str(nuplan_camera_type.value) not in sensor_dirs:
                continue
            cam = list(get_cameras(str(self._source_log_path), [str(nuplan_camera_type.value)]))[0]

            intrinsics_camera_matrix = np.array(pickle.loads(cam.intrinsic), dtype=np.float64)  # type: ignore
            intrinsic = PinholeIntrinsics.from_camera_matrix(intrinsics_camera_matrix)

            distortion_array = np.array(pickle.loads(cam.distortion), dtype=np.float64)  # type: ignore
            distortion = PinholeDistortion.from_array(distortion_array, copy=False)

            # NOTE: nuPlan stores rotation as a scalar-first quaternion [w, x, y, z].
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

    def _get_lidar_merged_metadata(self, sensor_dirs: Set[str]) -> LidarMergedMetadata:
        metadata: Dict = {}
        if "MergedPointCloud" in sensor_dirs:
            for lidar_type in NUPLAN_LIDAR_DICT.values():
                metadata[lidar_type] = LidarMetadata(
                    lidar_name=lidar_type.serialize(),
                    lidar_id=lidar_type,
                    lidar_to_imu_se3=PoseSE3.identity(),  # NOTE: Lidar extrinsics are unknown in nuPlan
                )
        return LidarMergedMetadata(metadata)

    # ------------------------------------------------------------------------------------------------------------------
    # Sensor extraction (upstream versions stat files under the sensor root)
    # ------------------------------------------------------------------------------------------------------------------

    def _extract_cameras(
        self,
        lidar_timestamp_us: int,
        metadatas: Dict[CameraID, PinholeCameraMetadata],
        members: Dict[str, TarMember],
    ) -> List[ParsedCamera]:
        camera_data_list: List[ParsedCamera] = []
        image_index = self._ensure_image_index()
        ego_interpolator = self._ensure_ego_interpolator()

        for camera_type, camera_channel in NUPLAN_CAMERA_MAPPING.items():
            if camera_type not in metadatas:
                continue

            match = image_index.find(str(camera_channel.value), lidar_timestamp_us)
            if match is None:
                continue

            image_timestamp_us, filename_jpg = match
            member = members.get(filename_jpg)
            if member is None:
                continue

            # Interpolate ego pose at the rolling-shutter-compensated camera timestamp
            compensated_timestamp_us = image_timestamp_us + NUPLAN_ROLLING_SHUTTER_S.time_us
            ego_pose = ego_interpolator.interpolate(compensated_timestamp_us)
            camera_to_global_se3 = rel_to_abs_se3(
                origin=ego_pose,
                pose_se3=metadatas[camera_type].camera_to_imu_se3,
            )

            # NOTE: original image timestamp, without rolling shutter compensation (see upstream).
            camera_data_list.append(
                ParsedCamera(
                    metadata=metadatas[camera_type],
                    timestamp=Timestamp.from_us(image_timestamp_us),
                    camera_to_global_se3=camera_to_global_se3,
                    byte_string=self._read_camera_jpeg(member),
                )
            )

        return camera_data_list

    def _prefetch_step(
        self,
        nuplan_lidar_pc: LidarPc,
        metadatas: Dict[CameraID, PinholeCameraMetadata],
        members: Dict[str, TarMember],
    ) -> None:
        """Announce the camera and lidar members of a later step to the kernel (read-ahead into the page cache)."""
        if self._reader is None:
            self._reader = TarMemberReader()
        wanted: List[TarMember] = []
        image_index = self._ensure_image_index()
        for camera_type, camera_channel in NUPLAN_CAMERA_MAPPING.items():
            if camera_type in metadatas:
                match = image_index.find(str(camera_channel.value), int(nuplan_lidar_pc.timestamp))
                if match is not None and match[1] in members:
                    wanted.append(members[match[1]])
        lidar_member = members.get(str(nuplan_lidar_pc.filename))
        if lidar_member is not None:
            wanted.append(lidar_member)
        for member in wanted:
            try:
                self._reader.prefetch(*member)
            except OSError as e:  # a read-ahead request is optional
                logger.debug("Read-ahead of %s failed: %s", member[0], e)

    def _extract_lidar(
        self,
        nuplan_lidar_pc: LidarPc,
        metadata: LidarMergedMetadata,
        members: Dict[str, TarMember],
    ) -> Optional[Lidar]:
        member = members.get(str(nuplan_lidar_pc.filename))
        if member is None:
            logger.debug("Lidar member not in tar index: %s", nuplan_lidar_pc.filename)
            return None

        point_cloud_3d, point_cloud_features = _pcd_bytes_to_point_cloud(self._read_member(member))
        return Lidar(
            timestamp=Timestamp.from_us(nuplan_lidar_pc.timestamp),
            timestamp_end=Timestamp.from_us(nuplan_lidar_pc.timestamp + NUPLAN_LIDAR_SWEEP_DURATION_US),
            metadata=metadata,
            point_cloud_3d=point_cloud_3d,
            point_cloud_features=point_cloud_features,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Base class overrides
    # ------------------------------------------------------------------------------------------------------------------

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass. Mirrors the upstream loop with tar-backed sensor IO."""
        members, sensor_dirs = self._ensure_members()
        pinhole_camera_metadatas = self._get_camera_metadatas(sensor_dirs)
        lidar_merged_metadata = self._get_lidar_merged_metadata(sensor_dirs)

        nuplan_log_db = NuPlanDB(str(self._nuplan_data_root), str(self._source_log_path), None)
        try:
            step_interval: int = int(TARGET_DT / NUPLAN_DEFAULT_DT)
            offset = _get_ideal_lidar_pc_offset(self._source_log_path, nuplan_log_db)
            num_steps = len(nuplan_log_db.lidar_pc)
            steps = range(offset, num_steps, step_interval)
            for k in range(min(_PREFETCH_STEPS, len(steps))):
                self._prefetch_step(nuplan_log_db.lidar_pc[steps[k]], pinhole_camera_metadatas, members)

            for k, lidar_pc_index in enumerate(steps):
                if k + _PREFETCH_STEPS < len(steps):
                    ahead = nuplan_log_db.lidar_pc[steps[k + _PREFETCH_STEPS]]
                    self._prefetch_step(ahead, pinhole_camera_metadatas, members)
                    del ahead
                nuplan_lidar_pc = nuplan_log_db.lidar_pc[lidar_pc_index]
                lidar_pc_token: str = nuplan_lidar_pc.token
                timestamp = Timestamp.from_us(nuplan_lidar_pc.timestamp)

                ego_state_se3 = _extract_nuplan_ego_state(nuplan_lidar_pc, NUPLAN_EGO_STATE_SE3_METADATA)
                box_detections_se3 = _extract_nuplan_box_detections(
                    nuplan_lidar_pc, self._source_log_path, timestamp, NUPLAN_BOX_DETECTIONS_SE3_METADATA
                )
                traffic_lights = _extract_nuplan_traffic_lights(nuplan_log_db, lidar_pc_token, timestamp)
                parsed_pinhole_cameras = self._extract_cameras(
                    int(nuplan_lidar_pc.timestamp), pinhole_camera_metadatas, members
                )
                lidar = self._extract_lidar(nuplan_lidar_pc, lidar_merged_metadata, members)
                custom_modality = _extract_nuplan_scenario_data(nuplan_lidar_pc)

                modalities: List[BaseModality] = [ego_state_se3, box_detections_se3, traffic_lights, custom_modality]
                modalities.extend(parsed_pinhole_cameras)
                if lidar is not None:
                    modalities.append(lidar)

                yield ModalitiesSync(timestamp=timestamp, modalities=modalities)
                del nuplan_lidar_pc
        finally:
            # NOTE: The nuPlanDB class has several internal references; release them properly.
            nuplan_log_db.detach_tables()
            nuplan_log_db.remove_ref()
            del nuplan_log_db

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        _, sensor_dirs = self._ensure_members()
        lidar_merged_metadata = self._get_lidar_merged_metadata(sensor_dirs)

        yield from self._iter_ego_states_se3(NUPLAN_EGO_STATE_SE3_METADATA)
        yield from self._iter_box_detections_se3(NUPLAN_BOX_DETECTIONS_SE3_METADATA)
        yield from self._iter_traffic_lights()
        yield from self._iter_lidars(lidar_merged_metadata)
        yield from self._iter_pinhole_cameras()

    def _iter_pinhole_cameras(self) -> Iterator[ParsedCamera]:
        """Yields pinhole camera observations for all cameras at native rate (~10Hz)."""
        members, sensor_dirs = self._ensure_members()
        pinhole_camera_metadatas = self._get_camera_metadatas(sensor_dirs)
        channel_to_camera_id = {str(v.value): k for k, v in NUPLAN_CAMERA_MAPPING.items()}
        ego_interpolator = self._ensure_ego_interpolator()
        log_file = str(self._source_log_path)

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

            compensated_timestamp_us = row["timestamp"] + NUPLAN_ROLLING_SHUTTER_S.time_us
            ego_pose = ego_interpolator.interpolate(compensated_timestamp_us)
            camera_to_global_se3 = rel_to_abs_se3(
                origin=ego_pose,
                pose_se3=camera_metadata.camera_to_imu_se3,
            )

            yield ParsedCamera(
                metadata=camera_metadata,
                timestamp=Timestamp.from_us(row["timestamp"]),
                camera_to_global_se3=camera_to_global_se3,
                byte_string=self._read_camera_jpeg(member),
            )

    def _iter_lidars(self, modality_metadata: LidarMergedMetadata) -> Iterator[Lidar]:  # type: ignore[override]
        """Yields all lidar sweeps at native rate (20Hz)."""
        members, _ = self._ensure_members()
        for row in iter_all_lidar_pc_from_db(str(self._source_log_path)):
            member = members.get(str(row["filename"]))
            if member is None:
                continue
            point_cloud_3d, point_cloud_features = _pcd_bytes_to_point_cloud(self._read_member(member))
            yield Lidar(
                timestamp=Timestamp.from_us(row["timestamp"]),
                timestamp_end=Timestamp.from_us(row["timestamp"] + NUPLAN_LIDAR_SWEEP_DURATION_US),
                metadata=modality_metadata,
                point_cloud_3d=point_cloud_3d,
                point_cloud_features=point_cloud_features,
            )


class NuplanTarParser(BaseDatasetParser):
    """Dataset parser for nuPlan with tar-archived sensor blobs."""

    def __init__(
        self,
        splits: List[str],
        nuplan_data_root: Union[Path, str],
        nuplan_maps_root: Union[Path, str],
        nuplan_archive_roots: Union[Path, str, List[Union[Path, str]]],
        log_names: Optional[List[str]] = None,
        sensor_logs: Optional[str] = None,
        jpeg_quality: Optional[int] = None,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> None:
        """
        :param splits: nuPlan splits to convert, e.g. ["nuplan_test"].
        :param nuplan_data_root: Root containing ``nuplan-v1.1/splits/...`` with the extracted .db files.
        :param nuplan_maps_root: Root directory of the extracted nuPlan maps.
        :param nuplan_archive_roots: The folder or folders with the sensor archives and their ``.index.parquet`` /
            ``.logs.txt`` files (:mod:`py123d.parser.nuplan.nuplan_tar_index`); a string may list several folders
            separated by ``os.pathsep``.
        :param log_names: Optional allow-list of log names to convert.
        :param sensor_logs: ``"with"``: only the logs that appear in an indexed sensor archive; ``"without"``: only
            the other logs (annotations only); None: all logs. Only about 1,500 of the ~17,000 nuPlan logs have
            sensor data.
        :param jpeg_quality: cv2 quality to re-encode camera JPEGs at, or None to store
            the original nuPlan bytes untouched.
        :param shard_index: This task's shard of the log list (SLURM-array parallelism).
            Shards are disjoint, so one writer per log dir is preserved.
        :param num_shards: Total number of shards.
        """
        assert nuplan_data_root is not None, "`nuplan_data_root` must be provided."
        assert nuplan_archive_roots, "`nuplan_archive_roots` must be provided."
        assert jpeg_quality is None or 1 <= int(jpeg_quality) <= 100, (
            f"jpeg_quality must be in [1, 100] or None, got {jpeg_quality}"
        )
        assert 0 <= int(shard_index) < int(num_shards), f"shard_index {shard_index} not in [0, {num_shards})"
        self._splits = list(splits)
        self._nuplan_data_root = Path(nuplan_data_root)
        self._nuplan_maps_root = Path(nuplan_maps_root) if nuplan_maps_root is not None else None
        if isinstance(nuplan_archive_roots, str):  # one folder, or several separated by os.pathsep (as in PATH)
            roots: List[Union[Path, str]] = [r for r in nuplan_archive_roots.split(os.pathsep) if r]
        elif isinstance(nuplan_archive_roots, Path):
            roots = [nuplan_archive_roots]
        else:
            roots = list(nuplan_archive_roots)
        self._nuplan_archive_roots = [Path(root) for root in roots]
        self._log_names = set(log_names) if log_names is not None else None
        assert sensor_logs in (None, "with", "without"), f"sensor_logs must be None, 'with' or 'without': {sensor_logs}"
        self._sensor_logs = sensor_logs
        self._jpeg_quality = int(jpeg_quality) if jpeg_quality is not None else None
        self._shard_index = int(shard_index)
        self._num_shards = int(num_shards)

        self._index = NuplanTarIndex(self._nuplan_archive_roots)
        self._split_log_path_pairs = self._collect_split_log_path_pairs()

    def _collect_split_log_path_pairs(self) -> List[Tuple[str, Path]]:
        """Collects the (split, log_path) pairs for the specified splits. Mirrors upstream."""
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
            valid_log_names = list(all_log_names & set(log_names_per_split[split_type]))

            if self._log_names is not None:
                valid_log_names = [log_name for log_name in valid_log_names if log_name in self._log_names]
            if self._sensor_logs is not None:
                with_sensors = self._index.log_names
                keep = self._sensor_logs == "with"
                valid_log_names = [log_name for log_name in valid_log_names if (log_name in with_sensors) == keep]

            for log_name in valid_log_names:
                split_log_path_pairs.append((split, nuplan_split_folder / f"{log_name}.db"))

        # Deterministic strided sharding over the sorted list: disjoint across
        # tasks, and interleaving spreads the long Vegas logs across shards.
        split_log_path_pairs.sort(key=lambda pair: (pair[0], pair[1].name))
        return split_log_path_pairs[self._shard_index :: self._num_shards]

    def get_log_parsers(self) -> List[NuplanTarLogParser]:  # type: ignore
        """Inherited, see superclass."""
        log_parsers: List[NuplanTarLogParser] = []
        without_sensors = 0
        for split, source_log_path in self._split_log_path_pairs:
            archives = self._index.archives_for_log(source_log_path.stem)
            if not archives:
                without_sensors += 1
            log_parsers.append(
                NuplanTarLogParser(
                    split=split,
                    source_log_path=source_log_path,
                    nuplan_data_root=self._nuplan_data_root,
                    archive_paths=archives,
                    jpeg_quality=self._jpeg_quality,
                )
            )
        if without_sensors:
            logger.warning(
                "%d of %d logs have no indexed sensor archive and will convert without camera/lidar data.",
                without_sensors,
                len(log_parsers),
            )
        return log_parsers

    def get_map_parsers(self) -> List[NuplanMapParser]:  # type: ignore
        """Inherited, see superclass."""
        if self._nuplan_maps_root is None or not self._nuplan_maps_root.is_dir():
            return []
        return [
            NuplanMapParser(nuplan_maps_root=self._nuplan_maps_root, location=location)
            for location in NUPLAN_MAP_LOCATIONS
        ]
