from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    Timestamp,
)
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.ftheta_camera import FThetaCameraMetadata
from py123d.datatypes.sensors.lidar import LidarMergedMetadata
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, BoundingBoxSE3Index, Vector3D, Vector3DIndex
from py123d.geometry.pose import PoseSE3
from py123d.geometry.transform import rel_to_abs_se3_array
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.ncore.ncore_download import NCoreDownloader
from py123d.parser.ncore.utils.ncore_constants import (
    NCORE_BOX_DETECTIONS_SE3_METADATA,
    NCORE_CAMERA_ID_MAPPING,
    NCORE_EGO_STATE_SE3_METADATA,
    NCORE_LIDAR_SENSOR_ID,
    NCORE_RIG_FRAME_ID,
    NCORE_SPLITS,
    NCORE_WORLD_FRAME_ID,
    resolve_ncore_label,
)
from py123d.parser.ncore.utils.ncore_helper import (
    cuboid_bbox_to_rig_se3_array,
    find_closest_index,
    ftheta_params_to_intrinsics,
)

if TYPE_CHECKING:
    from ncore.data import CuboidTrackObservation


logger = logging.getLogger(__name__)

DATASET_NAME = "ncore"
_LIDAR_WINDOW_US = 50_000
_LIDAR_SPIN_DURATION_US = 100_000


def _import_ncore_v4():
    """Lazy import of the ncore.data.v4 reader API."""
    try:
        from ncore.data.v4 import (
            CameraSensorComponent,
            CuboidsComponent,
            IntrinsicsComponent,
            LidarSensorComponent,
            PosesComponent,
            SequenceComponentGroupsReader,
        )
    except ImportError as exc:
        raise ImportError(
            "The nvidia-ncore package is required to parse NCore data. Install it via `pip install py123d[ncore]`."
        ) from exc
    return (
        SequenceComponentGroupsReader,
        PosesComponent,
        IntrinsicsComponent,
        CuboidsComponent,
        LidarSensorComponent,
        CameraSensorComponent,
    )


class NCoreParser(BaseDatasetParser):
    """Dataset parser for the NVIDIA PhysicalAI-Autonomous-Vehicles-NCore dataset."""

    def __init__(
        self,
        splits: List[str],
        ncore_data_root: Optional[Union[Path, str]] = None,
        max_clips: Optional[int] = None,
        downloader: Optional[NCoreDownloader] = None,
    ) -> None:
        """Initialize the NCore parser.

        :param splits: Dataset splits to process. Currently only ``"ncore_train"`` is shipped.
        :param ncore_data_root: Root directory of the downloaded NCore dataset (contains
            ``clips/``). Required when ``downloader`` is ``None``; ignored otherwise.
        :param max_clips: Optional cap on the number of clips to process. Applied on top
            of the downloader's own selection in streaming mode.
        :param downloader: Optional :class:`NCoreDownloader` used for streaming mode.
            When provided, each log parser pulls its assigned clip via
            :meth:`NCoreDownloader.download_single_clip` into a per-clip
            :class:`tempfile.TemporaryDirectory`, converts it, and deletes the temp dir
            before moving on. Clip selection comes from
            :meth:`NCoreDownloader.resolve_clip_ids`; no local ``ncore_data_root`` is
            required in this mode.
        """
        for split in splits:
            assert split in NCORE_SPLITS, f"Split {split} is not available. Available splits: {NCORE_SPLITS}"
        assert len(splits) > 0, "At least one split must be provided."

        self._splits = splits
        self._max_clips = max_clips
        self._downloader: Optional[NCoreDownloader] = downloader

        if downloader is not None:
            self._data_root: Optional[Path] = None
            self._clip_entries: List[Tuple[str, Optional[Path], str]] = self._collect_clips_streaming()
        else:
            assert ncore_data_root is not None, "`ncore_data_root` must be provided when `downloader` is None."
            data_root = Path(ncore_data_root)
            assert data_root.exists(), f"`ncore_data_root` path {data_root} does not exist."
            self._data_root = data_root
            self._clip_entries = self._collect_clips_local()

    def _collect_clips_local(self) -> List[Tuple[str, Optional[Path], str]]:
        """Discover clip manifests under ``{data_root}/clips/*/pai_*.json``.

        A clip is considered valid only when its sequence manifest, default component
        store, and lidar component store are all present on disk — partial downloads
        are skipped silently.
        """
        assert self._data_root is not None  # narrows type for mypy
        clips_dir = self._data_root / "clips"
        assert clips_dir.is_dir(), f"`clips/` directory not found under {self._data_root}."

        entries: List[Tuple[str, Optional[Path], str]] = []
        split = self._splits[0]
        for clip_dir in sorted(clips_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            clip_id = clip_dir.name
            manifest = clip_dir / f"pai_{clip_id}.json"
            default_store = clip_dir / f"pai_{clip_id}.ncore4.zarr.itar"
            lidar_store = clip_dir / f"pai_{clip_id}.ncore4-{NCORE_LIDAR_SENSOR_ID}.zarr.itar"
            if not (manifest.exists() and default_store.exists() and lidar_store.exists()):
                continue
            entries.append((clip_id, manifest, split))
            if self._max_clips is not None and len(entries) >= self._max_clips:
                break
        return entries

    def _collect_clips_streaming(self) -> List[Tuple[str, Optional[Path], str]]:
        """Enumerate clip UUIDs to stream via the configured downloader."""
        assert self._downloader is not None
        clip_ids = self._downloader.resolve_clip_ids()
        logger.info("NCore streaming: %d clips selected by downloader", len(clip_ids))

        if self._max_clips is not None:
            clip_ids = clip_ids[: self._max_clips]

        split = self._splits[0]
        return [(cid, None, split) for cid in clip_ids]

    def get_log_parsers(self) -> List[NCoreLogParser]:  # type: ignore[override]
        """Inherited, see superclass."""
        return [
            NCoreLogParser(
                data_root=self._data_root,
                clip_id=clip_id,
                sequence_manifest_path=manifest,
                split=split,
                downloader=self._downloader,
            )
            for clip_id, manifest, split in self._clip_entries
        ]

    def get_map_parsers(self) -> List[BaseMapParser]:  # type: ignore[override]
        """Inherited, see superclass. NCore does not include HD-map data."""
        return []


class NCoreLogParser(BaseLogParser):
    """Picklable handle for one NCore clip. All readers are opened lazily inside iterators."""

    def __init__(
        self,
        data_root: Optional[Path],
        clip_id: str,
        sequence_manifest_path: Optional[Path],
        split: str,
        downloader: Optional[NCoreDownloader] = None,
    ) -> None:
        self._data_root = data_root
        self._clip_id = clip_id
        self._sequence_manifest_path = sequence_manifest_path
        self._split = split
        self._downloader = downloader

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset=DATASET_NAME,
            split=self._split,
            log_name=self._clip_id,
            location=None,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Clip materialization (local path OR per-clip temp download)
    # ------------------------------------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def _resolved_clip(self) -> Iterator[Tuple[Path, Path]]:
        """Yields ``(data_root, sequence_manifest_path)`` for the duration of one iterator pass.

        In local mode these are the pre-set instance attributes. In streaming mode the
        clip is downloaded into a fresh ``tempfile.TemporaryDirectory`` which is deleted
        when the context manager exits (i.e. after the parser generator is exhausted and
        after ``_ClipContext.close()`` releases the zarr readers).
        """
        if self._downloader is None:
            assert self._data_root is not None and self._sequence_manifest_path is not None
            yield self._data_root, self._sequence_manifest_path
            return

        with tempfile.TemporaryDirectory(prefix=f"ncore_{self._clip_id}_") as tmp:
            tmp_root = Path(tmp)
            logger.info("Streaming NCore clip %s to %s", self._clip_id, tmp_root)
            manifest_path = self._downloader.download_single_clip(
                clip_id=self._clip_id,
                output_dir=tmp_root,
            )
            yield tmp_root, manifest_path

    # ------------------------------------------------------------------------------------------------------------------
    # Synchronized iteration (lidar-rate)
    # ------------------------------------------------------------------------------------------------------------------

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        with self._resolved_clip() as (data_root, manifest_path):
            ctx = _open_clip_context(manifest_path)
            try:
                ego_metadata = NCORE_EGO_STATE_SE3_METADATA
                det_metadata = NCORE_BOX_DETECTIONS_SE3_METADATA

                lidar_end_ts = ctx.lidar_frame_end_ts
                rig_poses_ts = ctx.rig_poses_ts
                rig_poses_se3 = ctx.rig_poses_se3
                cuboids = ctx.cuboids
                cuboid_obs_ts = np.asarray([obs.timestamp_us for obs in cuboids], dtype=np.int64)

                lidar_relative_path = self._lidar_relative_path()
                lidar_metadata = ctx.lidar_merged_metadata

                for lidar_ts in lidar_end_ts:
                    ts_us = int(lidar_ts)
                    timestamp = Timestamp.from_us(ts_us)

                    ego_state = _ego_state_from_rig_trajectory(rig_poses_se3, rig_poses_ts, ts_us, ego_metadata)
                    box_detections = _build_box_detections_in_window(
                        cuboids,
                        cuboid_obs_ts,
                        ts_us,
                        ctx.reference_to_rig,
                        rig_poses_se3,
                        rig_poses_ts,
                        det_metadata,
                    )

                    parsed_lidar = ParsedLidar(
                        metadata=lidar_metadata,
                        start_timestamp=timestamp,
                        end_timestamp=Timestamp.from_us(ts_us + _LIDAR_SPIN_DURATION_US),
                        dataset_root=data_root,
                        relative_path=lidar_relative_path,
                        iteration=ts_us,
                    )

                    parsed_cameras = _extract_cameras_at_ts(
                        ts_us,
                        ctx,
                        rig_poses_se3,
                        rig_poses_ts,
                    )

                    yield ModalitiesSync(
                        timestamp=timestamp,
                        modalities=[ego_state, box_detections, parsed_lidar, *parsed_cameras],
                    )
            finally:
                ctx.close()

    # ------------------------------------------------------------------------------------------------------------------
    # Asynchronous iteration (native-rate)
    # ------------------------------------------------------------------------------------------------------------------

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        with self._resolved_clip() as (data_root, manifest_path):
            ctx = _open_clip_context(manifest_path)
            try:
                ego_metadata = NCORE_EGO_STATE_SE3_METADATA
                det_metadata = NCORE_BOX_DETECTIONS_SE3_METADATA
                lidar_metadata = ctx.lidar_merged_metadata
                rig_poses_ts = ctx.rig_poses_ts
                rig_poses_se3 = ctx.rig_poses_se3
                lidar_relative_path = self._lidar_relative_path()

                # 1. Ego states at native rate (one per rig→world sample)
                for idx, ts in enumerate(rig_poses_ts):
                    yield _ego_state_at_index(rig_poses_se3, int(ts), idx, ego_metadata)

                # 2. Lidar spins at native rate (~10 Hz)
                for lidar_ts in ctx.lidar_frame_end_ts:
                    ts_us = int(lidar_ts)
                    yield ParsedLidar(
                        metadata=lidar_metadata,
                        start_timestamp=Timestamp.from_us(ts_us),
                        end_timestamp=Timestamp.from_us(ts_us + _LIDAR_SPIN_DURATION_US),
                        dataset_root=data_root,
                        relative_path=lidar_relative_path,
                        iteration=ts_us,
                    )

                # 3. Box detections grouped into lidar-rate windows (same as sync path).
                cuboid_obs_ts = np.asarray([obs.timestamp_us for obs in ctx.cuboids], dtype=np.int64)
                for lidar_ts in ctx.lidar_frame_end_ts:
                    ts_us = int(lidar_ts)
                    yield _build_box_detections_in_window(
                        ctx.cuboids,
                        cuboid_obs_ts,
                        ts_us,
                        ctx.reference_to_rig,
                        rig_poses_se3,
                        rig_poses_ts,
                        det_metadata,
                    )

                # 4. Camera frames at native rate (~30 fps per camera).
                for cam_name, cam_reader in ctx.camera_readers.items():
                    cam_id = NCORE_CAMERA_ID_MAPPING[cam_name]
                    cam_metadata = ctx.camera_metadatas[cam_id]
                    for cam_ts in cam_reader.frames_timestamps_us[:, 1]:
                        yield _build_parsed_camera(
                            cam_metadata,
                            int(cam_ts),
                            cam_reader,
                            rig_poses_se3,
                            rig_poses_ts,
                        )
            finally:
                ctx.close()

    def _lidar_relative_path(self) -> str:
        return f"clips/{self._clip_id}/pai_{self._clip_id}.ncore4-{NCORE_LIDAR_SENSOR_ID}.zarr.itar"


# ----------------------------------------------------------------------------------------------------------------------
# Clip-scoped readers + calibration (opened once per iterator call, closed in finally)
# ----------------------------------------------------------------------------------------------------------------------


class _ClipContext:
    """Holds all opened readers + decoded calibration for one iteration pass."""

    __slots__ = (
        "_sequence_reader",
        "rig_poses_se3",
        "rig_poses_ts",
        "reference_to_rig",
        "lidar_merged_metadata",
        "lidar_frame_end_ts",
        "camera_readers",
        "camera_metadatas",
        "cuboids",
    )

    def __init__(
        self,
        sequence_reader,
        rig_poses_se3: np.ndarray,
        rig_poses_ts: np.ndarray,
        reference_to_rig: Dict[str, PoseSE3],
        lidar_merged_metadata: LidarMergedMetadata,
        lidar_frame_end_ts: np.ndarray,
        camera_readers: Dict[str, object],
        camera_metadatas: Dict[CameraID, FThetaCameraMetadata],
        cuboids: List["CuboidTrackObservation"],
    ) -> None:
        self._sequence_reader = sequence_reader
        self.rig_poses_se3 = rig_poses_se3
        self.rig_poses_ts = rig_poses_ts
        self.reference_to_rig = reference_to_rig
        self.lidar_merged_metadata = lidar_merged_metadata
        self.lidar_frame_end_ts = lidar_frame_end_ts
        self.camera_readers = camera_readers
        self.camera_metadatas = camera_metadatas
        self.cuboids = cuboids

    def close(self) -> None:
        # SequenceComponentGroupsReader does not expose a public close; dropping references lets
        # garbage collection release the underlying zarr/itar file handles before the temp dir
        # (if any) is cleaned up by the surrounding context manager.
        self._sequence_reader = None
        self.camera_readers = {}

        # Drop the per-path LRU caches in the lidar loader so cleaned-up temp files don't
        # linger as stale reader references.
        from py123d.parser.ncore.ncore_sensor_io import _open_lidar_reader

        _open_lidar_reader.cache_clear()


def _open_clip_context(sequence_manifest_path: Path) -> _ClipContext:
    (
        SequenceComponentGroupsReader,
        PosesComponent,
        IntrinsicsComponent,
        CuboidsComponent,
        LidarSensorComponent,
        CameraSensorComponent,
    ) = _import_ncore_v4()

    seq_reader = SequenceComponentGroupsReader([sequence_manifest_path])

    # Poses: one rig→world trajectory + static sensor-to-rig poses.
    poses_readers = seq_reader.open_component_readers(PosesComponent.Reader)
    assert poses_readers, f"No poses component in {sequence_manifest_path}"
    poses_reader = next(iter(poses_readers.values()))
    dynamic_poses, dynamic_ts = poses_reader.get_dynamic_pose(NCORE_RIG_FRAME_ID, NCORE_WORLD_FRAME_ID)
    rig_poses_se3 = np.asarray(dynamic_poses, dtype=np.float64)
    rig_poses_ts = np.asarray(dynamic_ts, dtype=np.int64)

    static_pose_table: Dict[str, PoseSE3] = {NCORE_RIG_FRAME_ID: PoseSE3.identity()}
    for (src, tgt), static_matrix in poses_reader.get_static_poses():
        if tgt == NCORE_RIG_FRAME_ID:
            static_pose_table[src] = PoseSE3.from_transformation_matrix(np.asarray(static_matrix, dtype=np.float64))

    # Intrinsics: FTheta camera models + (unused) lidar model.
    intr_readers = seq_reader.open_component_readers(IntrinsicsComponent.Reader)
    assert intr_readers, f"No intrinsics component in {sequence_manifest_path}"
    intr_reader = next(iter(intr_readers.values()))

    # Lidar
    lidar_readers = seq_reader.open_component_readers(LidarSensorComponent.Reader)
    assert lidar_readers, f"No lidar component in {sequence_manifest_path}"
    lidar_reader = next(iter(lidar_readers.values()))
    lidar_frame_end_ts = np.asarray(lidar_reader.frames_timestamps_us[:, 1], dtype=np.int64)
    lidar_to_rig = static_pose_table.get(NCORE_LIDAR_SENSOR_ID, PoseSE3.identity())
    lidar_merged_metadata = LidarMergedMetadata(
        {
            LidarID.LIDAR_TOP: LidarMetadata(
                lidar_name=NCORE_LIDAR_SENSOR_ID,
                lidar_id=LidarID.LIDAR_TOP,
                lidar_to_imu_se3=lidar_to_rig,
            ),
        }
    )

    # Cameras: only keep those present in both the NCore store and the py123d camera ID mapping.
    cam_readers_raw = seq_reader.open_component_readers(CameraSensorComponent.Reader)
    camera_readers: Dict[str, object] = {}
    camera_metadatas: Dict[CameraID, FThetaCameraMetadata] = {}
    for cam_name, cam_reader in cam_readers_raw.items():
        cam_id = NCORE_CAMERA_ID_MAPPING.get(cam_name)
        if cam_id is None:
            continue
        cam_params = intr_reader.get_camera_model_parameters(cam_name)
        intrinsics, width, height = ftheta_params_to_intrinsics(cam_params)
        camera_to_rig = static_pose_table.get(cam_name, PoseSE3.identity())
        camera_metadatas[cam_id] = FThetaCameraMetadata(
            camera_name=cam_name,
            camera_id=cam_id,
            intrinsics=intrinsics,
            width=width,
            height=height,
            camera_to_imu_se3=camera_to_rig,
        )
        camera_readers[cam_name] = cam_reader

    # Cuboids (optional — some clips have no labels).
    cuboid_readers = seq_reader.open_component_readers(CuboidsComponent.Reader)
    cuboids: List["CuboidTrackObservation"] = []
    if cuboid_readers:
        cuboids_reader = next(iter(cuboid_readers.values()))
        cuboids = list(cuboids_reader.get_observations())
        cuboids.sort(key=lambda o: o.timestamp_us)

    return _ClipContext(
        sequence_reader=seq_reader,
        rig_poses_se3=rig_poses_se3,
        rig_poses_ts=rig_poses_ts,
        reference_to_rig=static_pose_table,
        lidar_merged_metadata=lidar_merged_metadata,
        lidar_frame_end_ts=lidar_frame_end_ts,
        camera_readers=camera_readers,
        camera_metadatas=camera_metadatas,
        cuboids=cuboids,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Per-modality builders
# ----------------------------------------------------------------------------------------------------------------------


def _nearest_rig_pose(rig_poses_se3: np.ndarray, rig_poses_ts: np.ndarray, ts_us: int) -> PoseSE3:
    idx = find_closest_index(rig_poses_ts, ts_us)
    return PoseSE3.from_transformation_matrix(rig_poses_se3[idx])


def _ego_state_from_rig_trajectory(
    rig_poses_se3: np.ndarray,
    rig_poses_ts: np.ndarray,
    ts_us: int,
    metadata: EgoStateSE3Metadata,
) -> EgoStateSE3:
    idx = find_closest_index(rig_poses_ts, ts_us)
    return _ego_state_at_index(rig_poses_se3, int(rig_poses_ts[idx]), idx, metadata)


def _ego_state_at_index(
    rig_poses_se3: np.ndarray,
    ts_us: int,
    idx: int,
    metadata: EgoStateSE3Metadata,
) -> EgoStateSE3:
    # NCore poses carry no velocity/acceleration — downstream `infer_ego_dynamics: true`
    # fills them via finite differences across successive samples.
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D.from_array(np.zeros(len(Vector3DIndex), dtype=np.float64)),
        acceleration=Vector3D.from_array(np.zeros(len(Vector3DIndex), dtype=np.float64)),
        angular_velocity=Vector3D.from_array(np.zeros(len(Vector3DIndex), dtype=np.float64)),
    )
    return EgoStateSE3.from_imu(
        imu_se3=PoseSE3.from_transformation_matrix(rig_poses_se3[idx]),
        metadata=metadata,
        dynamic_state_se3=dynamic_state,
        timestamp=Timestamp.from_us(ts_us),
    )


def _build_box_detections_in_window(
    cuboids: List["CuboidTrackObservation"],
    cuboid_obs_ts: np.ndarray,
    lidar_ts_us: int,
    reference_to_rig: Dict[str, PoseSE3],
    rig_poses_se3: np.ndarray,
    rig_poses_ts: np.ndarray,
    metadata: BoxDetectionsSE3Metadata,
) -> BoxDetectionsSE3:
    timestamp = Timestamp.from_us(lidar_ts_us)
    if len(cuboids) == 0:
        return BoxDetectionsSE3(box_detections=[], timestamp=timestamp, metadata=metadata)

    mask = (cuboid_obs_ts >= lidar_ts_us - _LIDAR_WINDOW_US) & (cuboid_obs_ts < lidar_ts_us + _LIDAR_WINDOW_US)
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size == 0:
        return BoxDetectionsSE3(box_detections=[], timestamp=timestamp, metadata=metadata)

    box_detections: List[BoxDetectionSE3] = []
    zero_velocity = Vector3D.from_array(np.zeros(len(Vector3DIndex), dtype=np.float64))
    for det_idx in selected_idx:
        obs = cuboids[int(det_idx)]
        ref_to_rig = reference_to_rig.get(obs.reference_frame_id, PoseSE3.identity())
        bbox_in_rig = cuboid_bbox_to_rig_se3_array(obs, ref_to_rig)

        rig_to_world_at_obs = _nearest_rig_pose(rig_poses_se3, rig_poses_ts, int(obs.reference_frame_timestamp_us))
        bbox_in_rig[BoundingBoxSE3Index.SE3] = rel_to_abs_se3_array(
            origin=rig_to_world_at_obs,
            pose_se3_array=bbox_in_rig[BoundingBoxSE3Index.SE3].reshape(1, -1),
        )

        label = resolve_ncore_label(obs.class_id)
        box_detections.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(label=label, track_token=str(obs.track_id)),
                bounding_box_se3=BoundingBoxSE3.from_array(bbox_in_rig),
                velocity_3d=zero_velocity,
            )
        )

    return BoxDetectionsSE3(box_detections=box_detections, timestamp=timestamp, metadata=metadata)


def _extract_cameras_at_ts(
    lidar_ts_us: int,
    ctx: _ClipContext,
    rig_poses_se3: np.ndarray,
    rig_poses_ts: np.ndarray,
) -> List[ParsedCamera]:
    cameras: List[ParsedCamera] = []
    for cam_name, cam_reader in ctx.camera_readers.items():
        cam_id = NCORE_CAMERA_ID_MAPPING[cam_name]
        cam_metadata = ctx.camera_metadatas[cam_id]
        frame_end_ts = np.asarray(cam_reader.frames_timestamps_us[:, 1], dtype=np.int64)
        if frame_end_ts.size == 0:
            continue
        frame_idx = find_closest_index(frame_end_ts, lidar_ts_us)
        cam_ts_us = int(frame_end_ts[frame_idx])
        cameras.append(_build_parsed_camera(cam_metadata, cam_ts_us, cam_reader, rig_poses_se3, rig_poses_ts))
    return cameras


def _build_parsed_camera(
    cam_metadata: FThetaCameraMetadata,
    cam_ts_us: int,
    cam_reader,
    rig_poses_se3: np.ndarray,
    rig_poses_ts: np.ndarray,
) -> ParsedCamera:
    image_data = cam_reader.get_frame_data(cam_ts_us)
    rig_to_world = _nearest_rig_pose(rig_poses_se3, rig_poses_ts, cam_ts_us)
    camera_to_global = rel_to_abs_se3(origin=rig_to_world, pose_se3=cam_metadata.camera_to_imu_se3)
    return ParsedCamera(
        metadata=cam_metadata,
        timestamp=Timestamp.from_us(cam_ts_us),
        camera_to_global_se3=camera_to_global,
        byte_string=bytes(image_data.get_encoded_image_data()),
    )
