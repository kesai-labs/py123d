"""Dataset parser for the NVIDIA PhysicalAI-Autonomous-Vehicles-NCore dataset.

NCore ships clips in NVIDIA's V4 component-based format — one UUID folder per clip
under ``clips/`` containing a ``pai_{clip_id}.json`` sequence manifest plus sibling
``.zarr.itar`` component stores for poses, intrinsics, cuboids, lidar, and each of
the 7 FTheta cameras.

Reading relies on the ``nvidia-ncore`` PyPI package (optional; install via
``pip install py123d[ncore]``). The parser instances are picklable — all zarr/tar
readers are opened lazily inside the iterator methods so the conversion pipeline can
ship ``NCoreLogParser`` objects across a process pool without touching open file
handles.

Two operating modes:

**Local** (default): clips live under ``ncore_data_root/clips/{uuid}/``, typically
downloaded up front via ``py123d-ncore-download``.

**Streaming**: set ``stream_enabled: true`` and the parser will pull each clip from
Hugging Face into a per-clip temp directory just-in-time, convert it, and delete the
temp dir before moving on. Useful when disk is tight or when converting a one-off
subset without committing 2.4 TB to permanent storage.

See :ref:`the NCore docs <ncore>` and https://github.com/NVIDIA/ncore for details
on the component layout.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from dataclasses import dataclass
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
from py123d.parser.ncore.download import (
    CAMERA_IDS as NCORE_DOWNLOAD_CAMERA_IDS,
)
from py123d.parser.ncore.download import (
    MODALITY_CHOICES as NCORE_DOWNLOAD_MODALITY_CHOICES,
)
from py123d.parser.ncore.download import (
    download_clip,
    list_all_clip_ids,
    resolve_hf_token,
)
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


@dataclass(frozen=True)
class _StreamConfig:
    """Settings carried on each log parser in streaming mode — serializable across processes."""

    hf_token: Optional[str]
    revision: str
    modality: str
    cameras: Optional[Tuple[str, ...]]
    temp_dir: Optional[str]
    max_workers: int


class NCoreParser(BaseDatasetParser):
    """Dataset parser for the NVIDIA PhysicalAI-Autonomous-Vehicles-NCore dataset."""

    def __init__(
        self,
        splits: List[str],
        ncore_data_root: Optional[Union[Path, str]] = None,
        max_clips: Optional[int] = None,
        stream_enabled: bool = False,
        stream_clip_ids: Optional[List[str]] = None,
        stream_hf_token: Optional[str] = None,
        stream_revision: str = "main",
        stream_modality: str = "all",
        stream_cameras: Optional[List[str]] = None,
        stream_temp_dir: Optional[Union[Path, str]] = None,
        stream_max_workers: int = 4,
    ) -> None:
        """Initialize the NCore parser.

        :param splits: Dataset splits to process. Currently only ``"ncore_train"`` is shipped.
        :param ncore_data_root: Root directory of the downloaded NCore dataset (contains ``clips/``).
            Required for local mode; ignored when ``stream_enabled=True``.
        :param max_clips: Optional cap on the number of clips to process.
        :param stream_enabled: If ``True``, download each clip to a temp directory at parse time
            and delete the temp dir afterward. No local ``ncore_data_root`` is required.
        :param stream_clip_ids: Optional explicit list of clip UUIDs to stream. If unset, the
            full clip catalog is listed from Hugging Face (truncated by ``max_clips``).
        :param stream_hf_token: Hugging Face access token. Falls back to ``$HF_TOKEN`` /
            ``$HUGGINGFACE_HUB_TOKEN`` if not provided.
        :param stream_revision: HF dataset revision to stream from (default ``"main"``).
        :param stream_modality: Which modalities to pull per clip: ``"all"`` / ``"metadata"`` /
            ``"lidar"`` / ``"cameras"``. Non-``"all"`` choices still include the sequence
            metadata + default component store so the clip remains loadable.
        :param stream_cameras: When ``stream_modality="cameras"``, restrict to these camera IDs.
        :param stream_temp_dir: Parent directory for per-clip temp folders. Defaults to the
            system temp location.
        :param stream_max_workers: Parallel HF download workers per clip.
        """
        for split in splits:
            assert split in NCORE_SPLITS, f"Split {split} is not available. Available splits: {NCORE_SPLITS}"
        assert len(splits) > 0, "At least one split must be provided."
        assert stream_modality in NCORE_DOWNLOAD_MODALITY_CHOICES, (
            f"stream_modality {stream_modality!r} must be one of {NCORE_DOWNLOAD_MODALITY_CHOICES}"
        )
        if stream_cameras is not None:
            for cam in stream_cameras:
                assert cam in NCORE_DOWNLOAD_CAMERA_IDS, (
                    f"stream_cameras entry {cam!r} is not a valid NCore camera ID; "
                    f"must be one of {NCORE_DOWNLOAD_CAMERA_IDS}"
                )

        self._splits = splits
        self._max_clips = max_clips
        self._stream_enabled = stream_enabled

        if stream_enabled:
            resolved_token = resolve_hf_token(stream_hf_token)
            if resolved_token is None:
                logger.warning(
                    "Streaming NCore without an HF token. The dataset is gated — set $HF_TOKEN "
                    "if clip downloads fail with 401/403."
                )
            self._stream_config: Optional[_StreamConfig] = _StreamConfig(
                hf_token=resolved_token,
                revision=stream_revision,
                modality=stream_modality,
                cameras=tuple(stream_cameras) if stream_cameras else None,
                temp_dir=str(stream_temp_dir) if stream_temp_dir is not None else None,
                max_workers=stream_max_workers,
            )
            self._data_root: Optional[Path] = None
            self._clip_entries: List[Tuple[str, Optional[Path], str]] = self._collect_clips_streaming(stream_clip_ids)
        else:
            assert ncore_data_root is not None, "`ncore_data_root` must be provided when `stream_enabled=False`."
            data_root = Path(ncore_data_root)
            assert data_root.exists(), f"`ncore_data_root` path {data_root} does not exist."
            self._stream_config = None
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

    def _collect_clips_streaming(self, stream_clip_ids: Optional[List[str]]) -> List[Tuple[str, Optional[Path], str]]:
        """Enumerate clip UUIDs to stream from HF; no local manifest path yet."""
        if stream_clip_ids:
            clip_ids: List[str] = list(stream_clip_ids)
        else:
            assert self._stream_config is not None
            clip_ids = list_all_clip_ids(token=self._stream_config.hf_token, revision=self._stream_config.revision)
            logger.info("NCore streaming: %d clips discovered on HF", len(clip_ids))

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
                stream_config=self._stream_config,
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
        stream_config: Optional[_StreamConfig] = None,
    ) -> None:
        self._data_root = data_root
        self._clip_id = clip_id
        self._sequence_manifest_path = sequence_manifest_path
        self._split = split
        self._stream_config = stream_config

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
        if self._stream_config is None:
            assert self._data_root is not None and self._sequence_manifest_path is not None
            yield self._data_root, self._sequence_manifest_path
            return

        with tempfile.TemporaryDirectory(prefix=f"ncore_{self._clip_id}_", dir=self._stream_config.temp_dir) as tmp:
            tmp_root = Path(tmp)
            logger.info("Streaming NCore clip %s to %s", self._clip_id, tmp_root)
            manifest_path = download_clip(
                clip_id=self._clip_id,
                output_dir=tmp_root,
                modality=self._stream_config.modality,
                cameras=list(self._stream_config.cameras) if self._stream_config.cameras else None,
                hf_token=self._stream_config.hf_token,
                revision=self._stream_config.revision,
                max_workers=self._stream_config.max_workers,
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
