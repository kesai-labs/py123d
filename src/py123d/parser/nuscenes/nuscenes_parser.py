"""Unified nuScenes dataset parser supporting both 2Hz (keyframe) and 10Hz (interpolated) modes.

The parser mode is determined by the split names passed to :class:`NuScenesParser`:
- Regular splits (e.g. ``nuscenes_train``) → 2Hz keyframe-only iteration
- Interpolated splits (e.g. ``nuscenes-interpolated_train``) → 10Hz with interpolated box detections
"""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    CameraID,
    EgoStateSE3,
    EgoStateSE3Metadata,
    LidarMergedMetadata,
    LogMetadata,
    PinholeCameraMetadata,
    Timestamp,
)
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.nuscenes.nuscenes_map_parser import NuScenesMapParser
from py123d.parser.nuscenes.utils.nuscenes_constants import (
    NUSCENES_BOX_DETECTIONS_SE3_METADATA,
    NUSCENES_DATA_SPLITS,
    NUSCENES_DATABASE_VERSION_MAPPING,
    NUSCENES_DT,
    NUSCENES_EGO_STATE_SE3_METADATA,
    NUSCENES_INTERPOLATED_DATA_SPLITS,
    NUSCENES_MAP_LOCATIONS,
    TARGET_DT,
)
from py123d.parser.nuscenes.utils.nuscenes_extraction import (
    _extract_camera_with_interpolated_pose,
    collect_camera_timelines,
    collect_ego_pose_timeline,
    collect_keyframe_samples,
    collect_lidar_sweep_timeline,
    extract_cameras_from_timeline,
    extract_ego_state_from_sample,
    extract_ego_state_from_sample_data,
    extract_lidar_from_sample_data,
    extract_nuscenes_box_detections,
    extract_nuscenes_cameras,
    extract_nuscenes_lidar,
    find_nearest_cameras_for_sweep,
    find_surrounding_keyframes,
    get_nuscenes_lidar_metadata_from_scene,
    get_nuscenes_pinhole_camera_metadata_from_scene,
    interpolate_box_detections,
    subsample_sweeps,
)

check_dependencies(["nuscenes"], "nuscenes")
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes

if TYPE_CHECKING:
    from py123d.parser.nuscenes.nuscenes_download import NuscenesDownloader

logger = logging.getLogger(__name__)

_ALL_SPLITS = NUSCENES_DATA_SPLITS + NUSCENES_INTERPOLATED_DATA_SPLITS

_NUSCENES_SPLIT_NAME_MAPPING: Dict[str, str] = {
    "nuscenes_train": "train",
    "nuscenes_val": "val",
    "nuscenes_test": "test",
    "nuscenes-mini_train": "mini_train",
    "nuscenes-mini_val": "mini_val",
    "nuscenes-interpolated_train": "train",
    "nuscenes-interpolated_val": "val",
    "nuscenes-interpolated_test": "test",
    "nuscenes-interpolated-mini_train": "mini_train",
    "nuscenes-interpolated-mini_val": "mini_val",
}


class NuScenesParser(BaseDatasetParser):
    """Dataset parser for the nuScenes dataset.

    Supports both native 2Hz keyframe iteration and 10Hz interpolated iteration,
    determined by the split names provided. Regular splits (``nuscenes_*``) use 2Hz;
    interpolated splits (``nuscenes-interpolated_*``) use 10Hz with intermediate lidar
    sweeps, real ego poses, and interpolated box detections (SLERP for rotations).
    """

    def __init__(
        self,
        splits: List[str],
        nuscenes_data_root: Optional[Union[Path, str]] = None,
        nuscenes_map_root: Optional[Union[Path, str]] = None,
        downloader: Optional["NuscenesDownloader"] = None,
        stream_archives: Optional[List[str]] = None,
    ) -> None:
        """Initializes the NuScenesParser.

        :param splits: List of dataset splits (e.g. ["nuscenes_train"] or ["nuscenes-interpolated_val"]).
        :param nuscenes_data_root: Root directory of the nuScenes data. Required in local
            mode (``downloader=None``); ignored when ``downloader`` is set.
        :param nuscenes_map_root: Root directory of the nuScenes map expansion. Optional
            — when ``None`` in local mode, :meth:`get_map_parsers` returns an empty list
            (map conversion is skipped). In streaming mode, ``None`` triggers an
            auto-detect: if the downloader materialized a ``maps/`` subdirectory under
            the streaming data root (e.g. because ``nuScenes-map-expansion-v1.3.zip``
            was in ``stream_archives``), that path is used; otherwise map conversion
            is skipped. Pass an explicit path to override.
        :param downloader: Optional :class:`~py123d.parser.nuscenes.nuscenes_download.NuscenesDownloader`
            used for streaming mode. When provided, the parser materializes the configured
            archives (``stream_archives``) into a session-scoped
            :class:`tempfile.TemporaryDirectory` at construction time, points
            ``nuscenes_data_root`` at it, and cleans the temp directory up when the parser
            is garbage collected.
        :param stream_archives: Archive subset to materialize in streaming mode. ``None``
            falls back to ``downloader.resolve_archives()``, i.e. the downloader's
            ``preset`` / ``archives`` / ``num_archives`` configuration (default in the
            streaming Hydra config: ``preset: trainval_one``). Ignored when
            ``downloader`` is ``None``.
        """
        for split in splits:
            assert split in _ALL_SPLITS, f"Split {split} is not available. Available splits: {_ALL_SPLITS}"

        self._splits: List[str] = splits
        self._downloader: Optional["NuscenesDownloader"] = downloader
        self._stream_temp_dir_handle: Optional[tempfile.TemporaryDirectory] = None

        if downloader is not None:
            self._nuscenes_data_root: Path = self._materialize_streaming_root(
                downloader=downloader, stream_archives=stream_archives
            )
        else:
            assert nuscenes_data_root is not None, "`nuscenes_data_root` must be provided when `downloader` is None."
            self._nuscenes_data_root = Path(nuscenes_data_root)

        # Resolve map_root. Priority: explicit user arg > streaming auto-detect > None.
        # The auto-detect covers the common case where `nuScenes-map-expansion-v1.3.zip`
        # was in the streaming archive set and extracted a populated `maps/` dir.
        # NuScenesMap (devkit) joins ``<dataroot>/maps/expansion/<location>.json``, so
        # ``nuscenes_map_root`` must be the *parent* of ``maps/`` — same convention as
        # ``nuscenes_data_root`` in local mode.
        if nuscenes_map_root is not None:
            self._nuscenes_map_root: Optional[Path] = Path(nuscenes_map_root)
        elif downloader is not None and (self._nuscenes_data_root / "maps" / "expansion").is_dir():
            self._nuscenes_map_root = self._nuscenes_data_root
            logger.info("nuScenes streaming: auto-detected map_root at %s", self._nuscenes_map_root)
        else:
            self._nuscenes_map_root = None

        self._nuscenes_dbs: Dict[str, NuScenes] = {}
        self._scene_tokens_per_split: Dict[str, List[str]] = self._collect_scene_tokens()

    def _materialize_streaming_root(
        self,
        downloader: "NuscenesDownloader",
        stream_archives: Optional[List[str]],
    ) -> Path:
        """Download + extract the configured archive subset into a managed temp directory.

        The temp directory is cleaned up in :meth:`__del__`. Mirrors the WOD motion
        streaming pattern: one shared data root for the parser's lifetime rather than
        per-scene temp dirs (nuScenes parser needs all metadata tables up front).
        """
        archives = list(stream_archives) if stream_archives else downloader.resolve_archives()
        self._stream_temp_dir_handle = tempfile.TemporaryDirectory(prefix="py123d-nuscenes-")
        tmp_root = Path(self._stream_temp_dir_handle.name)
        logger.info("nuScenes streaming temp dir: %s", tmp_root)
        downloader.materialize_archives(archives=archives, output_dir=tmp_root)
        return tmp_root

    def __del__(self) -> None:
        """Clean up the streaming temp directory when the parser is garbage collected."""
        # TODO(nuscenes): move to explicit context manager — __del__ semantics are unreliable
        # (GC ordering, exception swallowing). Mirrors the same TODO on WOD / pandaset parsers.
        handle = getattr(self, "_stream_temp_dir_handle", None)
        if handle is not None:
            try:
                handle.cleanup()
            except Exception:
                pass

    def _collect_scene_tokens(self) -> Dict[str, List[str]]:
        """Collects scene tokens for the specified splits."""
        scene_tokens_per_split: Dict[str, List[str]] = {}
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

            nuscenes_split = _NUSCENES_SPLIT_NAME_MAPPING[split]
            scene_names = scene_splits.get(nuscenes_split, [])
            scene_tokens = [scene["token"] for scene in nusc.scene if scene["name"] in scene_names]
            scene_tokens_per_split[split] = scene_tokens
        return scene_tokens_per_split

    def get_log_parsers(self) -> List[NuScenesLogParser]:  # type: ignore
        """Inherited, see superclass."""
        log_parsers: List[NuScenesLogParser] = []
        for split, scene_tokens in self._scene_tokens_per_split.items():
            database_version = NUSCENES_DATABASE_VERSION_MAPPING[split]
            nusc = self._nuscenes_dbs[database_version]
            is_interpolated = "interpolated" in split
            target_dt = TARGET_DT if is_interpolated else NUSCENES_DT

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
                        target_dt=target_dt,
                        nusc=nusc,
                    )
                )
        return log_parsers

    def get_map_parsers(self) -> List[NuScenesMapParser]:  # type: ignore
        """Inherited, see superclass.

        Returns an empty list when ``nuscenes_map_root`` is not configured — useful in
        streaming mode where the HD-map expansion is not downloaded.
        """
        if self._nuscenes_map_root is None:
            return []
        return [
            NuScenesMapParser(nuscenes_maps_root=self._nuscenes_map_root, location=location)
            for location in NUSCENES_MAP_LOCATIONS
        ]


class NuScenesLogParser(BaseLogParser):
    """Lightweight, picklable handle to one nuScenes scene/log.

    Parameterized by ``target_dt`` to support both 2Hz keyframe-only and 10Hz interpolated
    iteration modes within the same class.
    """

    def __init__(
        self,
        split: str,
        scene_token: str,
        scene_name: str,
        location: str,
        nuscenes_data_root: Path,
        database_version: str,
        target_dt: float,
        nusc: Optional[NuScenes] = None,
    ) -> None:
        self._split = split
        self._scene_token = scene_token
        self._scene_name = scene_name
        self._location = location
        self._nuscenes_data_root = nuscenes_data_root
        self._database_version = database_version
        self._target_dt = target_dt
        self._shared_nusc: Optional[NuScenes] = nusc
        self._owns_nusc: bool = False

    @property
    def _is_interpolated(self) -> bool:
        return self._target_dt < NUSCENES_DT

    def _get_or_load_nusc(self) -> NuScenes:
        """Returns the shared NuScenes DB, loading it lazily if needed."""
        if self._shared_nusc is None:
            self._shared_nusc = NuScenes(
                version=self._database_version,
                dataroot=str(self._nuscenes_data_root),
                verbose=False,
            )
            self._owns_nusc = True
        return self._shared_nusc

    def _release_nusc(self) -> None:
        """Releases the NuScenes DB if this parser owns it (i.e. it was not shared)."""
        if self._owns_nusc:
            self._shared_nusc = None
            self._owns_nusc = False
            gc.collect()

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude the NuScenes DB from pickle state for Ray/ProcessPool safety."""
        state = self.__dict__.copy()
        state["_shared_nusc"] = None
        state["_owns_nusc"] = False
        return state

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset="nuscenes",
            split=self._split,
            log_name=self._scene_name,
            location=self._location,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Sync iteration
    # ------------------------------------------------------------------------------------------------------------------

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        if self._is_interpolated:
            yield from self._iter_sync_interpolated()
        else:
            yield from self._iter_sync_keyframes()

    def _iter_sync_keyframes(self) -> Iterator[ModalitiesSync]:
        """Yields synchronized frames at 2Hz keyframe rate."""
        ego_metadata = NUSCENES_EGO_STATE_SE3_METADATA
        box_detections_metadata = NUSCENES_BOX_DETECTIONS_SE3_METADATA
        pinhole_cameras_metadata = get_nuscenes_pinhole_camera_metadata_from_scene(
            self._get_or_load_nusc, self._scene_token
        )
        lidar_metadata = get_nuscenes_lidar_metadata_from_scene(self._get_or_load_nusc, self._scene_token)

        nusc = self._get_or_load_nusc()
        try:
            can_bus = NuScenesCanBus(dataroot=str(self._nuscenes_data_root))
            scene = nusc.get("scene", self._scene_token)

            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                timestamp = Timestamp.from_us(sample["timestamp"])

                ego_state = extract_ego_state_from_sample(nusc, sample, can_bus, ego_metadata)
                box_detections = extract_nuscenes_box_detections(nusc, sample, box_detections_metadata)
                parsed_cameras = extract_nuscenes_cameras(
                    nusc=nusc,
                    sample=sample,
                    nuscenes_data_root=self._nuscenes_data_root,
                    pinhole_cameras_metadata=pinhole_cameras_metadata,
                )
                parsed_lidar = extract_nuscenes_lidar(
                    nusc=nusc,
                    sample=sample,
                    nuscenes_data_root=self._nuscenes_data_root,
                    lidar_metadata=lidar_metadata,
                )

                modalities: List[BaseModality] = [ego_state, box_detections]
                modalities.extend(parsed_cameras)
                if parsed_lidar is not None:
                    modalities.append(parsed_lidar)

                yield ModalitiesSync(timestamp=timestamp, modalities=modalities)
                sample_token = sample["next"]
        finally:
            self._release_nusc()

    def _iter_sync_interpolated(self) -> Iterator[ModalitiesSync]:
        """Yields synchronized frames at ~10Hz using intermediate lidar sweeps.

        For keyframe timestamps: uses original annotations and synchronized cameras.
        For non-keyframe timestamps: interpolates box detections (SLERP for rotations),
        selects nearest cameras, and uses the sweep's real ego pose.
        """
        ego_metadata = NUSCENES_EGO_STATE_SE3_METADATA
        box_detections_metadata = NUSCENES_BOX_DETECTIONS_SE3_METADATA
        pinhole_cameras_metadata = get_nuscenes_pinhole_camera_metadata_from_scene(
            self._get_or_load_nusc, self._scene_token
        )
        lidar_metadata = get_nuscenes_lidar_metadata_from_scene(self._get_or_load_nusc, self._scene_token)

        nusc = self._get_or_load_nusc()
        try:
            can_bus = NuScenesCanBus(dataroot=str(self._nuscenes_data_root))
            scene = nusc.get("scene", self._scene_token)

            lidar_timeline = collect_lidar_sweep_timeline(nusc, scene)
            keyframe_samples = collect_keyframe_samples(nusc, scene)

            keyframe_detections: Dict[str, BoxDetectionsSE3] = {}
            for sample in keyframe_samples:
                keyframe_detections[sample["token"]] = extract_nuscenes_box_detections(
                    nusc, sample, box_detections_metadata
                )

            selected_sweeps = subsample_sweeps(lidar_timeline)
            camera_timelines = collect_camera_timelines(nusc, scene)
            ego_pose_timeline = collect_ego_pose_timeline(nusc, scene)

            for sweep in selected_sweeps:
                timestamp = Timestamp.from_us(sweep["timestamp"])
                ego_state = extract_ego_state_from_sample_data(nusc, sweep, can_bus, self._scene_name, ego_metadata)

                if sweep["is_key_frame"] and sweep["sample_token"] in keyframe_detections:
                    box_detections = keyframe_detections[sweep["sample_token"]]
                else:
                    prev_kf, next_kf = find_surrounding_keyframes(sweep["timestamp"], keyframe_samples)
                    if prev_kf is not None and next_kf is not None:
                        delta = next_kf["timestamp"] - prev_kf["timestamp"]
                        t = (sweep["timestamp"] - prev_kf["timestamp"]) / delta
                        box_detections = interpolate_box_detections(
                            keyframe_detections[prev_kf["token"]],
                            keyframe_detections[next_kf["token"]],
                            t,
                            timestamp,
                        )
                    elif prev_kf is not None:
                        box_detections = keyframe_detections[prev_kf["token"]]
                    else:
                        box_detections = BoxDetectionsSE3(
                            box_detections=[], timestamp=timestamp, metadata=box_detections_metadata
                        )

                parsed_cameras = find_nearest_cameras_for_sweep(
                    nusc=nusc,
                    target_timestamp=sweep["timestamp"],
                    camera_timelines=camera_timelines,
                    nuscenes_data_root=self._nuscenes_data_root,
                    pinhole_cameras_metadata=pinhole_cameras_metadata,
                    ego_pose_timeline=ego_pose_timeline,
                )

                parsed_lidar = extract_lidar_from_sample_data(
                    sweep, nuscenes_data_root=self._nuscenes_data_root, lidar_metadata=lidar_metadata
                )

                modalities: List[BaseModality] = [ego_state, box_detections]
                modalities.extend(parsed_cameras)
                if parsed_lidar is not None:
                    modalities.append(parsed_lidar)

                yield ModalitiesSync(timestamp=timestamp, modalities=modalities)
        finally:
            self._release_nusc()

    # ------------------------------------------------------------------------------------------------------------------
    # Async iteration (native rates for 2Hz; subsampled to ~10Hz for interpolated mode)
    # ------------------------------------------------------------------------------------------------------------------

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        ego_metadata = NUSCENES_EGO_STATE_SE3_METADATA
        box_detections_metadata = NUSCENES_BOX_DETECTIONS_SE3_METADATA
        pinhole_cameras_metadata = get_nuscenes_pinhole_camera_metadata_from_scene(
            self._get_or_load_nusc, self._scene_token
        )
        lidar_metadata = get_nuscenes_lidar_metadata_from_scene(self._get_or_load_nusc, self._scene_token)

        try:
            yield from self._iter_ego_states_se3(ego_metadata)
            yield from self._iter_box_detections_se3(box_detections_metadata)
            yield from self._iter_lidars(lidar_metadata)
            if pinhole_cameras_metadata:
                for camera_type, camera_metadata in pinhole_cameras_metadata.items():
                    yield from self._iter_pinhole_cameras(camera_type, camera_metadata, pinhole_cameras_metadata)
        finally:
            self._release_nusc()

    def _iter_ego_states_se3(self, ego_metadata: EgoStateSE3Metadata) -> Iterator[EgoStateSE3]:
        """Yields ego states at full lidar sweep rate (~20Hz)."""
        nusc = self._get_or_load_nusc()
        can_bus = NuScenesCanBus(dataroot=str(self._nuscenes_data_root))
        scene = nusc.get("scene", self._scene_token)
        lidar_timeline = collect_lidar_sweep_timeline(nusc, scene)

        for sweep in lidar_timeline:
            yield extract_ego_state_from_sample_data(nusc, sweep, can_bus, self._scene_name, ego_metadata)

    def _iter_box_detections_se3(self, box_detections_metadata: BoxDetectionsSE3Metadata) -> Iterator[BoxDetectionsSE3]:
        """Yields box detections at keyframe rate (~2Hz), or interpolated at the lidar clock (~20Hz)."""
        nusc = self._get_or_load_nusc()
        scene = nusc.get("scene", self._scene_token)

        if not self._is_interpolated:
            # 2Hz: yield keyframe annotations directly
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                yield extract_nuscenes_box_detections(nusc, sample, box_detections_metadata)
                sample_token = sample["next"]
        else:
            # Interpolated: yield at every lidar sweep (~20Hz), interpolating between keyframes.
            # The deferred sync table will downsample to the desired iteration rate.
            keyframe_samples = collect_keyframe_samples(nusc, scene)
            keyframe_detections: Dict[str, BoxDetectionsSE3] = {}
            for sample in keyframe_samples:
                keyframe_detections[sample["token"]] = extract_nuscenes_box_detections(
                    nusc, sample, box_detections_metadata
                )

            lidar_timeline = collect_lidar_sweep_timeline(nusc, scene)

            for sweep in lidar_timeline:
                timestamp = Timestamp.from_us(sweep["timestamp"])
                if sweep["is_key_frame"] and sweep["sample_token"] in keyframe_detections:
                    # Re-stamp keyframe detections to the lidar sweep timestamp for consistency
                    kf_dets = keyframe_detections[sweep["sample_token"]]
                    yield BoxDetectionsSE3(
                        box_detections=kf_dets.box_detections,
                        timestamp=timestamp,
                        metadata=kf_dets.metadata,
                    )
                else:
                    prev_kf, next_kf = find_surrounding_keyframes(sweep["timestamp"], keyframe_samples)
                    if prev_kf is not None and next_kf is not None:
                        delta = next_kf["timestamp"] - prev_kf["timestamp"]
                        t = (sweep["timestamp"] - prev_kf["timestamp"]) / delta
                        yield interpolate_box_detections(
                            keyframe_detections[prev_kf["token"]],
                            keyframe_detections[next_kf["token"]],
                            t,
                            timestamp,
                        )
                    elif prev_kf is not None:
                        kf_dets = keyframe_detections[prev_kf["token"]]
                        yield BoxDetectionsSE3(
                            box_detections=kf_dets.box_detections,
                            timestamp=timestamp,
                            metadata=kf_dets.metadata,
                        )
                    else:
                        yield BoxDetectionsSE3(box_detections=[], timestamp=timestamp, metadata=box_detections_metadata)

    def _iter_pinhole_cameras(
        self,
        camera_type: CameraID,
        modality_metadata: PinholeCameraMetadata,
        pinhole_cameras_metadata: Dict[CameraID, PinholeCameraMetadata],
    ) -> Iterator[ParsedCamera]:
        """Yields pinhole camera observations for a specific camera at native rate (~12Hz).

        In interpolated mode, camera poses are computed by SLERP/LERP interpolation of the
        lidar-derived ego pose timeline (~20Hz) to the exact camera capture timestamp. This
        produces smoother poses and reduces lidar-to-camera projection misalignment.

        In keyframe mode, the discrete ego pose from the camera's own ego_pose_token is used.
        """
        target_camera_channel = modality_metadata.camera_name

        nusc = self._get_or_load_nusc()
        scene = nusc.get("scene", self._scene_token)
        camera_timelines = collect_camera_timelines(nusc, scene)
        timeline = camera_timelines.get(target_camera_channel, [])

        if self._is_interpolated:
            ego_pose_timeline = collect_ego_pose_timeline(nusc, scene)
            for cam_data in timeline:
                parsed_camera = _extract_camera_with_interpolated_pose(
                    cam_data, camera_type, self._nuscenes_data_root, pinhole_cameras_metadata, ego_pose_timeline
                )
                if parsed_camera is not None:
                    yield parsed_camera
        else:
            for cam_data in timeline:
                parsed_camera = extract_cameras_from_timeline(
                    nusc, cam_data, camera_type, self._nuscenes_data_root, pinhole_cameras_metadata
                )
                if parsed_camera is not None:
                    yield parsed_camera

    def _iter_lidars(self, lidar_metadata: LidarMergedMetadata) -> Iterator[ParsedLidar]:
        """Yields all lidar sweeps at native rate (~20Hz)."""
        nusc = self._get_or_load_nusc()
        scene = nusc.get("scene", self._scene_token)
        lidar_timeline = collect_lidar_sweep_timeline(nusc, scene)

        for sweep in lidar_timeline:
            parsed_lidar = extract_lidar_from_sample_data(
                sweep, nuscenes_data_root=self._nuscenes_data_root, lidar_metadata=lidar_metadata
            )
            if parsed_lidar is not None:
                yield parsed_lidar
