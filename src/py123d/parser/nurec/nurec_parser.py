from __future__ import annotations

import bisect
import concurrent.futures
import io
import json
import logging
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from typing_extensions import override

from py123d.common.utils.dependencies import check_dependencies
from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    EgoStateSE3,
    EgoStateSE3Metadata,
    LogMetadata,
    MapMetadata,
    Timestamp,
)
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D
from py123d.parser.base_dataset_parser import BaseDatasetParser, BaseLogParser, BaseMapParser, ModalitiesSync
from py123d.parser.nurec.nurec_download import NuRecDownloader
from py123d.parser.nurec.nurec_map_parser import (
    NUREC_MAP_SOURCES,
    NuRecMapParser,
    _clipgt_member,
    resolve_map_source,
)
from py123d.parser.nurec.nurec_splits import (
    NUREC_DEFAULT_SPLITS,
    NUREC_RELEASE_SPLITS,
    release_from_path,
    resolve_scene_lists,
    scenes_of_split,
    validate_splits,
)
from py123d.parser.registry import PhysicalAIAVBoxDetectionLabel

check_dependencies(["csaps"], "nurec")
import csaps

logger = logging.getLogger(__name__)

# All 1607 NuRec scenes are Hyperion 8.1 rigs on four distinct Mercedes-Benz platforms.
# Wheelbases measured from calibration_estimate.parquet across the full 26.04 release.
# platform_name lives in rig_json.rig.properties (26.04 branch only).
_NUREC_PLATFORM_WHEEL_BASE_M: Dict[str, float] = {
    "hy8.1_daimler_gls": 3.135,  # X167 GLS       — 949 scenes
    "hy8.1_daimler_c118": 2.730,  # C118 CLA Coupé — 319 scenes
    "hy8.1_daimler_s223v2.9": 3.216,  # W223 S-Class   — 240 scenes
    "hy8.1_daimler_s222v2.9": 3.165,  # W222 S-Class   —  75 scenes
    "hy8.1_daimler_c118v2.9": 2.730,  # C118 CLA Coupé —  20 scenes
    "hy8.1_daimler_s223v3": 3.216,  # W223 S-Class   —   2 scenes
    "hy8.1_daimler_s223v2.9_new": 3.216,  # W223 S-Class   —   2 scenes
}
# (bbox_length_m, wheelbase_m) — nearest-neighbour fallback when rig.properties is
# absent (main branch).  Lengths from rig_trajectories.json rig_bbox.dim[0].
_NUREC_VEHICLE_BY_LENGTH: List[Tuple[float, float]] = [
    (4.688, 2.730),  # CLA C118
    (5.207, 3.135),  # GLS X167
    (5.255, 3.165),  # S-Class W222
    (5.393, 3.216),  # S-Class W223
]
_NUREC_WHEEL_BASE_FALLBACK_M: float = 3.135  # GLS — most common (949/1607 scenes)

NUREC_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(PhysicalAIAVBoxDetectionLabel)


@dataclass(frozen=True)
class _TrackSample:
    timestamp_us: int
    pose: PoseSE3


@dataclass(frozen=True)
class _NuRecTrack:
    track_id: str
    label: PhysicalAIAVBoxDetectionLabel
    length: float
    width: float
    height: float
    samples: List[_TrackSample]
    timestamps_us: List[int]


@dataclass(frozen=True)
class _NuRecScene:
    """One scene selected for conversion."""

    usdz_path: Path
    split: str
    has_map: bool

    @property
    def name(self) -> str:
        return self.usdz_path.stem


_NUREC_LAYOUT_HINT = (
    "Scenes are expected at '<nurec_root>/**/<release>_release/<scene uuid>/<scene uuid>.usdz', "
    "as written by `py123d-download dataset=nurec-<...>`."
)


class NuRecParser(BaseDatasetParser):
    """Dataset parser for NuRec USDZ archives.

    Each scene converts into one log with ego states and tracked SE3 cuboids, resampled
    onto a uniform 10 Hz grid, and into one map. A scene that carries none of the map
    sources requested by ``map_source`` converts into a log without a map.

    The scenes to convert are selected by split. ``nurec-curated_train`` and
    ``nurec-curated_val`` (the default) are lists of named scenes, see ``scene_lists``.
    ``nurec-2601_train`` and ``nurec-2604_train`` each cover every scene of one release.

    Scenes are found below ``nurec_root`` by release and uuid, in the layout of the
    Hugging Face dataset: ``<release>_release/<scene uuid>/<scene uuid>.usdz``.
    """

    def __init__(
        self,
        nurec_root: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        num_sequences: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        splits: Optional[List[str]] = None,
        scene_lists: Optional[Dict[str, List[str]]] = None,
        min_traffic_duration_us: int = 0,
        smooth_track_positions: bool = False,
        downloader: Optional[NuRecDownloader] = None,
        map_source: str = "clip_gt_or_xodr",
        xodr_member: str = "map.xodr",
        xodr_interpolation_step_size: float = 1.0,
        xodr_connection_distance_threshold: float = 0.1,
        xodr_internal_only: bool = True,
        alignment_enabled: bool = True,
        alignment_source: str = "world_base",
        alignment_rig_member: str = "rig_trajectories.json",
        alignment_pose_record_member: str = "pose_record.json",
        alignment_record_index: int = 0,
        alignment_invert_rotation: bool = True,
        alignment_xy_origin_m: Optional[Tuple[float, float]] = None,
        alignment_z_origin: str = "world_base_altitude",
        alignment_z_origin_m: Optional[float] = None,
    ) -> None:
        """Initializes the NuRec parser.

        :param nurec_root: Directory, or list of directories, holding the downloaded releases.
            Ignored when ``downloader`` is set.
        :param num_sequences: Converts at most this many scenes per split.
        :param sample_random: Picks the ``num_sequences`` scenes at random instead of the first ones.
        :param seed: Random seed used when ``sample_random`` is set.
        :param splits: Splits to convert, defaults to ``nurec-curated_train`` and ``nurec-curated_val``.
        :param scene_lists: Mapping of split name to scene names, each ``"<release>/<scene uuid>"``.
            Defaults to the bundled train/val lists.
        :param min_traffic_duration_us: Drops tracks living shorter than this inside the scene window.
        :param smooth_track_positions: Smooths track positions with a cubic smoothing spline.
        :param downloader: Streams the scenes from Hugging Face instead of reading ``nurec_root``. It
            fetches the scenes selected by ``splits``, ``num_sequences``, ``sample_random`` and ``seed``,
            narrowed down by its own ``sequence_ids``, into a temporary directory.
        :param map_source: Which map a scene is converted from, see :class:`NuRecMapParser`. The
            ``xodr_*`` and ``alignment_*`` options are forwarded to it as well.
        """
        self._scene_lists = resolve_scene_lists(scene_lists)
        self._splits = validate_splits(splits if splits is not None else NUREC_DEFAULT_SPLITS, self._scene_lists)
        if map_source not in NUREC_MAP_SOURCES:
            raise ValueError(f"Unsupported NuRec map_source: {map_source!r}. Use one of {NUREC_MAP_SOURCES}.")
        self._num_sequences = num_sequences
        self._sample_random = sample_random
        self._seed = seed
        self._min_traffic_duration_us = min_traffic_duration_us
        self._smooth_track_positions = smooth_track_positions
        self._downloader = downloader
        self._map_source = map_source
        self._xodr_member = xodr_member
        # Forwarded verbatim to every NuRecMapParser; see its docstring for what each controls.
        self._map_parser_kwargs: Dict[str, object] = dict(
            map_source=map_source,
            xodr_member=xodr_member,
            xodr_interpolation_step_size=xodr_interpolation_step_size,
            xodr_connection_distance_threshold=xodr_connection_distance_threshold,
            xodr_internal_only=xodr_internal_only,
            alignment_enabled=alignment_enabled,
            alignment_source=alignment_source,
            alignment_rig_member=alignment_rig_member,
            alignment_pose_record_member=alignment_pose_record_member,
            alignment_record_index=alignment_record_index,
            alignment_invert_rotation=alignment_invert_rotation,
            alignment_xy_origin_m=alignment_xy_origin_m,
            alignment_z_origin=alignment_z_origin,
            alignment_z_origin_m=alignment_z_origin_m,
        )

        if downloader is not None:
            # The downloader fetches the scenes this parser converts.
            downloader.splits = self._splits
            downloader.scene_lists = self._scene_lists
            if num_sequences is not None:
                downloader.num_sequences = num_sequences
            if sample_random:
                downloader.sample_random = sample_random
                downloader.seed = seed
            if downloader.output_dir is None:
                self._tmp_dir = tempfile.TemporaryDirectory(prefix="nurec_streaming_")
                downloader.output_dir = Path(self._tmp_dir.name)
            downloader.download()
            # The downloaded files keep the release layout, so they are selected like local ones.
            roots = [Path(downloader.output_dir)]
        else:
            assert nurec_root is not None, "`nurec_root` must be provided when `downloader` is None."
            roots = [Path(nurec_root)] if isinstance(nurec_root, (str, Path)) else [Path(root) for root in nurec_root]

        self._scenes: List[_NuRecScene] = self._select_scenes(roots)

    @override
    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        return [
            NuRecMapParser(
                usdz_path=scene.usdz_path,
                location=scene.name,
                split=scene.split,
                log_name=scene.name,
                **self._map_parser_kwargs,
            )
            for scene in self._scenes
            if scene.has_map
        ]

    @override
    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        return [
            NuRecLogParser(
                usdz_path=scene.usdz_path,
                split=scene.split,
                min_traffic_duration_us=self._min_traffic_duration_us,
                smooth_track_positions=self._smooth_track_positions,
                has_map=scene.has_map,
            )
            for scene in self._scenes
        ]

    def _select_scenes(self, roots: List[Path]) -> List[_NuRecScene]:
        """Collects the scenes of the requested splits that are present below ``roots``."""
        usdz_by_scene = self._discover_usdz(roots)

        selected: List[Tuple[str, Path]] = []
        for split in self._splits:
            if split in NUREC_RELEASE_SPLITS:
                release = NUREC_RELEASE_SPLITS[split]
                usdz_paths = [
                    usdz_path
                    for (scene_release, _), usdz_path in sorted(usdz_by_scene.items())
                    if scene_release == release
                ]
            else:
                scenes = scenes_of_split(split, self._scene_lists)
                usdz_paths = [usdz_by_scene[scene] for scene in scenes if scene in usdz_by_scene]
                # A streaming downloader may fetch only some of the listed scenes on purpose.
                if len(usdz_paths) < len(scenes) and self._downloader is None:
                    logger.warning(
                        "NuRec split %s: %d of %d listed scene(s) not found under %s",
                        split,
                        len(scenes) - len(usdz_paths),
                        len(scenes),
                        [str(root) for root in roots],
                    )
            usdz_paths = self._limit_sequences(usdz_paths)
            logger.info("NuRec split %s: %d scene(s) selected", split, len(usdz_paths))
            selected.extend((split, usdz_path) for usdz_path in usdz_paths)

        if not selected:
            raise FileNotFoundError(
                f"No NuRec scenes found for splits {self._splits} under {[str(root) for root in roots]} "
                f"(releases found: {sorted({release for release, _ in usdz_by_scene})}). {_NUREC_LAYOUT_HINT}"
            )

        map_sources = self._resolve_map_sources([usdz_path for _, usdz_path in selected])
        return [
            _NuRecScene(usdz_path=usdz_path, split=split, has_map=map_sources[usdz_path] is not None)
            for split, usdz_path in selected
        ]

    def _discover_usdz(self, roots: List[Path]) -> Dict[Tuple[str, str], Path]:
        """Indexes the .usdz files below ``roots`` by (release, scene uuid)."""
        usdz_by_scene: Dict[Tuple[str, str], Path] = {}
        num_without_release = 0
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(f"NuRec root directory not found: {root}")
            for usdz_path in sorted(root.rglob("*.usdz")):
                release = release_from_path(usdz_path)
                if release is None:
                    num_without_release += 1
                else:
                    usdz_by_scene.setdefault((release, usdz_path.stem), usdz_path)

        if num_without_release > 0:
            logger.warning(
                "NuRec: ignoring %d .usdz file(s) under %s that are not inside a '<release>_release' directory. %s",
                num_without_release,
                [str(root) for root in roots],
                _NUREC_LAYOUT_HINT,
            )
        logger.info(
            "NuRec: discovered %d .usdz file(s) under %s, by release: %s",
            len(usdz_by_scene),
            [str(root) for root in roots],
            dict(sorted(Counter(release for release, _ in usdz_by_scene).items())),
        )
        return usdz_by_scene

    def _limit_sequences(self, usdz_paths: List[Path]) -> List[Path]:
        """Applies ``num_sequences`` to the scenes of one split."""
        if self._num_sequences is None or self._num_sequences >= len(usdz_paths):
            return usdz_paths
        if self._sample_random:
            return sorted(random.Random(self._seed).sample(usdz_paths, self._num_sequences))
        return usdz_paths[: self._num_sequences]

    def _resolve_map_sources(self, usdz_paths: List[Path]) -> Dict[Path, Optional[str]]:
        """Reads from each archive which map source it will be converted from, if any."""
        unique_usdz_paths = list(dict.fromkeys(usdz_paths))

        def _resolve(usdz_path: Path) -> Optional[str]:
            try:
                with ZipFile(usdz_path) as archive:
                    return resolve_map_source(set(archive.namelist()), self._map_source, self._xodr_member)
            except (OSError, BadZipFile) as error:
                logger.warning("NuRec scene %s: could not read %s (%s)", usdz_path.stem, usdz_path, error)
                return None

        # Only the archive's member list is read, which is quick but worth doing in parallel.
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            map_sources = dict(zip(unique_usdz_paths, pool.map(_resolve, unique_usdz_paths)))

        counts = Counter(map_sources.values())
        logger.info(
            "NuRec maps (map_source=%s): %d from clip_gt, %d from xodr, %d scene(s) without a map",
            self._map_source,
            counts["clip_gt"],
            counts["xodr"],
            counts[None],
        )
        return map_sources


class NuRecLogParser(BaseLogParser):
    """Log parser for one NuRec USDZ driving scene."""

    def __init__(
        self,
        usdz_path: Union[str, Path],
        split: str = "nurec-curated_train",
        min_traffic_duration_us: int = 0,
        smooth_track_positions: bool = False,
        has_map: bool = True,
    ) -> None:
        self._usdz_path = Path(usdz_path)
        self._split = split
        # False for a scene that carries none of the requested map sources.
        self._has_map = has_map
        self._smooth_track_positions = smooth_track_positions
        # AlpaSim drops tracks shorter than this within the scene window; 0 keeps all.
        self._min_traffic_duration_us = min_traffic_duration_us

    @property
    def _uuid(self) -> str:
        return self._usdz_path.stem

    @override
    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        map_metadata: Optional[MapMetadata] = None
        if self._has_map:
            map_metadata = MapMetadata(
                dataset="nurec",
                location=self._uuid,
                split=self._split,
                log_name=self._uuid,
                map_has_z=True,
                map_is_per_log=True,
            )
        return LogMetadata(
            dataset="nurec",
            split=self._split,
            log_name=self._uuid,
            location=self._uuid,
            map_metadata=map_metadata,
        )

    @override
    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        with ZipFile(self._usdz_path) as archive:
            rig_root = _read_required_json_member(archive, "rig_trajectories.json", self._uuid)
            tracks_root = _read_required_json_member(archive, "sequence_tracks.json", self._uuid)
            bbox_length_m = float(rig_root["rig_trajectories"][0]["rig_bbox"]["dim"][0])
            wheel_base_m = _rig_wheel_base_m(archive, bbox_length_m, self._uuid)

        # One clip is one vehicle's drive, so the single rig trajectory is the ego.
        rig_trajectories = rig_root["rig_trajectories"]
        if len(rig_trajectories) != 1:
            raise ValueError(f"NuRec scene {self._uuid}: expected one rig trajectory, found {len(rig_trajectories)}")
        rig = rig_trajectories[0]
        ego_metadata = _extract_ego_metadata(rig, wheel_base_m)
        tracks = _extract_tracks(tracks_root, smooth_positions=self._smooth_track_positions)

        # The rig timestamps are the window tracks are filtered against.
        rig_timestamps_us = [int(ts) for ts in rig["T_rig_world_timestamps_us"]]
        if rig_timestamps_us:
            tracks = _filter_short_tracks(
                tracks,
                rig_timestamps_us[0],
                rig_timestamps_us[-1],
                self._min_traffic_duration_us,
            )

        rig_poses = [PoseSE3.from_transformation_matrix(np.asarray(T, dtype=np.float64)) for T in rig["T_rig_worlds"]]

        # Uniform 10 Hz grid anchored at ts[1] - 100 ms; ts[0] is typically off-grid.
        frame_timestamps_us = _uniform_grid_us(rig_timestamps_us)

        for timestamp_us in frame_timestamps_us:
            timestamp = Timestamp.from_us(timestamp_us)
            ego_pose = _interpolate_pose_sequence(rig_timestamps_us, rig_poses, timestamp_us, extrapolate=True)
            if ego_pose is None:
                continue
            ego_state = EgoStateSE3.from_imu(
                imu_se3=ego_pose,
                metadata=ego_metadata,
                timestamp=timestamp,
            )

            detections = [
                detection
                for detection in (_track_detection_at_timestamp(track, timestamp_us) for track in tracks)
                if detection is not None
            ]

            yield ModalitiesSync(
                timestamp=timestamp,
                modalities=[
                    ego_state,
                    BoxDetectionsSE3(
                        box_detections=detections,
                        timestamp=timestamp,
                        metadata=NUREC_BOX_DETECTIONS_SE3_METADATA,
                    ),
                ],
            )


def _read_required_json_member(archive: ZipFile, member: str, uuid: str) -> Dict:
    """JSON payload of a required USDZ member.

    Raises a message naming the scene and member, instead of a bare zipfile
    ``KeyError``, so a batch run's logs are triageable across many scenes.
    """
    try:
        payload = archive.read(member)
    except KeyError:
        raise ValueError(f"NuRec scene {uuid}: archive has no required member {member!r}") from None
    return json.loads(payload)


def _uniform_grid_us(rig_timestamps_us: List[int]) -> List[int]:
    """Uniform 10 Hz frame timestamps spanning the rig trajectory.

    Recorded timestamps are nominally 10 Hz but drift by milliseconds, and cuboid
    tracks run on their own clock. The grid is anchored at the second rig
    timestamp, since the first one usually sits off the rhythm.
    """
    step_us = 100_000
    t0_us = rig_timestamps_us[1] - step_us
    num_steps = (rig_timestamps_us[-1] - t0_us) // step_us + 1
    return [t0_us + step * step_us for step in range(num_steps)]


def _rig_wheel_base_m(archive: ZipFile, bbox_length_m: float, uuid: str) -> float:
    """Wheel base resolved in priority order:

    1. calibration_estimate vehicle block (26.04 branch) — exact axle positions.
    2. calibration_estimate rig.properties.platform_name (26.04 fallback) — table lookup.
    3. Nearest-neighbour on bbox_length_m (main branch) — table lookup.
    4. Hard constant fallback.

    Steps 1-2 are skipped (falling through to 3-4) when calibration_estimate.parquet
    is absent or malformed. It is absent in scenes that ship no clipgt layers at all.
    """
    rig: Dict = {}
    try:
        frame = pd.read_parquet(io.BytesIO(archive.read(_clipgt_member("calibration_estimate"))))
        cal_row = frame["calibration_estimate"].iloc[0]
        rig_json_raw = cal_row["rig_json"] if "rig_json" in cal_row else cal_row
        rig = (json.loads(rig_json_raw) if isinstance(rig_json_raw, str) else rig_json_raw).get("rig", {})
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.warning(
            "NuRec %s: could not read calibration_estimate (%s); falling back to platform/bbox lookup", uuid, e
        )

    vehicle_entry = rig.get("vehicle")
    if vehicle_entry is not None:
        vehicle = vehicle_entry["value"] if "value" in vehicle_entry else vehicle_entry
        return float(vehicle["axleFront"]["position"]) - float(vehicle["axleRear"]["position"])

    platform_name = rig.get("properties", {}).get("platform_name", "")
    if platform_name in _NUREC_PLATFORM_WHEEL_BASE_M:
        wb = _NUREC_PLATFORM_WHEEL_BASE_M[platform_name]
        logger.debug("NuRec platform %r → wheel base %.3f m", platform_name, wb)
        return wb

    if _NUREC_VEHICLE_BY_LENGTH:
        length, wb = min(_NUREC_VEHICLE_BY_LENGTH, key=lambda v: abs(v[0] - bbox_length_m))
        logger.debug("NuRec bbox length %.3f m → nearest vehicle %.3f m → wheel base %.3f m", bbox_length_m, length, wb)
        return wb

    logger.warning("NuRec: could not determine wheel base; using fallback %.3f m", _NUREC_WHEEL_BASE_FALLBACK_M)
    return _NUREC_WHEEL_BASE_FALLBACK_M


def _extract_ego_metadata(rig: Dict, wheel_base_m: float) -> EgoStateSE3Metadata:
    """Ego vehicle metadata; rig_bbox gives the dims and the rig->center offset.

    That offset is around 1.4 m longitudinally, so a missing rig_bbox raises
    rather than defaulting a box into the wrong place.
    """
    bbox = rig.get("rig_bbox") or {}
    dims, centroid = bbox.get("dim"), bbox.get("centroid")
    if not dims or not centroid:
        raise ValueError(f"NuRec rig {rig.get('sequence_id')}: rig_bbox is missing its dim or centroid")
    return EgoStateSE3Metadata(
        vehicle_name="nurec_rig",
        length=float(dims[0]),
        width=float(dims[1]),
        height=float(dims[2]),
        wheel_base=wheel_base_m,
        center_to_imu_se3=PoseSE3(
            x=float(centroid[0]),
            y=float(centroid[1]),
            z=float(centroid[2]),
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        ),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def _extract_tracks(tracks_root: Dict, smooth_positions: bool = False) -> List[_NuRecTrack]:
    """Tracks from sequence_tracks.json with valid, time-sorted samples."""
    chunk = next(iter(tracks_root.values()))
    tracks_data = chunk["tracks_data"]
    dims_data = chunk.get("cuboidtracks_data", {})
    all_dims = dims_data.get("cuboids_dims", [])

    tracks: List[_NuRecTrack] = []
    for idx, track_id in enumerate(tracks_data.get("tracks_id", [])):
        poses = tracks_data["tracks_poses"][idx]
        timestamps_us = [int(ts) for ts in tracks_data["tracks_timestamps_us"][idx]]
        if len(poses) != len(timestamps_us):
            logger.warning("Skipping NuRec track %s with mismatched pose/timestamp counts", track_id)
            continue

        if idx >= len(all_dims):
            logger.warning("Skipping NuRec track %s with no cuboid dims", track_id)
            continue
        length, width, height = (float(all_dims[idx][0]), float(all_dims[idx][1]), float(all_dims[idx][2]))
        if length <= 0.0 or width <= 0.0 or height <= 0.0:
            logger.warning("Skipping NuRec track %s with invalid cuboid dims %s", track_id, all_dims[idx])
            continue

        valid_pairs = sorted(
            (
                (int(timestamp_us), list(pose))
                for pose, timestamp_us in zip(poses, timestamps_us)
                if _valid_track_pose(pose)
            ),
            key=lambda pair: pair[0],
        )
        if not valid_pairs:
            continue

        ts_arr = np.array([t for t, _ in valid_pairs], dtype=np.float64)
        if smooth_positions and len(valid_pairs) >= 4 and bool(np.all(np.diff(ts_arr) > 0)):
            # AlpaSim smooths positions only, never orientations.
            pos = np.array([pose[:3] for _, pose in valid_pairs], dtype=np.float64)
            css = csaps.CubicSmoothingSpline(ts_arr / 1e6, pos.T, normalizedsmooth=True)
            smoothed = css(ts_arr / 1e6).T
            max_error = float(np.max(np.abs(smoothed - pos)))
            if max_error > 1.0:
                logger.warning("Max error in cubic spline approximation: %.6f m for track_id=%s", max_error, track_id)
            valid_pairs = [(t, [*smoothed[i], *pose[3:]]) for i, (t, pose) in enumerate(valid_pairs)]

        samples = [
            _TrackSample(
                timestamp_us=t,
                pose=_nurec_track_pose_to_pose_se3(pose),
            )
            for t, pose in valid_pairs
        ]
        tracks.append(
            _NuRecTrack(
                track_id=str(track_id),
                label=_nurec_label(tracks_data["tracks_label_class"][idx]),
                length=length,
                width=width,
                height=height,
                samples=samples,
                timestamps_us=[sample.timestamp_us for sample in samples],
            )
        )

    return tracks


def _clipped_lifetime_us(track: _NuRecTrack, window_start_us: int, window_end_us: int) -> int:
    """Track lifetime inside the window, matching AlpaSim's clip-then-filter semantics."""
    inside = [ts for ts in track.timestamps_us if window_start_us <= ts <= window_end_us]
    if not inside:
        return 0
    return inside[-1] - inside[0] + 1


def _filter_short_tracks(
    tracks: List[_NuRecTrack],
    window_start_us: int,
    window_end_us: int,
    min_duration_us: int,
) -> List[_NuRecTrack]:
    """Drop tracks AlpaSim would not spawn: in-window lifetime below min_duration_us."""
    if min_duration_us <= 0:
        return tracks
    kept = [track for track in tracks if _clipped_lifetime_us(track, window_start_us, window_end_us) >= min_duration_us]
    dropped = len(tracks) - len(kept)
    if dropped:
        logger.info(
            "NuRec: dropped %d/%d tracks with in-window lifetime < %d us (AlpaSim parity)",
            dropped,
            len(tracks),
            min_duration_us,
        )
    return kept


def _slerp_quat_wxyz(qa: np.ndarray, qb: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation between two (qw,qx,qy,qz) quaternions."""
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        q = qa + alpha * (qb - qa)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        q = (np.sin((1.0 - alpha) * theta) * qa + np.sin(alpha * theta) * qb) / np.sin(theta)
    return q / np.linalg.norm(q)


def _interpolate_pose_pair(a: PoseSE3, b: PoseSE3, alpha: float) -> PoseSE3:
    """Position-lerp + quaternion-slerp between two poses at fraction alpha."""
    qa = np.array([a.qw, a.qx, a.qy, a.qz], dtype=np.float64)
    qb = np.array([b.qw, b.qx, b.qy, b.qz], dtype=np.float64)
    qw, qx, qy, qz = _slerp_quat_wxyz(qa, qb, alpha)
    return PoseSE3(
        x=a.x + alpha * (b.x - a.x),
        y=a.y + alpha * (b.y - a.y),
        z=a.z + alpha * (b.z - a.z),
        qw=float(qw),
        qx=float(qx),
        qy=float(qy),
        qz=float(qz),
    )


def _interpolate_pose_sequence(
    timestamps_us: List[int],
    poses: List[PoseSE3],
    target_us: int,
    extrapolate: bool = False,
) -> Optional[PoseSE3]:
    """Pose at target_us via position-lerp + quaternion-slerp; None outside the sequence.

    With extrapolate=True, targets outside the sequence extend the boundary
    segment's linear motion, with rotation clamped to the boundary sample.
    """
    if not timestamps_us:
        return None
    if target_us < timestamps_us[0] or target_us > timestamps_us[-1]:
        if not extrapolate or len(timestamps_us) < 2:
            return None
        lo, hi = (0, 1) if target_us < timestamps_us[0] else (len(timestamps_us) - 2, len(timestamps_us) - 1)
        span = timestamps_us[hi] - timestamps_us[lo]
        if span <= 0:
            return poses[lo]
        alpha = (target_us - timestamps_us[lo]) / span
        a, b = poses[lo], poses[hi]
        boundary = poses[0] if target_us < timestamps_us[0] else poses[-1]
        return PoseSE3(
            x=a.x + alpha * (b.x - a.x),
            y=a.y + alpha * (b.y - a.y),
            z=a.z + alpha * (b.z - a.z),
            qw=boundary.qw,
            qx=boundary.qx,
            qy=boundary.qy,
            qz=boundary.qz,
        )
    idx = bisect.bisect_left(timestamps_us, target_us)
    if idx < len(timestamps_us) and timestamps_us[idx] == target_us:
        return poses[idx]
    lo, hi = idx - 1, idx
    span = timestamps_us[hi] - timestamps_us[lo]
    if span <= 0:
        return poses[lo]
    alpha = (target_us - timestamps_us[lo]) / span
    return _interpolate_pose_pair(poses[lo], poses[hi], alpha)


def _track_detection_at_timestamp(track: _NuRecTrack, timestamp_us: int) -> Optional[BoxDetectionSE3]:
    """Detection at an exact timestamp, within the track's lifetime only; no extrapolation."""
    pose = _interpolate_pose_sequence(track.timestamps_us, [s.pose for s in track.samples], timestamp_us)
    if pose is None:
        return None
    sample_idx = _nearest_timestamp_index(track.timestamps_us, timestamp_us)
    return BoxDetectionSE3(
        attributes=BoxDetectionAttributes(
            label=track.label,
            track_token=track.track_id,
        ),
        bounding_box_se3=BoundingBoxSE3(
            center_se3=pose,
            length=track.length,
            width=track.width,
            height=track.height,
        ),
        velocity_3d=_track_velocity(track, sample_idx) if sample_idx is not None else None,
    )


def _nearest_timestamp_index(timestamps_us: List[int], target_us: int) -> Optional[int]:
    """Index of the timestamp nearest to target_us; None for an empty sequence."""
    insert_idx = bisect.bisect_left(timestamps_us, target_us)
    candidates = [idx for idx in (insert_idx, insert_idx - 1) if 0 <= idx < len(timestamps_us)]
    if not candidates:
        return None
    return min(candidates, key=lambda idx: abs(timestamps_us[idx] - target_us))


def _track_velocity(track: _NuRecTrack, sample_idx: int) -> Optional[Vector3D]:
    """Central-difference velocity around a sample; None when undefined."""
    if len(track.samples) < 2:
        return None

    prev_idx = max(0, sample_idx - 1)
    next_idx = min(len(track.samples) - 1, sample_idx + 1)
    if prev_idx == next_idx:
        return None

    prev_sample = track.samples[prev_idx]
    next_sample = track.samples[next_idx]
    dt_s = (next_sample.timestamp_us - prev_sample.timestamp_us) / 1_000_000.0
    if dt_s <= 0.0:
        return None

    return Vector3D(
        x=(next_sample.pose.x - prev_sample.pose.x) / dt_s,
        y=(next_sample.pose.y - prev_sample.pose.y) / dt_s,
        z=(next_sample.pose.z - prev_sample.pose.z) / dt_s,
    )


def _nurec_track_pose_to_pose_se3(pose: List[float]) -> PoseSE3:
    """PoseSE3 from a NuRec [x, y, z, qx, qy, qz, qw] track pose."""
    return PoseSE3(
        x=float(pose[0]),
        y=float(pose[1]),
        z=float(pose[2]),
        qw=float(pose[6]),
        qx=float(pose[3]),
        qy=float(pose[4]),
        qz=float(pose[5]),
    )


def _valid_track_pose(pose: List[float]) -> bool:
    return len(pose) == 7 and all(np.isfinite(float(value)) for value in pose)


def _nurec_label(label: str) -> PhysicalAIAVBoxDetectionLabel:
    """NuRec cuboid class string to its Physical AI AV label.

    NuRec uses that taxonomy, so the class strings match the enum member names.
    """
    return PhysicalAIAVBoxDetectionLabel[label.upper()]
