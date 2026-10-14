"""NuRec scenes converted together with the cameras and the lidar of their Physical AI AV clip.

Every NuRec scene reconstructs one clip of the Physical AI AV dataset, under the clip's id. The
scene carries the map, the rig trajectory and the tracked cuboids; the clip carries the seven
f-theta cameras and the lidar. This parser joins the two: frames are the clip's lidar spins, the
ego pose is the rig pose at the spin, so that map, boxes and sensors share the scene's world
frame, and the cameras are posed with the rig pose at their own timestamp.

The two sources run on different clocks: a clip's clock starts where the scene's rig trajectory
starts. The scene records the start of every lidar spin on its own clock, which confirms the
offset. Log timestamps are on the scene's clock, like those of a scene converted without sensors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import override

from py123d.datatypes import BoxDetectionsSE3, CameraID, EgoStateSE3, LogMetadata, Timestamp
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.ftheta_camera import FThetaCameraMetadata
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature
from py123d.geometry import PoseSE3
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import BaseLogParser, ModalitiesSync, ParsedCamera
from py123d.parser.nurec.nurec_parser import (
    NUREC_BOX_DETECTIONS_SE3_METADATA,
    NuRecLogParser,
    NuRecParser,
    _interpolate_pose_sequence,
    _read_scene,
    _track_detection_at_timestamp,
)
from py123d.parser.physical_ai_av.physical_ai_av_parser import (
    _LIDAR_SPIN_DURATION_US,
    PhysicalAIAVLogParser,
    _get_ftheta_camera_metadatas,
    _get_lidar_metadata,
    _SequentialFrameReader,
)
from py123d.parser.physical_ai_av.physical_ai_av_sensor_io import load_physical_ai_av_point_cloud_data_from_path
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import PHYSICAL_AI_AV_CAMERA_ID_MAPPING
from py123d.parser.physical_ai_av.utils.physical_ai_av_helper import find_closest_index
from py123d.parser.physical_ai_av.utils.physical_ai_av_zip import path_exists, read_parquet, resolve_clip_file

logger = logging.getLogger(__name__)

_LIDAR_NAME = "lidar_top_360fov"
_LIDAR_TIMESTAMPS_FEATURE = LidarFeature.TIMESTAMPS.serialize()
# Spins of the two sources are the same spins; a larger median mismatch means the clocks were not aligned.
_MAX_SPIN_RESIDUAL_US = 1_000


class NuRecPAIParser(NuRecParser):
    """Dataset parser for NuRec scenes with the sensors of their Physical AI AV clip.

    Scenes are selected like in :class:`NuRecParser`. Those that are not Physical AI AV clips,
    or whose clip files are not under ``physical_ai_av_data_root``, are left out and counted
    in a warning. The Physical AI AV files are expected in the flat layout
    ``py123d-download dataset=nurec-pai`` writes.
    """

    def __init__(
        self,
        physical_ai_av_data_root: Union[str, Path],
        cameras: Optional[List[str]] = None,
        jpeg_quality: int = 75,
        **nurec_kwargs,
    ) -> None:
        """Initializes the parser.

        :param physical_ai_av_data_root: Physical AI AV data root with the files of the clips.
        :param cameras: Physical AI AV camera names to convert, defaults to all seven.
        :param jpeg_quality: JPEG quality the camera frames are stored with.
        :param nurec_kwargs: The options of :class:`NuRecParser`.
        """
        super().__init__(**nurec_kwargs)
        self._data_root = Path(physical_ai_av_data_root)
        self._cameras = list(cameras) if cameras is not None else list(PHYSICAL_AI_AV_CAMERA_ID_MAPPING)
        unknown = set(self._cameras) - set(PHYSICAL_AI_AV_CAMERA_ID_MAPPING)
        assert not unknown, f"Unknown cameras {sorted(unknown)}; valid: {sorted(PHYSICAL_AI_AV_CAMERA_ID_MAPPING)}"
        assert 1 <= int(jpeg_quality) <= 100, f"jpeg_quality must be in [1, 100], got {jpeg_quality}"
        self._jpeg_quality = int(jpeg_quality)

        clip_index = pd.read_parquet(self._data_root / "clip_index.parquet")
        chunk_by_clip = {str(clip_id): int(chunk) for clip_id, chunk in clip_index["chunk"].items()}
        self._chunks: Dict[str, int] = {}
        num_without_clip, num_without_files = 0, 0
        for scene in self._scenes:
            chunk = chunk_by_clip.get(scene.name)
            if chunk is None:
                num_without_clip += 1
            elif not self._has_lidar_and_egomotion(scene.name, chunk):
                num_without_files += 1
            else:
                self._chunks[scene.name] = chunk
        if num_without_clip or num_without_files:
            logger.warning(
                "NuRec: leaving out %d scene(s) that are no Physical AI AV clip and %d whose clip's lidar or "
                "egomotion is not under %s",
                num_without_clip,
                num_without_files,
                self._data_root,
            )
        self._scenes = [scene for scene in self._scenes if scene.name in self._chunks]
        if not self._scenes:
            raise FileNotFoundError(
                f"None of the selected NuRec scenes has its Physical AI AV files under {self._data_root}."
            )
        logger.info("NuRec: %d scene(s) with Physical AI AV sensors", len(self._scenes))

    def _has_lidar_and_egomotion(self, clip_id: str, chunk: int) -> bool:
        return path_exists(resolve_clip_file(self._data_root, "lidar", _LIDAR_NAME, clip_id, "parquet", chunk)) and (
            path_exists(resolve_clip_file(self._data_root, "labels", "egomotion", clip_id, "parquet", chunk))
        )

    @override
    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        return [
            NuRecPAILogParser(
                usdz_path=scene.usdz_path,
                split=scene.split,
                has_map=scene.has_map,
                min_traffic_duration_us=self._min_traffic_duration_us,
                smooth_track_positions=self._smooth_track_positions,
                data_root=self._data_root,
                chunk=self._chunks[scene.name],
                cameras=self._cameras,
                jpeg_quality=self._jpeg_quality,
            )
            for scene in self._scenes
        ]


class NuRecPAILogParser(PhysicalAIAVLogParser):
    """Log parser for one NuRec scene, with the cameras and the lidar read from its Physical AI AV clip."""

    def __init__(
        self,
        usdz_path: Union[str, Path],
        split: str,
        has_map: bool,
        min_traffic_duration_us: int,
        smooth_track_positions: bool,
        data_root: Path,
        chunk: int,
        cameras: List[str],
        jpeg_quality: int,
    ) -> None:
        """
        :param usdz_path: Path of the scene's USDZ archive.
        :param split: Split the log belongs to.
        :param has_map: Whether the scene is converted with a map.
        :param min_traffic_duration_us: See :class:`NuRecLogParser`.
        :param smooth_track_positions: See :class:`NuRecLogParser`.
        :param data_root: Physical AI AV data root.
        :param chunk: Physical AI AV chunk of the clip.
        :param cameras: Physical AI AV camera names to convert.
        :param jpeg_quality: JPEG quality the camera frames are stored with.
        """
        self._nurec = NuRecLogParser(
            usdz_path=usdz_path,
            split=split,
            min_traffic_duration_us=min_traffic_duration_us,
            smooth_track_positions=smooth_track_positions,
            has_map=has_map,
        )
        super().__init__(data_root=data_root, clip_id=self._nurec._uuid, chunk=chunk, split=split)
        self._cameras = list(cameras)
        self._jpeg_quality = int(jpeg_quality)

    @override
    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return self._nurec.get_log_metadata()

    @override
    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Not supported: the sensors are joined to the scene on the lidar spins, in sync mode."""
        raise NotImplementedError("NuRecPAILogParser only supports the synchronous conversion.")

    @override
    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass.

        One frame per lidar spin of the clip, on the scene's clock: the rig pose at the spin start, the
        tracked cuboids at that time, the spin, and the frame of every camera nearest to the spin start.
        """
        rig, ego_metadata, tracks = _read_scene(
            self._nurec._usdz_path,
            self._clip_id,
            self._nurec._smooth_track_positions,
            self._nurec._min_traffic_duration_us,
        )
        rig_timestamps_us = [int(ts) for ts in rig["T_rig_world_timestamps_us"]]
        rig_poses = [PoseSE3.from_transformation_matrix(np.asarray(T, dtype=np.float64)) for T in rig["T_rig_worlds"]]

        lidar_path = self._clip_file("lidar", _LIDAR_NAME, "parquet")
        clip_spin_starts_us = self._read_lidar_timestamps(lidar_path)
        clock_offset_us = _clock_offset_us(
            rig_timestamps_us, _scene_spin_starts_us(rig, self._clip_id), clip_spin_starts_us
        )

        def rig_pose_at(scene_time_us: int) -> PoseSE3:
            pose = _interpolate_pose_sequence(rig_timestamps_us, rig_poses, scene_time_us, extrapolate=True)
            assert pose is not None, f"NuRec scene {self._clip_id}: no rig trajectory"
            return pose

        ftheta_metadatas = _get_ftheta_camera_metadatas(self._data_root, self._clip_id, self._chunk)
        lidar_metadata = _get_lidar_metadata(self._data_root, self._clip_id, self._chunk)
        camera_timestamps_us: Dict[str, npt.NDArray[np.int64]] = {}
        for camera_name in self._cameras:
            timestamps_path = self._clip_file("camera", camera_name, "timestamps.parquet")
            if path_exists(timestamps_path):
                camera_timestamps_us[camera_name] = read_parquet(timestamps_path)["timestamp"].to_numpy(dtype=np.int64)

        captures: Dict[str, _SequentialFrameReader] = {}
        try:
            for camera_name in camera_timestamps_us:
                video_path = self._clip_file("camera", camera_name, "mp4")
                if path_exists(video_path):
                    captures[camera_name] = _SequentialFrameReader(self._video_file(video_path))

            for spin_index, clip_spin_start_us in enumerate(clip_spin_starts_us):
                scene_time_us = int(clip_spin_start_us) + clock_offset_us
                timestamp = Timestamp.from_us(scene_time_us)
                ego_state = EgoStateSE3.from_imu(
                    imu_se3=rig_pose_at(scene_time_us), metadata=ego_metadata, timestamp=timestamp
                )
                detections = [
                    detection
                    for detection in (_track_detection_at_timestamp(track, scene_time_us) for track in tracks)
                    if detection is not None
                ]
                box_detections = BoxDetectionsSE3(
                    box_detections=detections, timestamp=timestamp, metadata=NUREC_BOX_DETECTIONS_SE3_METADATA
                )
                # Decoded here: the log writer loads a lazy lidar by dataset name, and the dataset is NuRec.
                point_cloud_3d, point_cloud_features = load_physical_ai_av_point_cloud_data_from_path(
                    lidar_path, spin_index, dict(lidar_metadata)
                )
                if _LIDAR_TIMESTAMPS_FEATURE in point_cloud_features:
                    point_cloud_features[_LIDAR_TIMESTAMPS_FEATURE] = (
                        point_cloud_features[_LIDAR_TIMESTAMPS_FEATURE] + clock_offset_us
                    )
                lidar = Lidar(
                    timestamp=timestamp,
                    timestamp_end=Timestamp.from_us(scene_time_us + _LIDAR_SPIN_DURATION_US),
                    metadata=lidar_metadata,
                    point_cloud_3d=point_cloud_3d,
                    point_cloud_features=point_cloud_features,
                )
                cameras = _extract_cameras(
                    clip_spin_start_us=int(clip_spin_start_us),
                    clock_offset_us=clock_offset_us,
                    rig_pose_at=rig_pose_at,
                    captures=captures,
                    camera_timestamps_us=camera_timestamps_us,
                    ftheta_metadatas=ftheta_metadatas,
                    jpeg_quality=self._jpeg_quality,
                )
                yield ModalitiesSync(timestamp=timestamp, modalities=[ego_state, box_detections, lidar, *cameras])
        finally:
            for capture in captures.values():
                capture.release()
            self._remove_staged_videos()


def _scene_spin_starts_us(rig: Dict, uuid: str) -> npt.NDArray[np.int64]:
    """The lidar spin start times a scene records, on the scene's clock."""
    lidar_frames = rig.get("lidars_frame_timestamps_us") or {}
    if len(lidar_frames) != 1:
        raise ValueError(f"NuRec scene {uuid}: expected one lidar in rig_trajectories.json, found {len(lidar_frames)}")
    return np.asarray(next(iter(lidar_frames.values())), dtype=np.int64)[:, 0]


def _clock_offset_us(
    rig_timestamps_us: List[int],
    scene_spin_starts_us: npt.NDArray[np.int64],
    clip_spin_starts_us: npt.NDArray[np.int64],
) -> int:
    """Scene time minus clip time.

    A clip's clock starts where the scene's rig trajectory starts, so the offset is the first rig
    timestamp. The lidar spins both sides record are the same spins, which confirms it.

    :param rig_timestamps_us: Timestamps of the scene's rig trajectory.
    :param scene_spin_starts_us: Spin start times the scene records, on its clock.
    :param clip_spin_starts_us: Spin start times of the clip, on its clock.
    :return: The offset to add to a clip time to get the scene time.
    """
    if not rig_timestamps_us or len(scene_spin_starts_us) == 0 or len(clip_spin_starts_us) == 0:
        raise ValueError("Cannot align the clocks of a scene and its clip without a rig trajectory and lidar spins.")
    offset_us = int(rig_timestamps_us[0])
    shifted = clip_spin_starts_us.astype(np.int64) + offset_us
    insert = np.searchsorted(scene_spin_starts_us, shifted)
    below = scene_spin_starts_us[np.clip(insert - 1, 0, len(scene_spin_starts_us) - 1)]
    above = scene_spin_starts_us[np.clip(insert, 0, len(scene_spin_starts_us) - 1)]
    residual_us = float(np.median(np.minimum(np.abs(below - shifted), np.abs(above - shifted))))
    if residual_us > _MAX_SPIN_RESIDUAL_US:
        raise ValueError(
            f"The lidar spins of the scene and of its clip do not align (median mismatch {residual_us:.0f} us "
            f"with the clip clock starting at the rig trajectory)."
        )
    return offset_us


def _extract_cameras(
    clip_spin_start_us: int,
    clock_offset_us: int,
    rig_pose_at,
    captures: Dict[str, _SequentialFrameReader],
    camera_timestamps_us: Dict[str, npt.NDArray[np.int64]],
    ftheta_metadatas: Dict[CameraID, FThetaCameraMetadata],
    jpeg_quality: int,
) -> List[ParsedCamera]:
    """The frame of every camera nearest to a lidar spin start, posed with the rig pose at the frame's time."""
    cameras: List[ParsedCamera] = []
    for camera_name, timestamps_us in camera_timestamps_us.items():
        camera_id = PHYSICAL_AI_AV_CAMERA_ID_MAPPING[camera_name]
        if camera_id not in ftheta_metadatas or camera_name not in captures:
            continue
        frame_index = find_closest_index(timestamps_us, clip_spin_start_us)
        frame = captures[camera_name].read_at(frame_index)
        if frame is None:
            continue
        camera_metadata = ftheta_metadatas[camera_id]
        scene_time_us = int(timestamps_us[frame_index]) + clock_offset_us
        _, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        cameras.append(
            ParsedCamera(
                metadata=camera_metadata,
                timestamp=Timestamp.from_us(scene_time_us),
                camera_to_global_se3=rel_to_abs_se3(
                    origin=rig_pose_at(scene_time_us), pose_se3=camera_metadata.camera_to_imu_se3
                ),
                byte_string=bytes(jpeg_bytes),
            )
        )
    return cameras
