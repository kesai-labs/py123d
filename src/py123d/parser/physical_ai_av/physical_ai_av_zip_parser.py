"""Physical AI AV parser that reads the chunk archives of the Hugging Face dataset directly (no extraction).

An alternative to :class:`~py123d.parser.physical_ai_av.physical_ai_av_parser.PhysicalAIAVParser` for the dataset as
downloaded: sensor and label features in ``<feature>.chunk_NNNN.zip`` archives (members stored uncompressed, read
with one ``pread`` each through :mod:`py123d.parser.physical_ai_av.physical_ai_av_zip_io`), calibration as per-chunk
parquet files. Differences from the extracted-tree parser:

- Lidar: each spin is decoded from its Draco blob with the named attributes intensity, point time and
  ``model_element`` (beam and column index; :func:`physical_ai_av_sensor_io.decode_physical_ai_av_spin`), so the
  kedec lidar codec uses its grid mode.
- Cameras: the HEVC videos are decoded once in order with PyAV; the kept images keep their YUV planes and color
  tags (no JPEG step). The kept images follow :mod:`physical_ai_av_camera_selection`: per lidar spin, one reference
  time (the time the lidar measures the front camera direction), one capture time per camera group, 10 images per
  second.
- Asynchronous conversion only (``async_conversion=true``): the ego states at their native rate, the boxes, cameras
  and lidar once per spin; the sync table is built afterwards with the lidar spins as reference.
- No radar (the radar covers only about a third of the chunks).

kedec: ``py123d-conversion dataset=physical-ai-av-zip codec=kedec async_conversion=true``.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from py123d.common.io.camera.yuv_frame import YUV420Frame
from py123d.datatypes import EgoStateSE3, LidarID, Timestamp
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.ftheta_camera import FThetaCameraMetadata
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import ModalitiesSync, ParsedCamera
from py123d.parser.physical_ai_av.physical_ai_av_camera_selection import (
    FRONT_CAMERA,
    camera_azimuth_deg,
    direction_delay_us,
    front_delay_us,
    reference_times,
    select_camera_frames,
)
from py123d.parser.physical_ai_av.physical_ai_av_parser import (
    _LIDAR_SPIN_DURATION_US,
    PhysicalAIAVLogParser,
    PhysicalAIAVParser,
    _dynamic_state_in_ego_frame,
    _extract_box_detections,
    _get_ego_state_metadata,
    _get_ftheta_camera_metadatas,
    _get_lidar_metadata,
)
from py123d.parser.physical_ai_av.physical_ai_av_sensor_io import decode_physical_ai_av_spin, spin_start_column
from py123d.parser.physical_ai_av.physical_ai_av_zip_io import ZipChunkReader, pai_chunk_zip_path, pai_member_name
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import (
    PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA,
    PHYSICAL_AI_AV_CAMERA_ID_MAPPING,
)
from py123d.parser.physical_ai_av.utils.physical_ai_av_helper import find_closest_index, quat_scalar_last_to_pose_se3

logger = logging.getLogger(__name__)

LIDAR_FEATURE = "lidar_top_360fov"


def _pose_at(ego_df: pd.DataFrame, ego_timestamps: np.ndarray, timestamp_us: int):
    row = ego_df.iloc[find_closest_index(ego_timestamps, timestamp_us)]
    return quat_scalar_last_to_pose_se3(
        qx=row["qx"], qy=row["qy"], qz=row["qz"], qw=row["qw"], x=row["x"], y=row["y"], z=row["z"]
    )


class _SequentialVideo:
    """Decodes a video in order and returns the frames with increasing indices."""

    def __init__(self, data: io.BytesIO) -> None:
        self._container = av.open(data)
        stream = self._container.streams.video[0]
        stream.thread_type = "AUTO"
        self._frames = self._container.decode(stream)
        self._next = 0
        self.ended = False

    def frame_at(self, index: int):
        """The decoded frame with this index, or None when the video ends before it or the index lies behind."""
        if index < self._next or self.ended:
            return None
        for frame in self._frames:
            self._next += 1
            if self._next - 1 == index:
                return frame
        self.ended = True
        return None

    def close(self) -> None:
        self._container.close()


class PhysicalAIAVZipLogParser(PhysicalAIAVLogParser):
    """Picklable handle to one clip whose data lie in the chunk archives."""

    def __init__(
        self, data_root: Path, clip_id: str, chunk: int, split: str, cameras: List[str], delay_sample_every: int = 10
    ) -> None:
        super().__init__(data_root=data_root, clip_id=clip_id, chunk=chunk, split=split)
        self._cameras = list(cameras)
        self._delay_sample_every = max(1, int(delay_sample_every))

    def _zip(self, group: str, feature: str) -> Path:
        return pai_chunk_zip_path(self._data_root, group, feature, self._chunk)

    def _parquet(self, reader: ZipChunkReader, group: str, feature: str, ext: str) -> Optional[pd.DataFrame]:
        zip_path, member = self._zip(group, feature), pai_member_name(self._clip_id, feature, ext)
        if not reader.contains(zip_path, member):
            return None
        return pq.read_table(reader.open_bytes(zip_path, member)).to_pandas()

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Not supported: this parser converts asynchronously (``async_conversion=true``)."""
        raise NotImplementedError(
            "PhysicalAIAVZipLogParser supports only iter_modalities_async (async_conversion=true)."
        )

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        reader = ZipChunkReader()
        videos: Dict[str, _SequentialVideo] = {}
        try:
            yield from self._iter(reader, videos)
        finally:
            for video in videos.values():
                video.close()
            reader.close()

    def _iter(self, reader: ZipChunkReader, videos: Dict[str, _SequentialVideo]) -> Iterator[BaseModality]:
        ego_metadata = _get_ego_state_metadata(self._data_root, self._clip_id, self._chunk)

        # Ego states at their native rate.
        ego_df = self._parquet(reader, "labels", "egomotion", "parquet")
        assert ego_df is not None, f"No egomotion for clip {self._clip_id} (chunk {self._chunk})"
        ego_df = ego_df.sort_values("timestamp").reset_index(drop=True)
        ego_timestamps = ego_df["timestamp"].to_numpy(dtype=np.int64)
        for _, row in ego_df.iterrows():
            pose = quat_scalar_last_to_pose_se3(
                qx=row["qx"], qy=row["qy"], qz=row["qz"], qw=row["qw"], x=row["x"], y=row["y"], z=row["z"]
            )
            yield EgoStateSE3.from_imu(
                imu_se3=pose,
                metadata=ego_metadata,
                dynamic_state_se3=_dynamic_state_in_ego_frame(row, pose),
                timestamp=Timestamp.from_us(int(row["timestamp"])),
            )

        # Lidar spins: the 10 Hz grid of the clip.
        lidar_zip = self._zip("lidar", LIDAR_FEATURE)
        lidar_member = pai_member_name(self._clip_id, LIDAR_FEATURE, "parquet")
        assert reader.contains(lidar_zip, lidar_member), f"No lidar for clip {self._clip_id} (chunk {self._chunk})"
        lidar_table = pq.read_table(reader.open_bytes(lidar_zip, lidar_member))
        start_column = spin_start_column(lidar_table.column_names)
        spin_starts = lidar_table[start_column].to_numpy().astype(np.int64)
        spin_ends = (
            lidar_table["spin_end_timestamp"].to_numpy().astype(np.int64)
            if "spin_end_timestamp" in lidar_table.column_names
            else spin_starts + _LIDAR_SPIN_DURATION_US
        )
        blobs = lidar_table["draco_encoded_pointcloud"]
        lidar_metadata = _get_lidar_metadata(self._data_root, self._clip_id, self._chunk)
        lidar_to_imu = dict(lidar_metadata)[LidarID.LIDAR_TOP].lidar_to_imu_se3

        # Calibration and camera image times.
        ftheta_metadatas = _get_ftheta_camera_metadatas(self._data_root, self._clip_id, self._chunk)
        camera_timestamps: Dict[str, np.ndarray] = {}
        for name in self._cameras:
            camera_id = PHYSICAL_AI_AV_CAMERA_ID_MAPPING[name]
            if camera_id not in ftheta_metadatas:
                continue
            if not reader.contains(self._zip("camera", name), pai_member_name(self._clip_id, name, "mp4")):
                continue
            ts_df = self._parquet(reader, "camera", name, "timestamps.parquet")
            if ts_df is not None and len(ts_df) > 0:
                camera_timestamps[name] = ts_df["timestamp"].to_numpy(dtype=np.int64)

        # Reference time per spin: the time the lidar measures the front camera direction.
        delay = None
        front_id = PHYSICAL_AI_AV_CAMERA_ID_MAPPING[FRONT_CAMERA]
        if front_id in ftheta_metadatas:
            azimuth = camera_azimuth_deg(ftheta_metadatas[front_id].camera_to_imu_se3.rotation_matrix)
            center_xy = (float(lidar_to_imu.x), float(lidar_to_imu.y))
            samples = []
            for k in range(0, len(spin_starts), self._delay_sample_every):
                xyz, features = decode_physical_ai_av_spin(blobs[k].as_py(), int(spin_starts[k]), lidar_to_imu)
                times = features.get(LidarFeature.TIMESTAMPS.serialize())
                if times is not None:
                    samples.append(direction_delay_us(xyz, times, int(spin_starts[k]), center_xy, azimuth))
            delay = front_delay_us(samples)
        if delay is None:
            logger.warning("Clip %s: no front direction delay; the reference times are the spin starts.", self._clip_id)
        selection = select_camera_frames(camera_timestamps, reference_times(spin_starts, delay))

        obstacle_df = self._parquet(reader, "labels", "obstacle.offline", "parquet")
        obs_timestamps = obstacle_df["timestamp_us"].to_numpy(dtype=np.int64) if obstacle_df is not None else None

        for name in camera_timestamps:
            videos[name] = _SequentialVideo(
                reader.open_bytes(self._zip("camera", name), pai_member_name(self._clip_id, name, "mp4"))
            )
        last_kept: Dict[str, int] = {name: -1 for name in camera_timestamps}

        for k, spin_start in enumerate(spin_starts):
            yield _extract_box_detections(
                obstacle_df=obstacle_df,
                obs_timestamps=obs_timestamps,
                lidar_ts=int(spin_start),
                ego_df=ego_df,
                ego_timestamps=ego_timestamps,
                ego_metadata=ego_metadata,
                metadata=PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA,
            )
            for name, timestamps in camera_timestamps.items():
                index = int(selection[name][k])
                if index < 0 or index <= last_kept[name]:  # no image, or the image of the previous spin
                    continue
                frame = videos[name].frame_at(index)
                if frame is None:
                    continue
                last_kept[name] = index
                metadata: FThetaCameraMetadata = ftheta_metadatas[PHYSICAL_AI_AV_CAMERA_ID_MAPPING[name]]
                camera_ts = int(timestamps[index])
                ego_pose = _pose_at(ego_df, ego_timestamps, camera_ts)
                yield ParsedCamera(
                    metadata=metadata,
                    timestamp=Timestamp.from_us(camera_ts),
                    camera_to_global_se3=rel_to_abs_se3(origin=ego_pose, pose_se3=metadata.camera_to_imu_se3),
                    frame=YUV420Frame.from_av_frame(frame),
                )
            xyz, features = decode_physical_ai_av_spin(blobs[k].as_py(), int(spin_start), lidar_to_imu)
            yield Lidar(
                timestamp=Timestamp.from_us(int(spin_start)),
                timestamp_end=Timestamp.from_us(int(spin_ends[k])),
                metadata=lidar_metadata,
                point_cloud_3d=xyz,
                point_cloud_features=features,
            )


def convertible_clips(reader: ZipChunkReader, data_root: Path, entries: List[Tuple[str, int, str]]):
    """Split (clip_id, chunk, split) entries into the clips whose lidar and egomotion members exist, and the others
    (about 4 to 5 % of the clips have no lidar or egomotion although the index marks them valid)."""
    convertible, skipped = [], []
    for clip_id, chunk, split in entries:
        has_lidar = reader.contains(
            pai_chunk_zip_path(data_root, "lidar", LIDAR_FEATURE, chunk),
            pai_member_name(clip_id, LIDAR_FEATURE, "parquet"),
        )
        has_ego = reader.contains(
            pai_chunk_zip_path(data_root, "labels", "egomotion", chunk),
            pai_member_name(clip_id, "egomotion", "parquet"),
        )
        (convertible if has_lidar and has_ego else skipped).append((clip_id, chunk, split))
    return convertible, skipped


class PhysicalAIAVZipParser(PhysicalAIAVParser):
    """Dataset parser for Physical AI AV from the chunk archives."""

    def __init__(
        self,
        splits: List[str],
        physical_ai_av_data_root: Union[Path, str],
        max_clips: Optional[int] = None,
        cameras: Optional[List[str]] = None,
        chunk_start: Optional[int] = None,
        chunk_end: Optional[int] = None,
        clip_ids: Optional[List[str]] = None,
        delay_sample_every: int = 10,
    ) -> None:
        """
        :param splits: The splits, e.g. ["physical-ai-av_train"].
        :param physical_ai_av_data_root: The dataset folder (chunk archives, calibration, ``clip_index.parquet``).
        :param max_clips: At most this many clips, or None.
        :param cameras: The cameras to convert; None: all seven.
        :param chunk_start: The first chunk (inclusive), or None.
        :param chunk_end: The last chunk (exclusive), or None.
        :param clip_ids: Only these clips, or None.
        :param delay_sample_every: Every n-th spin is decoded twice to measure the front direction delay.
        """
        cameras = list(PHYSICAL_AI_AV_CAMERA_ID_MAPPING) if cameras is None else list(cameras)
        unknown = set(cameras) - set(PHYSICAL_AI_AV_CAMERA_ID_MAPPING)
        assert not unknown, f"Unknown cameras {sorted(unknown)}; valid: {sorted(PHYSICAL_AI_AV_CAMERA_ID_MAPPING)}"
        self._cameras = cameras
        self._chunk_start = None if chunk_start is None else int(chunk_start)
        self._chunk_end = None if chunk_end is None else int(chunk_end)
        self._clip_ids: Optional[Set[str]] = None if clip_ids is None else set(clip_ids)
        self._delay_sample_every = int(delay_sample_every)
        super().__init__(splits=splits, physical_ai_av_data_root=physical_ai_av_data_root, max_clips=max_clips)

    def _discover_available_chunks(self) -> set:
        """The chunks with calibration and a complete lidar archive, within [chunk_start, chunk_end)."""
        available = set(super()._discover_available_chunks())
        lidar_dir = self._data_root / "lidar" / LIDAR_FEATURE
        lidar_chunks = (
            {int(p.name.split("chunk_")[1].split(".")[0]) for p in lidar_dir.glob(f"{LIDAR_FEATURE}.chunk_*.zip")}
            if lidar_dir.is_dir()
            else set()
        )
        available &= lidar_chunks
        if self._chunk_start is not None:
            available = {c for c in available if c >= self._chunk_start}
        if self._chunk_end is not None:
            available = {c for c in available if c < self._chunk_end}
        return available

    def _collect_clips(self) -> List[Tuple[str, int, str]]:
        """Inherited, see superclass. Only clips with lidar and egomotion members (and in ``clip_ids``)."""
        entries = super()._collect_clips()
        if self._clip_ids is not None:
            entries = [e for e in entries if e[0] in self._clip_ids]
        reader = ZipChunkReader()
        try:
            convertible, skipped = convertible_clips(reader, self._data_root, entries)
        finally:
            reader.close()
        if skipped:
            logger.warning(
                "%d of %d clips have no lidar or egomotion member and are skipped.", len(skipped), len(entries)
            )
        return sorted(convertible, key=lambda e: (e[1], e[0]))

    def get_log_parsers(self) -> List[PhysicalAIAVZipLogParser]:  # type: ignore[override]
        """Inherited, see superclass."""
        return [
            PhysicalAIAVZipLogParser(
                data_root=self._data_root,
                clip_id=clip_id,
                chunk=chunk,
                split=split,
                cameras=self._cameras,
                delay_sample_every=self._delay_sample_every,
            )
            for clip_id, chunk, split in self._clip_entries
        ]
