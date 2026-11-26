"""OpenDV-YouTube parser: one camera-only log per YouTube driving video.

The dataset (OpenDriveLab/DriveAGI, ``opendv``) is a list of YouTube videos with, per video, a part to discard at
the beginning and at the end (title screens, parking). Its toolkit downloads each video to
``videos/<YouTuber with "_" for " ">/<video id>.<ext>`` and extracts 10 images per second from the rest. This parser
keeps the same images without the JPEG step:

- Grid: the times ``start_discard + k x 100 ms`` (seconds after the first frame), for ``k < int(10 x (floor(duration)
  - start_discard - end_discard))`` as in the toolkit's frame extraction.
- Per grid time the decoded frame nearest by presentation time (a tie goes to the earlier frame), if it lies at most
  50 ms away; each frame is kept at most once. The kept frame keeps its YUV planes and color tags.
- Camera: one front camera (``PCAM_F0``) with the video's size, an identity pose and no intrinsics (unknown for
  YouTube videos), and no global pose (no ego motion is known).
- Time stamps: microseconds after the first frame of the video.

Expected folder: ``<opendv_data_root>/meta/OpenDV-YouTube.json`` (the toolkit's video list: ``videoid``, ``split``,
``start_discard``, ``end_discard``, ``youtuber``) and ``<opendv_data_root>/videos``. Videos that are not on disk are
skipped (YouTube removed some of them). Asynchronous conversion only (``async_conversion=true``); the sync table has
one row per kept image.

kedec: ``py123d-conversion dataset=opendv codec=kedec async_conversion=true``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import av
import numpy as np

from py123d.common.io.camera.yuv_frame import YUV420Frame
from py123d.datatypes import Timestamp
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata
from py123d.geometry import PoseSE3
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
)

logger = logging.getLogger(__name__)

OPENDV_DATASET = "opendv"
OPENDV_SPLITS = ("opendv_train", "opendv_val")
VIDEO_EXTENSIONS = ("mp4", "webm", "mkv")  # the toolkit's POSSIBLE_EXTS
CAMERA_NAME = "youtube_front"
RATE_HZ = 10
MAX_OFFSET_S = 0.05  # half the grid period


def grid_frame_count(duration_s: float, start_discard_s: float, end_discard_s: float, rate_hz: int = RATE_HZ) -> int:
    """The number of grid times of a video, as the toolkit's ``int(fps * (total_frames // video_fps - discard_begin -
    discard_end))`` (``total_frames // video_fps`` is the duration rounded down to whole seconds)."""
    return max(0, int(rate_hz * (math.floor(duration_s) - start_discard_s - end_discard_s)))


def grid_times_s(start_discard_s: float, count: int, rate_hz: int = RATE_HZ) -> np.ndarray:
    """The grid times in seconds after the first frame."""
    return start_discard_s + np.arange(count, dtype=np.float64) / rate_hz


def select_grid_frames(
    frames: Iterable[Tuple[float, Any]], grid_s: Sequence[float], max_offset_s: float = MAX_OFFSET_S
) -> Iterator[Tuple[int, float, Any]]:
    """Yield ``(k, time, frame)`` for each grid time ``grid_s[k]`` with the frame nearest to it (a tie goes to the
    earlier frame), if that frame lies at most ``max_offset_s`` away; each frame is yielded at most once.

    :param frames: ``(time, frame)`` pairs in decoding order with increasing times (seconds).
    :param grid_s: The increasing grid times (seconds).
    """
    n = len(grid_s)
    k = 0
    previous: Optional[Tuple[float, Any]] = None
    last_time: Optional[float] = None

    def choose(current: Optional[Tuple[float, Any]], target: float) -> Optional[Tuple[float, Any]]:
        if previous is None:
            return current
        if current is None or target - previous[0] <= current[0] - target:
            return previous
        return current

    for time, frame in frames:
        if k >= n:
            break
        while k < n and time >= grid_s[k]:
            best = choose((time, frame), grid_s[k])
            if abs(best[0] - grid_s[k]) <= max_offset_s and best[0] != last_time:
                yield k, best[0], best[1]
                last_time = best[0]
            k += 1
        previous = (time, frame)
    while k < n and previous is not None:  # the video ends before the grid: only the last frame is left
        if abs(previous[0] - grid_s[k]) <= max_offset_s and previous[0] != last_time:
            yield k, previous[0], previous[1]
            last_time = previous[0]
        k += 1


def find_video(videos_root: Path, video: Dict[str, Any]) -> Optional[Path]:
    """The downloaded file of a video (``<folder>/<videoid>.<ext>``), or None."""
    folder = video.get("folder") or str(video["youtuber"]).replace(" ", "_")
    for ext in VIDEO_EXTENSIONS:
        path = videos_root / folder / f"{video['videoid']}.{ext}"
        if path.is_file():
            return path
    return None


def _decoded_frames(container: "av.container.InputContainer", stream) -> Iterator[Tuple[float, Any]]:
    """``(seconds after the first frame, frame)`` in decoding order; frames without a time are skipped."""
    first: Optional[float] = None
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        if first is None:
            first = float(frame.time)
        yield float(frame.time) - first, frame


def _duration_s(container: "av.container.InputContainer", stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    raise ValueError("the video has no duration")


class OpenDVLogParser(BaseLogParser):
    """Picklable handle to one OpenDV video."""

    def __init__(
        self,
        video_path: Path,
        video_id: str,
        split: str,
        start_discard_s: float,
        end_discard_s: float,
        decode_threads: int = 4,
    ) -> None:
        self._video_path = Path(video_path)
        self._video_id = video_id
        self._split = split
        self._start_discard_s = float(start_discard_s)
        self._end_discard_s = float(end_discard_s)
        self._decode_threads = int(decode_threads)

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(dataset=OPENDV_DATASET, split=self._split, log_name=self._video_id, location=None)

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Not supported: this parser converts asynchronously (``async_conversion=true``)."""
        raise NotImplementedError("OpenDVLogParser supports only iter_modalities_async (async_conversion=true).")

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass. The kept images of the video, in time order."""
        container = av.open(str(self._video_path))
        try:
            stream = container.streams.video[0]
            if self._decode_threads > 1:
                # Frame threading; one decoder thread keeps the default mode (video izhGt1GnGFk failed with
                # "Invalid data" only in frame-threaded decoding).
                stream.thread_type = "AUTO"
                stream.codec_context.thread_count = self._decode_threads
            count = grid_frame_count(_duration_s(container, stream), self._start_discard_s, self._end_discard_s)
            if count == 0:
                logger.warning("Video %s: no grid time after the discarded parts.", self._video_id)
                return
            metadata = PinholeCameraMetadata(
                camera_name=CAMERA_NAME,
                camera_id=CameraID.PCAM_F0,
                intrinsics=None,
                distortion=None,
                width=int(stream.codec_context.width),
                height=int(stream.codec_context.height),
                camera_to_imu_se3=PoseSE3.identity(),
            )
            grid = grid_times_s(self._start_discard_s, count)
            kept = 0
            for _, time_s, frame in select_grid_frames(_decoded_frames(container, stream), grid):
                kept += 1
                yield ParsedCamera(
                    metadata=metadata,
                    timestamp=Timestamp.from_us(int(round(time_s * 1e6))),
                    camera_to_global_se3=None,
                    frame=YUV420Frame.from_av_frame(frame),
                )
            if kept < count:
                logger.warning("Video %s: %d of %d grid times have a frame within 50 ms.", self._video_id, kept, count)
        finally:
            container.close()


class OpenDVParser(BaseDatasetParser):
    """Dataset parser for OpenDV-YouTube from the downloaded videos."""

    def __init__(
        self,
        splits: List[str],
        opendv_data_root: Union[Path, str],
        video_ids: Optional[List[str]] = None,
        max_videos: Optional[int] = None,
        decode_threads: int = 4,
    ) -> None:
        """
        :param splits: The splits, ``opendv_train`` and/or ``opendv_val``.
        :param opendv_data_root: The folder with ``meta/OpenDV-YouTube.json`` and ``videos``.
        :param video_ids: Only these videos, or None.
        :param max_videos: At most this many videos (in list order), or None.
        :param decode_threads: Decoder threads per video (frame threading); 1 decodes without frame threading.
        """
        unknown = set(splits) - set(OPENDV_SPLITS)
        assert not unknown, f"Unknown splits {sorted(unknown)}; valid: {list(OPENDV_SPLITS)}"
        assert opendv_data_root is not None, "opendv_data_root is required (dataset_paths.opendv_data_root)"
        self._root = Path(opendv_data_root)
        self._splits = list(splits)
        self._video_ids = None if video_ids is None else set(video_ids)
        self._max_videos = max_videos
        self._decode_threads = int(decode_threads)
        self._entries = self._collect_videos()

    def _collect_videos(self) -> List[Tuple[Path, Dict[str, Any], str]]:
        with open(self._root / "meta" / "OpenDV-YouTube.json") as f:
            videos = json.load(f)
        entries, missing = [], 0
        for video in videos:
            split = f"{OPENDV_DATASET}_{str(video['split']).lower()}"
            if split not in self._splits or (self._video_ids is not None and video["videoid"] not in self._video_ids):
                continue
            path = find_video(self._root / "videos", video)
            if path is None:
                missing += 1
                continue
            entries.append((path, video, split))
        if missing:
            logger.warning("%d videos of the selected splits are not on disk and are skipped.", missing)
        if self._max_videos is not None:
            entries = entries[: self._max_videos]
        return entries

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass. OpenDV has no maps."""
        return []

    def get_log_parsers(self) -> List[OpenDVLogParser]:  # type: ignore[override]
        """Inherited, see superclass."""
        return [
            OpenDVLogParser(
                video_path=path,
                video_id=video["videoid"],
                split=split,
                start_discard_s=video["start_discard"],
                end_discard_s=video["end_discard"],
                decode_threads=self._decode_threads,
            )
            for path, video, split in self._entries
        ]
