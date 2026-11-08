"""Selection of camera images at a lower rate: the image nearest to each reference time, with hysteresis.

A camera that captures faster than the stored rate (20 or 30 images per second, stored at 10) keeps one image per
reference time. The reference times come from the dataset: for example the times at which the lidar measures the
camera's direction (one per lidar scan), or a uniform grid. :func:`select_nearest_frames` keeps the nearest image
per reference time; with hysteresis it keeps the image ``step`` images after the previously kept one unless the
nearest image is nearer by more than the hysteresis. Without hysteresis, a camera whose clock drifts against the
reference switches between two images whenever both are about equally near, which gives irregular intervals
(report camera-lidar-compression, Section 5); 0.5 ms of hysteresis removes these switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

FRAME_SELECTION_MODES = ("all", "reference", "grid")


@dataclass(frozen=True)
class FrameSelectionConfig:
    """How a parser selects camera images.

    :param mode: ``"all"`` keeps every image; ``"reference"`` keeps the image nearest to each reference time that the
        parser supplies (for example the time the lidar measures the camera direction); ``"grid"`` uses a uniform grid
        of reference times with period ``period_us`` and phase ``grid_phase_us``.
    :param period_us: Period of the kept images (100,000 us for 10 images per second).
    :param hysteresis_us: Hysteresis of the selection (0 for the nearest image).
    :param limit_us: A reference time without an image within this offset keeps no image (None: no limit).
    :param grid_phase_us: Phase of the grid in mode ``"grid"``: the grid times are ``k * period_us + grid_phase_us``.
    """

    mode: str = "all"
    period_us: int = 100_000
    hysteresis_us: int = 500
    limit_us: Optional[int] = None
    grid_phase_us: int = 0

    def __post_init__(self) -> None:
        assert self.mode in FRAME_SELECTION_MODES, f"mode must be one of {FRAME_SELECTION_MODES}, got {self.mode}."
        assert self.period_us > 0, f"period_us must be positive, got {self.period_us}."
        assert self.hysteresis_us >= 0, f"hysteresis_us must be >= 0, got {self.hysteresis_us}."
        assert self.limit_us is None or self.limit_us >= 0, f"limit_us must be >= 0, got {self.limit_us}."


def nearest_indices(frame_ts: npt.ArrayLike, query_ts: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Index of the nearest frame time for each query time; a tie goes to the earlier frame.

    :param frame_ts: Sorted frame times (at least one).
    :param query_ts: Query times.
    :return: One index per query time.
    """
    frame_ts = np.asarray(frame_ts, dtype=np.int64)
    query_ts = np.asarray(query_ts, dtype=np.int64)
    if len(frame_ts) == 0:
        raise ValueError("nearest_indices requires at least one frame time.")
    if len(frame_ts) == 1:
        return np.zeros(len(query_ts), dtype=np.int64)
    upper = np.clip(np.searchsorted(frame_ts, query_ts), 1, len(frame_ts) - 1)
    lower = upper - 1
    take_upper = np.abs(frame_ts[upper] - query_ts) < np.abs(frame_ts[lower] - query_ts)
    return np.where(take_upper, upper, lower).astype(np.int64)


def frames_per_reference(frame_ts: npt.ArrayLike, ref_ts: npt.ArrayLike) -> int:
    """The number of frames per reference period (2 for 20 Hz frames and 10 Hz references), at least 1."""
    frame_ts = np.asarray(frame_ts, dtype=np.int64)
    ref_ts = np.asarray(ref_ts, dtype=np.int64)
    if len(frame_ts) < 2 or len(ref_ts) < 2:
        return 1
    frame_period = float(np.median(np.diff(frame_ts)))
    ref_period = float(np.median(np.diff(ref_ts)))
    if frame_period <= 0:
        return 1
    return max(1, int(round(ref_period / frame_period)))


def select_nearest_frames(
    frame_ts: npt.ArrayLike,
    ref_ts: npt.ArrayLike,
    hysteresis_us: int = 0,
    step: Optional[int] = None,
    limit_us: Optional[int] = None,
) -> npt.NDArray[np.int64]:
    """The kept frame for each reference time.

    Without hysteresis, the nearest frame (a tie goes to the earlier one). With hysteresis, the frame ``step`` frames
    after the previously kept one, unless the nearest frame is nearer than it by more than ``hysteresis_us`` (as the
    rule of the report camera-lidar-compression, ``sync_analysis.select``). With a limit, a reference time whose kept
    frame is farther than ``limit_us`` keeps no frame (-1), and the next reference time starts from its nearest frame.

    :param frame_ts: Sorted frame times in us.
    :param ref_ts: Sorted reference times in us.
    :param hysteresis_us: The hysteresis in us (0: nearest frames).
    :param step: Frames per reference period; estimated from the median intervals when None.
    :param limit_us: Largest offset of a kept frame from its reference time, or None.
    :return: One frame index per reference time, -1 where no frame is kept. Two reference times may keep the same
        frame when frames are missing.
    """
    frame_ts = np.asarray(frame_ts, dtype=np.int64)
    ref_ts = np.asarray(ref_ts, dtype=np.int64)
    if len(ref_ts) == 0:
        return np.zeros(0, dtype=np.int64)
    if len(frame_ts) == 0:
        return np.full(len(ref_ts), -1, dtype=np.int64)
    near = nearest_indices(frame_ts, ref_ts)
    if step is None:
        step = frames_per_reference(frame_ts, ref_ts)
    out = near.copy()
    previous: Optional[int] = None
    n = len(frame_ts)
    for k in range(len(ref_ts)):
        ref = int(ref_ts[k])
        if hysteresis_us and previous is not None:
            candidate = previous + step
            if candidate < n and near[k] != candidate:
                candidate_offset = abs(int(frame_ts[candidate]) - ref)
                within_limit = limit_us is None or candidate_offset <= limit_us
                if within_limit and abs(int(frame_ts[near[k]]) - ref) + hysteresis_us >= candidate_offset:
                    out[k] = candidate
        if limit_us is not None and abs(int(frame_ts[out[k]]) - ref) > limit_us:
            out[k] = -1
            previous = None
        else:
            previous = int(out[k])
    return out


def kept_frames(selected: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """The sorted, unique frame indices of a selection (without the -1 entries)."""
    selected = np.asarray(selected, dtype=np.int64)
    return np.unique(selected[selected >= 0])


def match_frames(frame_ts: npt.ArrayLike, capture_ts: npt.ArrayLike, max_offset_us: int) -> npt.NDArray[np.int64]:
    """The frame of each capture time: the nearest frame if it lies strictly within ``max_offset_us``, else -1.

    Used to keep the images of other cameras at the capture times chosen for a reference camera (cameras that are
    triggered together capture within a few microseconds; 5,000 us separates them from the neighbor images).
    """
    frame_ts = np.asarray(frame_ts, dtype=np.int64)
    capture_ts = np.asarray(capture_ts, dtype=np.int64)
    if len(frame_ts) == 0:
        return np.full(len(capture_ts), -1, dtype=np.int64)
    near = nearest_indices(frame_ts, capture_ts)
    return np.where(np.abs(frame_ts[near] - capture_ts) < max_offset_us, near, -1).astype(np.int64)


def uniform_reference_grid(start_us: int, end_us: int, period_us: int, phase_us: int = 0) -> npt.NDArray[np.int64]:
    """The times ``k * period_us + phase_us`` in [start_us, end_us)."""
    if period_us <= 0:
        raise ValueError(f"period_us must be positive, got {period_us}.")
    first = start_us + ((phase_us - start_us) % period_us)
    return np.arange(first, end_us, period_us, dtype=np.int64)
