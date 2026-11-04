"""Selection of the camera images of a Physical AI AV clip for 10 images per second (kedec).

The cameras record 30 images per second; the lidar spins at 10 Hz. Per spin, one reference time for all seven
cameras: the time at which the lidar measures the direction of the front camera (front wide). It is the spin start
plus a delay, the median over sampled spins of the median time stamp of the points farther than 10 m within 1 degree
of the camera azimuth (as the KE:SAI rule of the report "camera-lidar-compression"). A clip without such points in
the sampled spins (seen for about 0.07 % of the clips: no lidar points within 1 degree of the camera azimuth, or only
points nearer than 10 m) uses the fixed delay of 50 ms: over 196 random clips the delay lies between 49.1 and 50.6 ms
(median 50.0 ms), because it is set by the lidar and camera mounting, not by the scene.

The cameras are triggered in two groups (front wide, front tele, rear left, rear right; cross left, cross right,
rear tele, the second group at a constant offset of up to 16.7 ms). In each group, the first camera present keeps
the image nearest to each reference time with a hysteresis of 0.5 ms (at most 26 ms away), and the other cameras of
the group keep their image of the same capture time (strictly within 5 ms), so each group keeps one capture time. A
camera whose images lie 5 ms or more (median) from those of its group leader in a clip (seen for the rear tele
camera) is selected on its own, with the same reference times, hysteresis and limit.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from py123d.common.frame_selection import match_frames, select_nearest_frames

FRONT_CAMERA = "camera_front_wide_120fov"
CAMERA_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("camera_front_wide_120fov", "camera_front_tele_30fov", "camera_rear_left_70fov", "camera_rear_right_70fov"),
    ("camera_cross_left_120fov", "camera_cross_right_120fov", "camera_rear_tele_30fov"),
)
FAR_M = 10.0
CONE_DEG = 1.0
HYSTERESIS_US = 500
LIMIT_US = 26_000
MATCH_US = 5_000
FALLBACK_DELAY_US = 50_000  # front direction delay of a clip without a measured value (see the module docstring)


def camera_azimuth_deg(camera_to_imu_rotation: npt.NDArray[np.float64]) -> float:
    """The azimuth (degrees, counterclockwise from the ego x axis) of a camera's optical axis (camera z axis)."""
    axis = np.asarray(camera_to_imu_rotation, dtype=np.float64) @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def direction_delay_us(
    xyz: npt.NDArray, point_times_us: npt.NDArray, spin_start_us: int, center_xy: Sequence[float], azimuth_deg: float
) -> Optional[float]:
    """The time after the spin start at which a spin measures an azimuth: the median time stamp of the points farther
    than 10 m (horizontal distance from the lidar) within 1 degree of the azimuth; None without such points."""
    dx = xyz[:, 0] - center_xy[0]
    dy = xyz[:, 1] - center_xy[1]
    far = np.hypot(dx, dy) > FAR_M
    if not far.any():
        return None
    azimuth = np.degrees(np.arctan2(dy[far], dx[far]))
    near = np.abs((azimuth - azimuth_deg + 180.0) % 360.0 - 180.0) < CONE_DEG
    if not near.any():
        return None
    return float(np.median(point_times_us[far][near].astype(np.int64) - int(spin_start_us)))


def front_delay_us(delays: Iterable[Optional[float]]) -> Optional[int]:
    """The median of the per-spin delays (spins without a value are left out); None without any value."""
    values = [d for d in delays if d is not None]
    return int(round(float(np.median(values)))) if values else None


def reference_times(spin_starts_us: npt.ArrayLike, delay_us: Optional[int]) -> npt.NDArray[np.int64]:
    """The reference time of each spin: its start plus the front direction delay (FALLBACK_DELAY_US for None)."""
    return np.asarray(spin_starts_us, dtype=np.int64) + int(FALLBACK_DELAY_US if delay_us is None else delay_us)


def _synchronized(frame_ts: npt.NDArray[np.int64], leader_ts: npt.NDArray[np.int64]) -> bool:
    """Whether a camera captures with its group leader: the median offset of its images from the nearest leader image
    lies within 5 ms."""
    if len(frame_ts) == 0 or len(leader_ts) == 0:
        return False
    nearest = np.clip(np.searchsorted(leader_ts, frame_ts), 1, len(leader_ts) - 1)
    offset = np.minimum(np.abs(frame_ts - leader_ts[nearest - 1]), np.abs(frame_ts - leader_ts[nearest]))
    return float(np.median(offset)) < MATCH_US


def select_camera_frames(
    camera_timestamps: Mapping[str, npt.ArrayLike],
    ref_ts: npt.ArrayLike,
    groups: Sequence[Sequence[str]] = CAMERA_GROUPS,
) -> Dict[str, npt.NDArray[np.int64]]:
    """The kept image of each camera per reference time (-1: none), with one capture time per camera group.

    :param camera_timestamps: Camera name -> sorted image times (us).
    :param ref_ts: The reference times (us), one per spin.
    :return: Camera name -> image index per reference time; a camera outside the groups keeps the image nearest to
        each reference time with the same hysteresis and limit.
    """
    ref_ts = np.asarray(ref_ts, dtype=np.int64)
    selected: Dict[str, npt.NDArray[np.int64]] = {}
    grouped = set()
    for group in groups:
        present = [name for name in group if name in camera_timestamps and len(camera_timestamps[name]) > 0]
        grouped.update(group)
        if not present:
            continue
        leader_ts = np.asarray(camera_timestamps[present[0]], dtype=np.int64)
        leader = select_nearest_frames(leader_ts, ref_ts, hysteresis_us=HYSTERESIS_US, limit_us=LIMIT_US)
        selected[present[0]] = leader
        kept = leader >= 0
        capture_ts = leader_ts[np.where(kept, leader, 0)]
        for name in present[1:]:
            member_ts = np.asarray(camera_timestamps[name], dtype=np.int64)
            if not _synchronized(member_ts, leader_ts):
                # Not triggered with its group in this clip (seen for the rear tele camera): its own selection.
                selected[name] = select_nearest_frames(
                    member_ts, ref_ts, hysteresis_us=HYSTERESIS_US, limit_us=LIMIT_US
                )
                continue
            member = np.full(len(ref_ts), -1, dtype=np.int64)
            member[kept] = match_frames(member_ts, capture_ts[kept], MATCH_US)
            selected[name] = member
    for name, timestamps in camera_timestamps.items():
        if name not in grouped:
            selected[name] = select_nearest_frames(
                np.asarray(timestamps, dtype=np.int64), ref_ts, hysteresis_us=HYSTERESIS_US, limit_us=LIMIT_US
            )
    return selected
