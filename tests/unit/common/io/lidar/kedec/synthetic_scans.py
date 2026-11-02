"""Synthetic lidar scans for the kedec tests: a spinning lidar with beams and columns, in the vehicle frame."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from py123d.geometry.pose import PoseSE3


def random_pose(rng: np.random.Generator, max_translation: float = 2.0) -> List[float]:
    """A random sensor pose (x, y, z, qw, qx, qy, qz) with a normalized quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    t = rng.uniform(-max_translation, max_translation, size=3)
    return [float(v) for v in np.concatenate([t, q])]


def to_vehicle_frame(points_sensor: npt.NDArray[np.float64], pose: List[float]) -> npt.NDArray[np.float64]:
    """Points in the vehicle frame from points in the sensor frame, with py123d's own pose convention."""
    pose_se3 = PoseSE3.from_list(pose)
    return points_sensor @ pose_se3.rotation_matrix.T + pose_se3.point_3d.array


def spinning_scan(
    rng: np.random.Generator,
    pose: List[float],
    n_beams: int = 32,
    n_columns: int = 360,
    max_range: float = 200.0,
    t0_us: int = 1_784_208_918_099_704,
    keep: float = 0.9,
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray]]:
    """A scan of a spinning lidar: beams between -25 and +15 degrees elevation, columns over 360 degrees azimuth,
    small angle noise, random ranges, a fraction of the returns missing. Returns float32 points in the vehicle frame
    and the features intensity, timestamps (absolute us), channel (beam), column and uid (the input index)."""
    beam_elevation = np.linspace(-25.0, 15.0, n_beams)
    beam_offset = rng.uniform(-0.5, 0.5, n_beams)
    column_azimuth = np.arange(n_columns) * (360.0 / n_columns) - 180.0
    beam, column = np.meshgrid(np.arange(n_beams), np.arange(n_columns), indexing="ij")
    beam, column = beam.ravel(), column.ravel()
    keep_mask = rng.random(len(beam)) < keep
    beam, column = beam[keep_mask], column[keep_mask]
    n = len(beam)
    elevation = np.radians(beam_elevation[beam] + rng.normal(0.0, 0.01, n))
    azimuth = np.radians(column_azimuth[column] + beam_offset[beam] + rng.normal(0.0, 0.01, n))
    r = rng.uniform(0.5, max_range, n)
    sensor = np.stack(
        [r * np.cos(elevation) * np.cos(azimuth), r * np.cos(elevation) * np.sin(azimuth), r * np.sin(elevation)], 1
    )
    points = to_vehicle_frame(sensor, pose).astype(np.float32)
    shuffle = rng.permutation(n)  # the stored order of a scan need not be beam order
    features = {
        "intensity": rng.integers(0, 256, n).astype(np.uint8),
        "timestamps": (t0_us + column * 277 + beam).astype(np.int64),
        "channel": beam.astype(np.uint8),
        "column": column.astype(np.uint16),
    }
    features = {name: values[shuffle] for name, values in features.items()}
    features["uid"] = np.arange(n, dtype=np.int64)
    return points[shuffle], features


def spherical_bound(
    r: npt.NDArray[np.float64], range_step: float = 0.02, angle_step: float = 0.008
) -> npt.NDArray[np.float64]:
    """Largest position error of the spherical mode at range r (with the second-order term and float32 output)."""
    angle = np.radians(angle_step / 2.0)
    return np.sqrt((range_step / 2.0) ** 2 + 2.0 * ((r + range_step / 2.0) * angle) ** 2) * (1 + 1e-6) + 1e-4


def grid_bound(
    r: npt.NDArray[np.float64], range_step: float = 0.02, side_step: float = 0.02
) -> npt.NDArray[np.float64]:
    """Largest position error of the grid mode at range r >= 0.5 m (sideways parts scaled by the range error)."""
    side = side_step / 2.0 * (1.0 + range_step / np.maximum(r, range_step)) * 1.001
    return np.sqrt((range_step / 2.0) ** 2 + 2.0 * side**2) + 1e-4
