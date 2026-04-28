"""Point cloud loader for the NVIDIA NCore V4 dataset.

NCore stores per-clip lidar frames in a ``pai_{clip_id}.ncore4-lidar_top_360fov.zarr.itar``
archive. Each frame is keyed by its end-of-frame timestamp (microseconds) and contains a
ray bundle (direction + per-ray timestamp) plus per-return data (distance_m, intensity,
valid mask). We decode one frame (the first return only), mask out invalid rays, convert
to 3D points in sensor frame, and transform into the ego/rig frame.

The loader is invoked lazily from :func:`py123d.common.io.lidar.path_lidar_io.load_point_cloud_data_from_path`
with the end-of-frame timestamp passed through as the ``index`` argument.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.datatypes.sensors.lidar import LidarFeature, LidarID, LidarMetadata
from py123d.geometry.transform import rel_to_abs_points_3d_array


def _import_ncore_v4():
    """Lazy import of nvidia-ncore with a clear install message."""
    try:
        from ncore.data.v4 import LidarSensorComponent, SequenceComponentGroupsReader
    except ImportError as exc:
        raise ImportError(
            "The nvidia-ncore package is required to load NCore data. Install it via `pip install py123d[ncore]`."
        ) from exc
    return SequenceComponentGroupsReader, LidarSensorComponent


@lru_cache(maxsize=2)
def _open_lidar_reader(zarr_itar_path: str):
    """Open one lidar store and return (sequence_reader, lidar_reader) — cached per-path.

    The ``SequenceComponentGroupsReader`` holds open file handles / decoded tar indexes,
    so we cache it to avoid paying the open cost on every 10 Hz spin. Cache size of 2
    is enough for the dispatcher (one current clip; one prefetched).
    """
    SequenceComponentGroupsReader, LidarSensorComponent = _import_ncore_v4()
    seq_reader = SequenceComponentGroupsReader([Path(zarr_itar_path)])
    readers = seq_reader.open_component_readers(LidarSensorComponent.Reader)
    if not readers:
        raise RuntimeError(f"No lidar component in NCore archive: {zarr_itar_path}")
    # Single-lidar dataset — take the one reader regardless of instance name.
    lidar_reader = next(iter(readers.values()))
    return seq_reader, lidar_reader


def load_ncore_point_cloud_data_from_path(
    zarr_itar_path: Union[Path, str],
    end_of_frame_ts_us: int,
    lidar_metadatas: Dict[LidarID, LidarMetadata],
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray]]:
    """Load a single NCore V4 lidar frame and return points in ego frame + features.

    :param zarr_itar_path: Absolute path to the per-clip lidar ``.zarr.itar`` archive.
    :param end_of_frame_ts_us: End-of-frame timestamp (microseconds). NCore keys frames
        by this timestamp; the NCoreLogParser pulls it from ``frames_timestamps_us[:, 1]``.
    :param lidar_metadatas: Mapping of LidarID → LidarMetadata (carries the sensor-to-rig extrinsic).
    :return: ``(xyz_ego[N,3] float32, {"intensity": uint8[N], "timestamp_us": int64[N]})``.
    """
    _, lidar_reader = _open_lidar_reader(str(zarr_itar_path))

    ts = int(end_of_frame_ts_us)
    # Per-ray data (sensor frame)
    direction = np.asarray(
        lidar_reader.get_frame_ray_bundle_data(ts, "direction"),
        dtype=np.float32,
    )  # [N, 3]
    per_ray_ts = np.asarray(
        lidar_reader.get_frame_ray_bundle_data(ts, "timestamp_us"),
        dtype=np.int64,
    )  # [N]
    # First-return data. Slice with return_index=0 to avoid loading all returns.
    distance = np.asarray(
        lidar_reader.get_frame_ray_bundle_return_data(ts, "distance_m", return_index=0),
        dtype=np.float32,
    )  # [N]
    intensity = np.asarray(
        lidar_reader.get_frame_ray_bundle_return_data(ts, "intensity", return_index=0),
        dtype=np.float32,
    )  # [N]
    valid_mask = lidar_reader.get_frame_ray_bundle_return_valid_mask(ts)[0]  # [N] bool

    direction = direction[valid_mask]
    distance = distance[valid_mask]
    intensity = intensity[valid_mask]
    per_ray_ts = per_ray_ts[valid_mask]

    xyz_sensor = direction * distance[:, None]
    xyz_ego = rel_to_abs_points_3d_array(
        origin=lidar_metadatas[LidarID.LIDAR_TOP].lidar_to_imu_se3,
        points_3d_array=xyz_sensor.astype(np.float64),
    ).astype(np.float32)

    features: Dict[str, npt.NDArray] = {
        LidarFeature.INTENSITY.serialize(): np.clip(intensity * 255.0, 0.0, 255.0).astype(np.uint8),
        LidarFeature.TIMESTAMPS.serialize(): per_ray_ts,
    }
    return xyz_ego, features
