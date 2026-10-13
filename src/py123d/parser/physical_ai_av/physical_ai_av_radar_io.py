"""Point cloud loader for the NVIDIA Physical AI AV radar.

PAIAV ships up to nine short-range radars, each stored per-clip at
``radar/<radar_name>/<clip_id>.<radar_name>.parquet``. Each parquet holds detections grouped by a
``scan_index`` (one radar scan per group); detections are in spherical sensor coordinates
(``azimuth``, ``elevation``, ``distance``) with ``radial_velocity``, ``rcs``, ``snr`` and
``exist_probb``. The per-scan ``timestamp`` column is clip-relative microseconds on the same clock as
the lidar ``reference_timestamp``.

We follow the merged model: for a reference (lidar-spin) timestamp, each radar's nearest scan is read,
converted spherical→Cartesian in the sensor frame, transformed into the ego/rig frame, mapped to
:class:`RadarFeature`, tagged with its :class:`RadarID`, and concatenated. The per-radar relative paths
are carried in ``load_kwargs["radar_paths"]`` (since the merged frame stores a single primary path).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from py123d.datatypes.sensors.radar import RADAR_FEATURE_DTYPES, RadarFeature, RadarID, RadarMetadata
from py123d.geometry.transform import rel_to_abs_points_3d_array
from py123d.parser.physical_ai_av.utils.physical_ai_av_zip import path_exists, read_parquet


@lru_cache(maxsize=16)
def _read_radar_parquet(parquet_path: str) -> pd.DataFrame:
    """Read and cache one radar parquet (one per clip+radar)."""
    return read_parquet(parquet_path)


def load_physical_ai_av_radar_point_cloud_data_from_path(
    primary_path: Union[Path, str],
    reference_timestamp_us: int,
    radar_metadatas: Dict[RadarID, RadarMetadata],
    sensor_root: Path,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray]]:
    """Loads and merges the PAIAV radar scans nearest to a reference timestamp into the ego frame.

    :param primary_path: Absolute path to the primary radar parquet (existence-checked upstream).
    :param reference_timestamp_us: Lidar-spin reference timestamp (clip-relative µs); the nearest radar
        scan per sensor is selected against the radar ``timestamp`` column.
    :param radar_metadatas: Per-:class:`RadarID` metadata (provides the sensor→ego extrinsics).
    :param sensor_root: Dataset sensor root, against which the relative radar paths are resolved.
    :param load_kwargs: Must carry ``"radar_paths"``: a ``{str(int(RadarID)): relative_path}`` mapping.
    :return: Tuple of (Nx3 float32 xyz in ego frame, features dict keyed by ``RadarFeature.serialize()``).
    """
    kw = load_kwargs or {}
    radar_paths: Dict[str, str] = kw.get("radar_paths", {})
    assert radar_paths, "Physical AI AV radar loading requires 'radar_paths' in load_kwargs."

    xyz_list: List[np.ndarray] = []
    features_list: Dict[str, List[np.ndarray]] = {}

    for radar_id_str, relative_path in radar_paths.items():
        radar_id = RadarID(int(radar_id_str))
        metadata = radar_metadatas.get(radar_id)
        if metadata is None:
            continue
        full_path = Path(sensor_root) / relative_path
        if not path_exists(full_path):
            continue

        df = _read_radar_parquet(str(full_path))
        scan_df = _select_nearest_scan(df, int(reference_timestamp_us))
        if scan_df is None or len(scan_df) == 0:
            continue

        azimuth = scan_df["azimuth"].to_numpy(dtype=np.float32)
        elevation = scan_df["elevation"].to_numpy(dtype=np.float32)
        distance = scan_df["distance"].to_numpy(dtype=np.float32)

        # Spherical (sensor frame) -> Cartesian: x forward, y left, z up.
        cos_el = np.cos(elevation)
        xyz_sensor = np.stack(
            [distance * cos_el * np.cos(azimuth), distance * cos_el * np.sin(azimuth), distance * np.sin(elevation)],
            axis=-1,
        ).astype(np.float64)

        extrinsic = metadata.radar_to_imu_se3
        xyz_ego = rel_to_abs_points_3d_array(origin=extrinsic, points_3d_array=xyz_sensor).astype(np.float32)
        xyz_list.append(xyz_ego)

        num_points = xyz_ego.shape[0]
        _append(features_list, RadarFeature.IDS, np.full(num_points, int(radar_id)))
        if "rcs" in scan_df:
            _append(features_list, RadarFeature.RCS, scan_df["rcs"].to_numpy())
        if "snr" in scan_df:
            _append(features_list, RadarFeature.SNR, scan_df["snr"].to_numpy())
        if "exist_probb" in scan_df:
            _append(features_list, RadarFeature.EXIST_PROBABILITY, scan_df["exist_probb"].to_numpy())
        if "sensor_timestamp" in scan_df:
            _append(features_list, RadarFeature.TIMESTAMPS, scan_df["sensor_timestamp"].to_numpy())

        # Decompose scalar radial velocity into an ego-frame (vx, vy) vector along the (ego-frame) ray.
        if "radial_velocity" in scan_df:
            radial = scan_df["radial_velocity"].to_numpy(dtype=np.float32)
            direction_ego = (extrinsic.rotation_matrix @ xyz_sensor.T).T.astype(np.float32)
            norm = np.linalg.norm(direction_ego[:, :2], axis=1, keepdims=True)
            unit_xy = np.divide(direction_ego[:, :2], norm, out=np.zeros_like(direction_ego[:, :2]), where=norm > 0)
            velocity_xy = unit_xy * radial[:, None]
            _append(features_list, RadarFeature.VELOCITY_X, velocity_xy[:, 0])
            _append(features_list, RadarFeature.VELOCITY_Y, velocity_xy[:, 1])

    if not xyz_list:
        return np.zeros((0, 3), dtype=np.float32), {}

    point_cloud_3d = np.concatenate(xyz_list, axis=0)
    point_cloud_features = {key: np.concatenate(values, axis=0) for key, values in features_list.items()}
    return point_cloud_3d, point_cloud_features


def _select_nearest_scan(df: pd.DataFrame, reference_timestamp_us: int) -> Optional[pd.DataFrame]:
    """Returns the detections of the scan whose ``timestamp`` is nearest to the reference, or None."""
    if "timestamp" not in df or len(df) == 0:
        return None
    scan_timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    unique_ts = np.unique(scan_timestamps)
    nearest_ts = unique_ts[int(np.argmin(np.abs(unique_ts - reference_timestamp_us)))]
    return df[scan_timestamps == nearest_ts]


def _append(features_list: Dict[str, List[np.ndarray]], feature: RadarFeature, values: np.ndarray) -> None:
    key = feature.serialize()
    features_list.setdefault(key, []).append(np.asarray(values).astype(RADAR_FEATURE_DTYPES[feature]))
