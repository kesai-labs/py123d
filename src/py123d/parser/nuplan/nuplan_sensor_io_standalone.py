from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from py123d.datatypes.sensors import LidarFeature
from py123d.parser.nuplan.utils.nuplan_constants import NUPLAN_LIDAR_DICT

# PCD type mapping: (size, type_char) -> numpy dtype
_PCD_TYPE_MAP = {
    (1, "I"): np.int8,
    (1, "U"): np.uint8,
    (2, "I"): np.int16,
    (2, "U"): np.uint16,
    (4, "I"): np.int32,
    (4, "U"): np.uint32,
    (4, "F"): np.float32,
    (8, "F"): np.float64,
    (8, "I"): np.int64,
    (8, "U"): np.uint64,
}

# Expected field order in the output array (matches nuplan's LidarPointCloud convention)
_NUPLAN_FIELD_ORDER = ["x", "y", "z", "intensity", "ring", "lidar_info"]


def _parse_pcd_header(header_bytes: bytes) -> Tuple[list, list, list, list, int, str]:
    """Parse a PCD file header, returning fields, sizes, types, counts, num_points, and data format."""
    fields = []
    sizes = []
    types = []
    counts = []
    num_points = 0
    data_format = "binary"

    for line in header_bytes.decode("ascii", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split()
        keyword = parts[0]
        if keyword == "FIELDS":
            fields = parts[1:]
        elif keyword == "SIZE":
            sizes = [int(s) for s in parts[1:]]
        elif keyword == "TYPE":
            types = parts[1:]
        elif keyword == "COUNT":
            counts = [int(c) for c in parts[1:]]
        elif keyword == "POINTS":
            num_points = int(parts[1])
        elif keyword == "DATA":
            data_format = parts[1].lower()

    if not counts:
        counts = [1] * len(fields)

    return fields, sizes, types, counts, num_points, data_format


def _load_pcd_binary(
    data: bytes, fields: list, sizes: list, types: list, counts: list, num_points: int
) -> Dict[str, np.ndarray]:
    """Load binary PCD data into a dict of {field_name: (num_points,) float64 array}."""
    point_size = sum(s * c for s, c in zip(sizes, counts))
    expected_size = point_size * num_points
    assert len(data) >= expected_size, f"PCD binary data too short: {len(data)} < {expected_size}"

    dt_fields = []
    for field, size, type_char, count in zip(fields, sizes, types, counts):
        np_dtype = _PCD_TYPE_MAP[(size, type_char)]
        if count == 1:
            dt_fields.append((field, np_dtype))
        else:
            dt_fields.append((field, np_dtype, (count,)))

    structured = np.frombuffer(data, dtype=np.dtype(dt_fields), count=num_points)

    field_arrays = {}
    for field in fields:
        arr = structured[field].astype(np.float64)
        if arr.ndim == 1:
            field_arrays[field] = arr
        else:
            for col in range(arr.shape[1]):
                field_arrays[f"{field}_{col}"] = arr[:, col]

    return field_arrays


def _load_pcd_from_bytes(raw: bytes) -> np.ndarray:
    """Parse a nuPlan binary PCD file and return a (6, num_points) float64 array.

    Output row order: x, y, z, intensity, ring, lidar_info (matching nuplan's LidarPointCloud convention).
    """
    header_end = raw.find(b"\nDATA ")
    assert header_end != -1, "Invalid PCD file: missing DATA line"

    data_line_end = raw.index(b"\n", header_end + 1)
    header_bytes = raw[: data_line_end + 1]
    body = raw[data_line_end + 1 :]

    fields, sizes, types, counts, num_points, data_format = _parse_pcd_header(header_bytes)
    assert data_format == "binary", f"nuPlan PCD files use binary format, got: {data_format}"

    field_arrays = _load_pcd_binary(body, fields, sizes, types, counts, num_points)

    # Reorder fields to match nuplan's LidarPointCloud convention
    return np.stack([field_arrays[name] for name in _NUPLAN_FIELD_ORDER], axis=0)


def load_nuplan_point_cloud_data_from_path(pcd_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads nuPlan Lidar point clouds from a ``.pcd`` file."""

    assert pcd_path.exists(), f"Lidar file not found: {pcd_path}"
    raw = pcd_path.read_bytes()

    # Shape: (6, N) with rows: x, y, z, intensity, ring, lidar_info
    merged_lidar_pc = _load_pcd_from_bytes(raw)
    lidar_ids = np.zeros(merged_lidar_pc.shape[1], dtype=np.uint8)

    for nuplan_lidar_id, lidar_id in NUPLAN_LIDAR_DICT.items():
        mask = merged_lidar_pc[-1, :] == nuplan_lidar_id
        lidar_ids[mask] = int(lidar_id)

    point_cloud_3d = merged_lidar_pc[:3, :].T.astype(np.float32)
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): merged_lidar_pc[3, :].astype(np.uint8),
        LidarFeature.CHANNEL.serialize(): merged_lidar_pc[4, :].astype(np.uint8),
        LidarFeature.IDS.serialize(): lidar_ids,
    }

    return point_cloud_3d, point_cloud_features
