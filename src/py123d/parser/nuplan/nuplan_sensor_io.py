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


def _load_pcd_binary(data: bytes, fields: list, sizes: list, types: list, counts: list, num_points: int) -> np.ndarray:
    """Load binary PCD data into a numpy structured array with native dtypes."""
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

    return np.frombuffer(data, dtype=np.dtype(dt_fields), count=num_points)


def _load_pcd_from_bytes(raw: bytes) -> np.ndarray:
    """Parse a nuPlan binary PCD file into a numpy structured array with native dtypes.

    Expected fields: x, y, z, intensity, ring, lidar_info.
    """
    header_end = raw.find(b"\nDATA ")
    assert header_end != -1, "Invalid PCD file: missing DATA line"

    data_line_end = raw.index(b"\n", header_end + 1)
    header_bytes = raw[: data_line_end + 1]
    body = raw[data_line_end + 1 :]

    fields, sizes, types, counts, num_points, data_format = _parse_pcd_header(header_bytes)
    assert data_format == "binary", f"nuPlan PCD files use binary format, got: {data_format}"

    return _load_pcd_binary(body, fields, sizes, types, counts, num_points)


def load_nuplan_point_cloud_data_from_path(pcd_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads nuPlan Lidar point clouds from a ``.pcd`` file."""

    assert pcd_path.exists(), f"Lidar file not found: {pcd_path}"
    raw = pcd_path.read_bytes()

    pcd = _load_pcd_from_bytes(raw)

    lidar_ids = np.zeros(len(pcd), dtype=np.uint8)
    lidar_info = pcd["lidar_info"]
    for nuplan_lidar_id, lidar_id in NUPLAN_LIDAR_DICT.items():
        lidar_ids[lidar_info == nuplan_lidar_id] = int(lidar_id)

    point_cloud_3d = np.column_stack((pcd["x"], pcd["y"], pcd["z"])).astype(np.float32)
    point_cloud_features = {
        LidarFeature.INTENSITY.serialize(): pcd["intensity"].astype(np.uint8),
        LidarFeature.CHANNEL.serialize(): pcd["ring"].astype(np.uint8),
        LidarFeature.IDS.serialize(): lidar_ids,
    }

    return point_cloud_3d, point_cloud_features
