"""The kedec point-cloud codec for lidar scans (format version 1).

kedec stores positions with fixed error bounds and all other per-point features without loss:

* ``"spherical"`` mode (every lidar): range, azimuth and elevation around the sensor, rounded to 2 cm and 0.008
  degrees by default (position error at most 2 cm up to about 175 m, 5.04 cm at 500 m); see
  :mod:`~py123d.common.io.lidar.kedec.spherical`.
* ``"grid"`` mode (scans that store a beam index ``channel`` and a column index ``column``): range and sideways
  deviations from per-scan beam and column tables, rounded to 2 cm (position error at most 1.73 cm); see
  :mod:`~py123d.common.io.lidar.kedec.grid`.

The mode ``"auto"`` uses the grid mode only when both indices are stored. Integer features (time stamps, intensity,
ring, ...) are stored exactly, other features as their bytes; every part is compressed with Zstandard (level 19 by
default). A merged cloud with sensor ids (feature ``ids``) is coded per sensor, each around its own sensor pose.
Points with non-finite coordinates are not stored; the blob head counts them.

The decoded points come in the stored order: sensor by sensor (ascending id), and per sensor in beam order (ring or
elevation row, then azimuth) or in (beam, column) order. The order of the input is not kept.

Blob layout: ``b"KEDC"``, version (uint8), reserved (uint8), head length (uint32, little-endian), the msgpack head,
and the compressed parts in the order of the head. The head contains everything that decoding requires, including
the sensor poses (so no log metadata is required), and a CRC32 of the parts.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import msgpack
import numpy as np
import numpy.typing as npt

from py123d.common.io.lidar.kedec import grid, spherical, zstd_parts
from py123d.common.io.lidar.kedec.int_coding import (
    decode_bytes_channel,
    decode_int_channel,
    encode_bytes_channel,
    encode_int_channel,
    is_integer_dtype,
)
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig

KEDEC_MAGIC = b"KEDC"
KEDEC_VERSION = 1
_PREFIX = struct.Struct("<4sBBI")  # magic, version, reserved, head length

IDS_FEATURE = "ids"
BEAM_FEATURE = "channel"
COLUMN_FEATURE = "column"
# Parts that are not features carry this prefix, so that they never collide with a feature name.
_CODEC_PART = "@"

_IDENTITY_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

Pose = Sequence[float]  # x, y, z, qw, qx, qy, qz (the list form of PoseSE3)


def is_kedec_binary(blob: bytes) -> bool:
    """Check if the given binary data is a kedec point cloud.

    :param blob: The binary data to check.
    :return: True if the data starts with the kedec magic bytes.
    """
    return blob[:4] == KEDEC_MAGIC


# ------------------------------------------------------------------------------------------------------------------
# Poses
# ------------------------------------------------------------------------------------------------------------------


def _pose_list(pose: Optional[Pose]) -> List[float]:
    values = [float(v) for v in (pose if pose is not None else _IDENTITY_POSE)]
    if len(values) != 7 or not np.all(np.isfinite(values)):
        raise ValueError(f"kedec: a sensor pose must be 7 finite values (x, y, z, qw, qx, qy, qz), got {pose}.")
    return values


def _rotation(pose: Sequence[float]) -> npt.NDArray[np.float64]:
    """The rotation matrix of the normalized quaternion (qw, qx, qy, qz) of a pose list."""
    w, x, y, z = np.asarray(pose[3:7], dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("kedec: a sensor pose has a zero quaternion.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _to_coding_frame(points: npt.NDArray[np.float64], pose: Sequence[float], frame: str) -> npt.NDArray[np.float64]:
    local = points - np.asarray(pose[:3], dtype=np.float64)
    return local @ _rotation(pose) if frame == "pose" else local


def _from_coding_frame(local: npt.NDArray[np.float64], pose: Sequence[float], frame: str) -> npt.NDArray[np.float64]:
    if frame == "pose":
        local = local @ _rotation(pose).T
    return local + np.asarray(pose[:3], dtype=np.float64)


# ------------------------------------------------------------------------------------------------------------------
# Encoding
# ------------------------------------------------------------------------------------------------------------------


def _is_index(values: Optional[npt.NDArray]) -> bool:
    """Whether a feature can serve as a beam or column index of the grid mode."""
    return (
        values is not None
        and values.ndim == 1
        and is_integer_dtype(values.dtype)
        and not np.issubdtype(values.dtype, np.bool_)
        and (len(values) == 0 or int(values.min()) >= 0)
    )


def _section_mode(features: Mapping[str, npt.NDArray], mode: str) -> str:
    has_grid = _is_index(features.get(BEAM_FEATURE)) and _is_index(features.get(COLUMN_FEATURE))
    if mode == "grid" and not has_grid:
        raise ValueError(
            f"kedec: the grid mode requires non-negative integer features {BEAM_FEATURE!r} and {COLUMN_FEATURE!r}."
        )
    if mode == "auto":
        return "grid" if has_grid else "spherical"
    return mode


class _PartWriter:
    """Collects the compressed parts of one section and their descriptors."""

    def __init__(self, level: int, int_choice: str) -> None:
        self.level = level
        self.int_choice = int_choice
        self.descriptors: List[List[Any]] = []
        self.data: List[bytes] = []

    def add_int(self, name: str, values: npt.NDArray) -> None:
        descriptor, compressed = encode_int_channel(values, self.level, self.int_choice)
        self.descriptors.append([name, "i", len(compressed), descriptor])
        self.data.append(compressed)

    def add_bytes(self, name: str, values: npt.NDArray) -> None:
        descriptor, compressed = encode_bytes_channel(values, self.level)
        self.descriptors.append([name, "b", len(compressed), descriptor])
        self.data.append(compressed)

    def add_raw(self, name: str, data: bytes) -> None:
        compressed = zstd_parts.compress(data, self.level)
        self.descriptors.append([name, "z", len(compressed), {"u": len(data)}])
        self.data.append(compressed)

    def add_feature(self, name: str, values: npt.NDArray) -> None:
        if values.ndim == 1 and is_integer_dtype(values.dtype):
            self.add_int(name, values)
        else:
            self.add_bytes(name, values)


def _encode_section(
    points: npt.NDArray[np.float64],
    features: Dict[str, npt.NDArray],
    sensor_id: Optional[int],
    pose: List[float],
    config: PointCloudCodecConfig,
) -> Tuple[Dict[str, Any], List[bytes]]:
    mode = _section_mode(features, config.kedec_mode)
    writer = _PartWriter(config.kedec_zstd_level, config.kedec_int_choice)
    if mode == "grid":
        frame = "pose"
        order, tables, channels, (n_beams, n_columns) = grid.encode_positions(
            _to_coding_frame(points, pose, frame),
            features[BEAM_FEATURE],
            features[COLUMN_FEATURE],
            config.kedec_range_step_m,
            config.kedec_side_step_m,
        )
        params: Dict[str, Any] = {
            "frame": frame,
            "range_step": config.kedec_range_step_m,
            "side_step": config.kedec_side_step_m,
            "n_beams": n_beams,
            "n_columns": n_columns,
        }
        writer.add_raw(_CODEC_PART + "tables", tables)
    else:
        frame = config.kedec_frame
        ring = features.get(BEAM_FEATURE)
        ring = ring if ring is not None and ring.ndim == 1 and is_integer_dtype(ring.dtype) else None
        order, channels = spherical.encode_positions(
            _to_coding_frame(points, pose, frame),
            ring,
            config.kedec_range_step_m,
            config.kedec_angle_step_deg,
            config.kedec_row_step_deg,
        )
        params = {
            "frame": frame,
            "range_step": config.kedec_range_step_m,
            "angle_step": config.kedec_angle_step_deg,
            "order": BEAM_FEATURE if ring is not None else "elevation_rows",
        }
    for name, values in channels.items():
        writer.add_int(_CODEC_PART + name, values)
    for name, values in features.items():
        writer.add_feature(name, values[order])
    section = {
        "id": sensor_id,
        "n": int(len(points)),
        "pose": pose,
        "mode": mode,
        "params": params,
        "parts": writer.descriptors,
    }
    return section, writer.data


def encode_point_cloud_as_kedec_binary(
    point_cloud_3d: npt.NDArray,
    point_cloud_features: Optional[Mapping[str, npt.NDArray]] = None,
    config: Optional[PointCloudCodecConfig] = None,
    sensor_poses: Optional[Mapping[int, Pose]] = None,
) -> bytes:
    """Encode a point cloud (xyz + optional features) into one kedec blob.

    :param point_cloud_3d: (N, 3) positions, in the frame of the sensor poses (the vehicle frame in py123d).
    :param point_cloud_features: Optional per-point features; the first axis of each array runs over the points.
    :param config: Codec settings (the ``kedec_*`` fields); defaults to :class:`PointCloudCodecConfig`.
    :param sensor_poses: Sensor id -> sensor pose (x, y, z, qw, qx, qy, qz) in the frame of the points. A cloud
        with an integer feature ``ids`` is coded per sensor id, each around its pose (identity if missing); a cloud
        without ``ids`` uses the only pose of a single-entry mapping, else the identity.
    :return: The kedec blob.
    """
    config = config if config is not None else PointCloudCodecConfig()
    points = np.asarray(point_cloud_3d)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"kedec: the point cloud must have shape (N, 3), got {points.shape}.")
    features = {name: np.asarray(values) for name, values in (point_cloud_features or {}).items()}
    for name, values in features.items():
        if values.ndim == 0 or values.shape[0] != len(points):
            raise ValueError(f"kedec: feature {name!r} has shape {values.shape}, expected {len(points)} points.")
        if values.dtype.hasobject:
            raise ValueError(f"kedec: feature {name!r} has dtype {values.dtype}, which cannot be stored.")
        if name.startswith(_CODEC_PART):
            raise ValueError(f"kedec: feature names must not start with {_CODEC_PART!r}, got {name!r}.")
    feature_list = [[name, values.dtype.str, list(values.shape[1:])] for name, values in features.items()]

    finite = np.isfinite(points).all(axis=1)
    dropped = int(len(points) - finite.sum())
    points = points[finite].astype(np.float64)
    if dropped:
        features = {name: values[finite] for name, values in features.items()}

    ids = features.get(IDS_FEATURE)
    split = ids is not None and ids.ndim == 1 and is_integer_dtype(ids.dtype) and not np.issubdtype(ids.dtype, np.bool_)
    poses = {int(sensor_id): pose for sensor_id, pose in (sensor_poses or {}).items()}
    sections: List[Dict[str, Any]] = []
    data: List[bytes] = []
    if split:
        assert ids is not None
        others = {name: values for name, values in features.items() if name != IDS_FEATURE}
        for sensor_id in np.unique(ids):
            mask = ids == sensor_id
            section, parts = _encode_section(
                points[mask],
                {name: values[mask] for name, values in others.items()},
                int(sensor_id),
                _pose_list(poses.get(int(sensor_id))),
                config,
            )
            sections.append(section)
            data.extend(parts)
    elif len(points):
        pose = _pose_list(next(iter(poses.values())) if len(poses) == 1 else None)
        section, parts = _encode_section(points, features, None, pose, config)
        sections.append(section)
        data.extend(parts)

    body = b"".join(data)
    head = {
        "n": int(len(points)),
        "dropped": dropped,
        "features": feature_list,
        "split": IDS_FEATURE if split else None,
        "sections": sections,
        "crc32": zlib.crc32(body),
    }
    packed = msgpack.packb(head, use_bin_type=True)
    return _PREFIX.pack(KEDEC_MAGIC, KEDEC_VERSION, 0, len(packed)) + packed + body


# ------------------------------------------------------------------------------------------------------------------
# Decoding
# ------------------------------------------------------------------------------------------------------------------


def read_kedec_head(blob: bytes) -> Tuple[Dict[str, Any], int]:
    """The head of a kedec blob and the offset of its first part.

    :raises ValueError: If the blob is not kedec, has an unknown version, or is truncated.
    """
    if len(blob) < _PREFIX.size or not is_kedec_binary(blob):
        raise ValueError("kedec: the blob does not start with the kedec magic bytes.")
    _, version, _, head_size = _PREFIX.unpack_from(blob)
    if version != KEDEC_VERSION:
        raise ValueError(f"kedec: unsupported format version {version} (this py123d reads version {KEDEC_VERSION}).")
    start = _PREFIX.size + head_size
    if start > len(blob):
        raise ValueError("kedec: the blob is truncated inside its head.")
    try:
        head = msgpack.unpackb(blob[_PREFIX.size : start], raw=False, strict_map_key=False)
    except Exception as error:  # msgpack raises several exception types for corrupt input
        raise ValueError(f"kedec: the blob head cannot be read ({error}).") from error
    if not isinstance(head, dict) or "sections" not in head or "features" not in head:
        raise ValueError("kedec: the blob head lacks its sections or features.")
    total = sum(int(part[2]) for section in head["sections"] for part in section["parts"])
    if start + total != len(blob):
        raise ValueError(f"kedec: the blob contains {len(blob) - start} bytes of parts, its head lists {total}.")
    if zlib.crc32(memoryview(blob)[start:]) != head.get("crc32"):
        raise ValueError("kedec: the checksum of the parts does not match (corrupt data).")
    return head, start


def load_point_cloud_from_kedec_binary(
    blob: bytes,
) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    """Decode a kedec blob into a point cloud (xyz) and optional features.

    :param blob: The kedec blob.
    :return: Tuple of (point_cloud_3d as Nx3 float32 array, features dict or None), in the stored order.
    :raises ValueError: If the blob is corrupt, truncated or of an unknown version.
    """
    head, offset = read_kedec_head(blob)
    view = memoryview(blob)
    feature_specs = [
        (str(name), np.dtype(dtype), tuple(int(v) for v in tail)) for name, dtype, tail in head["features"]
    ]
    specs_by_name = {name: (dtype, tail) for name, dtype, tail in feature_specs}
    split = head.get("split")

    points_parts: List[npt.NDArray[np.float64]] = []
    feature_parts: Dict[str, List[npt.NDArray]] = {name: [] for name, _, _ in feature_specs}
    for section in head["sections"]:
        n = int(section["n"])
        decoded: Dict[str, npt.NDArray] = {}  # features
        channels: Dict[str, npt.NDArray] = {}  # position channels, without the prefix
        raw_parts: Dict[str, bytes] = {}
        for name, kind, size, descriptor in section["parts"]:
            compressed = bytes(view[offset : offset + int(size)])
            offset += int(size)
            codec_part = name.startswith(_CODEC_PART)
            spec = specs_by_name.get(name)
            if not codec_part and spec is None:
                raise ValueError(f"kedec: the part {name!r} is not a listed feature.")
            if kind == "z" and codec_part:
                raw_parts[name[1:]] = zstd_parts.decompress(compressed, int(descriptor["u"]))
            elif kind == "i" and codec_part:
                channels[name[1:]] = decode_int_channel(descriptor, compressed, n, np.dtype(np.int64))
            elif kind == "i" and spec is not None:
                decoded[name] = decode_int_channel(descriptor, compressed, n, spec[0])
            elif kind == "b" and spec is not None:
                decoded[name] = decode_bytes_channel(descriptor, compressed, n, spec[0], spec[1])
            else:
                raise ValueError(f"kedec: unknown part {name!r} of kind {kind!r}.")

        params = section["params"]
        if section["mode"] == "grid":
            local = grid.decode_positions(
                raw_parts["tables"],
                channels,
                decoded[BEAM_FEATURE],
                decoded[COLUMN_FEATURE],
                int(params["n_beams"]),
                int(params["n_columns"]),
                float(params["range_step"]),
                float(params["side_step"]),
            )
        elif section["mode"] == "spherical":
            local = spherical.decode_positions(channels, float(params["range_step"]), float(params["angle_step"]))
        else:
            raise ValueError(f"kedec: unknown section mode {section['mode']!r}.")
        points_parts.append(_from_coding_frame(local, section["pose"], params["frame"]))

        for name, dtype, tail in feature_specs:
            if name == split:
                feature_parts[name].append(np.full(n, section["id"], dtype=dtype))
            elif name in decoded:
                feature_parts[name].append(decoded[name])
            else:
                raise ValueError(f"kedec: a section lacks the feature {name!r}.")

    if points_parts:
        point_cloud_3d = np.concatenate(points_parts).astype(np.float32)
    else:
        point_cloud_3d = np.zeros((0, 3), dtype=np.float32)
    if len(point_cloud_3d) != int(head["n"]):
        raise ValueError(f"kedec: decoded {len(point_cloud_3d)} points, the head lists {head['n']}.")
    if not feature_specs:
        return point_cloud_3d, None
    features: Dict[str, npt.NDArray] = {}
    for name, dtype, tail in feature_specs:
        parts = feature_parts[name]
        features[name] = np.concatenate(parts) if parts else np.zeros((0,) + tail, dtype=dtype)
    return point_cloud_3d, features
