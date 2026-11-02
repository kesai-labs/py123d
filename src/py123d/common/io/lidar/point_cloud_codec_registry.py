"""Registry of the binary point-cloud codecs of the Lidar and Radar writers and readers.

Each codec has a name (the value of ``LogWriterConfig.lidar_codec`` or ``radar_codec``), a test of its magic bytes,
an encoder and a decoder. Readers detect the codec of a blob from its first bytes, so a log can mix codecs and old
logs stay readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from py123d.common.io.lidar.draco_lidar_io import (
    encode_point_cloud_as_draco_binary,
    is_draco_binary,
    load_point_cloud_from_draco_binary,
)
from py123d.common.io.lidar.ipc_lidar_io import (
    encode_point_cloud_as_ipc_binary,
    is_ipc_binary,
    load_point_cloud_from_ipc_binary,
)
from py123d.common.io.lidar.kedec_lidar_io import (
    KEDEC_VERSION,
    encode_point_cloud_as_kedec_binary,
    is_kedec_binary,
    load_point_cloud_from_kedec_binary,
)
from py123d.common.io.lidar.laz_lidar_io import (
    encode_point_cloud_as_laz_binary,
    is_laz_binary,
    load_point_cloud_from_laz_binary,
)
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig

PointCloud = Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]
SensorPoses = Mapping[int, Sequence[float]]

# Schema-metadata key of an Arrow file that names the codec of its blobs (written for kedec logs).
CODEC_METADATA_KEY = b"codec"


@dataclass(frozen=True)
class PointCloudCodec:
    """One binary point-cloud codec.

    :param name: The name used in the log writer configuration.
    :param is_binary: Tests whether a blob was written by this codec (from its magic bytes).
    :param encode: ``encode(xyz, features, config, sensor_poses) -> bytes``.
    :param decode: ``decode(blob) -> (xyz, features)``.
    :param modalities: The modality types that may use the codec (``"lidar"``, ``"radar"``).
    :param version: The format version recorded in the schema metadata, or None.
    """

    name: str
    is_binary: Callable[[bytes], bool]
    encode: Callable[..., bytes]
    decode: Callable[[bytes], PointCloud]
    modalities: FrozenSet[str] = frozenset({"lidar", "radar"})
    version: Optional[int] = None


def _ipc_encoder(codec: Optional[str]) -> Callable[..., bytes]:
    def encode(xyz, features=None, config=None, sensor_poses=None) -> bytes:
        return encode_point_cloud_as_ipc_binary(xyz, features, codec=codec)  # type: ignore[arg-type]

    return encode


def _draco_encode(xyz, features=None, config=None, sensor_poses=None) -> bytes:
    return encode_point_cloud_as_draco_binary(xyz, features, config=config)


def _laz_encode(xyz, features=None, config=None, sensor_poses=None) -> bytes:
    return encode_point_cloud_as_laz_binary(xyz, features, config=config)


_REGISTRY: Dict[str, PointCloudCodec] = {}


def register_point_cloud_codec(codec: PointCloudCodec) -> None:
    """Add a codec to the registry (the name must be new)."""
    if codec.name in _REGISTRY:
        raise ValueError(f"A point-cloud codec named {codec.name!r} is already registered.")
    _REGISTRY[codec.name] = codec


for _codec in (
    PointCloudCodec("laz", is_laz_binary, _laz_encode, load_point_cloud_from_laz_binary),
    PointCloudCodec("draco", is_draco_binary, _draco_encode, load_point_cloud_from_draco_binary),
    PointCloudCodec("ipc_zstd", is_ipc_binary, _ipc_encoder("zstd"), load_point_cloud_from_ipc_binary),
    PointCloudCodec("ipc_lz4", is_ipc_binary, _ipc_encoder("lz4"), load_point_cloud_from_ipc_binary),
    PointCloudCodec("ipc", is_ipc_binary, _ipc_encoder(None), load_point_cloud_from_ipc_binary),
    PointCloudCodec(
        "kedec",
        is_kedec_binary,
        encode_point_cloud_as_kedec_binary,
        load_point_cloud_from_kedec_binary,
        modalities=frozenset({"lidar"}),
        version=KEDEC_VERSION,
    ),
):
    register_point_cloud_codec(_codec)


def point_cloud_codec_names(modality: str) -> FrozenSet[str]:
    """The names of the codecs that a modality type (``"lidar"`` or ``"radar"``) may use."""
    return frozenset(name for name, codec in _REGISTRY.items() if modality in codec.modalities)


def get_point_cloud_codec(name: str) -> PointCloudCodec:
    """The registered codec of the given name.

    :raises ValueError: If no codec of that name is registered.
    """
    codec = _REGISTRY.get(name)
    if codec is None:
        raise ValueError(f"Unknown point-cloud codec {name!r}; registered: {sorted(_REGISTRY)}.")
    return codec


def encode_point_cloud(
    name: str,
    point_cloud_3d: npt.NDArray,
    point_cloud_features: Optional[Mapping[str, npt.NDArray]] = None,
    config: Optional[PointCloudCodecConfig] = None,
    sensor_poses: Optional[SensorPoses] = None,
) -> bytes:
    """Encode a point cloud with the named codec.

    :param name: The codec name.
    :param point_cloud_3d: (N, 3) positions.
    :param point_cloud_features: Optional per-point features.
    :param config: Codec settings; defaults to :class:`PointCloudCodecConfig`.
    :param sensor_poses: Sensor id -> pose (x, y, z, qw, qx, qy, qz) of the sensors in the frame of the points;
        used by codecs that code positions around the sensor (kedec), ignored by the others.
    :return: The blob.
    """
    return get_point_cloud_codec(name).encode(point_cloud_3d, point_cloud_features, config, sensor_poses)


def detect_point_cloud_codec(blob: bytes) -> PointCloudCodec:
    """The codec of a blob, from its magic bytes (for the IPC variants: the first registered one).

    :raises ValueError: If no registered codec recognizes the blob.
    """
    for codec in _REGISTRY.values():
        if codec.is_binary(blob):
            return codec
    raise ValueError("Unknown point-cloud binary format (not one of: " + ", ".join(sorted(_REGISTRY)) + ").")


def decode_point_cloud(blob: bytes) -> PointCloud:
    """Decode a blob of any registered codec into (xyz, features)."""
    return detect_point_cloud_codec(blob).decode(blob)


def point_cloud_codec_metadata(name: str, config: Optional[PointCloudCodecConfig]) -> Dict[str, Any]:
    """The codec description written into the schema metadata of an Arrow file (key ``codec``).

    :param name: The codec name.
    :param config: The codec settings used.
    :return: A JSON-serializable mapping with the codec name, format version, writer and the kedec settings.
    """
    from py123d import __version__

    codec = get_point_cloud_codec(name)
    config = config if config is not None else PointCloudCodecConfig()
    description: Dict[str, Any] = {"codec": name, "version": codec.version, "encoder": f"py123d {__version__}"}
    if name == "kedec":
        description["options"] = {
            key: getattr(config, key) for key in config.__dataclass_fields__ if key.startswith("kedec_")
        }
    return description


def encode_codec_metadata(description: Mapping[str, Any]) -> bytes:
    """The schema-metadata value of a codec description (UTF-8 JSON)."""
    return json.dumps(dict(description), sort_keys=True).encode("utf-8")


def decode_codec_metadata(value: Optional[bytes]) -> Optional[Dict[str, Any]]:
    """The codec description of a schema-metadata value, or None if there is none."""
    return None if value is None else json.loads(value.decode("utf-8"))
