"""Tests of the point-cloud codec registry: the registered encoders give the same bytes as the codec modules."""

from __future__ import annotations

import numpy as np
import pytest

from py123d.common.io.lidar.draco_lidar_io import encode_point_cloud_as_draco_binary
from py123d.common.io.lidar.ipc_lidar_io import encode_point_cloud_as_ipc_binary
from py123d.common.io.lidar.kedec_lidar_io import encode_point_cloud_as_kedec_binary
from py123d.common.io.lidar.laz_lidar_io import encode_point_cloud_as_laz_binary
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig
from py123d.common.io.lidar.point_cloud_codec_registry import (
    PointCloudCodec,
    decode_codec_metadata,
    decode_point_cloud,
    detect_point_cloud_codec,
    encode_codec_metadata,
    encode_point_cloud,
    get_point_cloud_codec,
    point_cloud_codec_metadata,
    point_cloud_codec_names,
    register_point_cloud_codec,
)


def _cloud():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-30, 30, (500, 3)).astype(np.float32)
    features = {"intensity": rng.integers(0, 256, 500).astype(np.uint8)}
    return xyz, features


def test_names_per_modality():
    assert point_cloud_codec_names("lidar") == {"laz", "draco", "ipc_zstd", "ipc_lz4", "ipc", "kedec"}
    assert point_cloud_codec_names("radar") == {"laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"}


@pytest.mark.parametrize(
    "name, direct",
    [
        ("laz", lambda xyz, f, c: encode_point_cloud_as_laz_binary(xyz, f, config=c)),
        ("draco", lambda xyz, f, c: encode_point_cloud_as_draco_binary(xyz, f, config=c)),
        ("ipc", lambda xyz, f, c: encode_point_cloud_as_ipc_binary(xyz, f, codec=None)),
        ("ipc_zstd", lambda xyz, f, c: encode_point_cloud_as_ipc_binary(xyz, f, codec="zstd")),
        ("ipc_lz4", lambda xyz, f, c: encode_point_cloud_as_ipc_binary(xyz, f, codec="lz4")),
        ("kedec", lambda xyz, f, c: encode_point_cloud_as_kedec_binary(xyz, f, config=c)),
    ],
)
def test_registered_encoder_gives_the_same_bytes(name, direct):
    """The registry changes no byte of the encoders it wraps."""
    xyz, features = _cloud()
    config = PointCloudCodecConfig()
    assert encode_point_cloud(name, xyz, features, config=config) == direct(xyz, features, config)


@pytest.mark.parametrize("name", ["laz", "draco", "ipc", "ipc_zstd", "ipc_lz4", "kedec"])
def test_detection_and_decoding(name):
    xyz, features = _cloud()
    blob = encode_point_cloud(name, xyz, features)
    detected = detect_point_cloud_codec(blob)
    assert detected.name == name or (name.startswith("ipc") and detected.name.startswith("ipc"))
    decoded, decoded_features = decode_point_cloud(blob)
    assert decoded.shape == xyz.shape
    assert decoded_features is not None and sorted(decoded_features["intensity"]) == sorted(features["intensity"])


def test_unknown_blob_and_name():
    with pytest.raises(ValueError, match="Unknown point-cloud binary format"):
        decode_point_cloud(b"NOPE" + b"\x00" * 20)
    with pytest.raises(ValueError, match="Unknown point-cloud codec"):
        get_point_cloud_codec("zip")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_point_cloud_codec(get_point_cloud_codec("laz"))
    assert isinstance(get_point_cloud_codec("kedec"), PointCloudCodec)


def test_codec_metadata():
    config = PointCloudCodecConfig(kedec_angle_step_deg=0.01)
    description = point_cloud_codec_metadata("kedec", config)
    assert description["codec"] == "kedec" and description["version"] == 1
    assert description["encoder"].startswith("py123d ")
    assert description["options"]["kedec_angle_step_deg"] == 0.01
    assert decode_codec_metadata(encode_codec_metadata(description)) == description
    assert "options" not in point_cloud_codec_metadata("laz", None)
    assert decode_codec_metadata(None) is None
