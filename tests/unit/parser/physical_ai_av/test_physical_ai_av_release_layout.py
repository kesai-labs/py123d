"""Physical AI AV: reading the current dataset release and older revisions (synthetic data)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from py123d.datatypes.sensors.lidar import LidarFeature, LidarID, LidarMetadata
from py123d.geometry import PoseSE3
from py123d.parser.physical_ai_av import physical_ai_av_sensor_io as sensor_io
from py123d.parser.physical_ai_av.physical_ai_av_parser import PhysicalAIAVParser


def _data_root(tmp_path, split_in_clip_index: bool):
    """Clips a (train), b (val), c (train, chunk 1 without calibration on disk)."""
    root = tmp_path / "pai"
    clip_ids = pd.Index(["a", "b", "c"], name="clip_id")
    splits = ["train", "val", "train"]
    clip_index = pd.DataFrame({"clip_is_valid": [True] * 3, "chunk": [0, 0, 1]}, index=clip_ids)
    if split_in_clip_index:  # older revisions
        clip_index["split"] = splits
    else:  # current revision: the split is in splits/default.parquet
        (root / "splits").mkdir(parents=True)
        pd.DataFrame({"split": splits}, index=clip_ids).to_parquet(root / "splits" / "default.parquet")
    root.mkdir(exist_ok=True)
    clip_index.to_parquet(root / "clip_index.parquet")
    extrinsics = root / "calibration" / "sensor_extrinsics"
    extrinsics.mkdir(parents=True)
    (extrinsics / "sensor_extrinsics.chunk_0000.parquet").write_bytes(b"")
    return root


@pytest.mark.parametrize("split_in_clip_index", [False, True])
def test_clip_collection(tmp_path, split_in_clip_index):
    root = _data_root(tmp_path, split_in_clip_index)
    parser = PhysicalAIAVParser(["physical-ai-av_train", "physical-ai-av_val"], root)
    entries = [(p._clip_id, p._chunk, p._split) for p in parser.get_log_parsers()]
    assert entries == [("a", 0, "physical-ai-av_train"), ("b", 0, "physical-ai-av_val")]


def _write_spin_table(path, start_column: str) -> None:
    pd.DataFrame({"spin_index": [0], start_column: [1_000_000], "draco_encoded_pointcloud": [b"blob"]}).to_parquet(path)


def _lidar_metadatas():
    pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    return {LidarID.LIDAR_TOP: LidarMetadata(lidar_name="top", lidar_id=LidarID.LIDAR_TOP, lidar_to_imu_se3=pose)}


def test_spin_of_the_current_release_keeps_intensity_point_times_and_beams(tmp_path, monkeypatch):
    """Current spins store named attributes instead of colors: intensity, absolute point time, (beam, column)."""
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.5]], dtype=np.float32)
    attributes = [
        {"name": None, "data": points},
        {"name": "intensity", "data": np.array([[7], [200]], dtype=np.uint8)},
        {"name": "timestamp", "data": np.array([[1_000_010], [1_050_000]], dtype=np.uint32)},
        {"name": "model_element", "data": np.array([[0, 5], [127, 3599]], dtype=np.uint32)},
    ]
    mesh = SimpleNamespace(points=points, attributes=attributes, colors=None)
    monkeypatch.setattr(sensor_io.DracoPy, "decode", lambda blob: mesh)
    path = tmp_path / "clip.lidar_top_360fov.parquet"
    _write_spin_table(path, "spin_start_timestamp")

    xyz, features = sensor_io.load_physical_ai_av_point_cloud_data_from_path(path, 0, _lidar_metadatas())

    np.testing.assert_allclose(xyz, points + [1.0, 2.0, 3.0])
    assert features[LidarFeature.INTENSITY.serialize()].tolist() == [7, 200]
    assert features[LidarFeature.TIMESTAMPS.serialize()].tolist() == [1_000_010, 1_050_000]
    assert features[LidarFeature.TIMESTAMPS.serialize()].dtype == np.int64
    assert features[LidarFeature.CHANNEL.serialize()].tolist() == [0, 127]
    assert features[LidarFeature.CHANNEL.serialize()].dtype == np.uint8


def test_spin_of_an_older_revision_with_colors(tmp_path, monkeypatch):
    points = np.zeros((2, 3), dtype=np.float32)
    colors = np.array([[9, 0, 0], [10, 0xFF, 0xFF]], dtype=np.uint8)  # time codes 0 and 65535
    mesh = SimpleNamespace(points=points, attributes=[{"name": None, "data": points}], colors=colors)
    monkeypatch.setattr(sensor_io.DracoPy, "decode", lambda blob: mesh)
    path = tmp_path / "clip.lidar_top_360fov.parquet"
    _write_spin_table(path, "reference_timestamp")

    _, features = sensor_io.load_physical_ai_av_point_cloud_data_from_path(path, 0, _lidar_metadatas())

    assert features[LidarFeature.INTENSITY.serialize()].tolist() == [9, 10]
    assert features[LidarFeature.TIMESTAMPS.serialize()].tolist() == [1_105_000, 895_000]
    assert LidarFeature.CHANNEL.serialize() not in features
