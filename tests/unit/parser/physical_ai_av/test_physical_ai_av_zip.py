"""Physical AI AV from the chunk archives: zip reader, spin decoder, camera selection, clip collection."""

import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from py123d.datatypes.sensors.lidar import LidarFeature
from py123d.geometry import PoseSE3
from py123d.parser.physical_ai_av import physical_ai_av_sensor_io as sensor_io
from py123d.parser.physical_ai_av.physical_ai_av_camera_selection import (
    CAMERA_GROUPS,
    camera_azimuth_deg,
    direction_delay_us,
    front_delay_us,
    reference_times,
    select_camera_frames,
)
from py123d.parser.physical_ai_av.physical_ai_av_zip_io import ZipChunkReader, pai_chunk_zip_path, pai_member_name


def _stored_zip(path: Path, members: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


# ---------------------------------------------------------------------------------------------------------- zip reader


def test_zip_reader(tmp_path):
    path = pai_chunk_zip_path(tmp_path, "lidar", "lidar_top_360fov", 7)
    assert path == tmp_path / "lidar/lidar_top_360fov/lidar_top_360fov.chunk_0007.zip"
    payload = bytes(range(256)) * 100
    _stored_zip(path, {pai_member_name("clip", "lidar_top_360fov", "parquet"): payload, "other": b"x"})
    reader = ZipChunkReader()
    assert reader.contains(path, "clip.lidar_top_360fov.parquet") and not reader.contains(path, "missing")
    assert not reader.contains(tmp_path / "missing.zip", "clip.lidar_top_360fov.parquet")
    offset, size = reader.member_range(path, "clip.lidar_top_360fov.parquet")
    assert size == len(payload) and path.read_bytes()[offset : offset + size] == payload
    assert reader.read_bytes(path, "clip.lidar_top_360fov.parquet") == payload
    assert reader.open_bytes(path, "other").read() == b"x"
    reader.close()


def test_zip_reader_rejects_compressed_members(tmp_path):
    path = tmp_path / "c.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("m", b"y" * 1000)
    with pytest.raises(AssertionError):
        ZipChunkReader().read_bytes(path, "m")


# -------------------------------------------------------------------------------------------------------- spin decoder


def _mesh(points, attributes=None, colors=None):
    attrs = [{"name": None, "data": points}] + [{"name": k, "data": v} for k, v in (attributes or {}).items()]
    return SimpleNamespace(points=points, attributes=attrs, colors=colors)


def test_spin_with_named_attributes(monkeypatch):
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.5]])
    attributes = {
        "intensity": np.array([[7], [200]], dtype=np.uint8),
        "timestamp": np.array([[1_000_010], [1_050_000]], dtype=np.uint32),
        "model_element": np.array([[0, 5], [127, 3599]], dtype=np.uint16),
    }
    monkeypatch.setattr(sensor_io.DracoPy, "decode", lambda blob: _mesh(points, attributes))
    pose = PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    xyz, f = sensor_io.decode_physical_ai_av_spin(b"blob", 1_000_000, pose)
    np.testing.assert_allclose(xyz, points + [1.0, 2.0, 3.0])
    assert xyz.dtype == np.float32
    assert f[LidarFeature.INTENSITY.serialize()].tolist() == [7, 200]
    assert f[LidarFeature.TIMESTAMPS.serialize()].tolist() == [1_000_010, 1_050_000]
    assert f[LidarFeature.TIMESTAMPS.serialize()].dtype == np.int64
    assert (
        f[LidarFeature.CHANNEL.serialize()].tolist() == [0, 127]
        and f[LidarFeature.CHANNEL.serialize()].dtype == np.uint8
    )
    assert (
        f[LidarFeature.COLUMN.serialize()].tolist() == [5, 3599]
        and f[LidarFeature.COLUMN.serialize()].dtype == np.uint16
    )


def test_spin_of_an_older_revision_with_colors(monkeypatch):
    points = np.zeros((2, 3))
    colors = np.array([[9, 0, 0], [10, 0xFF, 0xFF]], dtype=np.uint8)  # time codes 0 and 65535
    mesh = SimpleNamespace(points=points, attributes=[{"name": None, "data": points}], colors=colors)
    monkeypatch.setattr(sensor_io.DracoPy, "decode", lambda blob: mesh)
    _, f = sensor_io.decode_physical_ai_av_spin(b"blob", 1_000_000, PoseSE3.identity())
    assert f[LidarFeature.INTENSITY.serialize()].tolist() == [9, 10]
    assert f[LidarFeature.TIMESTAMPS.serialize()].tolist() == [1_105_000, 895_000]
    assert LidarFeature.CHANNEL.serialize() not in f


def test_spin_start_column():
    assert sensor_io.spin_start_column(["spin_index", "spin_start_timestamp"]) == "spin_start_timestamp"
    assert sensor_io.spin_start_column(["reference_timestamp"]) == "reference_timestamp"
    with pytest.raises(ValueError):
        sensor_io.spin_start_column(["x"])


# ---------------------------------------------------------------------------------------------------- camera selection


def _rotating_spin(start_us, period_us=100_000, phase_deg=0.0, n=3600, far=True):
    """Points on a circle, measured clockwise from azimuth phase_deg at constant angular speed."""
    k = np.arange(n)
    azimuth = phase_deg - 360.0 * k / n
    r = 20.0 if far else 5.0
    xyz = np.stack([r * np.cos(np.radians(azimuth)), r * np.sin(np.radians(azimuth)), np.zeros(n)], axis=1)
    times = start_us + (k * period_us) // n
    return xyz, times


def test_camera_azimuth():
    assert camera_azimuth_deg(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])) == pytest.approx(0.0)  # z axis -> ego x
    assert camera_azimuth_deg(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])) == pytest.approx(90.0)  # z axis -> ego y


def test_direction_delay():
    xyz, times = _rotating_spin(1_000_000, phase_deg=180.0)
    # Azimuth 0 is reached half a turn after the start (180 degrees clockwise from 180).
    assert direction_delay_us(xyz, times, 1_000_000, (0.0, 0.0), 0.0) == pytest.approx(50_000, abs=50)
    near_xyz, near_times = _rotating_spin(1_000_000, far=False)
    assert direction_delay_us(near_xyz, near_times, 1_000_000, (0.0, 0.0), 0.0) is None
    assert front_delay_us([None, 49_000.0, 51_000.0, 50_000.0]) == 50_000
    assert front_delay_us([None]) is None
    np.testing.assert_array_equal(reference_times([0, 100_000], None), [0, 100_000])


def test_two_camera_groups_keep_one_capture_time_each():
    rng = np.random.default_rng(0)
    spins = 100_000 * np.arange(60) + rng.integers(-30, 30, 60)
    ref = reference_times(spins, 50_000)
    period = 33_333
    base = np.arange(-3, 190) * period + 7_000
    first = base + rng.integers(-20, 20, len(base))
    second = base + 16_000 + rng.integers(-20, 20, len(base))  # the second group at a constant offset
    stamps = {}
    for name in CAMERA_GROUPS[0]:
        stamps[name] = first + rng.integers(0, 100, len(base))
    for name in CAMERA_GROUPS[1]:
        stamps[name] = second + rng.integers(0, 100, len(base))
    stamps[CAMERA_GROUPS[1][2]] = np.delete(stamps[CAMERA_GROUPS[1][2]], [40, 41])  # two dropped frames
    selected = select_camera_frames(stamps, ref)
    for group in CAMERA_GROUPS:
        lead = stamps[group[0]][selected[group[0]]]
        assert (selected[group[0]] >= 0).all()
        assert np.abs(lead - ref).max() <= period // 2 + 600  # nearest image, hysteresis 0.5 ms
        assert set(np.diff(selected[group[0]])) <= {3}  # every third image: 10 of 30 per second
        for name in group[1:]:
            kept = selected[name] >= 0
            assert np.abs(stamps[name][selected[name][kept]] - lead[kept]).max() < 5_000
    assert (selected[CAMERA_GROUPS[1][2]] < 0).sum() <= 1  # a dropped frame at a capture time: no image


def test_missing_cameras_and_a_camera_outside_the_groups():
    ref = 100_000 * np.arange(5)
    selected = select_camera_frames({"camera_x": np.arange(0, 600_000, 33_333)}, ref)
    assert list(selected) == ["camera_x"] and (selected["camera_x"] >= 0).all()
    assert select_camera_frames({}, ref) == {}


# ---------------------------------------------------------------------------------------------------- clip collection


@pytest.fixture
def data_root(tmp_path):
    """Two chunks with calibration; chunk 0 has a lidar archive with clips a and b (b without egomotion); chunk 1 has
    no lidar archive. The split is in splits/default.parquet (current revision)."""
    root = tmp_path / "pai"
    index = pd.DataFrame(
        {"clip_is_valid": [True] * 3, "chunk": [0, 0, 1]}, index=pd.Index(["a", "b", "c"], name="clip_id")
    )
    (root / "splits").mkdir(parents=True)
    index.to_parquet(root / "clip_index.parquet")
    pd.DataFrame({"split": ["train", "val", "train"]}, index=index.index).to_parquet(
        root / "splits" / "default.parquet"
    )
    ext = root / "calibration" / "sensor_extrinsics"
    ext.mkdir(parents=True)
    for chunk in (0, 1):
        (ext / f"sensor_extrinsics.chunk_{chunk:04d}.parquet").write_bytes(b"")
    _stored_zip(
        pai_chunk_zip_path(root, "lidar", "lidar_top_360fov", 0),
        {"a.lidar_top_360fov.parquet": b"1", "b.lidar_top_360fov.parquet": b"2"},
    )
    _stored_zip(pai_chunk_zip_path(root, "labels", "egomotion", 0), {"a.egomotion.parquet": b"3"})
    (root / "lidar" / "lidar_top_360fov" / "lidar_top_360fov.chunk_0001.zip.part").write_bytes(b"")
    return root


def test_clip_collection(data_root):
    from py123d.parser.physical_ai_av.physical_ai_av_zip_parser import PhysicalAIAVZipParser

    parser = PhysicalAIAVZipParser(["physical-ai-av_train", "physical-ai-av_val"], data_root)
    assert [(p._clip_id, p._chunk, p._split) for p in parser.get_log_parsers()] == [("a", 0, "physical-ai-av_train")]
    assert PhysicalAIAVZipParser(["physical-ai-av_val"], data_root).get_log_parsers() == []
    assert PhysicalAIAVZipParser(["physical-ai-av_train"], data_root, chunk_start=1).get_log_parsers() == []
    assert PhysicalAIAVZipParser(["physical-ai-av_train"], data_root, clip_ids=["c"]).get_log_parsers() == []


def test_a_camera_not_synchronized_with_its_group_is_selected_on_its_own():
    ref = reference_times(100_000 * np.arange(30), 50_000)
    base = np.arange(-3, 95) * 33_333
    stamps = {name: base + 10_200 for name in CAMERA_GROUPS[1]}
    stamps["camera_rear_tele_30fov"] = base - 12_400  # its own trigger phase in this clip
    selected = select_camera_frames(stamps, ref)
    tele = stamps["camera_rear_tele_30fov"][selected["camera_rear_tele_30fov"]]
    assert (selected["camera_rear_tele_30fov"] >= 0).all()
    assert np.abs(tele - ref).max() <= 33_333 // 2 + 600
    lead = stamps["camera_cross_left_120fov"][selected["camera_cross_left_120fov"]]
    np.testing.assert_array_equal(stamps["camera_cross_right_120fov"][selected["camera_cross_right_120fov"]], lead)
