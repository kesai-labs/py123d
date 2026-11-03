"""Tests of transcoding a JPEG/IPC log into AV1 cameras (SVT-AV1) and kedec lidars."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter, SyncConfig
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader
from py123d.api.scene.arrow.transcode import (
    CameraSelection,
    GridSelection,
    TranscodeConfig,
    find_logs,
    transcode_log,
)
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.api.utils.arrow_metadata_utils import get_metadata_from_arrow_schema
from py123d.common.frame_selection import FrameSelectionConfig
from py123d.common.io.camera.video_codec_config import VideoCodecConfig
from py123d.common.io.codec_metadata import CODEC_METADATA_KEY, decode_codec_metadata
from py123d.datatypes import Timestamp
from py123d.datatypes.sensors.base_camera import CameraID
from py123d.datatypes.sensors.lidar import Lidar, LidarID, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeDistortion, PinholeIntrinsics
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D
from py123d.parser.base_dataset_parser import ParsedCamera

from ...scene.conftest import make_ego_metadata, make_log_metadata

T0 = 1_784_208_918_000_000
LIDAR_POSE = PoseSE3(x=1.2, y=0.0, z=1.9, qw=1.0, qx=0.0, qy=0.0, qz=0.0)


def _camera_metadata() -> PinholeCameraMetadata:
    return PinholeCameraMetadata(
        camera_name="front",
        camera_id=CameraID.PCAM_F0,
        intrinsics=PinholeIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=64.0),
        distortion=PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        width=160,
        height=128,
        camera_to_imu_se3=PoseSE3.identity(),
        has_arrival_time=True,
    )


def _image(i: int) -> np.ndarray:
    small = np.random.RandomState(i).randint(0, 256, size=(8, 10, 3)).astype(np.uint8)
    return cv2.resize(small, (160, 128), interpolation=cv2.INTER_CUBIC)


def _scan(i: int, metadata: LidarMetadata) -> Lidar:
    rng = np.random.default_rng(i)
    n = 2000
    az, el = rng.uniform(-np.pi, np.pi, n), rng.uniform(-0.4, 0.2, n)
    r = rng.uniform(1.0, 120.0, n)
    xyz = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], 1) + [1.2, 0.0, 1.9]
    start = T0 + i * 100_000
    features = {
        "intensity": rng.integers(0, 256, n).astype(np.uint8),
        "timestamps": start + rng.integers(0, 100_000, n).astype(np.int64),
        "uid": np.arange(n, dtype=np.int64) + i * n,
    }
    return Lidar(
        timestamp=Timestamp.from_us(start),
        timestamp_end=Timestamp.from_us(start + 100_000),
        metadata=metadata,
        point_cloud_3d=xyz.astype(np.float32),
        point_cloud_features=features,
        arrival_timestamp=Timestamp.from_us(start + 105_000),
    )


def _ego(ts_us: int) -> EgoStateSE3:
    dynamic = DynamicStateSE3(
        velocity=Vector3D(1.0, 0.0, 0.0), acceleration=Vector3D(0.0, 0.0, 0.0), angular_velocity=Vector3D(0.0, 0.0, 0.0)
    )
    return EgoStateSE3.from_imu(
        imu_se3=PoseSE3(x=ts_us * 1e-6, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        metadata=make_ego_metadata(),
        timestamp=Timestamp.from_us(ts_us),
        dynamic_state_se3=dynamic,
    )


@pytest.fixture(scope="module")
def source_log(tmp_path_factory) -> Path:
    return write_source_log(tmp_path_factory.mktemp("source"))


def write_source_log(root: Path) -> Path:
    """A JPEG/IPC log: lidar at 10 Hz (reference), a camera at 20 Hz with null poses on rows 2, 6, ..., ego at 100 Hz."""
    config = LogWriterConfig(camera_store_option="jpeg_binary", lidar_store_option="binary", lidar_codec="ipc_lz4")
    sync = SyncConfig(reference_column="lidar.lidar_top.timestamp_us", direction="forward")
    writer = ArrowLogWriter(config, logs_root=root, sensors_root=root, sync_config=sync)
    log_metadata = make_log_metadata()
    writer.reset(log_metadata)
    lidar_metadata = LidarMetadata("top", LidarID.LIDAR_TOP, LIDAR_POSE, has_arrival_time=True)
    camera_metadata = _camera_metadata()
    events = []
    for i in range(12):
        events.append((T0 + i * 100_000, 1, _scan(i, lidar_metadata)))
    for i in range(24):
        ts = T0 + 3_000 + i * 50_000 + (i % 3) * 400
        jpeg = cv2.imencode(".jpg", cv2.cvtColor(_image(i), cv2.COLOR_RGB2BGR))[1].tobytes()
        pose = PoseSE3(x=float(i), y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0) if i % 4 != 2 else None
        events.append(
            (
                ts,
                2,
                ParsedCamera(
                    camera_metadata,
                    Timestamp.from_us(ts),
                    pose,
                    byte_string=jpeg,
                    exposure_factor=1.0 + i / 100,
                    arrival_timestamp=Timestamp.from_us(ts + 30_000),
                ),
            )
        )
    for i in range(120):
        events.append((T0 + i * 10_000, 0, _ego(T0 + i * 10_000)))
    for _, _, modality in sorted(events, key=lambda e: (e[0], e[1])):
        writer.write_async(modality)
    writer.close()
    return root


def _table(path: Path) -> pa.Table:
    return pa.ipc.open_file(str(path)).read_all()


class EvenImages(CameraSelection):
    def select(self, log_dir, camera_timestamps):
        return {key: np.arange(0, len(ts), 2) for key, ts in camera_timestamps.items()}


def _config(**kwargs) -> TranscodeConfig:
    writer_config = LogWriterConfig(
        camera_store_option="av1",
        video_codec_config=VideoCodecConfig(backend="svt", svt_preset=12, svt_crf=30),
        lidar_store_option="binary",
        lidar_codec="kedec",
        encode_threads=2,
    )
    return TranscodeConfig(log_writer_config=writer_config, camera_threads=2, **kwargs)


@pytest.fixture(scope="module")
def transcoded(source_log, tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("target")
    log_metadata = make_log_metadata()
    assert transcode_log(source_log, target, log_metadata.split, log_metadata.log_name, _config(), EvenImages())
    return target


def _log_dir(root: Path) -> Path:
    log_metadata = make_log_metadata()
    return root / log_metadata.split / log_metadata.log_name


def test_find_logs(source_log):
    log_metadata = make_log_metadata()
    assert find_logs(source_log) == [(log_metadata.split, log_metadata.log_name)]
    assert find_logs(source_log, splits=["other"]) == []
    assert find_logs(source_log, log_names=[log_metadata.log_name]) == [(log_metadata.split, log_metadata.log_name)]


def test_other_files_are_copied_byte_for_byte(source_log, transcoded):
    source, target = _log_dir(source_log), _log_dir(transcoded)
    names = {p.name for p in source.iterdir()}
    assert names - {p.name for p in target.iterdir()} == set()
    for name in names:
        if name.startswith(("camera.", "lidar.")) or name == "sync.arrow":
            continue
        assert (source / name).read_bytes() == (target / name).read_bytes(), name
    assert (target / "camera.pcam_f0.av1.mp4").exists()
    assert not (transcoded / ".incomplete").exists()


def test_camera_rows_are_the_kept_source_rows(source_log, transcoded):
    source = _table(_log_dir(source_log) / "camera.pcam_f0.arrow")
    target = _table(_log_dir(transcoded) / "camera.pcam_f0.arrow")
    assert target.num_rows == 12
    for column in ("timestamp_us", "camera_to_global_se3", "exposure_factor", "arrival_us"):
        name = f"camera.pcam_f0.{column}"
        assert target.column(name).to_pylist() == source.column(name).to_pylist()[0::2], column
    assert target.column("camera.pcam_f0.data").to_pylist() == list(range(12))
    assert target.column("camera.pcam_f0.camera_to_global_se3").to_pylist()[1] is None  # null poses stay null
    description = decode_codec_metadata(target.schema.metadata[CODEC_METADATA_KEY])
    assert description["codec"] == "av1" and description["encoder"] == "libsvtav1"
    metadata = _camera_metadata()
    for row in (0, 5, 11):
        image = ArrowCameraReader.read_at_index(row, target, metadata, "test", log_dir=_log_dir(transcoded))
        reference = ArrowCameraReader.read_at_index(2 * row, source, metadata, "test", log_dir=_log_dir(source_log))
        assert image is not None and reference is not None
        mse = np.mean((image.image.astype(float) - reference.image.astype(float)) ** 2)
        assert 10 * np.log10(255**2 / mse) > 30


def test_lidar_rows_points_and_times(source_log, transcoded):
    source = _table(_log_dir(source_log) / "lidar.lidar_top.arrow")
    target = _table(_log_dir(transcoded) / "lidar.lidar_top.arrow")
    for column in ("timestamp_us", "end_timestamp_us", "arrival_us"):
        name = f"lidar.lidar_top.{column}"
        assert target.column(name).to_pylist() == source.column(name).to_pylist()
    metadata = get_metadata_from_arrow_schema(target.schema, LidarMetadata)
    for row in range(source.num_rows):
        a = ArrowLidarReader.read_at_index(row, source, metadata, "test")
        b = ArrowLidarReader.read_at_index(row, target, metadata, "test")
        uid = b.point_cloud_features["uid"] - row * 2000
        np.testing.assert_array_equal(b.point_cloud_features["timestamps"], a.point_cloud_features["timestamps"][uid])
        np.testing.assert_array_equal(b.point_cloud_features["intensity"], a.point_cloud_features["intensity"][uid])
        errors = np.linalg.norm(b.point_cloud_3d.astype(float) - a.point_cloud_3d[uid].astype(float), axis=1)
        assert len(errors) == len(a.point_cloud_3d) and errors.max() < 0.02
    assert decode_codec_metadata(target.schema.metadata[CODEC_METADATA_KEY])["codec"] == "kedec"


def test_sync_table(source_log, transcoded):
    source = _table(_log_dir(source_log) / "sync.arrow")
    target = _table(_log_dir(transcoded) / "sync.arrow")
    assert target.schema.metadata == source.schema.metadata
    for name in source.column_names:
        if name != "camera.pcam_f0":
            assert target.column(name).to_pylist() == source.column(name).to_pylist(), name
    # rows every 100 ms from T0; kept images at T0 + 3 ms + k * 100 ms (+ 0/400/800 us): image k for row k
    assert target.column("camera.pcam_f0").to_pylist() == list(range(12))


def test_existing_target_is_skipped_unless_forced(source_log, transcoded):
    log_metadata = make_log_metadata()
    assert not transcode_log(source_log, transcoded, log_metadata.split, log_metadata.log_name, _config(), EvenImages())
    with pytest.raises(ValueError, match="same directory"):
        transcode_log(source_log, source_log, log_metadata.split, log_metadata.log_name, _config(force=True))


def test_grid_selection_and_force(source_log, tmp_path):
    log_metadata = make_log_metadata()
    selection = GridSelection(FrameSelectionConfig(mode="grid", period_us=100_000, grid_phase_us=3_000))
    config = _config(force=True)
    assert transcode_log(source_log, tmp_path, log_metadata.split, log_metadata.log_name, config, selection)
    assert transcode_log(source_log, tmp_path, log_metadata.split, log_metadata.log_name, config, selection)
    timestamps = _table(_log_dir(tmp_path) / "camera.pcam_f0.arrow").column("camera.pcam_f0.timestamp_us").to_numpy()
    assert len(timestamps) == 12 and np.all(np.diff(timestamps) >= 99_000)


def test_grid_selection_keeps_the_first_image():
    """A grid time up to half a period before the first image selects it."""
    selection = GridSelection(FrameSelectionConfig(mode="grid", period_us=100_000, grid_phase_us=0))
    ts = 1_000_000 + 18_000 + np.arange(20) * 50_000  # images at phase 18 and 68 ms
    kept = selection.select(Path("."), {"camera.pcam_f0": ts})["camera.pcam_f0"]
    np.testing.assert_array_equal(kept, np.arange(0, 20, 2))


def test_failure_leaves_the_target_untouched(source_log, tmp_path, monkeypatch):
    from py123d.api.scene.arrow import transcode

    def broken(*args, **kwargs):
        raise RuntimeError("lidar failed")

    monkeypatch.setattr(transcode, "transcode_lidar", broken)
    log_metadata = make_log_metadata()
    with pytest.raises(RuntimeError):
        transcode_log(source_log, tmp_path, log_metadata.split, log_metadata.log_name, _config(), EvenImages())
    assert not (_log_dir(tmp_path)).exists()
    assert (tmp_path / ".incomplete" / log_metadata.split / log_metadata.log_name).exists()


def test_a_camera_file_named_after_another_camera_is_refused(source_log, tmp_path):
    """The new files are named after the metadata; a file whose metadata names another camera is not renamed."""
    renamed = tmp_path / "camera.pcam_l0.arrow"
    import shutil

    shutil.copy(_log_dir(source_log) / "camera.pcam_f0.arrow", renamed)
    from py123d.api.scene.arrow.transcode import transcode_camera

    with pytest.raises(ValueError, match="names the camera camera.pcam_f0"):
        transcode_camera(renamed, tmp_path, np.arange(2), _config())


def test_cameras_and_lidars_are_transcoded_at_the_same_time(source_log, tmp_path, monkeypatch):
    """The camera and the lidar of the log each wait for the other at a barrier; one after the other they would
    break it after the timeout."""
    import threading

    from py123d.api.scene.arrow import transcode

    barrier = threading.Barrier(2, timeout=30)
    camera, lidar = transcode.transcode_camera, transcode.transcode_lidar

    def camera_at_barrier(*args, **kwargs):
        barrier.wait()
        return camera(*args, **kwargs)

    def lidar_at_barrier(*args, **kwargs):
        barrier.wait()
        return lidar(*args, **kwargs)

    monkeypatch.setattr(transcode, "transcode_camera", camera_at_barrier)
    monkeypatch.setattr(transcode, "transcode_lidar", lidar_at_barrier)
    log_metadata = make_log_metadata()
    assert transcode_log(source_log, tmp_path, log_metadata.split, log_metadata.log_name, _config(), EvenImages())
    assert _table(_log_dir(tmp_path) / "camera.pcam_f0.arrow").num_rows == 12


def test_copies_of_write_protected_files_are_writable(source_log, tmp_path):
    """The copied files keep the content and the modification time, not the mode of a write-protected source."""
    import os
    import shutil
    import stat

    source = tmp_path / "source"
    shutil.copytree(source_log, source)
    log_dir = _log_dir(source)
    for path in log_dir.iterdir():
        path.chmod(0o444)
    log_metadata = make_log_metadata()
    target = tmp_path / "target"
    assert transcode_log(source, target, log_metadata.split, log_metadata.log_name, _config(), EvenImages())
    copied = [p for p in _log_dir(target).iterdir() if not p.name.startswith(("camera.", "lidar.", "sync."))]
    assert copied
    for path in copied:
        assert path.stat().st_mode & stat.S_IWUSR, path.name
        assert path.stat().st_mtime_ns == (log_dir / path.name).stat().st_mtime_ns, path.name
        assert path.read_bytes() == (log_dir / path.name).read_bytes()
    for path in log_dir.iterdir():
        path.chmod(0o644)
    assert os.access(_log_dir(target) / "sync.arrow", os.W_OK)


def _radar_log(root: Path) -> Path:
    """A log with ego states and one radar stored as IPC (zstd) blobs."""
    from py123d.datatypes.sensors.radar import Radar, RadarID, RadarMetadata

    config = LogWriterConfig(radar_store_option="binary", radar_codec="ipc_zstd")
    sync = SyncConfig(reference_column="ego_state_se3.timestamp_us", direction="forward")
    writer = ArrowLogWriter(config, logs_root=root, sensors_root=root, sync_config=sync)
    log_metadata = make_log_metadata()
    writer.reset(log_metadata)
    metadata = RadarMetadata(radar_name="RADAR_FRONT", radar_id=RadarID.RADAR_FRONT)
    rng = np.random.default_rng(3)
    events = []
    for i in range(30):
        events.append((T0 + i * 10_000, 0, _ego(T0 + i * 10_000)))
    for i in range(6):
        n = 40 + i
        xyz = (rng.normal(size=(n, 3)) * [30.0, 10.0, 1.0]).astype(np.float32)
        features = {
            "rcs": rng.normal(size=n).astype(np.float32),
            "radial_velocity": rng.normal(size=n).astype(np.float32),
        }
        events.append((T0 + i * 50_000, 1, Radar(Timestamp.from_us(T0 + i * 50_000), metadata, xyz, features)))
    for _, _, modality in sorted(events, key=lambda e: (e[0], e[1])):
        writer.write_async(modality)
    writer.close()
    return root / log_metadata.split / log_metadata.log_name


def test_radar_is_transcoded_with_a_radar_codec_and_copied_without(tmp_path):
    from py123d.api.scene.arrow.modalities.arrow_radar import ArrowRadarReader
    from py123d.datatypes.sensors.radar import RadarMetadata

    source_dir = _radar_log(tmp_path / "source")
    name = "radar.radar_front.arrow"
    log_metadata = make_log_metadata()
    writer_config = LogWriterConfig(radar_store_option="binary", radar_codec="kedec", encode_threads=2)
    kedec_root = tmp_path / "kedec"
    assert transcode_log(
        tmp_path / "source",
        kedec_root,
        log_metadata.split,
        log_metadata.log_name,
        TranscodeConfig(log_writer_config=writer_config),
    )
    source, target = _table(source_dir / name), _table(_log_dir(kedec_root) / name)
    assert decode_codec_metadata(target.schema.metadata[CODEC_METADATA_KEY])["codec"] == "kedec"
    assert (
        target.column("radar.radar_front.timestamp_us").to_pylist()
        == source.column("radar.radar_front.timestamp_us").to_pylist()
    )
    metadata = get_metadata_from_arrow_schema(target.schema, RadarMetadata)
    for row in range(source.num_rows):
        a = ArrowRadarReader.read_at_index(row, source, metadata, "test")
        b = ArrowRadarReader.read_at_index(row, target, metadata, "test")
        ia, ib = np.argsort(a.point_cloud_features["rcs"]), np.argsort(b.point_cloud_features["rcs"])
        np.testing.assert_array_equal(
            b.point_cloud_features["radial_velocity"][ib], a.point_cloud_features["radial_velocity"][ia]
        )
        assert (
            np.linalg.norm(b.point_cloud_3d[ib].astype(float) - a.point_cloud_3d[ia].astype(float), axis=1).max() < 0.02
        )
    copy_root = tmp_path / "copy"
    assert transcode_log(
        tmp_path / "source",
        copy_root,
        log_metadata.split,
        log_metadata.log_name,
        TranscodeConfig(log_writer_config=LogWriterConfig()),
    )
    assert (_log_dir(copy_root) / name).read_bytes() == (source_dir / name).read_bytes()
