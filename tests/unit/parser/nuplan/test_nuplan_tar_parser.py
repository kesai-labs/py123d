"""The nuPlan log parser that reads camera images and lidar sweeps from the sensor archives."""

import pickle
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pytest

pytest.importorskip("nuplan")

from py123d.datatypes import CameraID
from py123d.datatypes.sensors.lidar import LidarFeature, LidarID
from py123d.parser.nuplan.nuplan_tar_index import build_tar_index
from py123d.parser.nuplan.nuplan_tar_parser import NuplanTarLogParser
from py123d.parser.nuplan.utils.nuplan_constants import NUPLAN_LIDAR_SWEEP_DURATION_US, NUPLAN_ROLLING_SHUTTER_S
from py123d.parser.nuplan.utils.nuplan_sql_helper import get_interpolated_ego_pose_from_db

from .test_nuplan_tar_index import write_sensor_archive

LOG_NAME = "2021.05.25.14.24.08_veh-25_00934_01067"
OTHER_LOG_NAME = "2021.05.25.14.16.10_veh-35_01100_01664"

# Lidar sweeps at 20 Hz; the cameras take an image with every second sweep.
NUM_LIDAR_SWEEPS = 6
LIDAR_TIMESTAMPS_US = 1_000_000 + 50_000 * np.arange(NUM_LIDAR_SWEEPS)
IMAGE_TIMESTAMPS_US = LIDAR_TIMESTAMPS_US[::2] + 4_000
CHANNELS = ["CAM_F0", "CAM_L0", "CAM_R0"]  # the archive has no images of CAM_R0

POINTS = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)


def _token(number: int) -> bytes:
    return number.to_bytes(8, "big")


# A sweep ten seconds after the others, which the archives have no file of.
LATE_SWEEP_TOKEN = _token(2999)


def _image_file(channel: str, index: int) -> str:
    return f"{LOG_NAME}/{channel}/{channel.lower()}_{index}.jpg"


def _lidar_file(index: int) -> str:
    return f"{LOG_NAME}/MergedPointCloud/sweep_{index}.pcd"


def _pcd_content(points: np.ndarray) -> bytes:
    """A lidar sweep in the binary PCD layout of nuPlan."""
    header = (
        "VERSION 0.7\nFIELDS x y z intensity ring lidar_info\nSIZE 4 4 4 1 1 1\nTYPE F F F U U U\n"
        f"COUNT 1 1 1 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA binary\n"
    )
    body = np.zeros(
        len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "u1"), ("ring", "u1"), ("info", "u1")]
    )
    body["x"], body["y"], body["z"] = points.T
    body["intensity"] = 10 + np.arange(len(points))
    body["ring"] = 20 + np.arange(len(points))
    body["info"] = np.arange(len(points)) % 5  # the lidar a point comes from, 0 is the top lidar
    return header.encode("ascii") + body.tobytes()


@pytest.fixture
def log_database(tmp_path: Path) -> Path:
    """A nuPlan log database with the tables the sensor access reads."""
    database_path = tmp_path / "splits" / "test" / f"{LOG_NAME}.db"
    database_path.parent.mkdir(parents=True)
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE ego_pose (timestamp INTEGER, x REAL, y REAL, z REAL, qw REAL, qx REAL, qy REAL, qz REAL);
            CREATE TABLE camera (token BLOB PRIMARY KEY, log_token BLOB, channel TEXT, model TEXT, translation BLOB,
                                 rotation BLOB, intrinsic BLOB, distortion BLOB, width INTEGER, height INTEGER);
            CREATE TABLE image (token BLOB PRIMARY KEY, next_token BLOB, prev_token BLOB, ego_pose_token BLOB,
                                camera_token BLOB, filename_jpg TEXT, timestamp INTEGER);
            CREATE TABLE lidar_pc (token BLOB PRIMARY KEY, filename TEXT, timestamp INTEGER);
            """
        )
        for index in range(40):  # 100 Hz, driving along x
            connection.execute(
                "INSERT INTO ego_pose VALUES (?, ?, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)",
                (990_000 + 10_000 * index, 0.1 * index),
            )
        for camera_index, channel in enumerate(CHANNELS):
            connection.execute(
                "INSERT INTO camera VALUES (?, ?, ?, 'pinhole', ?, ?, ?, ?, 1920, 1080)",
                (
                    _token(100 + camera_index),
                    _token(1),
                    channel,
                    pickle.dumps([1.5, 0.1 * camera_index, 1.6]),
                    pickle.dumps([1.0, 0.0, 0.0, 0.0]),
                    pickle.dumps([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]]),
                    pickle.dumps([0.0, 0.0, 0.0, 0.0, 0.0]),
                ),
            )
            for image_index, timestamp_us in enumerate(IMAGE_TIMESTAMPS_US):
                connection.execute(
                    "INSERT INTO image VALUES (?, NULL, NULL, ?, ?, ?, ?)",
                    (
                        _token(1000 + 100 * camera_index + image_index),
                        _token(5000 + image_index),
                        _token(100 + camera_index),
                        _image_file(channel, image_index),
                        int(timestamp_us),
                    ),
                )
        lidar_sweeps = [
            (_token(2000 + index), _lidar_file(index), int(timestamp_us))
            for index, timestamp_us in enumerate(LIDAR_TIMESTAMPS_US)
        ]
        lidar_sweeps.append(
            (LATE_SWEEP_TOKEN, f"{LOG_NAME}/MergedPointCloud/late.pcd", int(LIDAR_TIMESTAMPS_US[-1]) + 10_000_000)
        )
        connection.executemany("INSERT INTO lidar_pc VALUES (?, ?, ?)", lidar_sweeps)
    return database_path


@pytest.fixture
def log_parser(log_database: Path, tmp_path: Path) -> NuplanTarLogParser:
    """Log parser over a camera and a lidar archive, each without one of the files the database lists."""
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    camera_files = {
        _image_file(channel, index): f"{channel}-image-{index}".encode()
        for channel in CHANNELS[:2]
        for index in range(len(IMAGE_TIMESTAMPS_US))
    }
    del camera_files[_image_file("CAM_L0", 1)]
    camera_files[f"{OTHER_LOG_NAME}/CAM_B0/other.jpg"] = b"image of another log"
    lidar_files = {_lidar_file(index): _pcd_content(POINTS + index) for index in range(NUM_LIDAR_SWEEPS)}
    del lidar_files[_lidar_file(3)]

    archive_paths = [
        write_sensor_archive(archive_root / "nuplan-v1.1_test_camera_0.zip", camera_files),
        write_sensor_archive(archive_root / "nuplan-v1.1_test_lidar_0.zip", lidar_files),
    ]
    for archive_path in archive_paths:
        build_tar_index(archive_path)
    return NuplanTarLogParser(
        split="nuplan_test",
        source_log_path=log_database,
        nuplan_data_root=tmp_path,
        nuplan_archive_root=archive_root,
        archive_paths=archive_paths,
    )


def test_sensors_follow_the_archives(log_parser: NuplanTarLogParser):
    """A log has the cameras and the lidar it has files of in the archives."""
    camera_metadatas = log_parser._get_camera_metadatas()

    assert set(camera_metadatas) == {CameraID.PCAM_F0, CameraID.PCAM_L0}
    assert camera_metadatas[CameraID.PCAM_L0].camera_name == "CAM_L0"
    assert camera_metadatas[CameraID.PCAM_L0].width == 1920
    np.testing.assert_allclose(camera_metadatas[CameraID.PCAM_L0].camera_to_imu_se3.array[:3], [1.5, 0.1, 1.6])
    assert LidarID.LIDAR_TOP in dict(log_parser._get_lidar_merged_metadata())


def test_log_without_archives_has_no_sensors(log_database: Path, tmp_path: Path):
    log_parser = NuplanTarLogParser("nuplan_test", log_database, tmp_path, tmp_path, archive_paths=[])

    assert log_parser._get_camera_metadatas() == {}
    assert dict(log_parser._get_lidar_merged_metadata()) == {}
    assert list(log_parser._iter_pinhole_cameras()) == []
    assert list(log_parser._iter_lidars(log_parser._get_lidar_merged_metadata())) == []


def test_cameras_of_a_lidar_sweep(log_parser: NuplanTarLogParser, log_database: Path):
    """The images taken with a lidar sweep come out of the archive, with the pose of the camera."""
    camera_metadatas = log_parser._get_camera_metadatas()

    cameras = log_parser._extract_cameras(_token(2000).hex(), camera_metadatas)

    assert [camera.metadata.camera_name for camera in cameras] == ["CAM_F0", "CAM_L0"]
    assert [camera._byte_string for camera in cameras] == [b"CAM_F0-image-0", b"CAM_L0-image-0"]
    assert all(camera.timestamp.time_us == IMAGE_TIMESTAMPS_US[0] for camera in cameras)
    ego_pose = get_interpolated_ego_pose_from_db(
        str(log_database), int(IMAGE_TIMESTAMPS_US[0]) + NUPLAN_ROLLING_SHUTTER_S.time_us
    )
    np.testing.assert_allclose(cameras[1].camera_to_global_se3.array[:3], ego_pose.array[:3] + [1.5, 0.1, 1.6])


def test_image_missing_in_the_archive_is_left_out(log_parser: NuplanTarLogParser):
    cameras = log_parser._extract_cameras(_token(2002).hex(), log_parser._get_camera_metadatas())

    assert [camera._byte_string for camera in cameras] == [b"CAM_F0-image-1"]


def test_sweep_without_images(log_parser: NuplanTarLogParser):
    """A sweep long after the last image has none within the matching window."""
    assert log_parser._extract_cameras(LATE_SWEEP_TOKEN.hex(), log_parser._get_camera_metadatas()) == []


def test_all_cameras_at_native_rate(log_parser: NuplanTarLogParser):
    cameras = list(log_parser._iter_pinhole_cameras())

    contents: List[bytes] = [camera._byte_string for camera in cameras]
    assert sorted(contents) == sorted(
        [f"CAM_F0-image-{index}".encode() for index in range(3)] + [b"CAM_L0-image-0", b"CAM_L0-image-2"]
    )
    assert [camera.timestamp.time_us for camera in cameras] == sorted(camera.timestamp.time_us for camera in cameras)


def test_all_lidar_sweeps_at_native_rate(log_parser: NuplanTarLogParser):
    lidars = list(log_parser._iter_lidars(log_parser._get_lidar_merged_metadata()))

    present_sweeps = [0, 1, 2, 4, 5]
    assert [lidar.timestamp.time_us for lidar in lidars] == [
        int(LIDAR_TIMESTAMPS_US[index]) for index in present_sweeps
    ]
    for lidar, sweep_index in zip(lidars, present_sweeps):
        assert lidar.timestamp_end.time_us == lidar.timestamp.time_us + NUPLAN_LIDAR_SWEEP_DURATION_US
        np.testing.assert_array_equal(lidar.point_cloud_3d, POINTS + sweep_index)
        np.testing.assert_array_equal(lidar.point_cloud_features[LidarFeature.INTENSITY.serialize()], [10, 11, 12])
        np.testing.assert_array_equal(lidar.point_cloud_features[LidarFeature.CHANNEL.serialize()], [20, 21, 22])
        np.testing.assert_array_equal(
            lidar.point_cloud_features[LidarFeature.IDS.serialize()],
            [int(LidarID.LIDAR_TOP), int(LidarID.LIDAR_SIDE_LEFT), int(LidarID.LIDAR_SIDE_RIGHT)],
        )


def test_log_parser_is_pickled_without_open_archives(log_parser: NuplanTarLogParser):
    """The log parsers are sent to the worker processes, which open the archives themselves."""
    assert len(list(log_parser._iter_pinhole_cameras())) == 5

    restored = pickle.loads(pickle.dumps(log_parser))

    assert restored._members is None and restored._reader is None
    assert len(list(restored._iter_pinhole_cameras())) == 5
