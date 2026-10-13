"""A small synthetic Physical AI AV dataset, written in the flat layout the parser reads."""

import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import cv2
import DracoPy
import numpy as np
import pandas as pd
import pytest

SPLIT = "physical-ai-av_train"
CHUNK = 0
CLIP_ID = "00000000-0000-4000-8000-000000000001"

CAMERA_NAMES = ("camera_front_wide_120fov", "camera_rear_tele_30fov")
LIDAR_NAME = "lidar_top_360fov"
RADAR_NAME = "radar_front_center_srr_0"

# The ego drives straight at a heading that is not zero, so a vector in the ego frame differs from one in the odometry frame.
HEADING = 0.6
SPEED = 18.0
ACCELERATION = 1.5

NUM_SPINS = 5
SPIN_STARTS_US = 50_000 + 100_000 * np.arange(NUM_SPINS)
NUM_CAMERA_FRAMES = 18
CAMERA_TIMESTAMPS_US = 40_000 + 33_333 * np.arange(NUM_CAMERA_FRAMES)
IMAGE_WIDTH, IMAGE_HEIGHT = 64, 48


def camera_frame_level(frame_index: int) -> int:
    """Gray level a camera frame is filled with, which tells the frames of a video apart."""
    return 10 + 12 * frame_index


def _write_parquet(data_frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_parquet(path)


def _clip_file(root: Path, group: str, sensor: str, clip_id: str, extension: str) -> Path:
    return root / group / sensor / f"{clip_id}.{sensor}.{extension}"


def _write_egomotion(root: Path, clip_id: str) -> None:
    time_s = np.arange(0.0, 0.7, 0.01)
    direction = np.array([np.cos(HEADING), np.sin(HEADING), 0.0])
    position = np.outer(SPEED * time_s + 0.5 * ACCELERATION * time_s**2, direction)
    velocity = np.outer(SPEED + ACCELERATION * time_s, direction)
    egomotion = pd.DataFrame(
        {
            "timestamp": np.round(time_s * 1e6).astype(np.int64),
            "qx": 0.0,
            "qy": 0.0,
            "qz": np.sin(HEADING / 2),
            "qw": np.cos(HEADING / 2),
            "x": position[:, 0],
            "y": position[:, 1],
            "z": position[:, 2],
            "vx": velocity[:, 0],
            "vy": velocity[:, 1],
            "vz": velocity[:, 2],
            "ax": ACCELERATION * direction[0],
            "ay": ACCELERATION * direction[1],
            "az": ACCELERATION * direction[2],
            "curvature": 0.0,
        }
    )
    _write_parquet(egomotion, _clip_file(root, "labels", "egomotion", clip_id, "parquet"))


def _write_lidar(root: Path, clip_id: str) -> None:
    random = np.random.default_rng(0)
    lidar = pd.DataFrame(
        {
            "spin_index": np.arange(NUM_SPINS),
            "spin_start_timestamp": SPIN_STARTS_US,
            "spin_end_timestamp": SPIN_STARTS_US + 99_994,
            "draco_encoded_pointcloud": [
                DracoPy.encode(random.uniform(-20.0, 20.0, size=(30, 3)).astype(np.float32)) for _ in range(NUM_SPINS)
            ],
        }
    )
    _write_parquet(lidar, _clip_file(root, "lidar", LIDAR_NAME, clip_id, "parquet"))


def _write_obstacles(root: Path, clip_id: str) -> None:
    timestamps_us = np.repeat(SPIN_STARTS_US + 20_000, 2)
    obstacles = pd.DataFrame(
        {
            "timestamp_us": timestamps_us.astype(np.uint64),
            "track_id": np.tile(["1", "2"], NUM_SPINS),
            "center_x": np.tile([10.0, -5.0], NUM_SPINS),
            "center_y": np.tile([2.0, -3.0], NUM_SPINS),
            "center_z": 0.8,
            "size_x": 4.5,
            "size_y": 1.9,
            "size_z": 1.6,
            "orientation_x": 0.0,
            "orientation_y": 0.0,
            "orientation_z": 0.0,
            "orientation_w": 1.0,
            "label_class": np.tile(["automobile", "person"], NUM_SPINS),
        }
    )
    _write_parquet(obstacles, _clip_file(root, "labels", "obstacle.offline", clip_id, "parquet"))


def _write_camera(root: Path, clip_id: str, camera_name: str) -> None:
    timestamps = pd.DataFrame(
        {"timestamp": CAMERA_TIMESTAMPS_US}, index=pd.RangeIndex(NUM_CAMERA_FRAMES, name="frame_index")
    )
    _write_parquet(timestamps, _clip_file(root, "camera", camera_name, clip_id, "timestamps.parquet"))
    video_path = _clip_file(root, "camera", camera_name, clip_id, "mp4")
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (IMAGE_WIDTH, IMAGE_HEIGHT))
    assert writer.isOpened(), "OpenCV cannot write the test video"
    for frame_index in range(NUM_CAMERA_FRAMES):
        writer.write(np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), camera_frame_level(frame_index), dtype=np.uint8))
    writer.release()


def _write_radar(root: Path, clip_id: str) -> None:
    # One scan of three detections near every lidar spin.
    scan_timestamps_us = np.repeat(SPIN_STARTS_US + 5_000, 3)
    radar = pd.DataFrame(
        {
            "timestamp": scan_timestamps_us,
            "sensor_timestamp": scan_timestamps_us - 2_000,
            "azimuth": np.tile([-0.3, 0.0, 0.3], NUM_SPINS).astype(np.float32),
            "elevation": np.float32(0.0),
            "distance": np.tile([12.0, 25.0, 40.0], NUM_SPINS).astype(np.float32),
            "radial_velocity": np.float32(-3.0),
            "rcs": np.float32(-10.0),
            "snr": np.float32(15.0),
            "exist_probb": np.uint8(100),
        },
        index=pd.RangeIndex(len(scan_timestamps_us), name="scan_index"),
    )
    _write_parquet(radar, _clip_file(root, "radar", RADAR_NAME, clip_id, "parquet"))


def _write_calibration(root: Path, clip_ids: List[str]) -> None:
    chunk_tag = f"chunk_{CHUNK:04d}"
    sensor_names = [*CAMERA_NAMES, LIDAR_NAME, RADAR_NAME]

    intrinsics: Dict[str, List] = {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "cx": IMAGE_WIDTH / 2,
        "cy": IMAGE_HEIGHT / 2,
        **{f"bw_poly_{index}": value for index, value in enumerate([0.0, 0.02, 0.0, 0.0, 0.0])},
        **{f"fw_poly_{index}": value for index, value in enumerate([0.0, 50.0, 0.0, 0.0, 0.0])},
    }
    intrinsics_index = pd.MultiIndex.from_product([clip_ids, CAMERA_NAMES], names=["clip_id", "camera_name"])
    _write_parquet(
        pd.DataFrame(intrinsics, index=intrinsics_index),
        root / "calibration" / "camera_intrinsics" / f"camera_intrinsics.{chunk_tag}.parquet",
    )

    extrinsics_index = pd.MultiIndex.from_product([clip_ids, sensor_names], names=["clip_id", "sensor_name"])
    extrinsics = pd.DataFrame(
        {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "x": 1.5, "y": 0.0, "z": 1.2}, index=extrinsics_index
    )
    _write_parquet(extrinsics, root / "calibration" / "sensor_extrinsics" / f"sensor_extrinsics.{chunk_tag}.parquet")

    dimensions = pd.DataFrame(
        {
            "length": 4.87,
            "width": 2.12,
            "height": 1.47,
            "rear_axle_to_bbox_center": 1.33,
            "wheelbase": 2.85,
            "track_width": 1.58,
        },
        index=pd.Index(clip_ids, name="clip_id"),
    )
    _write_parquet(dimensions, root / "calibration" / "vehicle_dimensions" / f"vehicle_dimensions.{chunk_tag}.parquet")


def write_physical_ai_av_root(root: Path, clip_ids: List[str]) -> Path:
    """Writes the dataset files of the given clips, every clip with the same content.

    :param root: Directory to write into.
    :param clip_ids: Names of the clips, all in chunk 0 of the train split.
    :return: The dataset root.
    """
    clip_index = pd.DataFrame(
        {"clip_is_valid": True, "chunk": CHUNK, "split": "train"}, index=pd.Index(clip_ids, name="clip_id")
    )
    _write_parquet(clip_index, root / "clip_index.parquet")
    _write_calibration(root, clip_ids)
    for clip_id in clip_ids:
        _write_egomotion(root, clip_id)
        _write_lidar(root, clip_id)
        _write_obstacles(root, clip_id)
        _write_radar(root, clip_id)
        for camera_name in CAMERA_NAMES:
            _write_camera(root, clip_id, camera_name)
    return root


@pytest.fixture
def physical_ai_av_root(tmp_path: Path) -> Path:
    """Dataset root with one clip."""
    return write_physical_ai_av_root(tmp_path / "physical_ai_av", [CLIP_ID])


def write_chunk_archives(flat_root: Path, zip_root: Path) -> Path:
    """Writes a dataset again in the layout it is published in: the files of the clips in one archive per sensor.

    :param flat_root: Dataset written by :func:`write_physical_ai_av_root`.
    :param zip_root: Directory to write into.
    :return: The dataset root with the archives.
    """
    shutil.copytree(flat_root / "calibration", zip_root / "calibration")
    shutil.copy(flat_root / "clip_index.parquet", zip_root / "clip_index.parquet")
    for group in ("camera", "lidar", "radar", "labels"):
        for sensor_dir in sorted((flat_root / group).iterdir()):
            archive_path = zip_root / group / sensor_dir.name / f"{sensor_dir.name}.chunk_{CHUNK:04d}.zip"
            archive_path.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for file_path in sorted(sensor_dir.iterdir()):
                    archive.write(file_path, arcname=file_path.name)
    return zip_root


@pytest.fixture
def physical_ai_av_zip_root(physical_ai_av_root: Path, tmp_path: Path) -> Path:
    """The dataset of ``physical_ai_av_root`` in chunk archives."""
    return write_chunk_archives(physical_ai_av_root, tmp_path / "physical_ai_av_zip")
