"""Equivalence tests for the in-memory sqlite caches of the tar parser.

The caches replace per-image devkit queries (fresh connection per call, full
ego_pose scans); these tests pin their behavior to the upstream helpers on a
synthetic log DB.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nuplan")  # the nuPlan devkit (extra "nuplan")

from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens  # noqa: E402

from py123d.parser.nuplan.nuplan_tar_parser import _CameraImageIndex, _EgoPoseInterpolator  # noqa: E402
from py123d.parser.nuplan.utils.nuplan_sql_helper import get_interpolated_ego_pose_from_db  # noqa: E402

EGO_T0_US = 1_000_000
EGO_DT_US = 10_000  # 100 Hz
NUM_EGO_POSES = 200

CHANNELS = ["CAM_F0", "CAM_L0"]
IMAGE_T0_US = EGO_T0_US + 5_000
IMAGE_DT_US = 100_000  # 10 Hz
NUM_IMAGES_PER_CHANNEL = 15

LIDAR_T0_US = EGO_T0_US
LIDAR_DT_US = 50_000  # 20 Hz
NUM_LIDAR_PCS = 30


def _token(i: int) -> bytes:
    return i.to_bytes(8, "big")


@pytest.fixture
def log_db(tmp_path: Path) -> Path:
    """A minimal nuPlan log DB: jittered ego poses, 2 camera channels, 20 Hz lidar_pcs."""
    db_path = tmp_path / "synthetic_log.db"
    rng = np.random.default_rng(0)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE ego_pose (timestamp INTEGER, x REAL, y REAL, z REAL,
                                   qw REAL, qx REAL, qy REAL, qz REAL);
            CREATE TABLE camera (token BLOB PRIMARY KEY, channel TEXT);
            CREATE TABLE image (token BLOB PRIMARY KEY, next_token BLOB, prev_token BLOB,
                                ego_pose_token BLOB, camera_token BLOB,
                                filename_jpg TEXT, timestamp INTEGER);
            CREATE TABLE lidar_pc (token BLOB PRIMARY KEY, timestamp INTEGER);
            """
        )
        for i in range(NUM_EGO_POSES):
            quaternion = rng.normal(size=4)
            quaternion /= np.linalg.norm(quaternion)
            quaternion *= np.sign(quaternion[0]) or 1.0
            connection.execute(
                "INSERT INTO ego_pose VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (EGO_T0_US + i * EGO_DT_US, float(i) * 0.5, float(i) * -0.25, 0.1 * i, *quaternion),
            )
        for c, channel in enumerate(CHANNELS):
            connection.execute("INSERT INTO camera VALUES (?, ?)", (_token(1000 + c), channel))
            for i in range(NUM_IMAGES_PER_CHANNEL):
                connection.execute(
                    "INSERT INTO image VALUES (?, NULL, NULL, ?, ?, ?, ?)",
                    (
                        _token(2000 + c * 100 + i),
                        _token(4000 + i),  # ego_pose_token must be non-NULL for Image.from_db_row
                        _token(1000 + c),
                        f"log/{channel}/{i}.jpg",
                        IMAGE_T0_US + i * IMAGE_DT_US,
                    ),
                )
        for i in range(NUM_LIDAR_PCS):
            connection.execute("INSERT INTO lidar_pc VALUES (?, ?)", (_token(3000 + i), LIDAR_T0_US + i * LIDAR_DT_US))
    return db_path


def test_ego_interpolator_matches_upstream(log_db: Path) -> None:
    interpolator = _EgoPoseInterpolator.from_log_db(log_db)
    last_ts = EGO_T0_US + (NUM_EGO_POSES - 1) * EGO_DT_US
    query_timestamps = [
        EGO_T0_US - 12_345,  # before first pose -> clamp
        EGO_T0_US,  # exact first
        EGO_T0_US + 3 * EGO_DT_US,  # exact interior
        EGO_T0_US + 3 * EGO_DT_US + 1,  # just past a pose
        EGO_T0_US + 7 * EGO_DT_US + 4_999,  # mid-bracket
        last_ts,  # exact last
        last_ts + 54_321,  # past last pose -> clamp
    ]
    for timestamp_us in query_timestamps:
        expected = get_interpolated_ego_pose_from_db(str(log_db), timestamp_us)
        actual = interpolator.interpolate(timestamp_us)
        np.testing.assert_allclose(actual.array, expected.array, atol=1e-12, err_msg=f"ts={timestamp_us}")


def test_ego_interpolator_rejects_empty_table(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE ego_pose (timestamp INTEGER, x REAL, y REAL, z REAL, qw REAL, qx REAL, qy REAL, qz REAL)"
        )
    with pytest.raises(ValueError, match="No ego poses"):
        _EgoPoseInterpolator.from_log_db(db_path)


def test_image_index_matches_devkit(log_db: Path) -> None:
    image_index = _CameraImageIndex.from_log_db(log_db)
    with sqlite3.connect(log_db) as connection:
        lidar_rows = connection.execute("SELECT token, timestamp FROM lidar_pc ORDER BY timestamp").fetchall()

    for token, lidar_timestamp_us in lidar_rows:
        for channel in CHANNELS:
            expected = list(get_images_from_lidar_tokens(str(log_db), [token.hex()], [channel]))
            actual = image_index.find(channel, lidar_timestamp_us)
            if not expected:
                assert actual is None, f"ts={lidar_timestamp_us} {channel}"
            else:
                assert len(expected) == 1  # devkit window holds at most one 10 Hz image
                assert actual == (expected[0].timestamp, expected[0].filename_jpg)


def test_image_index_unknown_channel_and_out_of_range(log_db: Path) -> None:
    image_index = _CameraImageIndex.from_log_db(log_db)
    assert image_index.find("CAM_R0", IMAGE_T0_US) is None
    assert image_index.find("CAM_F0", IMAGE_T0_US - 10 * IMAGE_DT_US) is None
