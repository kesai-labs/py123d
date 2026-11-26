"""Smoke tests of the kedec benchmark (py123d.script.run_kedec_benchmark)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarWriter
from py123d.datatypes import LogMetadata, Timestamp
from py123d.datatypes.sensors.lidar import Lidar, LidarID, LidarMetadata
from py123d.geometry.pose import PoseSE3
from py123d.script import run_kedec_benchmark as benchmark


def test_matched_crf_interpolates_in_log_size():
    svt = {30: {"bytes_per_image": 2000, "psnr_y_db": 40.0}, 34: {"bytes_per_image": 1000, "psnr_y_db": 38.0}}
    crf, psnr = benchmark.matched_crf(np.sqrt(2000 * 1000), svt)
    assert abs(crf - 32.0) < 1e-9 and abs(psnr - 39.0) < 1e-9
    assert benchmark.matched_crf(5000, svt) is None


def test_lidar_benchmark_on_a_small_log(tmp_path: Path):
    metadata = LidarMetadata("top", LidarID.LIDAR_TOP, PoseSE3(x=1.0, y=0.0, z=2.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0))
    writer = ArrowLidarWriter(tmp_path, metadata, LogMetadata("test", "test_train", "log", None), "binary", "ipc_lz4")
    rng = np.random.default_rng(0)
    for i in range(3):
        xyz = (rng.normal(size=(500, 3)) * 20 + [1.0, 0.0, 2.0]).astype(np.float32)
        features = {
            "intensity": rng.integers(0, 256, 500).astype(np.uint8),
            "timestamps": np.arange(500, dtype=np.int64) + i,
        }
        writer.write_modality(
            Lidar(Timestamp.from_us(i * 100_000), Timestamp.from_us(i * 100_000 + 99_000), metadata, xyz, features)
        )
    writer.close()
    result = benchmark.benchmark_lidar(tmp_path, every=1)["lidar.lidar_top"]
    assert result["scans"] == 3 and result["points_per_scan"] == 500
    position = result["modes"]["spherical_position"]
    assert position["mode"] == "spherical" and 0 < position["largest_error_mm"] < 20
    assert set(position["bytes_per_point_by_part"]) >= {"@r", "@a", "@e", "intensity", "timestamps"}
