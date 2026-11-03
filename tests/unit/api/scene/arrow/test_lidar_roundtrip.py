"""Roundtrip tests for Lidar writer and reader with binary codecs (IPC, Draco, LAZ, kedec)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader, ArrowLidarWriter
from py123d.datatypes import Timestamp
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarID, LidarMergedMetadata, LidarMetadata
from py123d.geometry.pose import PoseSE3

from ..conftest import make_log_metadata


def _make_lidar_metadata(lidar_id: LidarID = LidarID.LIDAR_TOP) -> LidarMetadata:
    return LidarMetadata(
        lidar_name="top_lidar",
        lidar_id=lidar_id,
        lidar_to_imu_se3=PoseSE3.identity(),
    )


def _make_merged_metadata() -> LidarMergedMetadata:
    top = LidarMetadata(lidar_name="top_lidar", lidar_id=LidarID.LIDAR_TOP)
    return LidarMergedMetadata(lidar_metadata_dict={LidarID.LIDAR_TOP: top})


def _make_point_cloud(num_points: int = 100) -> tuple:
    """Create a random point cloud with xyz + features."""
    rng = np.random.RandomState(42)
    xyz = rng.randn(num_points, 3).astype(np.float32)
    features = {
        LidarFeature.INTENSITY.serialize(): rng.randint(0, 255, size=num_points).astype(np.float32),
    }
    return xyz, features


def _make_lidar(ts_us: int, metadata=None, num_points: int = 100) -> Lidar:
    if metadata is None:
        metadata = _make_merged_metadata()
    xyz, features = _make_point_cloud(num_points)
    return Lidar(
        timestamp=Timestamp.from_us(ts_us),
        timestamp_end=Timestamp.from_us(ts_us + 50_000),
        metadata=metadata,
        point_cloud_3d=xyz,
        point_cloud_features=features,
    )


class TestLidarIpcBinaryRoundtrip:
    """Write lidar data with IPC binary codec, read back."""

    def _write_and_read(self, log_dir: Path, lidars: list, codec: str = "ipc") -> pa.Table:
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=log_dir,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec=codec,
        )
        for lidar in lidars:
            writer.write_modality(lidar)
        writer.close()
        file_path = log_dir / f"{metadata.modality_key}.arrow"
        return pa.ipc.open_file(str(file_path)).read_all()

    def test_single_frame_ipc(self, tmp_path: Path):
        lidar = _make_lidar(1000)
        table = self._write_and_read(tmp_path, [lidar], codec="ipc")
        assert table.num_rows == 1

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert isinstance(result, Lidar)
        assert result.timestamp.time_us == 1000
        assert result.point_cloud_3d.shape == (100, 3)

    def test_ipc_zstd_codec(self, tmp_path: Path):
        lidar = _make_lidar(1000)
        table = self._write_and_read(tmp_path, [lidar], codec="ipc_zstd")

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape == (100, 3)

    def test_ipc_lz4_codec(self, tmp_path: Path):
        lidar = _make_lidar(1000)
        table = self._write_and_read(tmp_path, [lidar], codec="ipc_lz4")

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape == (100, 3)

    def test_point_cloud_data_preserved(self, tmp_path: Path):
        """Verify xyz coordinates survive IPC roundtrip (lossless)."""
        lidar = _make_lidar(1000)
        original_xyz = lidar.point_cloud_3d.copy()
        table = self._write_and_read(tmp_path, [lidar], codec="ipc")

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        np.testing.assert_array_almost_equal(result.point_cloud_3d, original_xyz, decimal=5)

    def test_multiple_frames(self, tmp_path: Path):
        lidars = [_make_lidar(i * 100_000) for i in range(5)]
        table = self._write_and_read(tmp_path, lidars, codec="ipc")
        assert table.num_rows == 5

        metadata = _make_merged_metadata()
        for i in range(5):
            result = ArrowLidarReader.read_at_index(i, table, metadata, "test-dataset")
            assert result is not None
            assert result.timestamp.time_us == i * 100_000

    def test_timestamps_roundtrip(self, tmp_path: Path):
        """Verify start and end timestamps survive roundtrip."""
        lidar = _make_lidar(1000)
        table = self._write_and_read(tmp_path, [lidar], codec="ipc")

        metadata = _make_merged_metadata()
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.timestamp.time_us == 1000
        assert result.timestamp_end.time_us == 51_000


class TestLidarDracoRoundtrip:
    """Write lidar data with Draco codec, read back."""

    def test_draco_roundtrip(self, tmp_path: Path):
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec="draco",
        )
        lidar = _make_lidar(1000)
        writer.write_modality(lidar)
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape == (100, 3)
        # Draco is lossy for coordinates, just check approximate shape
        assert result.point_cloud_3d.dtype == np.float32


class TestLidarLazRoundtrip:
    """Write lidar data with LAZ codec, read back."""

    def test_laz_roundtrip(self, tmp_path: Path):
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec="laz",
        )
        lidar = _make_lidar(1000)
        writer.write_modality(lidar)
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape[1] == 3
        assert result.point_cloud_3d.dtype == np.float32


class TestLidarReadColumn:
    def test_read_column_timestamp_deserialized(self, tmp_path: Path):
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec="ipc",
        )
        writer.write_modality(_make_lidar(5000))
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        ts = ArrowLidarReader.read_column_at_index(0, table, metadata, "timestamp_us", "test-dataset", deserialize=True)
        assert isinstance(ts, Timestamp)
        assert ts.time_us == 5000

    def test_read_column_end_timestamp(self, tmp_path: Path):
        metadata = _make_merged_metadata()
        log_meta = make_log_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=log_meta,
            lidar_store_option="binary",
            lidar_codec="ipc",
        )
        writer.write_modality(_make_lidar(5000))
        writer.close()

        file_path = tmp_path / f"{metadata.modality_key}.arrow"
        table = pa.ipc.open_file(str(file_path)).read_all()

        ts = ArrowLidarReader.read_column_at_index(
            0, table, metadata, "end_timestamp_us", "test-dataset", deserialize=True
        )
        assert isinstance(ts, Timestamp)
        assert ts.time_us == 55_000  # 5000 + 50_000


# ------------------------------------------------------------------------------------------------------------------
# kedec
# ------------------------------------------------------------------------------------------------------------------


def _kedec_lidar(ts_us: int, metadata, num_points: int = 2000, seed: int = 0, ids=None) -> Lidar:
    rng = np.random.RandomState(seed)
    xyz = rng.uniform(-60.0, 60.0, (num_points, 3)).astype(np.float32)
    features = {
        LidarFeature.INTENSITY.serialize(): rng.randint(0, 256, num_points).astype(np.uint8),
        LidarFeature.TIMESTAMPS.serialize(): ts_us + rng.randint(0, 100_000, num_points).astype(np.int64),
        "uid": np.arange(num_points, dtype=np.int64),
    }
    if ids is not None:
        features[LidarFeature.IDS.serialize()] = ids
    return Lidar(
        timestamp=Timestamp.from_us(ts_us),
        timestamp_end=Timestamp.from_us(ts_us + 100_000),
        metadata=metadata,
        point_cloud_3d=xyz,
        point_cloud_features=features,
    )


class TestLidarKedecRoundtrip:
    """Write lidar data with the kedec codec, read back through the reader."""

    def _write(self, log_dir: Path, metadata, lidars, encode_threads: int = 1, config=None) -> pa.Table:
        writer = ArrowLidarWriter(
            log_dir=log_dir,
            metadata=metadata,
            log_metadata=make_log_metadata(),
            lidar_store_option="binary",
            lidar_codec="kedec",
            lidar_codec_config=config,
            encode_threads=encode_threads,
        )
        for lidar in lidars:
            writer.write_modality(lidar)
        writer.close()
        return pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()

    def test_roundtrip_within_bound_and_exact_features(self, tmp_path: Path):
        pose = PoseSE3(x=1.2, y=0.0, z=1.9, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        metadata = LidarMetadata(lidar_name="top", lidar_id=LidarID.LIDAR_TOP, lidar_to_imu_se3=pose)
        lidar = _kedec_lidar(1_000_000, metadata)
        table = self._write(tmp_path, metadata, [lidar])
        result = ArrowLidarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None and result.point_cloud_3d.dtype == np.float32
        uid = result.point_cloud_features["uid"]
        errors = np.linalg.norm(result.point_cloud_3d - lidar.point_cloud_3d[uid], axis=1)
        assert errors.max() < 0.02
        for name in ("intensity", "timestamps"):
            np.testing.assert_array_equal(result.point_cloud_features[name], lidar.point_cloud_features[name][uid])

    def test_codec_schema_metadata(self, tmp_path: Path):
        from py123d.common.io.lidar.point_cloud_codec_registry import CODEC_METADATA_KEY, decode_codec_metadata

        metadata = _make_lidar_metadata()
        self._write(tmp_path, metadata, [_kedec_lidar(0, metadata)])
        schema = pa.ipc.open_file(str(tmp_path / f"{metadata.modality_key}.arrow")).schema
        description = decode_codec_metadata(schema.metadata[CODEC_METADATA_KEY])
        assert description["codec"] == "kedec" and description["version"] == 1
        assert description["options"]["kedec_zstd_level"] == 19
        assert b"metadata" in schema.metadata

    def test_merged_cloud_uses_each_sensor_pose(self, tmp_path: Path):
        from py123d.common.io.lidar.kedec_lidar_io import read_kedec_head

        top_pose = PoseSE3(x=1.0, y=0.0, z=2.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        front_pose = PoseSE3(x=3.5, y=0.0, z=0.5, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        merged = LidarMergedMetadata(
            lidar_metadata_dict={
                LidarID.LIDAR_TOP: LidarMetadata("top", LidarID.LIDAR_TOP, top_pose),
                LidarID.LIDAR_FRONT: LidarMetadata("front", LidarID.LIDAR_FRONT, front_pose),
            }
        )
        ids = np.where(np.arange(2000) % 2 == 0, int(LidarID.LIDAR_TOP.value), int(LidarID.LIDAR_FRONT.value))
        lidar = _kedec_lidar(0, merged, ids=ids.astype(np.uint8))
        table = self._write(tmp_path, merged, [lidar])
        blob = table[f"{merged.modality_key}.data"][0].as_py()
        head, _ = read_kedec_head(blob)
        poses = {section["id"]: section["pose"] for section in head["sections"]}
        assert poses == {
            int(LidarID.LIDAR_TOP.value): top_pose.tolist(),
            int(LidarID.LIDAR_FRONT.value): front_pose.tolist(),
        }
        front = ArrowLidarReader.read_at_index(0, table, merged, "test-dataset", lidar_id=LidarID.LIDAR_FRONT)
        assert front is not None and len(front.point_cloud_3d) == 1000

    def test_threads_give_identical_files(self, tmp_path: Path):
        """Encoding in threads writes the same rows in the same order as encoding in the calling thread."""
        metadata = _make_lidar_metadata()
        lidars = [_kedec_lidar(i * 100_000, metadata, num_points=500, seed=i) for i in range(12)]
        (tmp_path / "one").mkdir()
        (tmp_path / "four").mkdir()
        one = self._write(tmp_path / "one", metadata, lidars, encode_threads=1)
        four = self._write(tmp_path / "four", metadata, lidars, encode_threads=4)
        assert one.equals(four)

    def test_row_count_includes_rows_being_encoded(self, tmp_path: Path):
        metadata = _make_lidar_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=make_log_metadata(),
            lidar_store_option="binary",
            lidar_codec="kedec",
            encode_threads=3,
        )
        for i in range(5):
            assert writer.row_count == i
            writer.write_modality(_kedec_lidar(i * 100_000, metadata, num_points=300, seed=i))
        assert writer.row_count == 5
        writer.close()
        assert writer.row_count == 5 and writer._executor is None

    def test_encoding_error_closes_the_file(self, tmp_path: Path):
        """An encoder error is raised; the rows before it are written and the threads stop."""
        metadata = _make_lidar_metadata()
        writer = ArrowLidarWriter(
            log_dir=tmp_path,
            metadata=metadata,
            log_metadata=make_log_metadata(),
            lidar_store_option="binary",
            lidar_codec="kedec",
            encode_threads=2,
        )
        writer.write_modality(_kedec_lidar(0, metadata, num_points=100))
        broken = _kedec_lidar(100_000, metadata, num_points=100)
        broken.point_cloud_features["@bad"] = np.zeros(100)
        with pytest.raises(ValueError):
            writer.write_modality(broken)
            writer.write_modality(_kedec_lidar(200_000, metadata, num_points=100))
            writer.close()
        if writer._executor is not None:
            with pytest.raises(ValueError):
                writer.close()
        assert writer._executor is None
        table = pa.ipc.open_file(str(tmp_path / f"{metadata.modality_key}.arrow")).read_all()
        assert table.num_rows == 1

    def test_kedec_is_a_lidar_and_a_radar_codec(self):
        from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig

        config = LogWriterConfig(
            lidar_store_option="binary", lidar_codec="kedec", radar_store_option="binary", radar_codec="kedec"
        )
        assert config.lidar_codec == "kedec" and config.radar_codec == "kedec"
        with pytest.raises(AssertionError):
            LogWriterConfig(radar_store_option="binary", radar_codec="not_a_codec")

    def test_log_writer_config_accepts_mappings(self):
        from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
        from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig

        config = LogWriterConfig(
            lidar_store_option="binary",
            lidar_codec="kedec",
            lidar_codec_config={"kedec_angle_step_deg": 0.01, "laz_scales": [0.02, 0.02, 0.02]},
            encode_threads=4,
        )
        assert isinstance(config.lidar_codec_config, PointCloudCodecConfig)
        assert config.lidar_codec_config.kedec_angle_step_deg == 0.01
        assert config.lidar_codec_config.laz_scales == (0.02, 0.02, 0.02)
        with pytest.raises(AssertionError):
            LogWriterConfig(encode_threads=0)
