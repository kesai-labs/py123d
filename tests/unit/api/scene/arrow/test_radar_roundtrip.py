"""Roundtrip tests for Radar writer and reader with binary codecs (IPC, Draco, LAZ).

Radar mirrors lidar's merged storage model (one merged cloud tagged by a per-point ``RadarFeature.IDS``
sensor id, splittable per-``RadarID`` on read), but carries a single snapshot timestamp rather than a
sweep window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_radar import ArrowRadarReader, ArrowRadarWriter
from py123d.datatypes import Timestamp
from py123d.datatypes.sensors.radar import (
    Radar,
    RadarFeature,
    RadarID,
    RadarMergedMetadata,
    RadarMetadata,
    get_individual_radar,
    get_merged_radar,
)
from py123d.geometry.pose import PoseSE3

from ..conftest import make_log_metadata


def _make_merged_metadata() -> RadarMergedMetadata:
    return RadarMergedMetadata(
        radar_metadata_dict={
            RadarID.RADAR_FRONT: RadarMetadata(radar_name="RADAR_FRONT", radar_id=RadarID.RADAR_FRONT),
            RadarID.RADAR_BACK_LEFT: RadarMetadata(radar_name="RADAR_BACK_LEFT", radar_id=RadarID.RADAR_BACK_LEFT),
        }
    )


def _make_point_cloud(num_points: int = 100) -> tuple:
    """Create a random merged radar cloud whose IDS feature mixes two RadarIDs."""
    rng = np.random.RandomState(42)
    xyz = rng.randn(num_points, 3).astype(np.float32)
    ids = np.where(np.arange(num_points) % 2 == 0, int(RadarID.RADAR_FRONT), int(RadarID.RADAR_BACK_LEFT)).astype(
        np.uint8
    )
    features = {
        RadarFeature.IDS.serialize(): ids,
        RadarFeature.RCS.serialize(): rng.randn(num_points).astype(np.float32),
        RadarFeature.VELOCITY_X.serialize(): rng.randn(num_points).astype(np.float32),
        RadarFeature.VELOCITY_Y.serialize(): rng.randn(num_points).astype(np.float32),
    }
    return xyz, features


def _make_radar(ts_us: int, metadata=None, num_points: int = 100) -> Radar:
    if metadata is None:
        metadata = _make_merged_metadata()
    xyz, features = _make_point_cloud(num_points)
    return Radar(
        timestamp=Timestamp.from_us(ts_us),
        metadata=metadata,
        point_cloud_3d=xyz,
        point_cloud_features=features,
    )


def _write_and_read(log_dir: Path, radars: list, codec: str) -> pa.Table:
    metadata = _make_merged_metadata()
    writer = ArrowRadarWriter(
        log_dir=log_dir,
        metadata=metadata,
        log_metadata=make_log_metadata(),
        radar_store_option="binary",
        radar_codec=codec,
    )
    for radar in radars:
        writer.write_modality(radar)
    writer.close()
    file_path = log_dir / f"{metadata.modality_key}.arrow"
    return pa.ipc.open_file(str(file_path)).read_all()


class TestRadarBinaryRoundtrip:
    def test_single_frame_ipc(self, tmp_path: Path):
        table = _write_and_read(tmp_path, [_make_radar(1000)], codec="ipc")
        assert table.num_rows == 1

        metadata = _make_merged_metadata()
        result = ArrowRadarReader.read_at_index(0, table, metadata, "test-dataset")
        assert result is not None
        assert isinstance(result, Radar)
        assert result.timestamp.time_us == 1000
        assert result.point_cloud_3d.shape == (100, 3)
        # No end timestamp on radar.
        assert not hasattr(result, "timestamp_end")

    def test_point_cloud_preserved_ipc(self, tmp_path: Path):
        radar = _make_radar(1000)
        original_xyz = radar.point_cloud_3d.copy()
        table = _write_and_read(tmp_path, [radar], codec="ipc")
        result = ArrowRadarReader.read_at_index(0, table, _make_merged_metadata(), "test-dataset")
        np.testing.assert_array_almost_equal(result.point_cloud_3d, original_xyz, decimal=5)
        np.testing.assert_array_almost_equal(result.velocity, radar.velocity, decimal=5)

    def test_radar_id_filter(self, tmp_path: Path):
        """Reading with a radar_id kwarg returns only that sensor's points."""
        table = _write_and_read(tmp_path, [_make_radar(1000, num_points=100)], codec="ipc")
        metadata = _make_merged_metadata()
        front = ArrowRadarReader.read_at_index(0, table, metadata, "test-dataset", radar_id=RadarID.RADAR_FRONT)
        assert front is not None
        assert front.point_cloud_3d.shape == (50, 3)
        assert np.all(front.ids == int(RadarID.RADAR_FRONT))
        assert isinstance(front.metadata, RadarMetadata)
        assert front.metadata.radar_id == RadarID.RADAR_FRONT

    def test_draco_roundtrip(self, tmp_path: Path):
        table = _write_and_read(tmp_path, [_make_radar(1000)], codec="draco")
        result = ArrowRadarReader.read_at_index(0, table, _make_merged_metadata(), "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape == (100, 3)
        assert result.point_cloud_3d.dtype == np.float32

    def test_laz_roundtrip(self, tmp_path: Path):
        table = _write_and_read(tmp_path, [_make_radar(1000)], codec="laz")
        result = ArrowRadarReader.read_at_index(0, table, _make_merged_metadata(), "test-dataset")
        assert result is not None
        assert result.point_cloud_3d.shape[1] == 3

    def test_kedec_roundtrip(self, tmp_path: Path):
        """kedec codes the positions to 2 cm (spherical, per radar in the ids feature) and keeps every other feature;
        it stores the detections in its own order, so they are matched by their RCS value."""
        from py123d.common.io.codec_metadata import CODEC_METADATA_KEY, decode_codec_metadata

        radar = _make_radar(1000)
        table = _write_and_read(tmp_path, [radar], codec="kedec")
        assert decode_codec_metadata(table.schema.metadata[CODEC_METADATA_KEY])["codec"] == "kedec"
        metadata = _make_merged_metadata()
        result = ArrowRadarReader.read_at_index(0, table, metadata, "test-dataset")
        rcs = RadarFeature.RCS.serialize()
        a = np.argsort(radar.point_cloud_features[rcs])
        b = np.argsort(result.point_cloud_features[rcs])
        for name, values in radar.point_cloud_features.items():
            np.testing.assert_array_equal(result.point_cloud_features[name][b], values[a])
        errors = np.linalg.norm(result.point_cloud_3d[b].astype(float) - radar.point_cloud_3d[a].astype(float), axis=1)
        assert errors.max() < 0.02
        front = ArrowRadarReader.read_at_index(0, table, metadata, "test-dataset", radar_id=RadarID.RADAR_FRONT)
        assert front.point_cloud_3d.shape == (50, 3) and np.all(front.ids == int(RadarID.RADAR_FRONT))

    def test_multiple_frames(self, tmp_path: Path):
        radars = [_make_radar(i * 100_000) for i in range(5)]
        table = _write_and_read(tmp_path, radars, codec="ipc")
        assert table.num_rows == 5
        metadata = _make_merged_metadata()
        for i in range(5):
            result = ArrowRadarReader.read_at_index(i, table, metadata, "test-dataset")
            assert result is not None
            assert result.timestamp.time_us == i * 100_000

    def test_read_column_timestamp(self, tmp_path: Path):
        table = _write_and_read(tmp_path, [_make_radar(5000)], codec="ipc")
        ts = ArrowRadarReader.read_column_at_index(
            0, table, _make_merged_metadata(), "timestamp_us", "test-dataset", deserialize=True
        )
        assert isinstance(ts, Timestamp)
        assert ts.time_us == 5000


class TestRadarMergeSplit:
    def test_merge_then_split(self):
        rng = np.random.RandomState(0)

        def single(radar_id: RadarID, n: int) -> Radar:
            xyz = rng.randn(n, 3).astype(np.float32)
            feats = {
                RadarFeature.IDS.serialize(): np.full(n, int(radar_id), np.uint8),
                RadarFeature.RCS.serialize(): rng.randn(n).astype(np.float32),
            }
            return Radar(
                timestamp=Timestamp.from_us(100),
                metadata=RadarMetadata(
                    radar_name=radar_id.serialize(), radar_id=radar_id, radar_to_imu_se3=PoseSE3.identity()
                ),
                point_cloud_3d=xyz,
                point_cloud_features=feats,
            )

        front = single(RadarID.RADAR_FRONT, 3)
        back_left = single(RadarID.RADAR_BACK_LEFT, 2)

        merged = get_merged_radar([front, back_left])
        assert merged is not None
        assert merged.is_merged
        assert merged.point_cloud_3d.shape == (5, 3)

        split = get_individual_radar(merged, RadarID.RADAR_BACK_LEFT)
        assert split is not None
        assert split.point_cloud_3d.shape == (2, 3)
        np.testing.assert_array_almost_equal(split.point_cloud_3d, back_left.point_cloud_3d)
