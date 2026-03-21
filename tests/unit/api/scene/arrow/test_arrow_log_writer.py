"""Tests for ArrowLogWriter — write_sync, write_async, sync table construction."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter, SyncConfig
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.datatypes import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D
from py123d.parser.base_dataset_parser import ModalitiesSync

from ..conftest import make_ego_metadata, make_log_metadata, make_traffic_light_metadata


def _make_ego(ts_us: int) -> EgoStateSE3:
    metadata = make_ego_metadata()
    pose = PoseSE3.identity()
    dynamic = DynamicStateSE3(
        velocity=Vector3D(1.0, 0.0, 0.0),
        acceleration=Vector3D(0.0, 0.0, 0.0),
        angular_velocity=Vector3D(0.0, 0.0, 0.0),
    )
    return EgoStateSE3.from_imu(
        imu_se3=pose, metadata=metadata, timestamp=Timestamp.from_us(ts_us), dynamic_state_se3=dynamic
    )


# ===========================================================================
# SyncConfig
# ===========================================================================


class TestSyncConfig:
    def test_reference_modality_parses(self):
        cfg = SyncConfig(reference_column="lidar.lidar_merged.timestamp_us")
        assert cfg.reference_modality == "lidar.lidar_merged"

    def test_reference_modality_simple_key(self):
        cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us")
        assert cfg.reference_modality == "ego_state_se3"

    def test_reference_modality_no_dot_raises(self):
        """W6: Keys without dots cause ValueError at construction time."""
        with pytest.raises(ValueError, match="reference_column must be"):
            SyncConfig(reference_column="nodot")

    def test_reference_column_empty_parts_raises(self):
        with pytest.raises(ValueError, match="reference_column must be"):
            SyncConfig(reference_column=".timestamp_us")

    def test_reference_column_wrong_field_raises(self):
        """W6: timestamp field must end with 'timestamp_us'."""
        with pytest.raises(ValueError, match="timestamp field must end with"):
            SyncConfig(reference_column="ego_state_se3.imu_se3")

    def test_reference_timestamp_field(self):
        cfg = SyncConfig(reference_column="lidar.lidar_merged.end_timestamp_us")
        assert cfg.reference_timestamp_field == "end_timestamp_us"

    def test_wrong_reference_column_raises_on_build(self, tmp_path: Path):
        """W6: If the reference column is not in the Arrow file, raise an explicit error."""
        config = LogWriterConfig()
        # Reference column points to a field that doesn't exist in the ego table
        sync_cfg = SyncConfig(reference_column="ego_state_se3.nonexistent_timestamp_us")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)
        writer.write_async(_make_ego(100_000))
        with pytest.raises(ValueError, match="Timestamp column.*not found"):
            writer.close()


# ===========================================================================
# Reset
# ===========================================================================


class TestReset:
    def _make_writer(self, tmp_path: Path) -> ArrowLogWriter:
        config = LogWriterConfig()
        return ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path)

    def test_creates_log_dir(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        log_meta = make_log_metadata()
        result = writer.reset(log_meta)
        assert result is True
        log_dir = tmp_path / log_meta.split / log_meta.log_name
        assert log_dir.exists()
        # Write at least one sync frame before close (close without writes requires SyncConfig)
        ego = _make_ego(0)
        writer.write_sync(ModalitiesSync(timestamp=Timestamp.from_us(0), modalities=[ego]))
        writer.close()

    def test_existing_sync_skips(self, tmp_path: Path):
        """If sync.arrow exists and force=False, reset returns False."""
        log_meta = make_log_metadata()
        log_dir = tmp_path / log_meta.split / log_meta.log_name
        log_dir.mkdir(parents=True)
        (log_dir / "sync.arrow").touch()

        writer = self._make_writer(tmp_path)
        result = writer.reset(log_meta)
        assert result is False

    def test_force_overwrites(self, tmp_path: Path):
        log_meta = make_log_metadata()
        log_dir = tmp_path / log_meta.split / log_meta.log_name
        log_dir.mkdir(parents=True)
        (log_dir / "sync.arrow").touch()

        config = LogWriterConfig(force_log_conversion=True)
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path)
        result = writer.reset(log_meta)
        assert result is True
        ego = _make_ego(0)
        writer.write_sync(ModalitiesSync(timestamp=Timestamp.from_us(0), modalities=[ego]))
        writer.close()

    def test_double_reset_asserts(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        log_meta = make_log_metadata()
        writer.reset(log_meta)
        with pytest.raises(AssertionError, match="already initialized"):
            writer.reset(log_meta)
        # Write a frame so close() can build sync table
        ego = _make_ego(0)
        writer.write_sync(ModalitiesSync(timestamp=Timestamp.from_us(0), modalities=[ego]))
        writer.close()


# ===========================================================================
# write_sync
# ===========================================================================


class TestWriteSync:
    def _make_writer(self, tmp_path: Path) -> ArrowLogWriter:
        config = LogWriterConfig()
        return ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path)

    def test_single_ego_frame(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        ego = _make_ego(100_000)
        sync = ModalitiesSync(timestamp=Timestamp.from_us(100_000), modalities=[ego])
        writer.write_sync(sync)
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 1
        assert "ego_state_se3" in sync_table.column_names

    def test_multiple_frames(self, tmp_path: Path):
        writer = self._make_writer(tmp_path)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(5):
            ego = _make_ego(i * 100_000)
            sync = ModalitiesSync(timestamp=Timestamp.from_us(i * 100_000), modalities=[ego])
            writer.write_sync(sync)
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 5

    def test_missing_modality_null_in_sync(self, tmp_path: Path):
        """When a modality is present in some frames but not others, sync should have nulls."""
        writer = self._make_writer(tmp_path)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        # Frame 0: ego only
        ego0 = _make_ego(0)
        writer.write_sync(ModalitiesSync(timestamp=Timestamp.from_us(0), modalities=[ego0]))

        # Frame 1: ego + traffic_light (introduce a new modality)
        from py123d.datatypes import TrafficLightDetection, TrafficLightDetections, TrafficLightStatus

        tl_meta = make_traffic_light_metadata()
        ego1 = _make_ego(100_000)
        tl1 = TrafficLightDetections(
            detections=[TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)],
            timestamp=Timestamp.from_us(100_000),
            metadata=tl_meta,
        )
        writer.write_sync(ModalitiesSync(timestamp=Timestamp.from_us(100_000), modalities=[ego1, tl1]))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 2

        # Traffic light column: frame 0 should be null, frame 1 should be 0
        tl_key = tl_meta.modality_key
        assert tl_key in sync_table.column_names
        assert sync_table[tl_key][0].as_py() is None
        assert sync_table[tl_key][1].as_py() == 0


# ===========================================================================
# write_async
# ===========================================================================


class TestWriteAsync:
    def test_async_creates_modality_file(self, tmp_path: Path):
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        ego = _make_ego(100_000)
        writer.write_async(ego)
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        assert (log_dir / "ego_state_se3.arrow").exists()
        assert (log_dir / "sync.arrow").exists()

    def test_sync_then_async_asserts(self, tmp_path: Path):
        """write_sync first then write_async should assert (sync_rows is not None)."""
        config = LogWriterConfig()
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        ego = _make_ego(100_000)
        writer.write_sync(ModalitiesSync(timestamp=Timestamp.from_us(100_000), modalities=[ego]))
        with pytest.raises(AssertionError, match="write_async.*after.*write_sync"):
            writer.write_async(_make_ego(200_000))
        writer.close()


# ===========================================================================
# Close
# ===========================================================================


class TestClose:
    def test_close_without_reset_noop(self, tmp_path: Path):
        """Close without reset should not crash."""
        config = LogWriterConfig()
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path)
        writer.close()  # Should be safe

    def test_close_writes_sync_from_rows(self, tmp_path: Path):
        config = LogWriterConfig()
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        ego = _make_ego(0)
        writer.write_sync(ModalitiesSync(timestamp=Timestamp.from_us(0), modalities=[ego]))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        assert (log_dir / "sync.arrow").exists()


# ===========================================================================
# Deferred sync table
# ===========================================================================


class TestDeferredSync:
    def test_forward_direction(self, tmp_path: Path):
        """Write ego frames async with forward sync, verify sync table built correctly."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us", direction="forward")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(5):
            writer.write_async(_make_ego(i * 100_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 5
        assert "ego_state_se3" in sync_table.column_names

    def test_backward_direction(self, tmp_path: Path):
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us", direction="backward")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(5):
            writer.write_async(_make_ego(i * 100_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 5

    def test_no_reference_raises(self, tmp_path: Path):
        """If reference modality is missing, raise ValueError."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="lidar.lidar_merged.timestamp_us")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        # Write ego data, but reference is lidar (which we don't write)
        writer.write_async(_make_ego(100_000))
        with pytest.raises(ValueError, match="Reference modality.*not found"):
            writer.close()

    def test_forward_sync_indices_are_correct(self, tmp_path: Path):
        """Verify forward sync picks the first observation in [ref_ts, next_ref_ts)."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us", direction="forward")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        # Ego timestamps: 0, 100_000, 200_000, 300_000, 400_000
        for i in range(5):
            writer.write_async(_make_ego(i * 100_000))

        # Traffic lights at 50_000, 150_000, 250_000 (shifted by half an interval)
        from py123d.datatypes import TrafficLightDetection, TrafficLightDetections, TrafficLightStatus

        tl_meta = make_traffic_light_metadata()
        for ts_us in [50_000, 150_000, 250_000]:
            tl = TrafficLightDetections(
                detections=[TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)],
                timestamp=Timestamp.from_us(ts_us),
                metadata=tl_meta,
            )
            writer.write_async(tl)
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        tl_col = sync_table.column(tl_meta.modality_key).to_pylist()

        # Forward: [0, 100k) -> tl@50k (idx 0), [100k, 200k) -> tl@150k (idx 1),
        #          [200k, 300k) -> tl@250k (idx 2), [300k, 400k) -> None, [400k, inf) -> None
        assert tl_col == [0, 1, 2, None, None]

    def test_backward_sync_indices_are_correct(self, tmp_path: Path):
        """Verify backward sync picks the last observation in (prev_ref_ts, ref_ts]."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us", direction="backward")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(5):
            writer.write_async(_make_ego(i * 100_000))

        from py123d.datatypes import TrafficLightDetection, TrafficLightDetections, TrafficLightStatus

        tl_meta = make_traffic_light_metadata()
        for ts_us in [50_000, 150_000, 250_000]:
            tl = TrafficLightDetections(
                detections=[TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)],
                timestamp=Timestamp.from_us(ts_us),
                metadata=tl_meta,
            )
            writer.write_async(tl)
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        tl_col = sync_table.column(tl_meta.modality_key).to_pylist()

        # Backward: (-inf, 0] -> None, (0, 100k] -> tl@50k (idx 0),
        #           (100k, 200k] -> tl@150k (idx 1), (200k, 300k] -> tl@250k (idx 2), (300k, 400k] -> None
        assert tl_col == [None, 0, 1, 2, None]

    def test_self_sync_identity(self, tmp_path: Path):
        """Reference modality synced against itself should produce identity mapping (0, 1, 2, ...)."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us", direction="forward")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        n = 10
        for i in range(n):
            writer.write_async(_make_ego(i * 100_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        ego_col = sync_table.column("ego_state_se3").to_pylist()
        assert ego_col == list(range(n))

    def test_non_monotonic_timestamps_raise(self, tmp_path: Path):
        """Non-monotonic timestamps are caught during close (writer validates before sync build)."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        writer.write_async(_make_ego(100_000))
        writer.write_async(_make_ego(0))

        with pytest.raises(ValueError, match="monotonically"):
            writer.close()

    def test_sync_timestamps_column(self, tmp_path: Path):
        """Sync table timestamp_us column should match reference modality timestamps."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(reference_column="ego_state_se3.timestamp_us", direction="forward")
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        expected_ts = [0, 100_000, 200_000]
        for ts in expected_ts:
            writer.write_async(_make_ego(ts))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        actual_ts = sync_table.column("sync.timestamp_us").to_pylist()
        assert actual_ts == expected_ts


# ===========================================================================
# Deferred sync table — stride / thinning
# ===========================================================================


class TestDeferredSyncStride:
    def test_stride_2_thins_sync_table(self, tmp_path: Path):
        """Stride=2 on 10 ego frames should produce 5 sync rows."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us", direction="forward", target_iteration_stride=2
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(10):
            writer.write_async(_make_ego(i * 100_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 5

    def test_stride_timestamps_column(self, tmp_path: Path):
        """Sync timestamps should correspond to kept reference timestamps only."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us", direction="forward", target_iteration_stride=2
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(10):
            writer.write_async(_make_ego(i * 100_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        actual_ts = sync_table.column("sync.timestamp_us").to_pylist()
        assert actual_ts == [0, 200_000, 400_000, 600_000, 800_000]

    def test_duration_based_stride(self, tmp_path: Path):
        """target_iteration_duration_s=0.1 on a 20Hz log should produce stride=2."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us",
            direction="forward",
            target_iteration_duration_s=0.1,
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        # 20 frames at 50ms intervals = 20Hz
        for i in range(20):
            writer.write_async(_make_ego(i * 50_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 10

    def test_duration_takes_priority_over_stride(self, tmp_path: Path):
        """When both are set, target_iteration_duration_s wins."""
        config = LogWriterConfig()
        # duration=0.2s on 10Hz → stride=2; explicit stride=5 should be ignored
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us",
            direction="forward",
            target_iteration_stride=5,
            target_iteration_duration_s=0.2,
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(10):
            writer.write_async(_make_ego(i * 100_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 5  # stride=2, not stride=5

    def test_stride_1_no_thinning(self, tmp_path: Path):
        """Stride=1 should produce the same result as no stride."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us", direction="forward", target_iteration_stride=1
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(5):
            writer.write_async(_make_ego(i * 100_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        assert sync_table.num_rows == 5

    def test_self_sync_with_stride(self, tmp_path: Path):
        """Reference modality against itself with stride=2 should produce 0, 2, 4, ... indices."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us", direction="forward", target_iteration_stride=2
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(10):
            writer.write_async(_make_ego(i * 100_000))
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        ego_col = sync_table.column("ego_state_se3").to_pylist()
        assert ego_col == [0, 2, 4, 6, 8]

    def test_stride_forward_indices(self, tmp_path: Path):
        """Forward sync with stride=2: wider intervals should capture more observations."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us", direction="forward", target_iteration_stride=2
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        # Ego at 0, 100k, 200k, 300k, 400k, 500k (6 frames, stride=2 → kept: 0, 200k, 400k)
        for i in range(6):
            writer.write_async(_make_ego(i * 100_000))

        # Traffic lights at 50k, 150k, 250k, 350k, 450k
        from py123d.datatypes import TrafficLightDetection, TrafficLightDetections, TrafficLightStatus

        tl_meta = make_traffic_light_metadata()
        for ts_us in [50_000, 150_000, 250_000, 350_000, 450_000]:
            tl = TrafficLightDetections(
                detections=[TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)],
                timestamp=Timestamp.from_us(ts_us),
                metadata=tl_meta,
            )
            writer.write_async(tl)
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        tl_col = sync_table.column(tl_meta.modality_key).to_pylist()

        # Kept ref timestamps: 0, 200k, 400k
        # Forward intervals: [0, 200k) → tl@50k (idx 0), [200k, 400k) → tl@250k (idx 2), [400k, inf) → tl@450k (idx 4)
        assert tl_col == [0, 2, 4]

    def test_stride_backward_indices(self, tmp_path: Path):
        """Backward sync with stride=2: wider intervals should capture observations."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us", direction="backward", target_iteration_stride=2
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(6):
            writer.write_async(_make_ego(i * 100_000))

        from py123d.datatypes import TrafficLightDetection, TrafficLightDetections, TrafficLightStatus

        tl_meta = make_traffic_light_metadata()
        for ts_us in [50_000, 150_000, 250_000, 350_000, 450_000]:
            tl = TrafficLightDetections(
                detections=[TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)],
                timestamp=Timestamp.from_us(ts_us),
                metadata=tl_meta,
            )
            writer.write_async(tl)
        writer.close()

        log_dir = tmp_path / log_meta.split / log_meta.log_name
        sync_table = pa.ipc.open_file(str(log_dir / "sync.arrow")).read_all()
        tl_col = sync_table.column(tl_meta.modality_key).to_pylist()

        # Kept ref timestamps: 0, 200k, 400k
        # Backward intervals: (-inf, 0] → None, (0, 200k] → tl@150k (idx 1), (200k, 400k] → tl@350k (idx 3)
        assert tl_col == [None, 1, 3]

    def test_infeasible_duration_raises(self, tmp_path: Path):
        """target_iteration_duration_s smaller than raw duration should raise ValueError."""
        config = LogWriterConfig()
        sync_cfg = SyncConfig(
            reference_column="ego_state_se3.timestamp_us",
            direction="forward",
            target_iteration_duration_s=0.01,  # 10ms on a 10Hz (100ms) log → impossible
        )
        writer = ArrowLogWriter(config, logs_root=tmp_path, sensors_root=tmp_path, sync_config=sync_cfg)
        log_meta = make_log_metadata()
        writer.reset(log_meta)

        for i in range(5):
            writer.write_async(_make_ego(i * 100_000))

        with pytest.raises(ValueError, match="Cannot achieve target_iteration_duration_s"):
            writer.close()

    def test_stride_validation_in_sync_config(self):
        """Invalid stride/duration values should raise at SyncConfig construction."""
        with pytest.raises(ValueError, match="target_iteration_stride must be >= 1"):
            SyncConfig(reference_column="ego_state_se3.timestamp_us", target_iteration_stride=0)

        with pytest.raises(ValueError, match="target_iteration_stride must be >= 1"):
            SyncConfig(reference_column="ego_state_se3.timestamp_us", target_iteration_stride=-1)

        with pytest.raises(ValueError, match="target_iteration_duration_s must be > 0"):
            SyncConfig(reference_column="ego_state_se3.timestamp_us", target_iteration_duration_s=0.0)

        with pytest.raises(ValueError, match="target_iteration_duration_s must be > 0"):
            SyncConfig(reference_column="ego_state_se3.timestamp_us", target_iteration_duration_s=-1.0)
