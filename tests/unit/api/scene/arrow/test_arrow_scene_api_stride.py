"""Tests for stride-aware iteration indexing in ArrowSceneAPI.

These tests create real Arrow files on disk to exercise the full API path including
_get_sync_index, get_all_iteration_timestamps, and get_scene_timestamp_boundaries.
"""

import uuid
from pathlib import Path
from typing import List, Optional

import msgpack
import numpy as np
import pyarrow as pa
import pytest
from pyarrow import ipc

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata

# --- Helpers ---


def _make_log_metadata(timestep_seconds: float = 0.1) -> LogMetadata:
    return LogMetadata(
        dataset="test-dataset",
        split="test-dataset_train",
        log_name="log_001",
        location="boston",
        timestep_seconds=timestep_seconds,
        map_metadata=None,
    )


def _write_sync_arrow(
    log_dir: Path,
    num_rows: int = 50,
    timestep_us: int = 100_000,
    log_metadata: Optional[LogMetadata] = None,
    ego_nulls: Optional[List[int]] = None,
) -> pa.Table:
    """Write a sync.arrow file and return the table."""
    timestamps = np.arange(num_rows, dtype=np.int64) * timestep_us
    uuids = [uuid.uuid4().bytes for _ in range(num_rows)]

    ego_indices: List[Optional[int]] = list(range(num_rows))
    if ego_nulls:
        for i in ego_nulls:
            ego_indices[i] = None

    schema = pa.schema(
        [
            pa.field("sync.uuid", pa.binary(16)),
            pa.field("sync.timestamp_us", pa.int64()),
            pa.field("ego_state_se3", pa.int64()),
        ]
    )

    if log_metadata is not None:
        schema = schema.with_metadata({b"metadata": msgpack.packb(log_metadata.to_dict(), use_bin_type=True)})

    table = pa.table(
        {
            "sync.uuid": pa.array(uuids, type=pa.binary(16)),
            "sync.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "ego_state_se3": pa.array(ego_indices, type=pa.int64()),
        },
        schema=schema,
    )

    with open(log_dir / "sync.arrow", "wb") as f:
        writer = ipc.new_file(f, table.schema)
        writer.write_table(table)
        writer.close()

    return table


def _write_ego_arrow(log_dir: Path, num_rows: int = 50, timestep_us: int = 100_000) -> None:
    """Write a minimal ego_state_se3.arrow table."""
    timestamps = np.arange(num_rows, dtype=np.int64) * timestep_us
    # Minimal ego data — just timestamps and a dummy pose
    dummy_pose = np.eye(4, dtype=np.float64).tobytes()

    schema = pa.schema(
        [
            pa.field("ego_state_se3.timestamp_us", pa.int64()),
            pa.field("ego_state_se3.imu_se3", pa.binary()),
        ]
    )

    table = pa.table(
        {
            "ego_state_se3.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "ego_state_se3.imu_se3": pa.array([dummy_pose] * num_rows, type=pa.binary()),
        },
        schema=schema,
    )

    with open(log_dir / "ego_state_se3.arrow", "wb") as f:
        writer = ipc.new_file(f, table.schema)
        writer.write_table(table)
        writer.close()


@pytest.fixture
def log_dir_10hz(tmp_path) -> Path:
    """Create a 10Hz log with 50 frames (5 seconds)."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    meta = _make_log_metadata(timestep_seconds=0.1)
    _write_sync_arrow(log_dir, num_rows=50, timestep_us=100_000, log_metadata=meta)
    _write_ego_arrow(log_dir, num_rows=50, timestep_us=100_000)
    return log_dir


# --- Tests ---


class TestGetSyncIndexWithStride:
    def test_stride_1_same_as_original(self, log_dir_10hz):
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=10,
            num_future_iterations=5,
            num_history_iterations=2,
            future_duration_s=0.5,
            history_duration_s=0.2,
            iteration_duration_s=0.1,
            target_iteration_stride=1,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)
        assert api._get_sync_index(0) == 10
        assert api._get_sync_index(1) == 11
        assert api._get_sync_index(-1) == 9
        assert api._get_sync_index(-2) == 8

    def test_stride_5(self, log_dir_10hz):
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=10,
            num_future_iterations=4,
            num_history_iterations=2,
            future_duration_s=2.0,
            history_duration_s=1.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)
        assert api._get_sync_index(0) == 10  # current frame
        assert api._get_sync_index(1) == 15  # 10 + 1*5
        assert api._get_sync_index(2) == 20  # 10 + 2*5
        assert api._get_sync_index(3) == 25  # 10 + 3*5
        assert api._get_sync_index(-1) == 5  # 10 + (-1)*5
        assert api._get_sync_index(-2) == 0  # 10 + (-2)*5

    def test_iteration_out_of_bounds(self, log_dir_10hz):
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=10,
            num_future_iterations=2,
            num_history_iterations=1,
            future_duration_s=1.0,
            history_duration_s=0.5,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)
        with pytest.raises(AssertionError):
            api._get_sync_index(3)  # only 0, 1, 2 are valid
        with pytest.raises(AssertionError):
            api._get_sync_index(-2)  # only -1 is valid


class TestGetAllIterationTimestampsWithStride:
    def test_timestamps_spaced_by_stride(self, log_dir_10hz):
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=0,
            num_future_iterations=4,
            num_history_iterations=0,
            future_duration_s=2.0,
            history_duration_s=0.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)
        timestamps = api.get_all_iteration_timestamps(include_history=False)

        # Should have 5 timestamps: iterations 0,1,2,3,4 → sync indices 0,5,10,15,20
        assert len(timestamps) == 5
        for i in range(len(timestamps) - 1):
            diff_us = timestamps[i + 1].time_us - timestamps[i].time_us
            assert diff_us == 500_000  # 0.5s = 500,000 us

    def test_timestamps_with_history(self, log_dir_10hz):
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=10,
            num_future_iterations=2,
            num_history_iterations=2,
            future_duration_s=1.0,
            history_duration_s=1.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)

        ts_no_hist = api.get_all_iteration_timestamps(include_history=False)
        ts_with_hist = api.get_all_iteration_timestamps(include_history=True)

        # Without history: iterations 0,1,2 → sync 10,15,20
        assert len(ts_no_hist) == 3
        # With history: iterations -2,-1,0,1,2 → sync 0,5,10,15,20
        assert len(ts_with_hist) == 5

    def test_stride_1_returns_all_frames(self, log_dir_10hz):
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=0,
            num_future_iterations=9,
            num_history_iterations=0,
            future_duration_s=0.9,
            history_duration_s=0.0,
            iteration_duration_s=0.1,
            target_iteration_stride=1,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)
        timestamps = api.get_all_iteration_timestamps()
        assert len(timestamps) == 10  # 0..9


class TestGetSceneTimestampBoundariesWithStride:
    def test_boundaries_correct(self, log_dir_10hz):
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=10,
            num_future_iterations=4,
            num_history_iterations=0,
            future_duration_s=2.0,
            history_duration_s=0.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)
        first, last = api.get_scene_timestamp_boundaries(include_history=False)

        # First: sync idx 10 → 10 * 100_000 = 1_000_000 us
        assert first.time_us == 1_000_000
        # Last: end_idx - 1 = 10 + 4*5 + 1 - 1 = 30 → 30 * 100_000 = 3_000_000 us
        assert last.time_us == 3_000_000

    def test_boundaries_with_history(self, log_dir_10hz):
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=10,
            num_future_iterations=2,
            num_history_iterations=2,
            future_duration_s=1.0,
            history_duration_s=1.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)
        first, last = api.get_scene_timestamp_boundaries(include_history=True)

        # With history: start_idx = 10 - 2*5 = 0, end_idx = 10 + 2*5 + 1 = 21
        assert first.time_us == 0  # sync idx 0
        assert last.time_us == 2_000_000  # sync idx 20


class TestNumberOfIterationsWithStride:
    def test_number_of_iterations_independent_of_stride(self, log_dir_10hz):
        """number_of_iterations is logical count, not raw frame count."""
        metadata = SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="abc",
            initial_idx=10,
            num_future_iterations=4,
            num_history_iterations=2,
            future_duration_s=2.0,
            history_duration_s=1.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        api = ArrowSceneAPI(log_dir_10hz, metadata)
        assert api.number_of_iterations == 5  # 4 future + 1 current
        assert api.number_of_history_iterations == 2
