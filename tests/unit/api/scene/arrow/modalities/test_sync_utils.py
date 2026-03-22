"""Tests for sync_utils.py — sync table utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from py123d.api.scene.arrow.modalities.sync_utils import (
    _get_scene_sync_range,
    get_all_modality_timestamps,
    get_modality_index_from_sync_index,
    get_modality_table,
    get_sync_table,
)
from py123d.datatypes.metadata import SceneMetadata

from ...conftest import make_ego_metadata, make_log_metadata, write_ego_arrow, write_sync_arrow


def _make_scene_metadata(**kwargs) -> SceneMetadata:
    defaults = dict(
        dataset="test-dataset",
        split="test-dataset_train",
        initial_uuid="00000000-0000-0000-0000-000000000001",
        initial_idx=0,
        num_future_iterations=9,
        num_history_iterations=0,
        future_duration_s=0.9,
        history_duration_s=0.0,
        iteration_duration_s=0.1,
        target_iteration_stride=1,
    )
    defaults.update(kwargs)
    return SceneMetadata(**defaults)


class TestGetModalityTable:
    def test_existing_modality(self, tmp_path: Path):
        ego_meta = make_ego_metadata()
        write_ego_arrow(tmp_path, num_rows=5, timestep_us=100_000, metadata=ego_meta)
        table = get_modality_table(tmp_path, ego_meta.modality_key)
        assert table is not None
        assert table.num_rows == 5

    def test_nonexistent_returns_none(self, tmp_path: Path):
        table = get_modality_table(tmp_path, "does_not_exist")
        assert table is None


class TestGetSyncTable:
    def test_existing_sync(self, tmp_path: Path):
        log_meta = make_log_metadata()
        write_sync_arrow(tmp_path, num_rows=5, timestep_us=100_000, log_metadata=log_meta)
        table = get_sync_table(tmp_path)
        assert table is not None
        assert table.num_rows == 5

    def test_missing_sync_asserts(self, tmp_path: Path):
        with pytest.raises((AssertionError, FileNotFoundError, OSError)):
            get_sync_table(tmp_path)


class TestGetModalityIndexFromSyncIndex:
    def test_valid_index(self, tmp_path: Path):
        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()
        indices = [0, 1, 2, None, 4]
        sync_table = write_sync_arrow(
            tmp_path,
            num_rows=5,
            timestep_us=100_000,
            log_metadata=log_meta,
            modality_columns={ego_meta.modality_key: indices},
        )
        result = get_modality_index_from_sync_index(sync_table, ego_meta.modality_key, 1)
        assert result == 1

    def test_null_returns_none(self, tmp_path: Path):
        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()
        indices = [0, 1, 2, None, 4]
        sync_table = write_sync_arrow(
            tmp_path,
            num_rows=5,
            timestep_us=100_000,
            log_metadata=log_meta,
            modality_columns={ego_meta.modality_key: indices},
        )
        result = get_modality_index_from_sync_index(sync_table, ego_meta.modality_key, 3)
        assert result is None


class TestGetSceneSyncRange:
    def test_no_history(self):
        meta = _make_scene_metadata(initial_idx=5, num_future_iterations=3, target_iteration_stride=1)
        start, end = _get_scene_sync_range(meta, include_history=False)
        assert start == 5
        assert end == meta.end_idx

    def test_with_history(self):
        meta = _make_scene_metadata(
            initial_idx=10, num_future_iterations=3, num_history_iterations=2, target_iteration_stride=1
        )
        start, end = _get_scene_sync_range(meta, include_history=True)
        assert start == 8  # 10 - 2*1
        assert end == meta.end_idx

    def test_stride_affects_history_start(self):
        meta = _make_scene_metadata(
            initial_idx=20, num_future_iterations=2, num_history_iterations=2, target_iteration_stride=5
        )
        start, end = _get_scene_sync_range(meta, include_history=True)
        assert start == 10  # 20 - 2*5


class TestGetAllModalityTimestamps:
    def test_basic(self, tmp_path: Path):
        """All modalities present — returns timestamps for every frame."""
        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()
        num_rows = 10
        timestep_us = 100_000

        write_ego_arrow(tmp_path, num_rows, timestep_us, ego_meta)
        ego_key = ego_meta.modality_key
        sync_table = write_sync_arrow(
            tmp_path,
            num_rows,
            timestep_us,
            log_meta,
            modality_columns={ego_key: list(range(num_rows))},
        )
        scene_meta = _make_scene_metadata()

        ts_col = f"{ego_key}.timestamp_us"
        timestamps = get_all_modality_timestamps(tmp_path, sync_table, scene_meta, ego_key, ts_col)
        assert len(timestamps) == num_rows
        assert timestamps[0].time_us == 0
        assert timestamps[-1].time_us == 9 * timestep_us

    def test_null_entries_skipped(self, tmp_path: Path):
        """Null sync entries should be skipped when scanning for first/last row."""
        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()
        num_rows = 5
        timestep_us = 100_000

        write_ego_arrow(tmp_path, num_rows, timestep_us, ego_meta)
        ego_key = ego_meta.modality_key
        # Null at indices 0 and 4
        indices = [None, 1, 2, 3, None]
        sync_table = write_sync_arrow(
            tmp_path,
            num_rows,
            timestep_us,
            log_meta,
            modality_columns={ego_key: indices},
        )
        scene_meta = _make_scene_metadata(num_future_iterations=4)

        ts_col = f"{ego_key}.timestamp_us"
        timestamps = get_all_modality_timestamps(tmp_path, sync_table, scene_meta, ego_key, ts_col)
        # Returns timestamps between row 1 and row 3 (inclusive) = 3 timestamps
        assert len(timestamps) == 3

    def test_returns_all_between_first_last(self, tmp_path: Path):
        """get_all_modality_timestamps returns ALL rows between first/last referenced row,
        not just the strided ones. This is intentional: it enables async data markers
        (e.g., all camera frames between two lidar sweeps) to be included."""
        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()
        num_rows = 10
        timestep_us = 100_000

        write_ego_arrow(tmp_path, num_rows, timestep_us, ego_meta)
        ego_key = ego_meta.modality_key
        # All modality indices present
        sync_table = write_sync_arrow(
            tmp_path,
            num_rows,
            timestep_us,
            log_meta,
            modality_columns={ego_key: list(range(num_rows))},
        )
        # Stride=2: logical frames are 0, 2, 4, 6, 8 → first_row=0, last_row=8
        # But the function returns ALL rows 0..8 = 9 timestamps, not just 5
        scene_meta = _make_scene_metadata(num_future_iterations=4, target_iteration_stride=2, iteration_duration_s=0.2)

        ts_col = f"{ego_key}.timestamp_us"
        timestamps = get_all_modality_timestamps(tmp_path, sync_table, scene_meta, ego_key, ts_col)
        # Intended: returns 9 timestamps (rows 0 through 8), not just the 5 strided ones.
        # This allows async data between strided frames to be discovered.
        assert len(timestamps) == 9

    def test_all_null_returns_empty(self, tmp_path: Path):
        """If all sync entries for the modality are None, return empty list."""
        log_meta = make_log_metadata()
        ego_meta = make_ego_metadata()
        num_rows = 5
        timestep_us = 100_000

        write_ego_arrow(tmp_path, num_rows, timestep_us, ego_meta)
        ego_key = ego_meta.modality_key
        sync_table = write_sync_arrow(
            tmp_path,
            num_rows,
            timestep_us,
            log_meta,
            modality_columns={ego_key: [None] * num_rows},
        )
        scene_meta = _make_scene_metadata(num_future_iterations=4)

        ts_col = f"{ego_key}.timestamp_us"
        timestamps = get_all_modality_timestamps(tmp_path, sync_table, scene_meta, ego_key, ts_col)
        assert len(timestamps) == 0
