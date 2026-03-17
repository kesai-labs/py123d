import logging
import uuid
from typing import List, Optional

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.utils.scene_builder_utils import (
    _scene_has_complete_modalities,
    generate_scene_metadatas,
    resolve_iteration_counts,
    resolve_iteration_stride,
)
from py123d.api.scene.scene_filter import SceneFilter
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata

# --- Helpers ---


def _make_log_metadata() -> LogMetadata:
    return LogMetadata(
        dataset="test-dataset",
        split="test-dataset_train",
        log_name="log_001",
        location="boston",
        timestep_seconds=0.1,
        map_metadata=None,
    )


def _make_sync_table(
    num_rows: int = 100,
    timestep_us: int = 100_000,
    camera_nulls: Optional[List[int]] = None,
) -> pa.Table:
    """Build a minimal sync table for stride testing."""
    timestamps = np.arange(num_rows, dtype=np.int64) * timestep_us
    uuids = [uuid.uuid4().bytes for _ in range(num_rows)]

    camera_indices: List[Optional[int]] = list(range(num_rows))
    if camera_nulls:
        for i in camera_nulls:
            camera_indices[i] = None

    return pa.table(
        {
            "sync.uuid": pa.array(uuids, type=pa.binary(16)),
            "sync.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "camera.front": pa.array(camera_indices, type=pa.int64()),
        },
    )


# --- TestResolveIterationStride ---


class TestResolveIterationStride:
    def test_neither_set(self):
        f = SceneFilter()
        result = resolve_iteration_stride(f, raw_iteration_duration_s=0.1)
        assert result == 1

    def test_stride_only(self):
        f = SceneFilter(target_iteration_stride=5)
        result = resolve_iteration_stride(f, raw_iteration_duration_s=0.1)
        assert result == 5

    def test_duration_only_exact(self):
        f = SceneFilter(target_iteration_duration_s=0.5)
        result = resolve_iteration_stride(f, raw_iteration_duration_s=0.1)
        assert result == 5

    def test_duration_only_exact_2hz(self):
        f = SceneFilter(target_iteration_duration_s=0.5)
        result = resolve_iteration_stride(f, raw_iteration_duration_s=0.5)
        assert result == 1  # already at target frequency

    def test_duration_takes_priority(self):
        f = SceneFilter(target_iteration_stride=3, target_iteration_duration_s=0.5)
        result = resolve_iteration_stride(f, raw_iteration_duration_s=0.1)
        assert result == 5  # duration wins, not stride=3

    def test_duration_cannot_upsample(self, caplog):
        f = SceneFilter(target_iteration_duration_s=0.05)
        with caplog.at_level(logging.DEBUG):
            result = resolve_iteration_stride(f, raw_iteration_duration_s=0.1)
        assert result is None
        assert "Cannot upsample" in caplog.text

    def test_duration_tolerance_ok(self):
        # 0.48 / 0.1 = 4.8 → round to 5, deviation = |4.8-5|/5 = 4% < 15%
        f = SceneFilter(target_iteration_duration_s=0.48)
        result = resolve_iteration_stride(f, raw_iteration_duration_s=0.1)
        assert result == 5

    def test_duration_tolerance_exceeded(self, caplog):
        # 0.18 / 0.1 = 1.8 → round to 2, deviation = |1.8-2|/2 = 10% < 15% → OK actually
        # Use a worse case: 0.35 / 0.1 = 3.5 → round to 4, deviation = |3.5-4|/4 = 12.5% → still OK
        # Even worse: 0.65 / 0.1 = 6.5 → round to 6, deviation = |6.5-6|/6 = 8.3% → OK
        # Need > 15%: 0.58 / 0.1 = 5.8 → round to 6, deviation = |5.8-6|/6 = 3.3% → OK
        # Actually: target / raw where round(target/raw) differs by > 15%
        # E.g. target=0.12 on raw=0.07 → 1.714 → round=2, dev=|1.714-2|/2=14.3% → borderline OK
        # target=0.07 on raw=0.04 → 1.75 → round=2, dev=12.5% → OK
        # Easier: use a case where it clearly exceeds.
        # target=0.26, raw=0.1 → 2.6 → round=3, dev=|2.6-3|/3=13.3% → still OK
        # target=0.24, raw=0.1 → 2.4 → round=2, dev=|2.4-2|/2=20% → exceeds!
        f = SceneFilter(target_iteration_duration_s=0.24)
        with caplog.at_level(logging.DEBUG):
            result = resolve_iteration_stride(f, raw_iteration_duration_s=0.1)
        assert result is None
        assert "not achievable" in caplog.text

    def test_stride_1_on_matching_log(self):
        f = SceneFilter(target_iteration_duration_s=0.5)
        result = resolve_iteration_stride(f, raw_iteration_duration_s=0.5)
        assert result == 1

    def test_stride_large(self):
        f = SceneFilter(target_iteration_stride=50)
        result = resolve_iteration_stride(f, raw_iteration_duration_s=0.1)
        assert result == 50


# --- TestResolveIterationCountsWithStride ---


class TestResolveIterationCountsWithStride:
    def test_duration_based_with_stride(self):
        # future_duration_s=2.0, stride=5, raw=0.1 → effective=0.5 → future=4
        f = SceneFilter(future_duration_s=2.0)
        future, history = resolve_iteration_counts(f, iteration_duration_s=0.1, stride=5)
        assert future == 4

    def test_history_duration_with_stride(self):
        # history_duration_s=1.0, stride=5, raw=0.1 → effective=0.5 → history=2
        f = SceneFilter(history_duration_s=1.0)
        future, history = resolve_iteration_counts(f, iteration_duration_s=0.1, stride=5)
        assert future is None
        assert history == 2

    def test_iteration_based_with_stride(self):
        # Iteration-based params are already in strided units
        f = SceneFilter(future_num_iterations=4, history_num_iterations=2)
        future, history = resolve_iteration_counts(f, iteration_duration_s=0.1, stride=5)
        assert future == 4
        assert history == 2

    def test_stride_1_unchanged(self):
        f = SceneFilter(future_duration_s=1.0, history_duration_s=0.5)
        future, history = resolve_iteration_counts(f, iteration_duration_s=0.1, stride=1)
        assert future == 10
        assert history == 5


# --- TestGenerateSceneMetadatasWithStride ---


class TestGenerateSceneMetadatasWithStride:
    def test_mode_b_sliding_window_stride_5(self):
        table = _make_sync_table(num_rows=100, timestep_us=100_000)
        meta = _make_log_metadata()
        scenes = generate_scene_metadatas(
            table, meta, future_iterations=4, history_iterations=0, iteration_duration_s=0.1, stride=5
        )
        # end_idx = 100 - 4*5 = 80, step_idx = max(4*5, 5) = 20
        # candidates: 0, 20, 40, 60
        assert len(scenes) == 4
        assert scenes[0].initial_idx == 0
        assert scenes[1].initial_idx == 20
        assert scenes[2].initial_idx == 40
        assert scenes[3].initial_idx == 60

        for s in scenes:
            assert s.num_future_iterations == 4
            assert s.target_iteration_stride == 5
            assert abs(s.iteration_duration_s - 0.5) < 1e-9
            assert abs(s.future_duration_s - 2.0) < 1e-9
            # end_idx should be within bounds
            assert s.end_idx <= 100

    def test_mode_b_with_history_stride_5(self):
        table = _make_sync_table(num_rows=100, timestep_us=100_000)
        meta = _make_log_metadata()
        scenes = generate_scene_metadatas(
            table, meta, future_iterations=4, history_iterations=2, iteration_duration_s=0.1, stride=5
        )
        # initial_idx = 2*5 = 10, end_idx = 100 - 4*5 = 80, step = 20
        # candidates: 10, 30, 50, 70
        assert len(scenes) == 4
        assert scenes[0].initial_idx == 10
        assert scenes[1].initial_idx == 30

        for s in scenes:
            assert s.num_history_iterations == 2
            assert abs(s.history_duration_s - 1.0) < 1e-9

    def test_mode_a_full_log_with_stride(self):
        table = _make_sync_table(num_rows=51, timestep_us=100_000)
        meta = _make_log_metadata()
        scenes = generate_scene_metadatas(
            table, meta, future_iterations=None, history_iterations=0, iteration_duration_s=0.1, stride=5
        )
        assert len(scenes) == 1
        assert scenes[0].initial_idx == 0
        # (51 - 0 - 1) // 5 = 10
        assert scenes[0].num_future_iterations == 10
        assert scenes[0].target_iteration_stride == 5
        assert abs(scenes[0].iteration_duration_s - 0.5) < 1e-9
        assert abs(scenes[0].future_duration_s - 5.0) < 1e-9

    def test_metadata_fields_with_stride(self):
        table = _make_sync_table(num_rows=100, timestep_us=100_000)
        meta = _make_log_metadata()
        scenes = generate_scene_metadatas(
            table, meta, future_iterations=4, history_iterations=0, iteration_duration_s=0.1, stride=5
        )
        s = scenes[0]
        assert s.target_iteration_stride == 5
        assert abs(s.iteration_duration_s - 0.5) < 1e-9  # 0.1 * 5
        assert abs(s.future_duration_s - 2.0) < 1e-9  # 4 * 0.5

    def test_stride_too_large_for_log(self):
        table = _make_sync_table(num_rows=10, timestep_us=100_000)
        meta = _make_log_metadata()
        # stride=50, future=4 → need 4*50=200 frames → no scenes
        scenes = generate_scene_metadatas(
            table, meta, future_iterations=4, history_iterations=0, iteration_duration_s=0.1, stride=50
        )
        assert len(scenes) == 0

    def test_stride_1_matches_original_behavior(self):
        table = _make_sync_table(num_rows=20, timestep_us=100_000)
        meta = _make_log_metadata()
        scenes_s1 = generate_scene_metadatas(
            table, meta, future_iterations=5, history_iterations=0, iteration_duration_s=0.1, stride=1
        )
        scenes_default = generate_scene_metadatas(
            table, meta, future_iterations=5, history_iterations=0, iteration_duration_s=0.1
        )
        assert len(scenes_s1) == len(scenes_default)
        for s1, sd in zip(scenes_s1, scenes_default):
            assert s1.initial_idx == sd.initial_idx
            assert s1.num_future_iterations == sd.num_future_iterations
            assert s1.end_idx == sd.end_idx

    def test_mode_b_with_uuid_filter_and_stride(self):
        table = _make_sync_table(num_rows=100, timestep_us=100_000)
        meta = _make_log_metadata()
        uuid_indices = {25, 50, 90}
        scenes = generate_scene_metadatas(
            table,
            meta,
            future_iterations=4,
            history_iterations=0,
            iteration_duration_s=0.1,
            scene_uuid_indices=uuid_indices,
            stride=5,
        )
        # end_idx = 100 - 4*5 = 80, initial_idx = 0
        # 25 < 80 → yes, 50 < 80 → yes, 90 >= 80 → no
        assert len(scenes) == 2
        assert scenes[0].initial_idx == 25
        assert scenes[1].initial_idx == 50


# --- TestSceneHasCompleteModalitiesWithStride ---


class TestSceneHasCompleteModalitiesWithStride:
    def test_strided_null_at_skipped_frame_ok(self):
        """Nulls at frames that are skipped by stride should not cause rejection."""
        # Stride=5, scene at idx=0, future=2 → accesses frames 0, 5, 10
        # Null at frame 3 should be fine (skipped)
        table = _make_sync_table(num_rows=20, camera_nulls=[3])
        scene = SceneMetadata(
            dataset="test",
            split="test_train",
            initial_uuid="a",
            initial_idx=0,
            num_future_iterations=2,
            num_history_iterations=0,
            future_duration_s=1.0,
            history_duration_s=0.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        assert _scene_has_complete_modalities(scene, table, ["camera.front"]) is True

    def test_strided_null_at_accessed_frame_rejected(self):
        """Nulls at frames accessed by stride should cause rejection."""
        # Stride=5, scene at idx=0, future=2 → accesses frames 0, 5, 10
        # Null at frame 5 → rejected
        table = _make_sync_table(num_rows=20, camera_nulls=[5])
        scene = SceneMetadata(
            dataset="test",
            split="test_train",
            initial_uuid="a",
            initial_idx=0,
            num_future_iterations=2,
            num_history_iterations=0,
            future_duration_s=1.0,
            history_duration_s=0.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        assert _scene_has_complete_modalities(scene, table, ["camera.front"]) is False

    def test_strided_with_history_null_check(self):
        """History frames are also checked at strided indices."""
        # Stride=5, scene at idx=10, history=2 → accesses history frames 0, 5 and current 10
        # Null at frame 0 → rejected
        table = _make_sync_table(num_rows=20, camera_nulls=[0])
        scene = SceneMetadata(
            dataset="test",
            split="test_train",
            initial_uuid="a",
            initial_idx=10,
            num_future_iterations=1,
            num_history_iterations=2,
            future_duration_s=0.5,
            history_duration_s=1.0,
            iteration_duration_s=0.5,
            target_iteration_stride=5,
        )
        assert _scene_has_complete_modalities(scene, table, ["camera.front"]) is False

    def test_stride_1_checks_all_frames(self):
        """With stride=1, all frames are checked (original behavior)."""
        table = _make_sync_table(num_rows=20, camera_nulls=[3])
        scene = SceneMetadata(
            dataset="test",
            split="test_train",
            initial_uuid="a",
            initial_idx=0,
            num_future_iterations=4,
            num_history_iterations=0,
            future_duration_s=0.4,
            history_duration_s=0.0,
            iteration_duration_s=0.1,
            target_iteration_stride=1,
        )
        assert _scene_has_complete_modalities(scene, table, ["camera.front"]) is False
