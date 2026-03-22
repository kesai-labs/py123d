import uuid
from pathlib import Path
from typing import List, Optional

import msgpack
import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_scene_builder import (
    ArrowSceneBuilder,
    _apply_post_filters,
    _discover_split_names,
    _extract_scenes_from_log_dir,
    _extract_scenes_from_log_dirs,
    _parse_valid_log_dirs,
)
from py123d.api.scene.arrow.utils.scene_builder_utils import (
    _get_columns_matching_type,
    _is_modality_pattern,
    _parse_modality_pattern,
    check_log_passes_metadata_filters,
    filter_scene_metadata_candidates,
    generate_scene_metadatas,
    infer_iteration_duration_s,
    resolve_iteration_counts,
    resolve_scene_uuid_indices,
    scene_uuids_to_binary,
)
from py123d.api.scene.scene_filter import SceneFilter, _validate_modality_requirement
from py123d.common.execution import SequentialExecutor
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata

# --- Fixtures ---


def _make_log_metadata(
    dataset: str = "test-dataset",
    split: str = "test-dataset_train",
    log_name: str = "log_001",
    location: Optional[str] = "boston",
    map_metadata: Optional[MapMetadata] = None,
) -> LogMetadata:
    return LogMetadata(
        dataset=dataset,
        split=split,
        log_name=log_name,
        location=location,
        map_metadata=map_metadata,
    )


def _make_sync_table(
    num_rows: int = 20,
    timestep_us: int = 100_000,
    camera_nulls: Optional[List[int]] = None,
    lidar_nulls: Optional[List[int]] = None,
    jitter_us: int = 0,
    log_metadata: Optional[LogMetadata] = None,
) -> pa.Table:
    """Build a minimal sync table for testing.

    :param num_rows: Number of rows.
    :param timestep_us: Timestep between rows in microseconds (default 100ms = 10Hz).
    :param camera_nulls: Row indices where camera column should be null.
    :param lidar_nulls: Row indices where lidar column should be null.
    :param jitter_us: Random jitter to add to timestamps.
    :param log_metadata: LogMetadata to embed in schema metadata.
    """
    rng = np.random.RandomState(42)
    timestamps = np.arange(num_rows, dtype=np.int64) * timestep_us
    if jitter_us > 0:
        timestamps += rng.randint(-jitter_us, jitter_us + 1, size=num_rows)
        timestamps[0] = 0  # keep first timestamp clean

    uuids = [uuid.uuid4().bytes for _ in range(num_rows)]

    camera_indices: List[Optional[int]] = list(range(num_rows))
    if camera_nulls:
        for i in camera_nulls:
            camera_indices[i] = None

    lidar_indices: List[Optional[int]] = list(range(num_rows))
    if lidar_nulls:
        for i in lidar_nulls:
            lidar_indices[i] = None

    schema = pa.schema(
        [
            pa.field("sync.uuid", pa.binary(16)),
            pa.field("sync.timestamp_us", pa.int64()),
            pa.field("camera.front", pa.int64()),
            pa.field("lidar.top", pa.int64()),
        ]
    )

    if log_metadata is not None:
        existing = {}
        existing[b"metadata"] = msgpack.packb(log_metadata.to_dict(), use_bin_type=True)
        schema = schema.with_metadata(existing)

    table = pa.table(
        {
            "sync.uuid": pa.array(uuids, type=pa.binary(16)),
            "sync.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "camera.front": pa.array(camera_indices, type=pa.int64()),
            "lidar.top": pa.array(lidar_indices, type=pa.int64()),
        },
        schema=schema,
    )
    return table


def _make_multi_camera_sync_table(
    num_rows: int = 20,
    timestep_us: int = 100_000,
    front_nulls: Optional[List[int]] = None,
    rear_nulls: Optional[List[int]] = None,
    lidar_nulls: Optional[List[int]] = None,
) -> pa.Table:
    """Build a sync table with two camera columns and one lidar column.

    :param num_rows: Number of rows.
    :param timestep_us: Timestep between rows in microseconds.
    :param front_nulls: Row indices where camera.front should be null.
    :param rear_nulls: Row indices where camera.rear should be null.
    :param lidar_nulls: Row indices where lidar.top should be null.
    """
    timestamps = np.arange(num_rows, dtype=np.int64) * timestep_us
    uuids = [uuid.uuid4().bytes for _ in range(num_rows)]

    def _make_column(nulls: Optional[List[int]]) -> List[Optional[int]]:
        col: List[Optional[int]] = list(range(num_rows))
        if nulls:
            for i in nulls:
                col[i] = None
        return col

    schema = pa.schema(
        [
            pa.field("sync.uuid", pa.binary(16)),
            pa.field("sync.timestamp_us", pa.int64()),
            pa.field("camera.front", pa.int64()),
            pa.field("camera.rear", pa.int64()),
            pa.field("lidar.top", pa.int64()),
        ]
    )
    table = pa.table(
        {
            "sync.uuid": pa.array(uuids, type=pa.binary(16)),
            "sync.timestamp_us": pa.array(timestamps, type=pa.int64()),
            "camera.front": pa.array(_make_column(front_nulls), type=pa.int64()),
            "camera.rear": pa.array(_make_column(rear_nulls), type=pa.int64()),
            "lidar.top": pa.array(_make_column(lidar_nulls), type=pa.int64()),
        },
        schema=schema,
    )
    return table


def _write_demo_log(tmp_path: Path, split_name: str = "test-dataset_train", log_name: str = "log_001") -> Path:
    """Write a minimal demo log directory with sync.arrow to disk."""
    log_dir = tmp_path / "logs" / split_name / log_name
    log_dir.mkdir(parents=True)

    log_metadata = _make_log_metadata(split=split_name, log_name=log_name)
    sync_table = _make_sync_table(log_metadata=log_metadata)

    from pyarrow import ipc

    with open(log_dir / "sync.arrow", "wb") as f:
        writer = ipc.new_file(f, sync_table.schema)
        writer.write_table(sync_table)
        writer.close()

    return log_dir


# --- Tests ---


class TestInferIterationDuration:
    def test_uniform_timestamps(self):
        table = _make_sync_table(num_rows=10, timestep_us=100_000)
        result = infer_iteration_duration_s(table)
        assert abs(result - 0.1) < 1e-9

    def test_jittery_timestamps(self):
        table = _make_sync_table(num_rows=100, timestep_us=100_000, jitter_us=5_000)
        result = infer_iteration_duration_s(table)
        assert abs(result - 0.1) < 0.01  # median should be close to 0.1

    def test_different_frequency(self):
        table = _make_sync_table(num_rows=10, timestep_us=500_000)  # 2Hz
        result = infer_iteration_duration_s(table)
        assert abs(result - 0.5) < 1e-9

    def test_single_row_raises(self):
        table = _make_sync_table(num_rows=1)
        with pytest.raises(ValueError, match="fewer than 2 rows"):
            infer_iteration_duration_s(table)


class TestResolveIterationCounts:
    def test_duration_based(self):
        f = SceneFilter(future_duration_s=1.0, history_duration_s=0.5)
        future, history = resolve_iteration_counts(f, iteration_duration_s=0.1)
        assert future == 10
        assert history == 5

    def test_iteration_based(self):
        f = SceneFilter(future_num_iterations=8, history_num_iterations=3)
        future, history = resolve_iteration_counts(f, iteration_duration_s=0.1)
        assert future == 8
        assert history == 3

    def test_duration_takes_priority(self):
        f = SceneFilter(
            future_duration_s=1.0, future_num_iterations=99, history_duration_s=0.5, history_num_iterations=99
        )
        future, history = resolve_iteration_counts(f, iteration_duration_s=0.1)
        assert future == 10
        assert history == 5

    def test_neither_set(self):
        f = SceneFilter()
        future, history = resolve_iteration_counts(f, iteration_duration_s=0.1)
        assert future is None
        assert history == 0


class TestCheckLogPassesMetadataFilters:
    def test_passes_with_no_filters(self):
        meta = _make_log_metadata()
        result = check_log_passes_metadata_filters(meta, ["sync.uuid", "camera.front"], SceneFilter())
        assert result is True

    def test_location_filter(self):
        meta = _make_log_metadata(location="boston")
        result = check_log_passes_metadata_filters(meta, ["sync.uuid"], SceneFilter(log_locations=["boston"]))
        assert result is True

        result = check_log_passes_metadata_filters(meta, ["sync.uuid"], SceneFilter(log_locations=["pittsburgh"]))
        assert result is False

    def test_has_map_filter(self):
        meta_with_map = _make_log_metadata(
            map_metadata=MapMetadata(dataset="test", location="boston", map_has_z=False, map_is_per_log=False)
        )
        meta_no_map = _make_log_metadata(map_metadata=None)

        assert check_log_passes_metadata_filters(meta_with_map, [], SceneFilter(has_map=True)) is True
        assert check_log_passes_metadata_filters(meta_no_map, [], SceneFilter(has_map=True)) is False
        assert check_log_passes_metadata_filters(meta_no_map, [], SceneFilter(has_map=False)) is True
        assert check_log_passes_metadata_filters(meta_with_map, [], SceneFilter(has_map=False)) is False

    def test_required_modalities(self):
        columns = ["sync.uuid", "sync.timestamp_us", "camera.front", "lidar.top"]
        f = SceneFilter(required_scene_modalities=["camera.front"])
        assert check_log_passes_metadata_filters(_make_log_metadata(), columns, f) is True

        f = SceneFilter(required_scene_modalities=["camera.rear"])
        assert check_log_passes_metadata_filters(_make_log_metadata(), columns, f) is False

    def test_required_modalities_any_type(self):
        columns = ["sync.uuid", "sync.timestamp_us", "camera.front", "lidar.top"]
        meta = _make_log_metadata()

        # "camera:any" passes when at least one camera column exists
        f = SceneFilter(required_scene_modalities=["camera:any"])
        assert check_log_passes_metadata_filters(meta, columns, f) is True

        # "radar:any" fails when no radar columns exist
        f = SceneFilter(required_scene_modalities=["radar:any"])
        assert check_log_passes_metadata_filters(meta, columns, f) is False

    def test_required_modalities_all_type(self):
        columns = ["sync.uuid", "sync.timestamp_us", "camera.front", "lidar.top"]
        meta = _make_log_metadata()

        # "camera:all" at log level passes if at least one camera column exists
        f = SceneFilter(required_scene_modalities=["camera:all"])
        assert check_log_passes_metadata_filters(meta, columns, f) is True

        # "radar:all" fails when no radar columns exist
        f = SceneFilter(required_scene_modalities=["radar:all"])
        assert check_log_passes_metadata_filters(meta, columns, f) is False

    def test_required_modalities_type_matches_no_id(self):
        """Type pattern matches modalities without an ID (e.g., ego_state_se3)."""
        columns = ["sync.uuid", "sync.timestamp_us", "ego_state_se3", "camera.front"]
        meta = _make_log_metadata()

        f = SceneFilter(required_scene_modalities=["ego_state_se3:any"])
        assert check_log_passes_metadata_filters(meta, columns, f) is True

    def test_required_modalities_invalid_pattern(self):
        with pytest.raises(ValueError, match="Invalid modality pattern"):
            SceneFilter(required_scene_modalities=["camera:invalid"])

        with pytest.raises(ValueError, match="Invalid modality pattern"):
            SceneFilter(required_scene_modalities=["camera:any:3"])


class TestValidateModalityRequirement:
    def test_exact_key_with_dot(self):
        _validate_modality_requirement("camera.pcam_f0")  # should not raise

    def test_exact_key_without_dot(self):
        _validate_modality_requirement("ego_state_se3")  # should not raise

    def test_valid_any_pattern(self):
        _validate_modality_requirement("camera:any")  # should not raise

    def test_valid_all_pattern(self):
        _validate_modality_requirement("camera:all")  # should not raise

    def test_invalid_quantifier(self):
        with pytest.raises(ValueError, match="Invalid modality pattern"):
            _validate_modality_requirement("camera:some")

    def test_too_many_colons(self):
        with pytest.raises(ValueError, match="Invalid modality pattern"):
            _validate_modality_requirement("camera:any:3")

    def test_empty_quantifier(self):
        with pytest.raises(ValueError, match="Invalid modality pattern"):
            _validate_modality_requirement("camera:")


class TestModalityMatchingHelpers:
    def test_is_modality_pattern(self):
        assert _is_modality_pattern("camera:any") is True
        assert _is_modality_pattern("camera:all") is True
        assert _is_modality_pattern("camera.front") is False
        assert _is_modality_pattern("ego_state_se3") is False

    def test_parse_modality_pattern(self):
        assert _parse_modality_pattern("camera:any") == ("camera", "any")
        assert _parse_modality_pattern("lidar:all") == ("lidar", "all")

    def test_get_columns_matching_type_with_ids(self):
        columns = {"sync.uuid", "sync.timestamp_us", "camera.front", "camera.rear", "lidar.top"}
        result = _get_columns_matching_type("camera", columns)
        assert sorted(result) == ["camera.front", "camera.rear"]

    def test_get_columns_matching_type_without_id(self):
        columns = {"sync.uuid", "ego_state_se3", "camera.front"}
        result = _get_columns_matching_type("ego_state_se3", columns)
        assert result == ["ego_state_se3"]

    def test_get_columns_matching_type_no_match(self):
        columns = {"sync.uuid", "camera.front", "lidar.top"}
        result = _get_columns_matching_type("radar", columns)
        assert result == []

    def test_get_columns_matching_type_no_prefix_collision(self):
        """'camera' should not match 'camera_info' — only exact or 'camera.' prefix."""
        columns = {"camera.front", "camera_info"}
        result = _get_columns_matching_type("camera", columns)
        assert result == ["camera.front"]


class TestSceneFilterModalityValidation:
    def test_deduplicates_scene_modalities(self):
        f = SceneFilter(required_scene_modalities=["camera.front", "camera.front", "lidar:any"])
        assert f.required_scene_modalities == ["camera.front", "lidar:any"]

    def test_rejects_invalid_scene_modality_pattern(self):
        with pytest.raises(ValueError, match="Invalid modality pattern"):
            SceneFilter(required_scene_modalities=["camera:invalid"])

    def test_accepts_mixed_exact_and_pattern(self):
        f = SceneFilter(required_scene_modalities=["camera.front", "lidar:any", "camera:all"])
        assert f.required_scene_modalities == ["camera.front", "lidar:any", "camera:all"]


class TestGenerateSceneMetadatas:
    def test_full_log_no_duration(self):
        table = _make_sync_table(num_rows=20, timestep_us=100_000)
        meta = _make_log_metadata()
        scenes = generate_scene_metadatas(
            table, meta, future_iterations=None, history_iterations=0, iteration_duration_s=0.1
        )
        assert len(scenes) == 1
        assert scenes[0].num_future_iterations == 19
        assert scenes[0].num_history_iterations == 0
        assert abs(scenes[0].future_duration_s - 1.9) < 1e-9

    def test_sliding_window(self):
        table = _make_sync_table(num_rows=20, timestep_us=100_000)
        meta = _make_log_metadata()
        scenes = generate_scene_metadatas(
            table, meta, future_iterations=5, history_iterations=0, iteration_duration_s=0.1
        )
        # Window of 5 future iterations, stepping by 5: indices 0, 5, 10 (15 is >= 20-5=15 boundary)
        assert len(scenes) == 3
        assert scenes[0].initial_idx == 0
        assert scenes[1].initial_idx == 5
        assert scenes[2].initial_idx == 10
        for s in scenes:
            assert s.num_future_iterations == 5

    def test_with_history(self):
        table = _make_sync_table(num_rows=20, timestep_us=100_000)
        meta = _make_log_metadata()
        scenes = generate_scene_metadatas(
            table, meta, future_iterations=5, history_iterations=2, iteration_duration_s=0.1
        )
        # start_idx=2, end_idx=15, step=5 → indices 2, 7, 12
        assert len(scenes) == 3
        assert scenes[0].initial_idx == 2
        for s in scenes:
            assert s.num_history_iterations == 2

    def test_with_uuid_filter(self):
        table = _make_sync_table(num_rows=20, timestep_us=100_000)
        meta = _make_log_metadata()
        # Pre-filter to only index 5
        uuid_indices = {5}
        scenes = generate_scene_metadatas(
            table,
            meta,
            future_iterations=3,
            history_iterations=0,
            iteration_duration_s=0.1,
            scene_uuid_indices=uuid_indices,
        )
        assert len(scenes) == 1
        assert scenes[0].initial_idx == 5


class TestFilterScenes:
    def _make_candidates(self, num: int = 5) -> List[SceneMetadata]:
        return [
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid=str(uuid.uuid4()),
                initial_idx=i * 5,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            )
            for i in range(num)
        ]

    def test_timestamp_threshold(self):
        table = _make_sync_table(num_rows=30, timestep_us=100_000)
        candidates = self._make_candidates(5)  # at indices 0, 5, 10, 15, 20
        f = SceneFilter(timestamp_threshold_s=1.5)  # only keep scenes >= 1.5s apart
        result = filter_scene_metadata_candidates(candidates, f, table)
        # 0 → keep, 5 (0.5s gap) → skip, 10 (1.0s) → skip, 15 (1.5s) → keep, 20 (0.5s from 15) → skip
        assert len(result) == 2
        assert result[0].initial_idx == 0
        assert result[1].initial_idx == 15

    def test_required_scene_modalities_with_nulls(self):
        table = _make_sync_table(num_rows=20, timestep_us=100_000, camera_nulls=[2, 3])
        candidates = [
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="a",
                initial_idx=0,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="b",
                initial_idx=5,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
        ]
        f = SceneFilter(required_scene_modalities=["camera.front"])
        result = filter_scene_metadata_candidates(candidates, f, table)
        # First scene (idx 0-4) has nulls at 2,3 → filtered out. Second (idx 5-9) is clean.
        assert len(result) == 1
        assert result[0].initial_idx == 5

    def test_iteration_threshold(self):
        table = _make_sync_table(num_rows=30, timestep_us=100_000)
        candidates = self._make_candidates(5)  # at indices 0, 5, 10, 15, 20
        f = SceneFilter(iteration_threshold=12)  # only keep scenes >= 12 iterations apart
        result = filter_scene_metadata_candidates(candidates, f, table)
        # 0 → keep, 5 (5 gap) → skip, 10 (10) → skip, 15 (15) → keep, 20 (5 from 15) → skip
        assert len(result) == 2
        assert result[0].initial_idx == 0
        assert result[1].initial_idx == 15

    def test_timestamp_threshold_takes_priority_over_iteration_threshold(self):
        table = _make_sync_table(num_rows=30, timestep_us=100_000)
        candidates = self._make_candidates(5)  # at indices 0, 5, 10, 15, 20
        # timestamp_threshold_s=1.5 keeps 0 and 15; iteration_threshold=3 would keep 0, 5, 10, 15, 20
        f = SceneFilter(timestamp_threshold_s=1.5, iteration_threshold=3)
        result = filter_scene_metadata_candidates(candidates, f, table)
        assert len(result) == 2
        assert result[0].initial_idx == 0
        assert result[1].initial_idx == 15

    def test_required_scene_modalities_any_type(self):
        """'camera:any' keeps scenes where at least one camera column is complete."""
        # camera.front has nulls at [2, 3], camera.rear has nulls at [2, 3, 6, 7]
        table = _make_multi_camera_sync_table(num_rows=20, front_nulls=[2, 3], rear_nulls=[2, 3, 6, 7])
        candidates = [
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="a",
                initial_idx=0,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="b",
                initial_idx=5,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="c",
                initial_idx=10,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
        ]
        f = SceneFilter(required_scene_modalities=["camera:any"])
        result = filter_scene_metadata_candidates(candidates, f, table)
        # Scene at idx 0 (0-4): both cameras have nulls → neither complete → filtered out
        # Scene at idx 5 (5-9): front is complete, rear has nulls at 6,7 → front passes → kept
        # Scene at idx 10 (10-14): both cameras complete → kept
        assert len(result) == 2
        assert result[0].initial_idx == 5
        assert result[1].initial_idx == 10

    def test_required_scene_modalities_all_type(self):
        """'camera:all' keeps scenes where ALL camera columns are complete."""
        # camera.front is clean, camera.rear has nulls at [6, 7]
        table = _make_multi_camera_sync_table(num_rows=20, rear_nulls=[6, 7])
        candidates = [
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="a",
                initial_idx=0,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="b",
                initial_idx=5,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="c",
                initial_idx=10,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
        ]
        f = SceneFilter(required_scene_modalities=["camera:all"])
        result = filter_scene_metadata_candidates(candidates, f, table)
        # Scene at idx 0 (0-4): front ok, rear ok → kept
        # Scene at idx 5 (5-9): front ok, rear has nulls at 6,7 → filtered out
        # Scene at idx 10 (10-14): both ok → kept
        assert len(result) == 2
        assert result[0].initial_idx == 0
        assert result[1].initial_idx == 10

    def test_required_scene_modalities_any_type_no_match(self):
        """'radar:any' when no radar columns exist keeps no scenes."""
        table = _make_multi_camera_sync_table(num_rows=20)
        candidates = [
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="a",
                initial_idx=0,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
        ]
        f = SceneFilter(required_scene_modalities=["radar:any"])
        result = filter_scene_metadata_candidates(candidates, f, table)
        assert len(result) == 0

    def test_required_scene_modalities_all_type_no_match(self):
        """'radar:all' when no radar columns exist keeps no scenes."""
        table = _make_multi_camera_sync_table(num_rows=20)
        candidates = [
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="a",
                initial_idx=0,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
        ]
        # "radar:all" expands to zero columns → all_complete_keys is empty → no filtering → keeps scene
        f = SceneFilter(required_scene_modalities=["radar:all"])
        result = filter_scene_metadata_candidates(candidates, f, table)
        assert len(result) == 1

    def test_required_scene_modalities_combined_exact_and_pattern(self):
        """Combining exact key and type pattern in the same filter."""
        # camera.front has nulls at [2,3], camera.rear is clean, lidar.top is clean
        table = _make_multi_camera_sync_table(num_rows=20, front_nulls=[2, 3])
        candidates = [
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="a",
                initial_idx=0,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="b",
                initial_idx=5,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
        ]
        # Require exact lidar.top AND at least one camera
        f = SceneFilter(required_scene_modalities=["lidar.top", "camera:any"])
        result = filter_scene_metadata_candidates(candidates, f, table)
        # idx 0: lidar ok, camera.front has nulls but camera.rear is complete → camera:any passes → kept
        # idx 5: lidar ok, both cameras clean → kept
        assert len(result) == 2

    def test_required_scene_modalities_mixed_any_and_all(self):
        """Combining 'camera:all' and 'lidar:any' in the same filter."""
        # camera.front has nulls at [6,7], camera.rear is clean, lidar.top is clean
        table = _make_multi_camera_sync_table(num_rows=20, front_nulls=[6, 7])
        candidates = [
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="a",
                initial_idx=0,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="b",
                initial_idx=5,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
            SceneMetadata(
                dataset="test",
                split="test_train",
                initial_uuid="c",
                initial_idx=10,
                num_future_iterations=4,
                num_history_iterations=0,
                future_duration_s=0.4,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
        ]
        f = SceneFilter(required_scene_modalities=["camera:all", "lidar:any"])
        result = filter_scene_metadata_candidates(candidates, f, table)
        # idx 0 (0-4): camera.front ok, camera.rear ok, lidar ok → kept
        # idx 5 (5-9): camera.front has nulls at 6,7 → camera:all fails → filtered out
        # idx 10 (10-14): all clean → kept
        assert len(result) == 2
        assert result[0].initial_idx == 0
        assert result[1].initial_idx == 10

    def test_no_filters_passes_all(self):
        table = _make_sync_table(num_rows=20)
        candidates = self._make_candidates(3)
        result = filter_scene_metadata_candidates(candidates, SceneFilter(), table)
        assert len(result) == 3


class TestResolveSceneUuidIndices:
    def test_finds_matching_uuids(self):
        table = _make_sync_table(num_rows=10)
        from py123d.common.utils.uuid_utils import convert_to_str_uuid

        target_uuid = convert_to_str_uuid(table["sync.uuid"][3].as_py())
        target_binary = scene_uuids_to_binary([target_uuid])
        result = resolve_scene_uuid_indices(table, target_binary)
        assert result is not None
        assert 3 in result

    def test_no_matches_returns_none(self):
        table = _make_sync_table(num_rows=10)
        target_binary = scene_uuids_to_binary(["00000000-0000-0000-0000-000000000000"])
        result = resolve_scene_uuid_indices(table, target_binary)
        assert result is None


class TestSceneFilterUuidValidation:
    def test_accepts_valid_uuid_strings(self):
        f = SceneFilter(scene_uuids=["12345678-1234-5678-1234-567812345678"])
        assert f.scene_uuids == ["12345678-1234-5678-1234-567812345678"]

    def test_accepts_uuid_objects(self):
        import uuid

        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        f = SceneFilter(scene_uuids=[u])
        assert f.scene_uuids == ["12345678-1234-5678-1234-567812345678"]

    def test_rejects_invalid_uuid(self):
        with pytest.raises(ValueError, match="Invalid UUID"):
            SceneFilter(scene_uuids=["not-a-uuid"])


class TestCategory1LogDiscovery:
    def test_discovers_logs(self, tmp_path):
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_002")
        _write_demo_log(tmp_path, split_name="other-dataset_val", log_name="log_003")

        logs_root = tmp_path / "logs"

        # No filter → find all
        result = _parse_valid_log_dirs(logs_root, SceneFilter())
        assert len(result) == 3

        # Filter by dataset
        result = _parse_valid_log_dirs(logs_root, SceneFilter(datasets=["test-dataset"]))
        assert len(result) == 2

        # Filter by split type
        result = _parse_valid_log_dirs(logs_root, SceneFilter(split_types=["val"]))
        assert len(result) == 1
        assert result[0].name == "log_003"

        # Filter by log name
        result = _parse_valid_log_dirs(logs_root, SceneFilter(log_names=["log_001"]))
        assert len(result) == 1
        assert result[0].name == "log_001"


class TestCategory4PostFiltering:
    def test_chunking(self):
        # Create mock scenes (just need list length behavior)
        scenes = [None] * 10  # type: ignore

        f = SceneFilter(num_chunks=3, chunk_idx=0)
        result = _apply_post_filters(scenes, f)
        assert len(result) == 3

        f = SceneFilter(num_chunks=3, chunk_idx=2)
        result = _apply_post_filters(scenes, f)
        assert len(result) == 4  # last chunk gets remainder

    def test_max_num_scenes(self):
        scenes = [None] * 10  # type: ignore
        f = SceneFilter(max_num_scenes=3)
        result = _apply_post_filters(scenes, f)
        assert len(result) == 3

    def test_custom_filter_fn(self):
        scenes = list(range(10))  # type: ignore[list-item]
        f = SceneFilter(custom_filter_fns=[lambda s: s % 2 == 0])
        result = _apply_post_filters(scenes, f)  # type: ignore[arg-type]
        assert result == [0, 2, 4, 6, 8]

    def test_multiple_custom_filter_fns(self):
        scenes = list(range(10))  # type: ignore[list-item]
        f = SceneFilter(custom_filter_fns=[lambda s: s % 2 == 0, lambda s: s > 3])
        result = _apply_post_filters(scenes, f)  # type: ignore[arg-type]
        assert result == [4, 6, 8]

    def test_shuffle_changes_order(self):
        import random

        random.seed(42)
        scenes = list(range(20))  # type: ignore[list-item]
        f = SceneFilter(shuffle=True)
        result = _apply_post_filters(list(scenes), f)  # type: ignore[arg-type]
        assert len(result) == 20
        assert set(result) == set(scenes)
        # With seed 42 and 20 elements, the shuffled order should differ from sorted
        assert result != scenes

    def test_shuffle_then_max(self):
        import random

        random.seed(0)
        scenes = list(range(20))  # type: ignore[list-item]
        f = SceneFilter(shuffle=True, max_num_scenes=5)
        result = _apply_post_filters(list(scenes), f)  # type: ignore[arg-type]
        assert len(result) == 5

    def test_empty_scenes(self):
        result = _apply_post_filters([], SceneFilter())
        assert result == []

    def test_chunking_middle_chunk(self):
        scenes = [None] * 10  # type: ignore
        f = SceneFilter(num_chunks=3, chunk_idx=1)
        result = _apply_post_filters(scenes, f)
        assert len(result) == 3


class TestDiscoverSplitNames:
    def test_discovers_train_val(self, tmp_path):
        logs_root = tmp_path / "logs"
        (logs_root / "dataset-a_train").mkdir(parents=True)
        (logs_root / "dataset-a_val").mkdir(parents=True)
        (logs_root / "dataset-b_test").mkdir(parents=True)
        # A non-directory file should be skipped
        (logs_root / "stray_file.txt").touch()

        result = _discover_split_names(logs_root, SceneFilter())
        assert sorted(result) == ["dataset-a_train", "dataset-a_val", "dataset-b_test"]

    def test_filter_by_dataset(self, tmp_path):
        logs_root = tmp_path / "logs"
        (logs_root / "dataset-a_train").mkdir(parents=True)
        (logs_root / "dataset-b_val").mkdir(parents=True)

        result = _discover_split_names(logs_root, SceneFilter(datasets=["dataset-a"]))
        assert result == ["dataset-a_train"]

    def test_filter_by_split_type(self, tmp_path):
        logs_root = tmp_path / "logs"
        (logs_root / "dataset-a_train").mkdir(parents=True)
        (logs_root / "dataset-a_val").mkdir(parents=True)
        (logs_root / "dataset-a_test").mkdir(parents=True)

        result = _discover_split_names(logs_root, SceneFilter(split_types=["val"]))
        assert result == ["dataset-a_val"]

    def test_skips_non_matching_split_types(self, tmp_path):
        logs_root = tmp_path / "logs"
        (logs_root / "dataset-a_train").mkdir(parents=True)

        result = _discover_split_names(logs_root, SceneFilter(split_types=["val"]))
        assert result == []


class TestExtractScenesFromLogDir:
    def test_extracts_scenes_single_log(self, tmp_path):
        log_dir = _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        scenes = _extract_scenes_from_log_dir(log_dir, SceneFilter(), maps_root=tmp_path / "maps")
        assert len(scenes) == 1  # full log, no windowing → one scene

    def test_returns_empty_on_error(self, tmp_path):
        # Create a log dir with an invalid sync.arrow (empty file)
        log_dir = tmp_path / "logs" / "test-dataset_train" / "bad_log"
        log_dir.mkdir(parents=True)
        (log_dir / "sync.arrow").write_bytes(b"not valid arrow data")

        scenes = _extract_scenes_from_log_dir(log_dir, SceneFilter(), maps_root=tmp_path / "maps")
        assert scenes == []

    def test_with_future_duration(self, tmp_path):
        log_dir = _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        f = SceneFilter(future_duration_s=0.5)  # 5 iterations at 10Hz
        scenes = _extract_scenes_from_log_dir(log_dir, f, maps_root=tmp_path / "maps")
        assert len(scenes) > 1
        for s in scenes:
            assert s.scene_metadata.num_future_iterations == 5


class TestExtractScenesFromLogDirs:
    def test_batch_extraction(self, tmp_path):
        log_dir_1 = _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        log_dir_2 = _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_002")

        scenes = _extract_scenes_from_log_dirs(
            [log_dir_1, log_dir_2], filter=SceneFilter(), maps_root=tmp_path / "maps"
        )
        assert len(scenes) == 2  # one scene per log (full log, no windowing)

    def test_empty_list(self):
        scenes = _extract_scenes_from_log_dirs([], filter=SceneFilter(), maps_root=None)
        assert scenes == []


class TestArrowSceneBuilderInit:
    def test_init_with_explicit_paths(self, tmp_path):
        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        logs_root.mkdir()
        maps_root.mkdir()

        builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
        assert builder._logs_root == logs_root
        assert builder._maps_root == maps_root

    def test_init_accepts_string_paths(self, tmp_path):
        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        logs_root.mkdir()
        maps_root.mkdir()

        builder = ArrowSceneBuilder(logs_root=str(logs_root), maps_root=str(maps_root))
        assert builder._logs_root == logs_root
        assert builder._maps_root == maps_root


class TestArrowSceneBuilderGetScenes:
    def test_get_scenes_returns_scenes(self, tmp_path):
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_002")

        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        maps_root.mkdir(exist_ok=True)

        builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
        executor = SequentialExecutor()
        scenes = builder.get_scenes(SceneFilter(), executor)
        assert len(scenes) == 2

    def test_get_scenes_with_dataset_filter(self, tmp_path):
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        _write_demo_log(tmp_path, split_name="other-dataset_val", log_name="log_002")

        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        maps_root.mkdir(exist_ok=True)

        builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
        executor = SequentialExecutor()
        scenes = builder.get_scenes(SceneFilter(datasets=["test-dataset"]), executor)
        assert len(scenes) == 1

    def test_get_scenes_empty_when_no_logs(self, tmp_path):
        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        logs_root.mkdir()
        maps_root.mkdir()

        builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
        executor = SequentialExecutor()
        scenes = builder.get_scenes(SceneFilter(), executor)
        assert scenes == []

    def test_get_scenes_with_max_num_scenes(self, tmp_path):
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_002")
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_003")

        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        maps_root.mkdir(exist_ok=True)

        builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
        executor = SequentialExecutor()
        scenes = builder.get_scenes(SceneFilter(max_num_scenes=2), executor)
        assert len(scenes) == 2

    def test_get_scenes_with_log_name_filter(self, tmp_path):
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_002")

        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        maps_root.mkdir(exist_ok=True)

        builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
        executor = SequentialExecutor()
        scenes = builder.get_scenes(SceneFilter(log_names=["log_002"]), executor)
        assert len(scenes) == 1

    def test_get_scenes_with_split_names_filter(self, tmp_path):
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")
        _write_demo_log(tmp_path, split_name="test-dataset_val", log_name="log_002")

        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        maps_root.mkdir(exist_ok=True)

        builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
        executor = SequentialExecutor()
        scenes = builder.get_scenes(SceneFilter(split_names=["test-dataset_val"]), executor)
        assert len(scenes) == 1

    def test_get_scenes_with_future_duration(self, tmp_path):
        _write_demo_log(tmp_path, split_name="test-dataset_train", log_name="log_001")

        logs_root = tmp_path / "logs"
        maps_root = tmp_path / "maps"
        maps_root.mkdir(exist_ok=True)

        builder = ArrowSceneBuilder(logs_root=logs_root, maps_root=maps_root)
        executor = SequentialExecutor()
        # 20 rows at 10Hz, future_duration=0.5s → 5 iterations per window
        scenes = builder.get_scenes(SceneFilter(future_duration_s=0.5), executor)
        assert len(scenes) > 1
        for s in scenes:
            assert s.scene_metadata.num_future_iterations == 5
