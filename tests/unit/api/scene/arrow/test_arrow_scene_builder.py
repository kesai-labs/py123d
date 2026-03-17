import uuid
from pathlib import Path
from typing import List, Optional

import msgpack
import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_scene_builder import _parse_valid_log_dirs
from py123d.api.scene.arrow.utils.scene_builder_utils import (
    check_log_passes_metadata_filters,
    filter_scene_metadata_candidates,
    generate_scene_metadatas,
    infer_iteration_duration_s,
    resolve_iteration_counts,
    resolve_scene_uuid_indices,
    scene_uuids_to_binary,
)
from py123d.api.scene.scene_filter import SceneFilter
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata

# --- Fixtures ---


def _make_log_metadata(
    dataset: str = "test-dataset",
    split: str = "test-dataset_train",
    log_name: str = "log_001",
    location: Optional[str] = "boston",
    timestep_seconds: float = 0.1,
    map_metadata: Optional[MapMetadata] = None,
) -> LogMetadata:
    return LogMetadata(
        dataset=dataset,
        split=split,
        log_name=log_name,
        location=location,
        timestep_seconds=timestep_seconds,
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
        f = SceneFilter(required_log_modalities=["camera.front"])
        assert check_log_passes_metadata_filters(_make_log_metadata(), columns, f) is True

        f = SceneFilter(required_log_modalities=["camera.rear"])
        assert check_log_passes_metadata_filters(_make_log_metadata(), columns, f) is False


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
        from py123d.api.scene.arrow.arrow_scene_builder import _apply_post_filters

        # Create mock scenes (just need list length behavior)
        scenes = [None] * 10  # type: ignore

        f = SceneFilter(num_chunks=3, chunk_idx=0)
        result = _apply_post_filters(scenes, f)
        assert len(result) == 3

        f = SceneFilter(num_chunks=3, chunk_idx=2)
        result = _apply_post_filters(scenes, f)
        assert len(result) == 4  # last chunk gets remainder

    def test_max_num_scenes(self):
        from py123d.api.scene.arrow.arrow_scene_builder import _apply_post_filters

        scenes = [None] * 10  # type: ignore
        f = SceneFilter(max_num_scenes=3)
        result = _apply_post_filters(scenes, f)
        assert len(result) == 3
