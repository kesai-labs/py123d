"""The lazy scene path enumerates exactly what the eager path enumerates.

Both paths resolve the same filter semantics — strides, scopes, quantifiers,
post-filters — from different representations, so every filter shape is checked
against the eager result rather than against a hand-written expectation.
"""

import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import ipc

from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder, LazyArrowSceneBuilder
from py123d.api.scene.arrow.lazy_scene_sequence import LazySceneSequence
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.sequential_executor import SequentialExecutor
from py123d.datatypes import LogMetadata

SPLIT_NAME = "test-dataset_train"


def _write_log(
    logs_root: Path,
    log_name: str,
    num_rows: int = 24,
    timestep_us: int = 100_000,
    front_nulls: Optional[List[int]] = None,
    rear_nulls: Optional[List[int]] = None,
    lidar_nulls: Optional[List[int]] = None,
) -> Path:
    """Write one log whose modality columns carry the requested gaps.

    :param logs_root: Root the split directory is created under.
    :param log_name: Name of the log directory.
    :param num_rows: Rows in the sync table.
    :param timestep_us: Microseconds between rows.
    :param front_nulls: Rows where ``camera.front`` is null.
    :param rear_nulls: Rows where ``camera.rear`` is null.
    :param lidar_nulls: Rows where ``lidar.top`` is null.
    :return: The log directory.
    """
    log_dir = logs_root / SPLIT_NAME / log_name
    log_dir.mkdir(parents=True)

    def column(nulls: Optional[List[int]]) -> List[Optional[int]]:
        values: List[Optional[int]] = list(range(num_rows))
        for index in nulls or []:
            values[index] = None
        return values

    log_metadata = LogMetadata(
        dataset="test-dataset",
        split=SPLIT_NAME,
        log_name=log_name,
        location="boston",
        map_metadata=None,
    )
    schema = pa.schema(
        [
            pa.field("sync.uuid", pa.binary(16)),
            pa.field("sync.timestamp_us", pa.int64()),
            pa.field("camera.front", pa.int64()),
            pa.field("camera.rear", pa.int64()),
            pa.field("lidar.top", pa.int64()),
        ]
    )
    import msgpack

    schema = schema.with_metadata({b"metadata": msgpack.packb(log_metadata.to_dict(), use_bin_type=True)})
    table = pa.table(
        {
            "sync.uuid": pa.array([uuid.uuid4().bytes for _ in range(num_rows)], type=pa.binary(16)),
            "sync.timestamp_us": pa.array(np.arange(num_rows, dtype=np.int64) * timestep_us, type=pa.int64()),
            "camera.front": pa.array(column(front_nulls), type=pa.int64()),
            "camera.rear": pa.array(column(rear_nulls), type=pa.int64()),
            "lidar.top": pa.array(column(lidar_nulls), type=pa.int64()),
        },
        schema=schema,
    )
    with open(log_dir / "sync.arrow", "wb") as file:
        writer = ipc.new_file(file, table.schema)
        writer.write_table(table)
        writer.close()
    return log_dir


@pytest.fixture
def builders(tmp_path: Path) -> Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder]:
    """An eager and a lazy builder over the same logs, with differently shaped gaps.

    :param tmp_path: Temporary directory the logs are written to.
    :return: The eager builder and the lazy one.
    """
    logs_root = tmp_path / "logs"
    _write_log(logs_root, "log_complete")
    _write_log(logs_root, "log_front_gaps", front_nulls=[0, 5, 11, 23])
    _write_log(logs_root, "log_rear_gaps", rear_nulls=[2, 3, 4, 17])
    _write_log(logs_root, "log_lidar_gaps", lidar_nulls=list(range(0, 24, 3)))
    _write_log(logs_root, "log_short", num_rows=6)
    _write_log(logs_root, "log_no_cameras", front_nulls=list(range(24)), rear_nulls=list(range(24)))
    return (
        ArrowSceneBuilder(logs_root=logs_root, maps_root=tmp_path / "maps"),
        LazyArrowSceneBuilder(logs_root=logs_root, maps_root=tmp_path / "maps"),
    )


def _identity(scene: SceneAPI) -> Tuple:
    """Everything about a scene that enumeration decides.

    :param scene: The scene to describe.
    :return: The scene's identity as a comparable tuple.
    """
    metadata = scene.scene_metadata
    return (
        scene.log_name,
        metadata.initial_idx,
        metadata.initial_uuid,
        metadata.num_future_iterations,
        metadata.num_history_iterations,
        metadata.target_iteration_stride,
        round(metadata.iteration_duration_s, 9),
        round(metadata.future_duration_s, 9),
        round(metadata.history_duration_s, 9),
    )


FILTERS = {
    "no requirements": dict(history_num_iterations=1, future_num_iterations=4),
    "exact key": dict(history_num_iterations=1, future_num_iterations=4, required_scene_modalities=["lidar.top"]),
    "type all": dict(history_num_iterations=1, future_num_iterations=4, required_scene_modalities=["camera:all"]),
    "type any": dict(history_num_iterations=1, future_num_iterations=4, required_scene_modalities=["camera:any"]),
    "scope initial": dict(
        history_num_iterations=2, future_num_iterations=3, required_scene_modalities=["camera:all@initial"]
    ),
    "scope history": dict(
        history_num_iterations=2, future_num_iterations=3, required_scene_modalities=["camera:all@history"]
    ),
    "scope future": dict(
        history_num_iterations=2, future_num_iterations=3, required_scene_modalities=["camera:any@future"]
    ),
    "scope combined": dict(
        history_num_iterations=2, future_num_iterations=2, required_scene_modalities=["camera:all@initial+future"]
    ),
    "several requirements": dict(
        history_num_iterations=1,
        future_num_iterations=2,
        required_scene_modalities=["lidar.top@initial", "camera:all@initial", "camera:any"],
    ),
    "absent modality": dict(history_num_iterations=0, future_num_iterations=1, required_scene_modalities=["radar:all"]),
    "stride": dict(history_num_iterations=1, future_num_iterations=2, target_iteration_stride=3),
    "stride with gaps": dict(
        history_num_iterations=1,
        future_num_iterations=2,
        target_iteration_stride=2,
        required_scene_modalities=["camera:all"],
    ),
    "step by iterations": dict(history_num_iterations=1, future_num_iterations=3, iteration_threshold=4),
    "step by seconds": dict(history_num_iterations=1, future_num_iterations=3, timestamp_threshold_s=0.5),
    "scenes to log end": dict(history_num_iterations=2, required_scene_modalities=["camera:all@history+initial"]),
    "capped": dict(history_num_iterations=0, future_num_iterations=2, max_num_scenes=7),
    "chunked": dict(history_num_iterations=0, future_num_iterations=2, num_chunks=3, chunk_idx=1),
    "last chunk": dict(history_num_iterations=0, future_num_iterations=2, num_chunks=3, chunk_idx=2),
    "one log": dict(history_num_iterations=1, future_num_iterations=2, log_names=["log_front_gaps"]),
}


@pytest.mark.parametrize("name", list(FILTERS))
def test_lazy_enumeration_matches_eager(builders: Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder], name: str) -> None:
    """Both paths select the same scenes, in the same order, with the same metadata."""
    eager_builder, lazy_builder = builders
    scene_filter = SceneFilter(split_names=[SPLIT_NAME], shuffle=False, **FILTERS[name])

    eager = eager_builder.get_scenes(scene_filter, SequentialExecutor())
    lazy = lazy_builder.get_scenes(scene_filter, SequentialExecutor())

    assert [_identity(scene) for scene in lazy] == [_identity(scene) for scene in eager]


def test_custom_filter_functions_still_apply(builders: Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder]) -> None:
    """A filter function takes a scene, so the lazy path materializes and filters."""
    eager_builder, lazy_builder = builders
    scene_filter = SceneFilter(
        split_names=[SPLIT_NAME],
        shuffle=False,
        history_num_iterations=1,
        future_num_iterations=2,
        custom_filter_fns=[lambda scene: scene.log_name == "log_short"],
    )

    eager = eager_builder.get_scenes(scene_filter, SequentialExecutor())
    lazy = lazy_builder.get_scenes(scene_filter, SequentialExecutor())

    assert len(eager) > 0
    assert [_identity(scene) for scene in lazy] == [_identity(scene) for scene in eager]


def test_custom_anchor_filter_functions_match(builders: Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder]) -> None:
    """Anchor filter functions select the same scenes on the eager and the lazy path."""
    eager_builder, lazy_builder = builders

    def keep_even_anchors(context) -> np.ndarray:
        return context.anchors % 2 == 0

    scene_filter = SceneFilter(
        split_names=[SPLIT_NAME],
        shuffle=False,
        history_num_iterations=1,
        future_num_iterations=2,
        custom_anchor_filter_fns=[keep_even_anchors],
    )

    eager = eager_builder.get_scenes(scene_filter, SequentialExecutor())
    lazy = lazy_builder.get_scenes(scene_filter, SequentialExecutor())

    assert len(eager) > 0
    assert all(scene.scene_metadata.initial_idx % 2 == 0 for scene in eager)
    assert [_identity(scene) for scene in lazy] == [_identity(scene) for scene in eager]
    assert isinstance(lazy, LazySceneSequence)


def test_custom_anchor_filter_context_contents(builders: Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder]) -> None:
    """A filter function sees only the anchors that survived the built-in filters, with the log's context."""
    _, lazy_builder = builders
    seen = []

    def record(context) -> np.ndarray:
        seen.append(context)
        return np.ones(len(context.anchors), dtype=bool)

    scene_filter = SceneFilter(
        split_names=[SPLIT_NAME],
        shuffle=False,
        history_num_iterations=1,
        future_num_iterations=2,
        target_iteration_stride=2,
        required_scene_modalities=["camera:all"],
        log_names=["log_front_gaps"],
        custom_anchor_filter_fns=[record],
    )

    lazy = lazy_builder.get_scenes(scene_filter, SequentialExecutor())

    assert len(seen) == 1
    context = seen[0]
    assert context.log_dir.name == "log_front_gaps"
    assert context.history_iterations == 1
    assert context.future_iterations == 2
    assert context.stride == 2
    assert context.anchors.dtype == np.int64
    assert len(context.anchors) == len(lazy)
    assert list(context.anchors) == [scene.scene_metadata.initial_idx for scene in lazy]


def test_anchor_filter_result_shape_is_validated(
    builders: Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder], caplog
) -> None:
    """A scalar result must not silently broadcast over all anchors; the log is rejected instead."""
    import logging

    eager_builder, lazy_builder = builders
    scene_filter = SceneFilter(
        split_names=[SPLIT_NAME],
        shuffle=False,
        history_num_iterations=1,
        future_num_iterations=2,
        custom_anchor_filter_fns=[lambda context: np.array(True)],
    )

    with caplog.at_level(logging.WARNING):
        lazy = lazy_builder.get_scenes(scene_filter, SequentialExecutor())
        eager = eager_builder.get_scenes(scene_filter, SequentialExecutor())

    assert len(lazy) == 0
    assert len(eager) == 0
    assert "shape" in caplog.text


def test_anchor_keys_match_the_materialized_scenes(builders: Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder]) -> None:
    """The pairing keys come from the index, so they must equal what the scenes report."""
    eager_builder, lazy_builder = builders
    scene_filter = SceneFilter(
        split_names=[SPLIT_NAME], shuffle=False, history_num_iterations=1, future_num_iterations=2
    )

    lazy = lazy_builder.get_scenes(scene_filter, SequentialExecutor())
    assert isinstance(lazy, LazySceneSequence)

    expected = [(scene.log_name, scene.get_timestamp_at_iteration(0).time_us) for scene in lazy]
    assert lazy.anchor_keys() == expected


def test_indexing_is_stable_and_bounded(builders: Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder]) -> None:
    """Indexing builds a scene per call, and out-of-range access is an error."""
    eager_builder, lazy_builder = builders
    scene_filter = SceneFilter(
        split_names=[SPLIT_NAME], shuffle=False, history_num_iterations=1, future_num_iterations=2
    )

    lazy = lazy_builder.get_scenes(scene_filter, SequentialExecutor())

    assert _identity(lazy[3]) == _identity(lazy[3])
    assert _identity(lazy[-1]) == _identity(lazy[len(lazy) - 1])
    assert [_identity(scene) for scene in lazy[2:5]] == [_identity(lazy[index]) for index in (2, 3, 4)]
    with pytest.raises(IndexError):
        lazy[len(lazy)]


def test_shuffling_draws_the_same_permutation(builders: Tuple[ArrowSceneBuilder, LazyArrowSceneBuilder]) -> None:
    """Shuffling positions rather than scenes must reproduce the eager order."""
    import random

    eager_builder, lazy_builder = builders
    scene_filter = SceneFilter(
        split_names=[SPLIT_NAME], shuffle=True, history_num_iterations=1, future_num_iterations=2
    )

    random.seed(7)
    eager = eager_builder.get_scenes(scene_filter, SequentialExecutor())
    random.seed(7)
    lazy = lazy_builder.get_scenes(scene_filter, SequentialExecutor())

    assert [_identity(scene) for scene in lazy] == [_identity(scene) for scene in eager]
