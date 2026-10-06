import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import pyarrow as pa

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.lazy_scene_sequence import (
    LazySceneSequence,
    LogSceneIndex,
    apply_lazy_post_filters,
    build_log_scene_index,
)
from py123d.api.scene.arrow.utils.scene_builder_utils import (
    check_log_passes_metadata_filters,
    filter_scene_metadata_candidates,
    generate_scene_metadatas,
    infer_iteration_duration_s,
    resolve_iteration_counts,
    resolve_iteration_stride,
    resolve_scene_step_size,
    resolve_scene_uuid_indices,
    scene_uuids_to_binary,
)
from py123d.api.scene.scene_api import SceneAPI
from py123d.api.scene.scene_builder import SceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.api.utils.arrow_metadata_utils import get_metadata_from_arrow_schema
from py123d.api.utils.cache_source_utils import check_cache_source_modalities
from py123d.common.execution import Executor
from py123d.common.execution.utils import executor_map_chunked_list
from py123d.common.runtime import get_dataset_paths
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata

logger = logging.getLogger(__name__)

_DISCOVERY_THREADS = 8
"""Threads scanning directories during log discovery; the work is filesystem lookups, not Python."""


class ArrowSceneBuilder(SceneBuilder):
    """Builds scenes from Arrow log directories using a category-based pipeline."""

    def __init__(
        self,
        logs_root: Optional[Union[str, Path]] = None,
        maps_root: Optional[Union[str, Path]] = None,
    ):
        """Initialize the ArrowSceneBuilder.

        :param logs_root: Root directory for log files, defaults to None (uses PY123D_DATA_ROOT).
        :param maps_root: Root directory for map files, defaults to None (uses PY123D_DATA_ROOT).
        """
        if logs_root is None:
            logs_root = get_dataset_paths().py123d_logs_root
        if maps_root is None:
            maps_root = get_dataset_paths().py123d_maps_root

        assert logs_root is not None, "logs_root must be provided or PY123D_DATA_ROOT must be set."
        assert maps_root is not None, "maps_root must be provided or PY123D_DATA_ROOT must be set."
        self._logs_root = Path(logs_root)
        self._maps_root = Path(maps_root)

    def get_scenes(self, filter: SceneFilter, executor: Executor) -> Sequence[SceneAPI]:
        """Inherited, see superclass."""

        # Category 1: Log discovery (filesystem-only)
        log_paths = _parse_valid_log_dirs(self._logs_root, filter)
        if len(log_paths) == 0:
            return []

        # Categories 2 & 3: Metadata filtering + scene generation (parallelized across logs)
        # Pre-convert scene UUIDs to binary once (shared across all executor workers)
        target_uuids_binary = scene_uuids_to_binary(filter.scene_uuids) if filter.scene_uuids is not None else None
        scenes: List[SceneAPI] = executor_map_chunked_list(
            executor,
            partial(
                _extract_scenes_from_log_dirs,
                filter=filter,
                maps_root=self._maps_root,
                target_uuids_binary=target_uuids_binary,
            ),
            log_paths,
            name="Scene extraction",
        )

        # Category 4: Post-filtering
        scenes = _apply_post_filters(scenes, filter)
        return scenes


class LazyArrowSceneBuilder(ArrowSceneBuilder):
    """Builds scenes from Arrow log directories, one scene at a time on access.

    Enumeration keeps a split as a few numpy arrays per log and constructs a
    :class:`~py123d.api.scene.scene_api.SceneAPI` when the returned sequence is
    indexed, rather than one per candidate frame up front. Suited to splits
    whose scene count makes an object per scene expensive; the returned
    sequence is otherwise interchangeable with the eager builder's list.
    """

    def get_scenes(self, filter: SceneFilter, executor: Executor) -> Sequence[SceneAPI]:
        """Inherited, see superclass."""

        # Category 1: Log discovery (filesystem-only)
        log_paths = _parse_valid_log_dirs(self._logs_root, filter)
        if len(log_paths) == 0:
            return []

        # Categories 2 & 3, as per-log columns rather than scene objects
        target_uuids_binary = scene_uuids_to_binary(filter.scene_uuids) if filter.scene_uuids is not None else None
        log_indices = executor_map_chunked_list(
            executor,
            partial(_index_log_dirs, filter=filter, maps_root=self._maps_root, target_uuids_binary=target_uuids_binary),
            log_paths,
            name="Scene indexing",
        )

        # Category 4: Post-filtering
        return apply_lazy_post_filters(LazySceneSequence(log_indices), filter)


# --- Category 1: Log discovery ---


def _parse_valid_log_dirs(logs_root: Path, filter: SceneFilter) -> List[Path]:
    """Discover valid log directories based on Category 1 filter criteria (filesystem-only).

    A log directory is any directory containing ``sync.arrow`` at any depth
    beneath a split directory.

    :param logs_root: Root directory containing split subdirectories.
    :param filter: The scene filter.
    :return: Sorted list of valid log directory paths.
    """
    split_names = filter.split_names if filter.split_names is not None else _discover_split_names(logs_root, filter)
    log_names = set(filter.log_names) if filter.log_names is not None else None
    roots = [logs_root / split_name for split_name in split_names if (logs_root / split_name).is_dir()]

    log_paths: List[Path] = [root for root in roots if (root / "sync.arrow").exists()]
    frontier = [root for root in roots if root not in log_paths]
    with ThreadPoolExecutor(max_workers=_DISCOVERY_THREADS) as executor:
        while frontier:
            next_frontier: List[Path] = []
            for found, subdirectories in executor.map(_scan_for_logs, frontier):
                log_paths.extend(found)
                next_frontier.extend(subdirectories)
            frontier = next_frontier

    if log_names is not None:
        log_paths = [path for path in log_paths if path.name in log_names]
    return sorted(log_paths)


def _scan_for_logs(directory: Path) -> Tuple[List[Path], List[Path]]:
    """Split a directory's subdirectories into logs and directories still to search.

    Probing for ``sync.arrow`` directly costs one lookup, where listing a log
    directory would read every modality file it holds.

    :param directory: The directory to scan.
    :return: Tuple of (log directories found, subdirectories to descend into).
    """
    logs: List[Path] = []
    subdirectories: List[Path] = []
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return logs, subdirectories
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        # A log has no nested logs, so a match ends the descent.
        if os.path.exists(os.path.join(entry.path, "sync.arrow")):
            logs.append(Path(entry.path))
        else:
            subdirectories.append(Path(entry.path))
    return logs, subdirectories


def _discover_split_names(logs_root: Path, filter: SceneFilter) -> List[str]:
    """Discover split names from the filesystem based on dataset and split_type filters."""
    split_types = set(filter.split_types) if filter.split_types else {"train", "val", "test"}
    split_names: List[str] = []
    for split in logs_root.iterdir():
        split_name = split.name
        dataset_name = split_name.split("_")[0]
        if filter.datasets is not None and dataset_name not in filter.datasets:
            continue
        if split.is_dir() and any(split_type in split_name for split_type in split_types):
            split_names.append(split_name)
    return split_names


# --- Categories 2 & 3: Per-log scene extraction ---


def _index_log_dirs(
    log_dirs: List[Path],
    filter: SceneFilter,
    maps_root: Optional[Path] = None,
    target_uuids_binary: Optional[pa.Array] = None,
) -> List[LogSceneIndex]:
    """Index multiple log directories (chunked batch wrapper).

    :param log_dirs: List of log directory paths to index.
    :param filter: The scene filter.
    :param maps_root: Maps directory the built scenes resolve their map under, defaults to None
    :param target_uuids_binary: Pre-converted binary(16) Arrow array of target UUIDs, or None.
    :return: One index per log that contributes at least one scene.
    """
    indices = [build_log_scene_index(log_dir, filter, target_uuids_binary, maps_root=maps_root) for log_dir in log_dirs]
    return [index for index in indices if index is not None]


def _extract_scenes_from_log_dirs(
    log_dirs: List[Path],
    filter: SceneFilter,
    maps_root: Optional[Path],
    target_uuids_binary: Optional[pa.Array] = None,
) -> List[SceneAPI]:
    """Extract scenes from multiple log directories (chunked batch wrapper).

    :param log_dirs: List of log directory paths to process.
    :param filter: The scene filter.
    :param maps_root: Root directory for map files.
    :param target_uuids_binary: Pre-converted binary(16) Arrow array of target UUIDs, or None.
    :return: Combined list of SceneAPI objects from all logs.
    """
    scenes: List[SceneAPI] = []
    for log_dir in log_dirs:
        scenes.extend(_extract_scenes_from_log_dir(log_dir, filter, maps_root, target_uuids_binary))
    return scenes


def _extract_scenes_from_log_dir(
    log_dir: Path,
    filter: SceneFilter,
    maps_root: Optional[Path],
    target_uuids_binary: Optional[pa.Array] = None,
) -> List[SceneAPI]:
    """Extract scenes from a single log directory (Categories 2 & 3).

    :param log_dir: Path to the log directory.
    :param filter: The scene filter.
    :param maps_root: Root directory for map files.
    :param target_uuids_binary: Pre-converted binary(16) Arrow array of target UUIDs, or None.
    :return: List of SceneAPI objects for this log.
    :raises StaleModalityError: If a source modality changed after a derived modality was written.
    """
    check_cache_source_modalities(log_dir)

    try:
        scene_metadatas = _get_scene_metadatas_from_log(log_dir, filter, target_uuids_binary)
    except Exception as e:
        logger.warning("Error extracting scenes from %s: %s", log_dir, e)
        logger.debug("Full traceback for %s:", log_dir, exc_info=True)
        return []

    scenes: List[SceneAPI] = []
    for scene_metadata in scene_metadatas:
        scenes.append(ArrowSceneAPI(log_dir=log_dir, scene_metadata=scene_metadata, maps_root=maps_root))
    return scenes


def _get_scene_metadatas_from_log(
    log_dir: Path,
    filter: SceneFilter,
    target_uuids_binary: Optional[pa.Array] = None,
) -> List[SceneMetadata]:
    """Get scene metadatas from a log directory using the three-phase pipeline.

    Phase 1 (Category 2): Log-level metadata filtering
    Phase 2 (Category 3a/b): UUID pre-filtering + candidate scene generation
    Phase 3 (Category 3c): Scene-level filtering
    """
    sync_table = get_lru_cached_arrow_table(str(log_dir / "sync.arrow"))
    log_metadata = get_metadata_from_arrow_schema(sync_table.schema, LogMetadata)

    # Phase 1: Category 2 — metadata-level early rejection
    if not check_log_passes_metadata_filters(log_metadata, sync_table.column_names, filter):
        return []

    # Infer iteration duration and resolve stride
    iteration_duration_s = infer_iteration_duration_s(sync_table)
    stride = resolve_iteration_stride(filter, iteration_duration_s)
    if stride is None:
        return []  # Log incompatible with requested stride — warning already logged
    future_iterations, history_iterations = resolve_iteration_counts(filter, iteration_duration_s, stride)
    step_idx = resolve_scene_step_size(filter, iteration_duration_s, stride)

    # Phase 2: Category 3a — UUID pre-filtering (skip full scan if UUIDs specified)
    scene_uuid_indices = None
    if target_uuids_binary is not None:
        scene_uuid_indices = resolve_scene_uuid_indices(sync_table, target_uuids_binary)
        if scene_uuid_indices is None:
            return []

    # Phase 2: Category 3b — candidate scene generation
    candidates = generate_scene_metadatas(
        sync_table,
        log_metadata,
        future_iterations,
        history_iterations,
        iteration_duration_s,
        scene_uuid_indices,
        stride,
        step_idx,
    )

    # Phase 3: Category 3c — scene-level filtering
    result = filter_scene_metadata_candidates(candidates, filter, sync_table, log_dir)
    return result


# --- Category 4: Post-filtering ---


def _apply_post_filters(scenes: List[SceneAPI], filter: SceneFilter) -> List[SceneAPI]:
    """Apply post-filtering options (Category 4) to the collected scenes.

    :param scenes: List of SceneAPI objects.
    :param filter: The scene filter.
    :return: Filtered list of SceneAPI objects.
    """
    # 4.1 Custom filter functions
    if filter.custom_filter_fns is not None:
        for fn in filter.custom_filter_fns:
            scenes = [s for s in scenes if fn(s)]

    # 4.2 Chunking
    if filter.num_chunks is not None and filter.chunk_idx is not None:
        chunk_size = max(1, len(scenes) // filter.num_chunks)
        start = filter.chunk_idx * chunk_size
        end = start + chunk_size if filter.chunk_idx < filter.num_chunks - 1 else len(scenes)
        scenes = scenes[start:end]

    # 4.3 Shuffle and cap max number of scenes
    if filter.shuffle:
        random.shuffle(scenes)

    if filter.max_num_scenes is not None:
        scenes = scenes[: filter.max_num_scenes]

    return scenes
