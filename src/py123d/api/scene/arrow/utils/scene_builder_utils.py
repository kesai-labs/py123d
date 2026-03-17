"""Utility functions for ArrowSceneBuilderV3, organized by SceneFilter category.

Category 2: Metadata & log-level filtering
Category 3: Scene generation and scene-level filtering
"""

import logging
from typing import List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa

from py123d.api.scene.scene_filter import SceneFilter
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.common.utils.uuid_utils import convert_to_bytes_uuid, convert_to_str_uuid
from py123d.datatypes.metadata.log_metadata import LogMetadata

logger = logging.getLogger(__name__)


# --- Shared utilities ---


def infer_iteration_duration_s(sync_table: pa.Table) -> float:
    """Infer iteration duration from the sync table's timestamp column using the median of consecutive diffs.

    :param sync_table: The sync Arrow table with a ``sync.timestamp_us`` column.
    :return: Median iteration duration in seconds.
    :raises ValueError: If the sync table has fewer than 2 rows.
    """
    if sync_table.num_rows < 2:
        raise ValueError("Cannot infer iteration duration from a sync table with fewer than 2 rows.")

    timestamps_us = sync_table["sync.timestamp_us"].to_numpy()
    diffs_us = np.diff(timestamps_us)

    # NOTE @DanielDauner: We are using the median of all timestamp diffs, in case of outliers that could affect mean.
    iteration_duration_s = float(np.median(diffs_us)) / 1_000_000.0
    return iteration_duration_s


def resolve_iteration_counts(filter: SceneFilter, iteration_duration_s: float) -> Tuple[Optional[int], int]:
    """Resolve future/history iteration counts from filter parameters.

    Duration-based parameters take priority over iteration-based parameters.

    :param filter: The scene filter.
    :param iteration_duration_s: Inferred iteration duration in seconds.
    :return: Tuple of (future_iterations or None for full log, history_iterations).
    """
    # Future iterations
    if filter.future_duration_s is not None:
        future_iterations: Optional[int] = round(filter.future_duration_s / iteration_duration_s)
    elif filter.future_num_iterations is not None:
        future_iterations = filter.future_num_iterations
    else:
        future_iterations = None

    # History iterations
    if filter.history_duration_s is not None:
        history_iterations = round(filter.history_duration_s / iteration_duration_s)
    elif filter.history_num_iterations is not None:
        history_iterations = filter.history_num_iterations
    else:
        history_iterations = 0

    return future_iterations, history_iterations


# --- Category 2: Metadata & log-level filtering ---


def check_log_passes_metadata_filters(
    log_metadata: LogMetadata, sync_column_names: List[str], filter: SceneFilter
) -> bool:
    """Check whether a log passes all metadata-level filters (Category 2).

    Uses only log/map metadata and sync column names — no row data is read.

    :param log_metadata: The log's metadata.
    :param sync_column_names: Column names from the sync table.
    :param filter: The scene filter.
    :return: True if the log passes all filters.
    """
    # 2.1 Map-related
    map_meta = log_metadata.map_metadata

    if filter.has_map is True and map_meta is None:
        return False

    if filter.has_map is False and map_meta is not None:
        return False

    if filter.map_has_z is not None and map_meta is not None:
        if filter.map_has_z != map_meta.map_has_z:
            return False

    if filter.map_locations is not None:
        map_location = map_meta.location if map_meta is not None else None
        if map_location not in filter.map_locations:
            return False

    if filter.map_version is not None and map_meta is not None:
        if map_meta.version != filter.map_version:
            return False

    # 2.2 Log-related
    if filter.log_locations is not None:
        if log_metadata.location not in filter.log_locations:
            return False

    if filter.log_version is not None:
        if log_metadata.version != filter.log_version:
            return False

    if filter.required_log_modalities is not None:
        sync_column_set = set(sync_column_names)
        for modality_key in filter.required_log_modalities:
            if modality_key not in sync_column_set:
                return False

    return True


# --- Category 3a: Scene UUID pre-filtering ---


def scene_uuids_to_binary(scene_uuids: List[str]) -> pa.Array:
    """Convert a list of UUID or UUID strings to a binary(16) Arrow array."""
    return pa.array([convert_to_bytes_uuid(s) for s in scene_uuids], type=pa.binary(16))


def resolve_scene_uuid_indices(sync_table: pa.Table, target_uuids_binary: pa.Array) -> Optional[Set[int]]:
    """Look up sync table row indices matching the given binary UUID array.

    Uses Arrow-native ``isin`` for efficient matching without per-row Python conversion.

    :param sync_table: The sync Arrow table.
    :param target_uuids_binary: Pre-converted binary(16) Arrow array of target UUIDs.
    :return: Set of matching row indices, or None if no UUIDs were found.
    """
    mask = pa.compute.is_in(sync_table["sync.uuid"], value_set=target_uuids_binary)  # type: ignore
    indices = pa.compute.indices_nonzero(mask).to_pylist()  # type: ignore
    result: Optional[Set[int]] = set(indices) if len(indices) > 0 else None
    return result


# --- Category 3b: Candidate scene generation ---


def generate_scene_metadatas(
    sync_table: pa.Table,
    log_metadata: LogMetadata,
    future_iterations: Optional[int],
    history_iterations: int,
    iteration_duration_s: float,
    scene_uuid_indices: Optional[Set[int]] = None,
) -> List[SceneMetadata]:
    """Generate candidate SceneMetadata objects via temporal slicing.

    NOTE @DanielDauner: This function assumes that the sync table is sorted by time and that iteration duration
    is constant. We also needs this function to return metadatas in order to apply scene-level filters in the next step.

    :param sync_table: The sync Arrow table.
    :param log_metadata: The log metadata.
    :param future_iterations: Number of future iterations per scene, or None for full log.
    :param history_iterations: Number of history iterations per scene.
    :param iteration_duration_s: Inferred iteration duration in seconds.
    :param scene_uuid_indices: If provided, only generate scenes at these indices.
    :return: List of candidate SceneMetadata objects.
    """
    num_log_iterations = sync_table.num_rows
    uuid_column = sync_table["sync.uuid"]
    initial_idx = history_iterations

    if future_iterations is None:
        # Mode A: No future duration — each scene spans from its start index to the end of the log.
        # Without UUIDs: single scene from initial_idx.
        # With UUIDs: one scene per UUID position.
        if scene_uuid_indices is not None:
            candidate_indices = sorted(idx for idx in scene_uuid_indices if idx >= initial_idx)
        else:
            candidate_indices = [initial_idx]

        scene_metadatas: List[SceneMetadata] = []
        for idx in candidate_indices:
            num_future = max(num_log_iterations - idx - 1, 0)
            scene_metadatas.append(
                SceneMetadata(
                    dataset=log_metadata.dataset,
                    split=log_metadata.split,
                    initial_uuid=convert_to_str_uuid(uuid_column[idx].as_py()),
                    initial_idx=idx,
                    num_future_iterations=num_future,
                    num_history_iterations=history_iterations,
                    future_duration_s=num_future * iteration_duration_s,
                    history_duration_s=history_iterations * iteration_duration_s,
                    iteration_duration_s=iteration_duration_s,
                )
            )

    else:
        # Mode B: With future duration — each scene has fixed future and history iteration counts.
        # Without UUIDs: sliding window.
        # With UUIDs: scenes start at each UUID position, but only if a full future can fit until the end of the log.
        end_idx = num_log_iterations - future_iterations
        step_idx = max(future_iterations, 1)
        scene_metadatas: List[SceneMetadata] = []

        if scene_uuid_indices is not None:
            candidate_indices = sorted(idx for idx in scene_uuid_indices if initial_idx <= idx < end_idx)
        else:
            candidate_indices = list(range(initial_idx, end_idx, step_idx))

        for idx in candidate_indices:
            scene_metadatas.append(
                SceneMetadata(
                    dataset=log_metadata.dataset,
                    split=log_metadata.split,
                    initial_uuid=convert_to_str_uuid(uuid_column[idx].as_py()),
                    initial_idx=idx,
                    num_future_iterations=future_iterations,
                    num_history_iterations=history_iterations,
                    future_duration_s=future_iterations * iteration_duration_s,
                    history_duration_s=history_iterations * iteration_duration_s,
                    iteration_duration_s=iteration_duration_s,
                )
            )

    return scene_metadatas


# --- Category 3c: Scene-level filtering ---


def filter_scene_metadata_candidates(
    scene_metadatas: List[SceneMetadata],
    filter: SceneFilter,
    sync_table: pa.Table,
) -> List[SceneMetadata]:
    """Filter candidate scenes by scene-level criteria (Category 3).

    :param scene_metadatas: List of candidate SceneMetadata objects.
    :param filter: The scene filter.
    :param sync_table: The sync Arrow table.
    :return: Filtered list of SceneMetadata objects.
    """

    # 1. Required scene modalities: verify no nulls in scene's frame range
    if filter.required_scene_modalities is not None:
        sync_column_names = set(sync_table.column_names)
        modality_keys = [k for k in filter.required_scene_modalities if k in sync_column_names]
        if modality_keys:
            scene_metadatas = [
                s for s in scene_metadatas if _scene_has_complete_modalities(s, sync_table, modality_keys)
            ]

    # 2. Timestamp threshold: enforce minimum time gap between consecutive scenes
    #    timestamp_threshold_s takes priority over iteration_threshold.
    if filter.timestamp_threshold_s is not None:
        timestamps_us = sync_table["sync.timestamp_us"].to_numpy()
        filtered: List[SceneMetadata] = []
        for scene in scene_metadatas:
            if len(filtered) > 0:
                time_delta_s = float(timestamps_us[scene.initial_idx] - timestamps_us[filtered[-1].initial_idx]) / 1e6
                if time_delta_s < filter.timestamp_threshold_s:
                    continue
            filtered.append(scene)
        scene_metadatas = filtered
    elif filter.iteration_threshold is not None:
        filtered = []
        for scene in scene_metadatas:
            if len(filtered) > 0:
                iteration_delta = scene.initial_idx - filtered[-1].initial_idx
                if iteration_delta < filter.iteration_threshold:
                    continue
            filtered.append(scene)
        scene_metadatas = filtered

    return scene_metadatas


def _scene_has_complete_modalities(
    scene: SceneMetadata,
    sync_table: pa.Table,
    modality_keys: List[str],
) -> bool:
    """Check that all requested modality columns have no null values in the scene's frame range.

    :param scene: The scene metadata.
    :param sync_table: The sync Arrow table.
    :param modality_keys: List of sync table column names to check.
    :return: True if all modalities are complete (no nulls).
    """
    start = scene.initial_idx - scene.num_history_iterations
    length = scene.num_future_iterations + 1 + scene.num_history_iterations
    for key in modality_keys:
        column_slice = sync_table.column(key).slice(start, length)
        if column_slice.null_count > 0:
            return False
    return True
