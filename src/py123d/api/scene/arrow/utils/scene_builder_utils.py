"""Utility functions for ArrowSceneBuilderV3, organized by SceneFilter category.

Category 2: Metadata & log-level filtering
Category 3: Scene generation and scene-level filtering
"""

import logging
from pathlib import Path
from typing import FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.utils.route_utils import (
    ROUTE_POSITION_KEY,
    read_remaining_route_m,
)
from py123d.api.scene.scene_filter import VALID_MODALITY_SCOPES, AnchorFilterContext, SceneFilter
from py123d.common.utils.uuid_utils import convert_to_bytes_uuid, convert_to_str_uuid
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.metadata.log_metadata import LogMetadata

logger = logging.getLogger(__name__)


# --- Shared utilities ---


def infer_iteration_duration_from_timestamps_us(timestamps_us: np.ndarray) -> float:
    """Infer iteration duration from a 1-D array of timestamps in microseconds.

    Uses the median of consecutive diffs to be robust against outliers.

    :param timestamps_us: Sorted int64 timestamp array.
    :return: Median iteration duration in seconds.
    :raises ValueError: If the array has fewer than 2 elements.
    """
    if len(timestamps_us) < 2:
        raise ValueError("Cannot infer iteration duration from fewer than 2 timestamps.")
    diffs_us = np.diff(timestamps_us)
    iteration_duration_s = float(np.median(diffs_us)) / 1_000_000.0
    return iteration_duration_s


def compute_stride_from_duration(
    target_duration_s: float,
    raw_duration_s: float,
    tolerance: float = 0.15,
) -> Optional[int]:
    """Compute an integer stride from a target iteration duration and the raw duration.

    Returns ``None`` if the target cannot be achieved within *tolerance*
    (e.g. upsampling or deviation exceeds the threshold).

    :param target_duration_s: Desired iteration duration in seconds.
    :param raw_duration_s: Raw (native) iteration duration in seconds.
    :param tolerance: Maximum relative deviation allowed (default 15 %).
    :return: The integer stride, or ``None`` if infeasible.
    """
    raw_stride = target_duration_s / raw_duration_s
    stride = round(raw_stride)
    result: Optional[int] = stride

    if stride < 1:
        logger.debug(
            "Cannot upsample: target_duration_s=%.4fs < raw duration=%.4fs.",
            target_duration_s,
            raw_duration_s,
        )
        result = None
    elif abs(raw_stride - stride) / stride > tolerance:
        logger.debug(
            "Target duration %.4fs not achievable with raw duration %.4fs (computed stride=%d, deviation=%.1f%%).",
            target_duration_s,
            raw_duration_s,
            stride,
            abs(raw_stride - stride) / stride * 100,
        )
        result = None

    return result


def infer_iteration_duration_s(sync_table: pa.Table) -> float:
    """Infer iteration duration from the sync table's timestamp column using the median of consecutive diffs.

    :param sync_table: The sync Arrow table with a ``sync.timestamp_us`` column.
    :return: Median iteration duration in seconds.
    :raises ValueError: If the sync table has fewer than 2 rows.
    """
    if sync_table.num_rows < 2:
        raise ValueError("Cannot infer iteration duration from a sync table with fewer than 2 rows.")
    timestamps_us = sync_table["sync.timestamp_us"].to_numpy()
    return infer_iteration_duration_from_timestamps_us(timestamps_us)


def resolve_iteration_stride(filter: SceneFilter, raw_iteration_duration_s: float) -> Optional[int]:
    """Resolve the iteration stride from filter parameters.

    ``target_iteration_duration_s`` takes priority over ``target_iteration_stride``.
    Returns ``None`` when the stride is infeasible for the given log (caller should skip).

    :param filter: The scene filter.
    :param raw_iteration_duration_s: Raw (native) iteration duration in seconds.
    :return: The resolved stride, or None if the target is infeasible.
    """
    stride: Optional[int] = 1
    if filter.target_iteration_duration_s is not None:
        stride = compute_stride_from_duration(filter.target_iteration_duration_s, raw_iteration_duration_s)
    elif filter.target_iteration_stride is not None:
        stride = filter.target_iteration_stride

    return stride


def resolve_scene_step_size(filter: SceneFilter, iteration_duration_s: float, stride: int) -> int:
    """Resolve the step (in raw frames) between consecutive scene anchors.

    Priority: ``timestamp_threshold_s`` > ``iteration_threshold`` > default (1).
    Result is clamped to ``>= 1`` to guarantee forward progress.

    :param filter: The scene filter.
    :param iteration_duration_s: Raw (native) iteration duration in seconds.
    :param stride: Resolved logical-iteration stride in raw frames.
    :return: Step between scene anchors in raw frames.
    """
    if filter.timestamp_threshold_s is not None:
        step = round(filter.timestamp_threshold_s / iteration_duration_s)
    elif filter.iteration_threshold is not None:
        # iteration_threshold is in logical iterations; multiply by stride for raw frames.
        step = round(filter.iteration_threshold * stride)
    else:
        step = 1
    return max(1, step)


def resolve_iteration_counts(
    filter: SceneFilter, iteration_duration_s: float, stride: int = 1
) -> Tuple[Optional[int], int]:
    """Resolve future/history iteration counts from filter parameters.

    Duration-based parameters take priority over iteration-based parameters.
    When stride > 1, duration-based params are converted using the effective per-iteration
    duration (``iteration_duration_s * stride``).

    :param filter: The scene filter.
    :param iteration_duration_s: Raw (native) iteration duration in seconds.
    :param stride: Iteration stride (number of raw frames per logical iteration).
    :return: Tuple of (future_iterations or None for full log, history_iterations).
    """
    effective_duration_s = iteration_duration_s * stride

    # Future iterations
    if filter.future_duration_s is not None:
        future_iterations: Optional[int] = round(filter.future_duration_s / effective_duration_s)
    elif filter.future_num_iterations is not None:
        future_iterations = filter.future_num_iterations
    else:
        future_iterations = None

    # History iterations
    if filter.history_duration_s is not None:
        history_iterations = round(filter.history_duration_s / effective_duration_s)
    elif filter.history_num_iterations is not None:
        history_iterations = filter.history_num_iterations
    else:
        history_iterations = 0

    return future_iterations, history_iterations


# --- Modality matching helpers ---


def _is_modality_pattern(requirement: str) -> bool:
    """Check whether a modality requirement string is a type pattern (contains ``:``)."""
    return ":" in requirement


def _parse_modality_pattern(requirement: str) -> Tuple[str, str]:
    """Extract (modality_type, quantifier) from a pattern string like ``"camera:any"``.

    :param requirement: A pattern requirement containing ``:``.
    :return: Tuple of (modality_type, quantifier).
    """
    modality_type, quantifier = requirement.split(":")
    return modality_type, quantifier


def _get_columns_matching_type(modality_type: str, column_names: Set[str]) -> List[str]:
    """Return all sync column names that belong to a modality type.

    A column matches if it equals the type name exactly (e.g., ``"ego_state_se3"``)
    or starts with ``"<type>."`` (e.g., ``"camera.pcam_f0"`` matches type ``"camera"``).

    :param modality_type: The modality type prefix to match (e.g., ``"camera"``).
    :param column_names: Set of sync table column names.
    :return: List of matching column names.
    """
    prefix = modality_type + "."
    result = [col for col in column_names if col == modality_type or col.startswith(prefix)]
    return result


def _split_scope(requirement: str) -> Tuple[str, FrozenSet[str]]:
    """Split a requirement into its (body, scope), defaulting to the whole-scene scope.

    The scope segments are assumed already validated (see ``_validate_modality_requirement``).

    :param requirement: A requirement string, possibly with an ``@scope`` suffix.
    :return: Tuple of (body without the ``@scope`` suffix, set of scope segments).
    """
    body, separator, scope_str = requirement.partition("@")
    scope = frozenset(scope_str.split("+")) if separator else VALID_MODALITY_SCOPES
    return body, scope


def _resolve_requirement(requirement: str, sync_column_set: Set[str]) -> Tuple[List[str], str, FrozenSet[str]]:
    """Resolve a requirement to its (candidate columns, quantifier, scope).

    An exact key is modeled as quantifier ``"all"`` over a single-element column list; a type pattern
    expands to all columns matching the type with its declared quantifier.

    :param requirement: A single requirement string (possibly with an ``@scope`` suffix).
    :param sync_column_set: Set of available sync column names.
    :return: Tuple of (candidate columns present in the table, quantifier, scope segments).
    """
    body, scope = _split_scope(requirement)
    if _is_modality_pattern(body):
        modality_type, quantifier = _parse_modality_pattern(body)
        columns = _get_columns_matching_type(modality_type, sync_column_set)
    else:
        quantifier = "all"
        columns = [body] if body in sync_column_set else []
    return columns, quantifier, scope


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

    if filter.required_scene_modalities is not None:
        sync_column_set = set(sync_column_names)
        for req_str in filter.required_scene_modalities:
            body = req_str.split("@", 1)[0]  # Existence is scope-independent; ignore any @scope suffix.
            if _is_modality_pattern(body):
                modality_type, _ = _parse_modality_pattern(body)
                if len(_get_columns_matching_type(modality_type, sync_column_set)) == 0:
                    return False
            elif body not in sync_column_set:
                return False

    if filter.min_remaining_route_m is not None and ROUTE_POSITION_KEY not in sync_column_names:
        logger.warning(
            "Log '%s' has no '%s' modality, required by min_remaining_route_m; the log is rejected. "
            "Reconvert it with write_route enabled or backfill it.",
            log_metadata.log_name,
            ROUTE_POSITION_KEY,
        )
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
    uuid_column = sync_table["sync.uuid"]
    # PyArrow >= 18 stores UUIDs as extension<arrow.uuid>; is_in doesn't support extension types,
    # so cast to the underlying binary(16) storage type.
    if hasattr(uuid_column.type, "storage_type"):
        uuid_column = uuid_column.cast(uuid_column.type.storage_type)
    mask = pa.compute.is_in(uuid_column, value_set=target_uuids_binary)  # type: ignore
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
    stride: int = 1,
    step_idx: int = 1,
) -> List[SceneMetadata]:
    """Generate candidate SceneMetadata objects via temporal slicing.

    NOTE @DanielDauner: This function assumes that the sync table is sorted by time and that iteration duration
    is constant. We also needs this function to return metadatas in order to apply scene-level filters in the next step.

    :param sync_table: The sync Arrow table.
    :param log_metadata: The log metadata.
    :param future_iterations: Number of future iterations per scene, or None for full log.
    :param history_iterations: Number of history iterations per scene.
    :param iteration_duration_s: Raw (native) iteration duration in seconds.
    :param scene_uuid_indices: If provided, only generate scenes at these indices (``step_idx`` ignored).
    :param stride: Iteration stride (number of raw frames per logical iteration).
    :param step_idx: Step in raw frames between consecutive scene anchors when ``scene_uuid_indices`` is None.
        Defaults to ``1`` (maximum-overlap sliding window).
    :return: List of candidate SceneMetadata objects.
    """
    step_idx = max(1, step_idx)
    num_log_iterations = sync_table.num_rows
    uuid_column = sync_table["sync.uuid"]
    initial_idx = history_iterations * stride
    effective_duration_s = iteration_duration_s * stride

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
            num_future = max((num_log_iterations - idx - 1) // stride, 0)
            scene_metadatas.append(
                SceneMetadata(
                    dataset=log_metadata.dataset,
                    split=log_metadata.split,
                    initial_uuid=convert_to_str_uuid(uuid_column[idx].as_py()),
                    initial_idx=idx,
                    num_future_iterations=num_future,
                    num_history_iterations=history_iterations,
                    future_duration_s=num_future * effective_duration_s,
                    history_duration_s=history_iterations * effective_duration_s,
                    iteration_duration_s=effective_duration_s,
                    target_iteration_stride=stride,
                )
            )

    else:
        # Mode B: With future duration — each scene has fixed future and history iteration counts.
        # Without UUIDs: sliding window stepping by ``step_idx`` raw frames between anchors.
        # With UUIDs: scenes start at each UUID position, but only if a full future can fit until the end of the log.
        end_idx = num_log_iterations - future_iterations * stride
        scene_metadatas: List[SceneMetadata] = []

        if scene_uuid_indices is not None:
            # UUIDs override stepping: one scene per UUID position, ``step_idx`` is ignored.
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
                    future_duration_s=future_iterations * effective_duration_s,
                    history_duration_s=history_iterations * effective_duration_s,
                    iteration_duration_s=effective_duration_s,
                    target_iteration_stride=stride,
                )
            )

    return scene_metadatas


# --- Category 3c: Scene-level filtering ---


def keep_anchors_with_min_remaining_route(
    sync_table: pa.Table,
    anchors: np.ndarray,
    min_remaining_route_m: float,
    log_dir: Path,
) -> np.ndarray:
    """Select the anchors with enough route left, read from the route_position modality's
    ``remaining_m`` column. Anchors without a route position (ego absent) are dropped —
    their remaining route cannot be established.

    Deliberately reads only the column, never the modality metadata: the polyline blob
    stays undecoded during scene enumeration.

    :param sync_table: The sync Arrow table, with the ``route_position`` index column.
    :param anchors: Candidate anchor rows, int64.
    :param min_remaining_route_m: Minimum remaining route in meters.
    :param log_dir: The log directory holding ``route_position.arrow``.
    :return: Boolean array over ``anchors``, True for the scenes to keep.
    """
    remaining_by_row = read_remaining_route_m(log_dir)
    if remaining_by_row is None:
        return np.zeros(len(anchors), dtype=bool)

    row_indices = sync_table[ROUTE_POSITION_KEY].to_numpy(zero_copy_only=False)  # float64, NaN where null
    anchor_rows = row_indices[anchors]
    keep = ~np.isnan(anchor_rows)
    keep[keep] = remaining_by_row[anchor_rows[keep].astype(np.int64)] >= min_remaining_route_m
    return keep


def filter_scene_metadata_candidates(
    scene_metadatas: List[SceneMetadata],
    filter: SceneFilter,
    sync_table: pa.Table,
    log_dir: Optional[Path] = None,
) -> List[SceneMetadata]:
    """Filter candidate scenes by scene-level criteria (Category 3).

    :param scene_metadatas: List of candidate SceneMetadata objects.
    :param filter: The scene filter.
    :param sync_table: The sync Arrow table.
    :param log_dir: The log directory; required when the filter reads per-frame data
        beyond the sync table (min_remaining_route_m) or carries custom anchor
        filter functions.
    :return: Filtered list of SceneMetadata objects.
    """

    # 1. Required scene modalities: verify no nulls within each requirement's temporal scope.
    result = scene_metadatas
    if filter.required_scene_modalities is not None:
        sync_column_set = set(sync_table.column_names)
        for req_str in filter.required_scene_modalities:
            columns, quantifier, scope = _resolve_requirement(req_str, sync_column_set)
            if quantifier == "all":
                result = [s for s in result if _scene_has_complete_modalities(s, sync_table, columns, scope)]
            else:  # "any"
                result = [s for s in result if _scene_has_any_complete_modality(s, sync_table, columns, scope)]

    # 2. Remaining route after each scene's anchor frame.
    if filter.min_remaining_route_m is not None and len(result) > 0:
        assert log_dir is not None, "min_remaining_route_m filtering requires the log_dir."
        anchors = np.array([scene.initial_idx for scene in result], dtype=np.int64)
        keep = keep_anchors_with_min_remaining_route(sync_table, anchors, filter.min_remaining_route_m, log_dir)
        result = [scene for scene, kept in zip(result, keep) if kept]

    # 3. Custom anchor filter functions, grouped by future count: candidates
    # differ in it only when scenes run to the log's end.
    if filter.custom_anchor_filter_fns is not None and len(result) > 0:
        assert log_dir is not None, "custom_anchor_filter_fns filtering requires the log_dir."
        anchors = np.array([scene.initial_idx for scene in result], dtype=np.int64)
        futures = np.array([scene.num_future_iterations for scene in result], dtype=np.int64)
        keep = np.ones(len(result), dtype=bool)
        for future_iterations in np.unique(futures):
            group = np.flatnonzero(futures == future_iterations)
            for filter_fn in filter.custom_anchor_filter_fns:
                kept_group = group[keep[group]]
                if len(kept_group) == 0:
                    break
                context = AnchorFilterContext(
                    log_dir=log_dir,
                    sync_table=sync_table,
                    anchors=anchors[kept_group],
                    history_iterations=result[0].num_history_iterations,
                    future_iterations=int(future_iterations),
                    stride=result[0].target_iteration_stride,
                )
                mask = np.asarray(filter_fn(context), dtype=bool)
                if mask.shape != context.anchors.shape:
                    raise ValueError(
                        f"custom anchor filter returned shape {mask.shape} for {len(context.anchors)} anchors"
                    )
                keep[kept_group] = mask
        result = [scene for scene, kept in zip(result, keep) if kept]

    return result


def _scope_sync_indices(scene: SceneMetadata, scope: FrozenSet[str]) -> List[int]:
    """Resolve the (sorted) union of sync-table indices a scope must check for a scene.

    The ``history`` / ``initial`` / ``future`` segments are the non-overlapping pieces of the scene's
    frame range, so the default scope (all segments) reproduces the whole-scene range exactly.

    :param scene: The scene metadata (anchor at ``initial_idx`` == logical iteration 0).
    :param scope: Temporal segments from :data:`~py123d.api.scene.scene_filter.VALID_MODALITY_SCOPES`.
    :return: Sorted list of sync-table indices to check for completeness.
    """
    stride = scene.target_iteration_stride
    initial = scene.initial_idx
    end = initial + scene.num_future_iterations * stride + 1
    indices: Set[int] = set()
    if "history" in scope:
        indices.update(range(initial - scene.num_history_iterations * stride, initial, stride))
    if "initial" in scope:
        indices.add(initial)
    if "future" in scope:
        indices.update(range(initial + stride, end, stride))
    return sorted(indices)


def _scene_has_complete_modalities(
    scene: SceneMetadata,
    sync_table: pa.Table,
    modality_keys: List[str],
    scope: FrozenSet[str] = VALID_MODALITY_SCOPES,
) -> bool:
    """Check that all requested modality columns have no null values within the scope's frames.

    Vacuously True when ``modality_keys`` is empty (an absent exact key or a type with no matching
    columns does not filter at the scene level — the log-level existence check handles those).

    :param scene: The scene metadata.
    :param sync_table: The sync Arrow table.
    :param modality_keys: List of sync table column names to check.
    :param scope: Temporal segments selecting which frames to check.
    :return: True if all modalities are complete (no nulls at the scoped indices).
    """
    indices = _scope_sync_indices(scene, scope)
    result = True
    for key in modality_keys:
        column = sync_table.column(key)
        if any(column[i].as_py() is None for i in indices):
            result = False
    return result


def _scene_has_any_complete_modality(
    scene: SceneMetadata,
    sync_table: pa.Table,
    modality_keys: List[str],
    scope: FrozenSet[str] = VALID_MODALITY_SCOPES,
) -> bool:
    """Check that at least one of the given modality columns is complete within the scope's frames.

    Returns False when ``modality_keys`` is empty (a type with no matching columns keeps no scenes).

    :param scene: The scene metadata.
    :param sync_table: The sync Arrow table.
    :param modality_keys: List of sync table column names to check.
    :param scope: Temporal segments selecting which frames to check.
    :return: True if at least one modality is complete (no nulls at the scoped indices).
    """
    indices = _scope_sync_indices(scene, scope)
    result = False
    for key in modality_keys:
        column = sync_table.column(key)
        if all(column[i].as_py() is not None for i in indices):
            result = True
    return result
