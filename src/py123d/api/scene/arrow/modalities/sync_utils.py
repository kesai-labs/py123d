"""Shared sync-table utilities for Arrow modality readers.

These functions resolve the mapping from sync-table iterations to per-modality row indices.
"""

import threading
import weakref
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.time.timestamp import Timestamp


def get_modality_table(log_dir: Path, modality_key: str) -> Optional[pa.Table]:
    """Load the Arrow table for the given modality, or None if the file does not exist.

    :param log_dir: Path to the log directory.
    :param modality_key: The modality file stem (e.g. ``"ego_state_se3"``).
    :return: The cached Arrow table, or None.
    """
    file_path = log_dir / f"{modality_key}.arrow"
    table: Optional[pa.Table] = None
    if file_path.exists():
        table = get_lru_cached_arrow_table(file_path)
    return table


# Timestamp columns as NumPy arrays, cached per table object and column, with whether they never
# decrease. A lookup by timestamp runs once per frame when a log is replayed; converting the whole
# column on every call (a copy across all record batches) made each lookup cost grow with the log's
# length, so replaying a log took time quadratic in its length. pyarrow tables are not hashable, so
# entries are keyed by id() and removed when their table is garbage-collected.
_TIMESTAMP_ARRAYS: Dict[int, Dict[str, Tuple[npt.NDArray, bool]]] = {}
_TIMESTAMP_ARRAYS_LOCK = threading.Lock()


def get_timestamp_array(table: pa.Table, column: str) -> Tuple[npt.NDArray, bool]:
    """Returns the values of a timestamp column as a read-only NumPy array, and whether they never decrease.

    Computed once per table object and column, and cached until the table is garbage-collected.

    :param table: The Arrow table.
    :param column: The timestamp column name (e.g. ``"lidar.lidar_top.timestamp_us"``).
    :return: Tuple of (values, sorted), where sorted is True when no value is smaller than its predecessor.
    """
    key = id(table)
    with _TIMESTAMP_ARRAYS_LOCK:
        cached = _TIMESTAMP_ARRAYS.get(key, {}).get(column)
    if cached is None:
        values = table[column].to_numpy()
        # A column with nulls converts to floats with NaN; NaN compares False, so it counts as unsorted.
        is_sorted = bool(len(values) < 2 or np.all(values[1:] >= values[:-1]))
        values.setflags(write=False)
        cached = (values, is_sorted)
        with _TIMESTAMP_ARRAYS_LOCK:
            if key not in _TIMESTAMP_ARRAYS:
                _TIMESTAMP_ARRAYS[key] = {}
                weakref.finalize(table, _TIMESTAMP_ARRAYS.pop, key, None)
            _TIMESTAMP_ARRAYS[key][column] = cached
    return cached


def find_timestamp_index(
    ts_array: npt.NDArray,
    target_us: int,
    criteria: Literal["exact", "nearest", "forward", "backward"],
    is_sorted: bool,
) -> Optional[int]:
    """Returns the row index matching a timestamp.

    ``"exact"`` is the first row equal to the target, ``"nearest"`` the first row with the smallest absolute
    difference, ``"forward"`` the first row at or after the target and ``"backward"`` the last row at or before
    it, all in row order. On sorted values a binary search finds them (O(log n)); otherwise the values are
    scanned in full, which gives the same results.

    :param ts_array: The timestamps in microseconds, e.g. from :func:`get_timestamp_array`.
    :param target_us: The target timestamp in microseconds.
    :param criteria: The matching strategy.
    :param is_sorted: Whether the values never decrease.
    :return: The row index, or None if no row matches.
    """
    n = len(ts_array)
    index: Optional[int] = None
    if not is_sorted:
        if criteria == "exact":
            indices = np.where(ts_array == target_us)[0]
            if len(indices) > 0:
                index = int(indices[0])
        elif criteria == "nearest":
            if n > 0:
                index = int(np.argmin(np.abs(ts_array - target_us)))
        elif criteria == "forward":
            indices = np.where(ts_array >= target_us)[0]
            if len(indices) > 0:
                index = int(indices[0])
        elif criteria == "backward":
            indices = np.where(ts_array <= target_us)[0]
            if len(indices) > 0:
                index = int(indices[-1])
        return index

    first_at_or_after = int(np.searchsorted(ts_array, target_us, side="left"))
    if criteria == "exact":
        if first_at_or_after < n and ts_array[first_at_or_after] == target_us:
            index = first_at_or_after
    elif criteria == "forward":
        if first_at_or_after < n:
            index = first_at_or_after
    elif criteria == "backward":
        last_at_or_before = int(np.searchsorted(ts_array, target_us, side="right")) - 1
        if last_at_or_before >= 0:
            index = last_at_or_before
    elif criteria == "nearest" and n > 0:
        # Candidates: the first row at or after the target and the first row holding the value just before
        # it. On equal distance the earlier row wins, as with argmin over all rows.
        if first_at_or_after == n:
            index = int(np.searchsorted(ts_array, ts_array[n - 1], side="left"))
        elif first_at_or_after == 0:
            index = 0
        else:
            before = ts_array[first_at_or_after - 1]
            if target_us - before <= ts_array[first_at_or_after] - target_us:
                index = int(np.searchsorted(ts_array, before, side="left"))
            else:
                index = first_at_or_after
    return index


def get_sync_table(log_dir: Path) -> pa.Table:
    """Load the sync table. This must always exist.

    :param log_dir: Path to the log directory.
    :return: The sync Arrow table.
    :raises AssertionError: If ``sync.arrow`` is missing.
    """
    file_path = log_dir / "sync.arrow"
    table = get_lru_cached_arrow_table(file_path)
    assert table is not None, f"sync.arrow not found in {log_dir}"
    return table


def get_modality_index_from_sync_index(sync_table: pa.Table, column_name: str, idx: int) -> Optional[int]:
    """Extracts the row index from a sync table column.

    :param sync_table: The sync Arrow table.
    :param column_name: The sync column name for the modality.
    :param idx: The row index into the sync table.
    :return: The referenced row index, or None if the modality has no observation at this sync frame.
    """
    return sync_table[column_name][idx].as_py()


def _get_scene_sync_range(scene_metadata: SceneMetadata, include_history: bool = False) -> tuple:
    """Returns the (start_idx, end_idx) range into the sync table for the scene.

    :param scene_metadata: The scene metadata defining the iteration range.
    :param include_history: If True, extend the range to include history iterations.
    :return: Tuple of (start_idx, end_idx) where end_idx is exclusive.
    """
    start_idx = (
        scene_metadata.initial_idx - scene_metadata.num_history_iterations * scene_metadata.target_iteration_stride
        if include_history
        else scene_metadata.initial_idx
    )
    end_idx = scene_metadata.end_idx  # exclusive
    return start_idx, end_idx


def get_all_modality_timestamps(
    log_dir: Path,
    sync_table: pa.Table,
    scene_metadata: SceneMetadata,
    modality_key: str,
    timestamp_column: str,
    include_history: bool = False,
) -> List[Timestamp]:
    """Batch-read all timestamps for a modality within the current scene.

    Finds the first and last referenced row indices in the sync table for the scene range,
    then returns all timestamps from the modality table between those rows (inclusive).

    :param log_dir: Path to the log directory.
    :param sync_table: The sync Arrow table.
    :param scene_metadata: The scene metadata defining the iteration range.
    :param modality_key: The sync table column name / modality table name.
    :param timestamp_column: The column name in the modality table containing timestamps.
    :param include_history: If True, include history iterations before the scene start.
    :return: All timestamps in the modality table within the scene range, ordered by time.
    """
    modality_table = get_modality_table(log_dir, modality_key)
    if modality_table is None:
        return []

    start_idx, end_idx = _get_scene_sync_range(scene_metadata, include_history)
    stride = scene_metadata.target_iteration_stride

    # Find first referenced row index (scan forward through strided frames)
    first_row: Optional[int] = None
    for i in range(start_idx, end_idx, stride):
        first_row = get_modality_index_from_sync_index(sync_table, modality_key, i)
        if first_row is not None:
            break

    if first_row is None:
        return []

    # Find last referenced row index (scan backward through strided frames)
    last_row: Optional[int] = None
    for i in range(start_idx + ((end_idx - 1 - start_idx) // stride) * stride, start_idx - 1, -stride):
        last_row = get_modality_index_from_sync_index(sync_table, modality_key, i)
        if last_row is not None:
            break

    if last_row is None:
        return []

    # Read all timestamps between first and last rows (inclusive)
    ts_column = modality_table[timestamp_column]
    return [Timestamp.from_us(ts_column[i].as_py()) for i in range(first_row, last_row + 1)]
