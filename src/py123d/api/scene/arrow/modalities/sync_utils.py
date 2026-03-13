"""Shared sync-table utilities for Arrow modality readers.

These functions resolve the mapping from sync-table iterations to per-modality row indices.
"""

from pathlib import Path
from typing import List, Optional

import pyarrow as pa

from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.datatypes.time.timestamp import Timestamp


def get_modality_table(log_dir: Path, modality_key: str) -> Optional[pa.Table]:
    """Load the Arrow table for the given modality, or None if the file does not exist.

    :param log_dir: Path to the log directory.
    :param modality_key: The modality file stem (e.g. ``"ego_state_se3"``).
    :return: The cached Arrow table, or None.
    """
    file_path = log_dir / f"{modality_key}.arrow"
    if file_path.exists():
        return get_lru_cached_arrow_table(file_path)
    return None


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


def get_first_sync_index(sync_table: pa.Table, column_name: str, idx: int) -> Optional[int]:
    """Extracts the first row index from a sync table column.

    Handles both scalar (``pa.int64``) and list-typed (``pa.list_(pa.int64())``) sync columns.
    For list-typed columns, returns the first element of the list (earliest observation in the interval).

    :param sync_table: The sync Arrow table.
    :param column_name: The sync column name for the modality.
    :param idx: The row index into the sync table.
    :return: The first referenced row index, or None.
    """
    value = sync_table[column_name][idx].as_py()
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if len(value) > 0 else None
    return value


def get_last_sync_index(sync_table: pa.Table, column_name: str, idx: int) -> Optional[int]:
    """Extracts the last row index from a sync table column.

    Handles both scalar (``pa.int64``) and list-typed (``pa.list_(pa.int64())``) sync columns.
    For list-typed columns, returns the last element of the list (latest observation in the interval).

    :param sync_table: The sync Arrow table.
    :param column_name: The sync column name for the modality.
    :param idx: The row index into the sync table.
    :return: The last referenced row index, or None.
    """
    value = sync_table[column_name][idx].as_py()
    if value is None:
        return None
    if isinstance(value, list):
        return value[-1] if len(value) > 0 else None
    return value


def get_all_modality_timestamps(
    log_dir: Path,
    sync_table: pa.Table,
    scene_metadata: SceneMetadata,
    modality_key: str,
    timestamp_column: str,
) -> List[Timestamp]:
    """Batch-read all timestamps for a modality within the current scene.

    Finds the first and last referenced row indices in the sync table for the scene range,
    then returns all timestamps from the modality table between those rows (inclusive).
    This correctly handles async modalities (e.g. lidar) where multiple entries may exist
    per sync frame.

    :param log_dir: Path to the log directory.
    :param sync_table: The sync Arrow table.
    :param scene_metadata: The scene metadata defining the iteration range.
    :param modality_key: The sync table column name / modality table name.
    :param timestamp_column: The column name in the modality table containing timestamps.
    :return: All timestamps in the modality table within the scene range, ordered by time.
    """
    modality_table = get_modality_table(log_dir, modality_key)
    if modality_table is None:
        return []

    initial_idx = scene_metadata.initial_idx
    end_idx = scene_metadata.end_idx  # exclusive

    # Find first referenced row index (scan forward)
    first_row: Optional[int] = None
    for i in range(initial_idx, end_idx):
        first_row = get_first_sync_index(sync_table, modality_key, i)
        if first_row is not None:
            break

    if first_row is None:
        return []

    # Find last referenced row index (scan backward)
    last_row: Optional[int] = None
    for i in range(end_idx - 1, initial_idx - 1, -1):
        last_row = get_last_sync_index(sync_table, modality_key, i)
        if last_row is not None:
            break

    if last_row is None:
        return []

    # Read all timestamps between first and last rows (inclusive)
    ts_column = modality_table[timestamp_column]
    return [Timestamp.from_us(ts_column[i].as_py()) for i in range(first_row, last_row + 1)]
