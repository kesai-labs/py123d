import threading
from pathlib import Path
from typing import Final, Union

import pyarrow as pa
from cachetools import LRUCache

# TODO: Tune parameters and add to config?
MAX_LRU_CACHED_TABLES: Final[int] = 50_000


# ---------------------------------------------------------------------------
# Internal cache — keeps both the NativeFile (mmap) and the Table alive.
# On LRU eviction, the NativeFile is closed, releasing the mmap slot.
# ---------------------------------------------------------------------------


class _MmapLRUCache(LRUCache):
    """LRU cache that closes the underlying memory map on eviction."""

    def popitem(self) -> tuple:
        key, (source, _table) = super().popitem()
        if not source.closed:
            source.close()
        return key, (source, _table)


class _ArrowMmapStore:
    def __init__(self, maxsize: int) -> None:
        self._cache: _MmapLRUCache = _MmapLRUCache(maxsize=maxsize)
        self._lock = threading.Lock()

    def get(self, path: str) -> pa.Table:
        with self._lock:
            entry = self._cache.get(path)
            if entry is not None:
                return entry[1]

            # Open without a `with` block — source must stay open so the
            # table's buffers remain valid (zero-copy, backed by the mmap).
            source = pa.memory_map(path, "rb")
            table: pa.Table = pa.ipc.open_file(source).read_all()
            self._cache[path] = (source, table)
            return table


_store = _ArrowMmapStore(maxsize=MAX_LRU_CACHED_TABLES)


# ---------------------------------------------------------------------------
# Public API — unchanged signatures
# ---------------------------------------------------------------------------


def open_arrow_table(arrow_file_path: Union[str, Path]) -> pa.Table:
    """Open an `.arrow` file as memory map.

    :param arrow_file_path: The file path, defined as string or Path.
    :return: The memory-mapped arrow table.
    """
    with pa.memory_map(str(arrow_file_path), "rb") as source:
        table: pa.Table = pa.ipc.open_file(source).read_all()
    return table


def open_arrow_schema(arrow_file_path: Union[str, Path]) -> pa.Schema:
    """Loads an `.arrow` file schema.

    :param arrow_file_path: The file path, defined as string or Path.
    :return: The arrow schema.
    """
    with pa.memory_map(str(arrow_file_path), "rb") as source:
        schema: pa.Schema = pa.ipc.open_file(source).schema
    return schema


def read_arrow_table(arrow_file_path: Union[str, Path]) -> pa.Table:
    """Reads an arrow table from the file path.

    :param arrow_file_path: The file path, defined as string or Path.
    :return: The arrow table.
    """
    with pa.OSFile(str(arrow_file_path), "r") as source:
        table: pa.Table = pa.ipc.open_file(source).read_all()
    return table


def write_arrow_table(table: pa.Table, arrow_file_path: Union[str, Path]) -> None:
    """Writes an arrow table to the file path.

    :param table: The arrow table to write.
    :param arrow_file_path: The file path, defined as string or Path.
    """
    with pa.OSFile(str(arrow_file_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def get_lru_cached_arrow_table(arrow_file_path: Union[str, Path]) -> pa.Table:
    """Get a zero-copy memory-mapped arrow table from the LRU cache, or load
    it from disk on first access.

    The mmap slot is held open for as long as the entry remains in the cache.
    On LRU eviction the underlying NativeFile is closed, releasing the slot.
    This bounds the number of open memory maps to MAX_LRU_CACHED_TABLES
    regardless of how many API objects exist or how many unique paths are
    accessed over the lifetime of the process.

    Thread-safe: a single lock guards cache lookup and insertion together,
    preventing duplicate mmaps when two threads request the same path
    simultaneously.

    :param arrow_file_path: The path to the arrow file.
    :return: The cached memory-mapped arrow table.
    """
    return _store.get(str(arrow_file_path))
