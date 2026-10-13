import threading
from pathlib import Path
from typing import Dict, Optional, Union

import pyarrow as pa
from cachetools import LRUCache

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
        # Per-path locks so distinct files can load concurrently. The global lock
        # only guards cache bookkeeping; holding it across the (potentially
        # seconds-long) IPC load would serialize cold opens of unrelated files.
        self._load_locks: Dict[str, threading.Lock] = {}

    def get(self, path: str) -> pa.Table:
        with self._lock:
            entry = self._cache.get(path)
            if entry is not None:
                return entry[1]
            load_lock = self._load_locks.setdefault(path, threading.Lock())

        with load_lock:
            # Another thread may have finished loading this path while we waited.
            with self._lock:
                entry = self._cache.get(path)
                if entry is not None:
                    return entry[1]

            # Open without a `with` block — source must stay open so the
            # table's buffers remain valid (zero-copy, backed by the mmap).
            source = pa.memory_map(path, "rb")
            try:
                table: pa.Table = pa.ipc.open_file(source).read_all()
            except BaseException:
                source.close()
                with self._lock:
                    self._load_locks.pop(path, None)
                raise
            # The load lock must only be released for other threads after the cache
            # insert: popping it earlier opens a window in which a late arrival misses
            # the cache, creates a fresh lock, and performs a duplicate load whose
            # cache overwrite would drop this (source, table) pair without closing it.
            with self._lock:
                self._cache[path] = (source, table)
                self._load_locks.pop(path, None)
            return table

    def clear(self) -> None:
        """Close every cached memory map and empty the cache."""
        with self._lock:
            while self._cache:
                self._cache.popitem()


# Lazy init: the cache size is resolved on first access from
# py123d.common.runtime (Hydra/env-var driven), not at import time.
_store: Optional[_ArrowMmapStore] = None
_store_init_lock = threading.Lock()


def _get_store() -> _ArrowMmapStore:
    global _store  # noqa: PLW0603
    if _store is None:
        with _store_init_lock:
            if _store is None:
                from py123d.common.runtime import get_cache_settings

                _store = _ArrowMmapStore(maxsize=get_cache_settings().max_lru_cached_tables)
    return _store


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
    This bounds the number of open memory maps to
    ``cache_settings.max_lru_cached_tables`` regardless of how many API objects
    exist or how many unique paths are accessed over the lifetime of the process.

    Thread-safe: a single lock guards cache lookup and insertion together,
    preventing duplicate mmaps when two threads request the same path
    simultaneously.

    :param arrow_file_path: The path to the arrow file.
    :return: The cached memory-mapped arrow table.
    """
    return _get_store().get(str(arrow_file_path))


def clear_arrow_table_cache() -> None:
    """Empty the cache behind :func:`get_lru_cached_arrow_table` and close its memory maps.

    Call it after rewriting an `.arrow` file that was read before, since the cache would keep
    returning the old table, or to release the maps of files that will not be read again.
    """
    if _store is not None:
        _store.clear()
