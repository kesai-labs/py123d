from functools import lru_cache
from pathlib import Path
from typing import Final, Union

import pyarrow as pa

# TODO: Tune Parameters and add to config?
MAX_LRU_CACHED_TABLES: Final[int] = 1_000


def open_arrow_table(arrow_file_path: Union[str, Path]) -> pa.Table:
    """Open an `.arrow` file as memory map.

    :param arrow_file_path: The file path, defined as string or Path.
    :return: The memory-mapped arrow table.s
    """

    with pa.memory_map(str(arrow_file_path), "rb") as source:
        table: pa.Table = pa.ipc.open_file(source).read_all()
    return table


def open_arrow_schema(arrow_file_path: Union[str, Path]) -> pa.Schema:
    """Loads an `.arrow` file schema.

    :param arrow_file_path: The file path, defined as string or Path.
    :return: The memory-mapped arrow schema.
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
        # with pa.ipc.new_file(sink, table.schema, options=options) as writer:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


@lru_cache(maxsize=MAX_LRU_CACHED_TABLES)
def get_lru_cached_arrow_table(arrow_file_path: Union[str, Path]) -> pa.Table:
    """Get a memory-mapped arrow table from the LRU cache or load it from disk.

    :param arrow_file_path: The path to the arrow file.
    :return: The cached memory-mapped arrow table.
    """

    # NOTE @DanielDauner: The number of memory maps that a process can have is limited by the
    # linux kernel parameter /proc/sys/vm/max_map_count (default: 65530 in most distributions).
    # Thus, we cache memory-mapped arrow tables with an LRU cache to avoid
    # hitting this limit, specifically since many scenes/routines access the same table.
    # During cache eviction, the functools implementation calls __del__ on the
    # evicted cache entry, which closes the memory map, if no other references to the table exist.
    # Thus it is beneficial to keep track of all references to the table, otherwise the memory map
    # will not be closed and the limit can still be hit.
    # Not fully satisfied with this solution. Please reach out if you have a better idea.

    return open_arrow_table(str(arrow_file_path))
