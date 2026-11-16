from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Literal, Optional

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.sync_utils import find_timestamp_index, get_timestamp_array
from py123d.datatypes import BaseModality, BaseModalityMetadata, Timestamp

# Default flush threshold, well under Arrow's 2 GiB per-column limit for binary/string data.
DEFAULT_MAX_BATCH_BYTES: int = 128 * 1024 * 1024


def _row_variable_width_nbytes(row: Dict[str, Any], columns: FrozenSet[str]) -> int:
    """Sum the size of a row's variable-width (binary/string) values.

    Only these count against Arrow's 2 GiB per-column offset limit; fixed-width columns are
    already bounded by the row cap.

    :param row: A buffered row, where each value is a single-element list.
    :param columns: The schema columns. Keys outside these are dropped on flush, so they are
        excluded here to keep the budget aligned with what is actually written.
    :return: The total size in bytes of the row's binary and string values.
    """
    nbytes = 0
    for column, value in row.items():
        if column not in columns or not isinstance(value, (list, tuple)) or len(value) == 0:
            continue
        item = value[0]
        if isinstance(item, bytes):
            nbytes += len(item)
        elif isinstance(item, str):
            # str.isascii() reads a cached flag, so long ASCII values skip encoding a throwaway copy.
            nbytes += len(item) if item.isascii() else len(item.encode("utf-8"))
    return nbytes


class ArrowBaseModalityWriter:
    """Manages a single Arrow IPC file for one modality."""

    def __init__(
        self,
        file_path: Path,
        schema: pa.Schema,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    ) -> None:
        def _get_compression() -> Optional[pa.Codec]:
            """Returns the IPC compression codec, or None if no compression is configured."""
            if ipc_compression is not None:
                return pa.Codec(ipc_compression, compression_level=ipc_compression_level)
            return None

        self._file_path = file_path
        self._schema = schema
        self._schema_columns: FrozenSet[str] = frozenset(schema.names)
        self._row_count: int = 0
        self._max_batch_size = max_batch_size
        self.set_max_batch_bytes(max_batch_bytes)
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_bytes: int = 0
        self._source = pa.OSFile(str(file_path), "wb")
        options = pa.ipc.IpcWriteOptions(compression=_get_compression())
        self._writer = pa.ipc.new_file(self._source, schema=schema, options=options)

    @property
    def row_count(self) -> int:
        """Returns the total number of rows written (including buffered)."""
        return self._row_count

    def set_max_batch_bytes(self, max_batch_bytes: int) -> None:
        """Set the buffer's byte budget, taking effect on the next buffered row.

        Lets the log writer apply a configured budget uniformly, without every modality writer
        forwarding the argument through its own constructor.

        :param max_batch_bytes: The maximum buffered size in bytes before an automatic flush.
        :raises ValueError: If the budget is not positive.
        """
        if max_batch_bytes <= 0:
            raise ValueError(f"max_batch_bytes must be > 0, got {max_batch_bytes}.")
        self._max_batch_bytes = max_batch_bytes

    def write_batch(self, data: Dict[str, Any]) -> None:
        """Buffer a single row and flush when the row or byte budget is reached."""
        self._row_count += 1

        if self._max_batch_size is None:
            batch = pa.record_batch(data, schema=self._schema)
            self._writer.write_batch(batch)  # type: ignore
            return

        self._buffer.append(data)
        self._buffer_bytes += _row_variable_width_nbytes(data, self._schema_columns)
        if len(self._buffer) >= self._max_batch_size or self._buffer_bytes >= self._max_batch_bytes:
            self._flush_buffer()

    def write_modality(self, modality: BaseModality) -> None:
        """Writes modality data to the Arrow file. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement write_modality()")

    def _flush_buffer(self) -> None:
        """Write buffered rows as a single record batch.

        Each buffered row is a dict where every value is a single-element list (one row).
        We concatenate these lists to form a multi-row batch.
        """
        if not self._buffer:
            return
        merged = {col: [] for col in self._schema.names}
        for row in self._buffer:
            for col in self._schema.names:
                if col in row.keys():
                    merged[col].append(row[col][0])
                else:
                    merged[col].append(None)

        batch = pa.record_batch(merged, schema=self._schema)
        self._writer.write_batch(batch)  # type: ignore
        self._buffer.clear()
        self._buffer_bytes = 0

    def close(self) -> None:
        self._flush_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._source is not None:
            self._source.close()
            self._source = None
        if self._row_count > 1:
            self._validate_timestamp_order()

    def _validate_timestamp_order(self) -> None:
        """Read back the written file and verify that timestamps are monotonically increasing.

        :raises ValueError: If any timestamp is less than or equal to its predecessor.
        """
        ts_column: Optional[str] = None
        for name in self._schema.names:
            if name.endswith(".timestamp_us"):
                ts_column = name
                break
        if ts_column is None:
            raise ValueError(
                f"No timestamp column (ending with '.timestamp_us') found in schema of '{self._file_path.name}'. "
                f"Available columns: {self._schema.names}"
            )

        with pa.memory_map(str(self._file_path), "rb") as source:
            table = pa.ipc.open_file(source).read_all()
        ts_array = table.column(ts_column).to_numpy()
        diffs = np.diff(ts_array)
        violating = np.where(diffs < 0)[0]
        if len(violating) > 0:
            idx = int(violating[0])
            raise ValueError(
                f"Timestamps must be monotonically increasing in '{self._file_path.name}'. "
                f"Got {ts_array[idx + 1]} after {ts_array[idx]} at row {idx + 1}."
            )


class ArrowBaseModalityReader(ABC):
    """Base class for stateless Arrow modality readers.

    All readers follow a common 3-step pattern:
    1. Load the modality table from ``log_dir`` using ``metadata.modality_key``.
    2. Resolve the row index via the sync table.
    3. Deserialize the row into a domain object.
    """

    @staticmethod
    @abstractmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[BaseModality]:
        """Reads and deserializes a modality from the given table at the specified row index."""

    @staticmethod
    @abstractmethod
    def read_column_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        column: str,
        dataset: str,
        deserialize: bool = False,
        **kwargs,
    ) -> Optional[Any]:
        """Return a single column value at the given row index.

        The default implementation returns the raw value (via ``as_py()``).
        Subclasses can override to support deserialization.

        :param index: The row index in the Arrow table.
        :param table: The Arrow modality table.
        :param metadata: The modality metadata.
        :param column: The field name (e.g. ``"imu_se3"``, ``"timestamp_us"``).
        :param deserialize: If True, deserialize the value to its domain type.
        :return: The column value, or None if the column is not present.
        """

    @classmethod
    def read_at_timestamp(
        cls,
        timestamp: Timestamp,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        criteria: Literal["exact", "nearest", "forward", "backward"] = "exact",
        **kwargs,
    ) -> Optional[BaseModality]:
        """Read a modality entry by timestamp lookup.

        :param timestamp: The target timestamp.
        :param table: The Arrow modality table.
        :param metadata: The modality metadata.
        :param dataset: The dataset name (for sensor path resolution).
        :param criteria: Matching strategy: ``"exact"`` requires an exact match,
            ``"nearest"`` picks the entry with the smallest absolute time difference,
            ``"forward"`` picks the first entry at or after the target,
            ``"backward"`` picks the last entry at or before the target.
        :param kwargs: Additional keyword arguments forwarded to :meth:`read_at_index`.
        :return: The deserialized modality, or None if no matching timestamp is found.
        """
        ts_column = f"{metadata.modality_key}.timestamp_us"
        ts_array, is_sorted = get_timestamp_array(table, ts_column)
        index = find_timestamp_index(ts_array, timestamp.time_us, criteria, is_sorted)

        modality: Optional[BaseModality] = None
        if index is not None:
            modality = cls.read_at_index(index, table, metadata, dataset, **kwargs)
        return modality
