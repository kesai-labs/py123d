from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pyarrow as pa


class BaseModalityWriter:
    """Manages a single Arrow IPC file for one modality."""

    def __init__(
        self,
        file_path: Path,
        schema: pa.Schema,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ) -> None:
        def _get_compression() -> Optional[pa.Codec]:
            """Returns the IPC compression codec, or None if no compression is configured."""
            if ipc_compression is not None:
                return pa.Codec(ipc_compression, compression_level=ipc_compression_level)
            return None

        self._file_path = file_path
        self._schema = schema
        self._row_count: int = 0
        self._max_batch_size = max_batch_size
        self._buffer: List[Dict[str, Any]] = []
        self._source = pa.OSFile(str(file_path), "wb")
        options = pa.ipc.IpcWriteOptions(compression=_get_compression())
        self._writer = pa.ipc.new_file(self._source, schema=schema, options=options)

    @property
    def row_count(self) -> int:
        """Returns the total number of rows written (including buffered)."""
        return self._row_count

    def write_batch(self, data: Dict[str, Any]) -> None:
        """Buffer a single row and flush when the batch size is reached."""
        self._row_count += 1

        if self._max_batch_size is None:
            batch = pa.record_batch(data, schema=self._schema)
            self._writer.write_batch(batch)  # type: ignore
            return

        self._buffer.append(data)
        if len(self._buffer) >= self._max_batch_size:
            self._flush_buffer()

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

    def close(self) -> None:
        self._flush_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._source is not None:
            self._source.close()
            self._source = None
