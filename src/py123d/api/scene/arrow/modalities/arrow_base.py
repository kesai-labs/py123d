from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.sync_utils import get_all_modality_timestamps
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.time.timestamp import Timestamp


class ArrowBaseModalityWriter:
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

    def close(self) -> None:
        self._flush_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._source is not None:
            self._source.close()
            self._source = None


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
    ) -> Optional[BaseModality]:
        pass

    @staticmethod
    def read_all_timestamps(
        log_dir: Path,
        sync_table: pa.Table,
        scene_metadata: SceneMetadata,
        metadata: BaseModalityMetadata,
    ) -> List[Timestamp]:
        modality_key = metadata.modality_key
        return get_all_modality_timestamps(
            log_dir, sync_table, scene_metadata, modality_key, f"{modality_key}.timestamp_us"
        )
