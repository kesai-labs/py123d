from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
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
        **kwargs,
    ) -> Optional[BaseModality]:
        """Reads and deserializes a modality from the given table at the specified row index."""

    @staticmethod
    def get_column_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        column: str,
        deserialize: bool = False,
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
        full_column_name = f"{metadata.modality_key}.{column}"
        if full_column_name not in table.column_names:
            return None
        return table[full_column_name][index].as_py()

    @staticmethod
    def read_all_timestamps(
        log_dir: Path,
        sync_table: pa.Table,
        scene_metadata: SceneMetadata,
        metadata: BaseModalityMetadata,
    ) -> List[Timestamp]:
        modality_key = metadata.modality_key
        return get_all_modality_timestamps(
            log_dir,
            sync_table,
            scene_metadata,
            modality_key,
            f"{modality_key}.timestamp_us",
        )

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
        ts_array = table[ts_column].to_numpy()
        target_us = timestamp.time_us

        index: Optional[int] = None
        if criteria == "exact":
            indices = np.where(ts_array == target_us)[0]
            if len(indices) > 0:
                index = int(indices[0])
        elif criteria == "nearest":
            if len(ts_array) > 0:
                index = int(np.argmin(np.abs(ts_array - target_us)))
        elif criteria == "forward":
            indices = np.where(ts_array >= target_us)[0]
            if len(indices) > 0:
                index = int(indices[0])
        elif criteria == "backward":
            indices = np.where(ts_array <= target_us)[0]
            if len(indices) > 0:
                index = int(indices[-1])

        modality: Optional[BaseModality] = None
        if index is not None:
            modality = cls.read_at_index(index, table, metadata, dataset, **kwargs)
        return modality

    # @classmethod
    # def read_all_within_iterations(
    #     cls,
    #     table: pa.Table,
    #     sync_table: pa.Table,
    #     metadata: BaseModalityMetadata,
    #     dataset: str,
    #     start_iteration: int,
    #     end_iteration: int,
    #     initial_idx: int,
    #     **kwargs,
    # ) -> Iterator[BaseModality]:
    #     """Yield all modality entries for sync iterations in ``[start_iteration, end_iteration)``.

    #     :param table: The Arrow modality table.
    #     :param sync_table: The sync Arrow table.
    #     :param metadata: The modality metadata.
    #     :param dataset: The dataset name.
    #     :param start_iteration: First iteration (inclusive).
    #     :param end_iteration: Last iteration (exclusive).
    #     :param initial_idx: The scene's initial sync index (added to iterations).
    #     :param kwargs: Additional keyword arguments forwarded to :meth:`read_at_index`.
    #     """
    #     from py123d.api.scene.arrow.modalities.sync_utils import get_modality_index_from_sync_index

    #     modality_key = metadata.modality_key
    #     if modality_key not in sync_table.column_names:
    #         return

    #     for iteration in range(start_iteration, end_iteration):
    #         sync_index = initial_idx + iteration
    #         modality_index = get_modality_index_from_sync_index(sync_table, modality_key, sync_index)
    #         if modality_index is not None:
    #             modality = cls.read_at_index(modality_index, table, metadata, dataset, **kwargs)
    #             if modality is not None:
    #                 yield modality

    # @classmethod
    # def read_all_within_timestamps(
    #     cls,
    #     table: pa.Table,
    #     metadata: BaseModalityMetadata,
    #     dataset: str,
    #     start_timestamp: Timestamp,
    #     end_timestamp: Timestamp,
    #     **kwargs,
    # ) -> Iterator[BaseModality]:
    #     """Yield all modality entries with timestamps in ``[start_timestamp, end_timestamp]``.

    #     :param table: The Arrow modality table.
    #     :param metadata: The modality metadata.
    #     :param dataset: The dataset name.
    #     :param start_timestamp: Earliest timestamp (inclusive).
    #     :param end_timestamp: Latest timestamp (inclusive).
    #     :param kwargs: Additional keyword arguments forwarded to :meth:`read_at_index`.
    #     """
    #     ts_column = f"{metadata.modality_key}.timestamp_us"
    #     ts_array = table[ts_column].to_numpy()

    #     mask = (ts_array >= start_timestamp.time_us) & (ts_array <= end_timestamp.time_us)
    #     indices = np.where(mask)[0]

    #     for index in indices:
    #         modality = cls.read_at_index(int(index), table, metadata, dataset, **kwargs)
    #         if modality is not None:
    #             yield modality
