import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import ArrowBoxDetectionsSE3Writer
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraWriter
from py123d.api.scene.arrow.modalities.arrow_custom_modality import ArrowCustomModalityWriter
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Writer
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarWriter
from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections import ArrowTrafficLightDetectionsWriter
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.api.scene.base_log_writer import BaseLogWriter
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.utils.uuid_utils import create_deterministic_uuid
from py123d.datatypes import LogMetadata
from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionsMetadata
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, LidarMetadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.parser.base_dataset_parser import ModalitiesSync

# Sync table column names (plain strings replacing the deleted ModalitySchema)
_SYNC_COL_UUID = "sync.uuid"
_SYNC_COL_TIMESTAMP_US = "sync.timestamp_us"


def _get_uuid_arrow_type() -> pa.DataType:
    """Gets the appropriate Arrow UUID data type based on pyarrow version."""
    if pa.__version__ >= "18.0.0":
        return pa.uuid()
    else:
        return pa.binary(16)


@dataclass(frozen=True)
class SyncConfig:
    """Configuration for deferred sync table construction.

    :param reference_column: Fully qualified column name used as the sync reference,
        e.g. ``"lidar.lidar_merged.timestamp_us"``.
    :param direction: Sync direction. ``"forward"`` uses intervals ``[ref_i, ref_{i+1})``,
        ``"backward"`` uses intervals ``(ref_{i-1}, ref_i]``.
    """

    reference_column: str
    direction: Literal["forward", "backward"] = "forward"

    @property
    def reference_modality(self) -> str:
        """The modality name, e.g. ``"lidar.lidar_merged"``."""
        parts = self.reference_column.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"reference_column must be '<modality>.<timestamp_field>', got '{self.reference_column}'")
        return parts[0]

    @property
    def reference_timestamp_field(self) -> str:
        """The timestamp field name, e.g. ``"timestamp_us"``."""
        return self.reference_column.rsplit(".", 1)[-1]


@dataclass
class ArrowLogWriterState:
    log_dir: Path
    log_metadata: LogMetadata
    modality_writers: Dict[str, ArrowBaseModalityWriter] = field(default_factory=dict)
    sync_rows: Optional[List[Dict[str, Union[bytes, int, List[int]]]]] = None


class ArrowLogWriter(BaseLogWriter):
    def __init__(
        self,
        log_writer_config: LogWriterConfig,
        logs_root: Union[str, Path],
        sensors_root: Union[str, Path],
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
        sync_config: Optional[SyncConfig] = None,
    ) -> None:
        """Initializes the :class:`ArrowLogWriter`.

        :param log_writer_config: The log writer configuration.
        :param logs_root: The root directory for logs.
        :param sensors_root: The root directory for sensors (e.g. MP4 video files).
        :param ipc_compression: The IPC compression method, defaults to None.
        :param ipc_compression_level: The IPC compression level, defaults to None.
        :param sync_config: Configuration for deferred sync table construction, defaults to None.
        """
        self._log_writer_config = log_writer_config
        self._logs_root = Path(logs_root)
        self._sensors_root = Path(sensors_root)
        self._ipc_compression: Optional[Literal["lz4", "zstd"]] = ipc_compression
        self._ipc_compression_level: Optional[int] = ipc_compression_level
        self._sync_config: Optional[SyncConfig] = sync_config

        self._state: Optional[ArrowLogWriterState] = None

    # Internal helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _close_writers(self) -> None:
        """Close all open modality writers."""
        if self._state is not None:
            for writer in self._state.modality_writers.values():
                writer.close()
            self._state.modality_writers.clear()

    def _include_modality(self, modality_metadata: BaseModalityMetadata) -> bool:
        """Determine whether to include a modality based on the dataset converter config."""
        _include_modality: bool = True
        if (modality_metadata.modality_key in self._log_writer_config.exclude_modality_keys) or (
            modality_metadata.modality_type.serialize() in self._log_writer_config.exclude_modality_types
        ):
            _include_modality = False
        return _include_modality

    def _build_modality_writer(self, modality_metadata: BaseModalityMetadata) -> ArrowBaseModalityWriter:
        """Create the Arrow writer(s) for a single modality metadata entry."""
        assert self._state is not None, "Log writer state is not initialized."
        modality_writer: Optional[ArrowBaseModalityWriter] = None

        if isinstance(modality_metadata, EgoStateSE3Metadata):
            modality_writer = ArrowEgoStateSE3Writer(
                log_dir=self._state.log_dir,
                metadata=modality_metadata,
                ipc_compression=self._ipc_compression,
                ipc_compression_level=self._ipc_compression_level,
            )

        elif isinstance(modality_metadata, BoxDetectionsSE3Metadata):
            modality_writer = ArrowBoxDetectionsSE3Writer(
                log_dir=self._state.log_dir,
                metadata=modality_metadata,
                ipc_compression=self._ipc_compression,
                ipc_compression_level=self._ipc_compression_level,
            )

        elif isinstance(modality_metadata, TrafficLightDetectionsMetadata):
            modality_writer = ArrowTrafficLightDetectionsWriter(
                log_dir=self._state.log_dir,
                metadata=modality_metadata,
                ipc_compression=self._ipc_compression,
                ipc_compression_level=self._ipc_compression_level,
            )

        elif isinstance(modality_metadata, BaseCameraMetadata):
            modality_writer = ArrowCameraWriter(
                log_dir=self._state.log_dir,
                metadata=modality_metadata,
                camera_codec=self._log_writer_config.camera_store_option,
                ipc_compression=self._ipc_compression,
                ipc_compression_level=self._ipc_compression_level,
            )

        elif isinstance(modality_metadata, (LidarMergedMetadata, LidarMetadata)):
            modality_writer = ArrowLidarWriter(
                log_dir=self._state.log_dir,
                metadata=modality_metadata,
                log_metadata=self._state.log_metadata,
                lidar_store_option=self._log_writer_config.lidar_store_option,
                lidar_codec=self._log_writer_config.lidar_codec,
                ipc_compression=self._ipc_compression,
                ipc_compression_level=self._ipc_compression_level,
            )

        elif isinstance(modality_metadata, CustomModalityMetadata):
            modality_writer = ArrowCustomModalityWriter(
                log_dir=self._state.log_dir,
                metadata=modality_metadata,
                ipc_compression=self._ipc_compression,
                ipc_compression_level=self._ipc_compression_level,
            )
        else:
            raise ValueError(f"Unsupported modality metadata type: {type(modality_metadata)}")

        assert modality_writer is not None, (
            f"Modality writer for {modality_metadata.modality_key} should be initialized at this point."
        )
        return modality_writer

    def _write_single_modality(self, modality: BaseModality) -> Optional[int]:
        """Write a single modality and return its row index, or None if skipped."""
        assert self._state is not None, "Log writer is not initialized. Call reset() first."
        row_idx: Optional[int] = None
        if self._include_modality(modality.metadata):
            if modality.metadata.modality_key not in self._state.modality_writers.keys():
                self._state.modality_writers[modality.metadata.modality_key] = self._build_modality_writer(
                    modality.metadata
                )
            modality_writer = self._state.modality_writers.get(modality.metadata.modality_key)
            assert modality_writer is not None, (
                f"Modality writer for '{modality.metadata.modality_key}' should be initialized at this point."
            )
            row_idx = modality_writer.row_count
            modality_writer.write_modality(modality)
        return row_idx

    # Base class method implementations
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self, log_metadata: LogMetadata) -> bool:
        """Inherited, see superclass."""
        assert self._state is None, "Log writer is already initialized. Call close() before reset()."
        log_dir: Path = self._logs_root / log_metadata.split / log_metadata.log_name
        sync_file_path = log_dir / "sync.arrow"
        _write_log: bool = False
        if not sync_file_path.exists() or self._log_writer_config.force_log_conversion:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._state = ArrowLogWriterState(log_dir=log_dir, log_metadata=log_metadata)
            _write_log = True
        return _write_log

    def write_sync(self, modalities_sync: ModalitiesSync) -> None:
        """Inherited, see superclass."""
        assert self._state is not None, "Log writer is not initialized. Call reset() first."

        timestamp = modalities_sync.timestamp
        uuid = create_deterministic_uuid(
            split=self._state.log_metadata.split,
            log_name=self._state.log_metadata.log_name,
            timestamp_us=timestamp.time_us,
        )
        sync_row_indices: Dict[str, Union[bytes, int, List[int]]] = {
            "sync.uuid": uuid.bytes,
            "sync.timestamp_us": [timestamp.time_us],
        }
        for modality in modalities_sync.modalities:
            row_idx = self._write_single_modality(modality)
            if row_idx is not None:
                sync_row_indices[modality.metadata.modality_key] = [row_idx]

        if self._state.sync_rows is None:
            self._state.sync_rows = []
        assert isinstance(self._state.sync_rows, list), "Expected sync_rows to be a list."
        self._state.sync_rows.append(sync_row_indices)

    def write_async(self, modality: BaseModality) -> None:
        """Inherited, see superclass."""
        assert self._state is not None, "Log writer is not initialized. Call reset() first."
        assert self._state.sync_rows is None, (
            "Calling ``write_async`` after ``write_sync`` is not supported in the current implementation."
        )
        self._write_single_modality(modality)

    def close(self) -> None:
        """Inherited, see superclass."""
        if self._state is not None:
            self._close_writers()

            if self._state.sync_rows is not None:
                self._build_sync_table_from_rows()
            else:
                # Modality writers are already closed so Arrow files are finalized and readable.
                # Read back the files to build the sync table.
                self._build_deferred_sync_table()

        self._state = None

    # ------------------------------------------------------------------------------------------------------------------
    # Sync table helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _write_sync_arrow_file(self, schema: pa.Schema, rows: List[Dict[str, Any]]) -> None:
        """Write the sync Arrow IPC file from pre-built rows.

        This is intentionally kept separate from modality writers — it writes directly
        to ``sync.arrow`` via a standalone IPC writer.

        :param schema: The Arrow schema for the sync table.
        :param rows: Each element is a single-row dict (values are single-element lists).
        """
        assert self._state is not None
        schema = add_metadata_to_arrow_schema(schema, self._state.log_metadata)
        sync_path = self._state.log_dir / "sync.arrow"

        options = None
        if self._ipc_compression is not None:
            options = pa.ipc.IpcWriteOptions(
                compression=pa.Codec(self._ipc_compression, compression_level=self._ipc_compression_level)
            )

        # Merge single-row dicts into one multi-row dict so the entire sync
        # table is written as a single record batch (avoids 1-row-per-batch
        # chunking which hurts downstream read performance).
        merged: Dict[str, List[Any]] = {col: [] for col in schema.names}
        for row in rows:
            for col in schema.names:
                merged[col].append(row[col][0])

        source = pa.OSFile(str(sync_path), "wb")
        writer = pa.ipc.new_file(source, schema=schema, options=options)
        try:
            writer.write_batch(pa.record_batch(merged, schema=schema))
        finally:
            writer.close()
            source.close()

    def _build_sync_table_from_rows(self) -> None:
        """Build the sync table from rows collected via :meth:`write_sync`.

        In sync mode each row has a scalar ``int64`` index per modality (one-to-one mapping).
        """
        assert self._state is not None
        assert self._state.sync_rows is not None

        if not self._state.sync_rows:
            return

        # Collect all modality keys that appear across rows (preserving insertion order).
        modality_keys: List[str] = []
        seen: set = set()
        for row in self._state.sync_rows:
            for key in row:
                if key not in seen and key not in {_SYNC_COL_UUID, _SYNC_COL_TIMESTAMP_US}:
                    modality_keys.append(key)
                    seen.add(key)

        # Build schema: uuid, timestamp_us, then one int64 column per modality.
        uuid_type = _get_uuid_arrow_type()
        fields: List[pa.Field] = [
            pa.field(_SYNC_COL_UUID, uuid_type),
            pa.field(_SYNC_COL_TIMESTAMP_US, pa.int64()),
        ]
        for key in modality_keys:
            fields.append(pa.field(key, pa.int64()))
        schema = pa.schema(fields)

        # Normalize rows: ensure every row has all columns (None for missing modalities).
        normalized_rows: List[Dict[str, Any]] = []
        for row in self._state.sync_rows:
            normalized: Dict[str, Any] = {
                _SYNC_COL_UUID: [row[_SYNC_COL_UUID]],
                _SYNC_COL_TIMESTAMP_US: row[_SYNC_COL_TIMESTAMP_US],
            }
            for key in modality_keys:
                normalized[key] = row.get(key, [None])
            normalized_rows.append(normalized)

        self._write_sync_arrow_file(schema, normalized_rows)

    def _build_deferred_sync_table(self) -> None:
        """Build the sync table by reading timestamps from the written Arrow files.

        After all modality writers have been closed, this method scans the log directory
        for ``*.arrow`` files, extracts timestamp columns (columns ending in
        ``timestamp_us``), and builds the sync table using the reference modality
        from :class:`SyncConfig`.

        Lidars have two timestamp columns (``timestamp_us`` and ``end_timestamp_us``).
        Which one to use is determined by the :attr:`SyncConfig.reference_column` when
        the lidar is the reference modality. For non-reference lidar addons, the first
        ``*timestamp_us`` column is used.
        """
        assert self._state is not None
        assert self._sync_config is not None, "SyncConfig is required for deferred sync."

        ref_modality_key = self._sync_config.reference_modality
        ref_timestamp_field = self._sync_config.reference_timestamp_field
        direction = self._sync_config.direction

        # Read timestamps from all written Arrow files
        timestamp_logs: Dict[str, List[Tuple[int, int]]] = {}  # modality_key -> [(row_idx, ts_us)]

        for arrow_path in sorted(self._state.log_dir.glob("*.arrow")):
            if arrow_path.name == "sync.arrow":
                continue

            modality_key = arrow_path.stem
            reader = pa.ipc.open_file(arrow_path)
            table = reader.read_all()

            # Find the timestamp column: use the specific field for the reference modality,
            # otherwise pick the first column ending in "timestamp_us".
            ts_col_name = None
            if modality_key == ref_modality_key:
                ts_col_name = f"{modality_key}.{ref_timestamp_field}"
            else:
                for col_name in table.column_names:
                    if col_name.endswith("timestamp_us"):
                        ts_col_name = col_name
                        break

            if ts_col_name is None or ts_col_name not in table.column_names:
                continue

            timestamps = table.column(ts_col_name).to_pylist()
            timestamp_logs[modality_key] = [(i, ts) for i, ts in enumerate(timestamps)]

        # Extract reference timestamps
        ref_entries = timestamp_logs.get(ref_modality_key, [])
        if not ref_entries:
            return

        ref_timestamps_us = [ts for _, ts in ref_entries]
        addon_names = list(timestamp_logs.keys())

        # Build schema: uuid, timestamp_us, then one int64 column per modality.
        uuid_type = _get_uuid_arrow_type()
        fields: List[pa.Field] = [
            pa.field(_SYNC_COL_UUID, uuid_type),
            pa.field(_SYNC_COL_TIMESTAMP_US, pa.int64()),
        ]
        for addon in addon_names:
            fields.append(pa.field(addon, pa.int64()))
        schema = pa.schema(fields)

        # Pre-extract sorted timestamp arrays for efficient bisect lookups
        addon_timestamps: Dict[str, List[int]] = {}
        for addon in addon_names:
            addon_timestamps[addon] = [ts for _, ts in timestamp_logs[addon]]

        # Build one sync row per reference timestamp, picking the single closest
        # modality observation within the interval defined by the sync direction.
        sync_rows: List[Dict[str, Any]] = []
        for ref_idx, (_, ref_ts) in enumerate(ref_entries):
            sync_addon_data: Dict[str, Optional[int]] = {}

            for addon in addon_names:
                ts_list = addon_timestamps[addon]

                if direction == "forward":
                    # Interval: [ref_ts, next_ref_ts) — pick first (closest to sync ts)
                    next_ts = ref_timestamps_us[ref_idx + 1] if ref_idx + 1 < len(ref_entries) else None
                    lo = bisect.bisect_left(ts_list, ref_ts)
                    hi = bisect.bisect_left(ts_list, next_ts) if next_ts is not None else len(ts_list)
                    best_idx = timestamp_logs[addon][lo][0] if lo < hi else None
                else:
                    # Interval: (prev_ref_ts, ref_ts] — pick last (closest to sync ts)
                    prev_ts = ref_timestamps_us[ref_idx - 1] if ref_idx > 0 else None
                    lo = bisect.bisect_right(ts_list, prev_ts) if prev_ts is not None else 0
                    hi = bisect.bisect_right(ts_list, ref_ts)
                    best_idx = timestamp_logs[addon][hi - 1][0] if lo < hi else None

                sync_addon_data[addon] = best_idx

            sync_uuid = create_deterministic_uuid(
                split=self._state.log_metadata.split,
                log_name=self._state.log_metadata.log_name,
                timestamp_us=ref_ts,
            )

            sync_row: Dict[str, Any] = {
                _SYNC_COL_UUID: [sync_uuid.bytes],
                _SYNC_COL_TIMESTAMP_US: [ref_ts],
            }
            for addon, best_index in sync_addon_data.items():
                sync_row[addon] = [best_index]
            sync_rows.append(sync_row)

        self._write_sync_arrow_file(schema, sync_rows)
