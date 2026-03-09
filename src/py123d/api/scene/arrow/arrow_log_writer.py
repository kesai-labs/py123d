import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pyarrow as pa

from py123d.api.scene.abstract_log_writer import AbstractLogWriter
from py123d.api.scene.arrow.modalities.arrow_box_detections_se3 import ArrowBoxDetectionsSE3Writer
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowFisheyeMEICameraWriter, ArrowPinholeCameraWriter
from py123d.api.scene.arrow.modalities.arrow_custom_modality import ArrowCustomModalityWriter
from py123d.api.scene.arrow.modalities.arrow_ego_state_se3 import ArrowEgoStateSE3Writer
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarWriter
from py123d.api.scene.arrow.modalities.arrow_traffic_light_detections_writer import ArrowTrafficLightDetectionsWriter
from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.utils.uuid_utils import create_deterministic_uuid
from py123d.datatypes import LogMetadata, PinholeCameraMetadata
from py123d.datatypes.custom.custom_modality import CustomModalityMetadata
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionsMetadata
from py123d.datatypes.metadata.base_metadata import BaseModalityMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata
from py123d.datatypes.sensors.lidar import LidarMergedMetadata, LidarMetadata
from py123d.datatypes.vehicle_state.ego_metadata import EgoStateSE3Metadata
from py123d.parser.abstract_dataset_parser import ParsedFrame, ParsedModality
from py123d.parser.dataset_converter_config import DatasetConverterConfig

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
        e.g. ``"lidar.lidar_merged.start_timestamp_us"``.
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
        """The timestamp field name, e.g. ``"start_timestamp_us"``."""
        return self.reference_column.rsplit(".", 1)[-1]


@dataclass
class ArrowLogWriterState:
    log_dir: Path
    log_metadata: LogMetadata
    deferred_sync: bool = False
    modality_writers: Dict[str, BaseModalityWriter] = field(default_factory=dict)


class ArrowLogWriter(AbstractLogWriter):
    def __init__(
        self,
        dataset_converter_config: DatasetConverterConfig,
        logs_root: Union[str, Path],
        sensors_root: Union[str, Path],
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
        sync_config: Optional[SyncConfig] = None,
    ) -> None:
        """Initializes the :class:`ArrowLogWriter`.

        :param dataset_converter_config: The dataset converter configuration.
        :param logs_root: The root directory for logs.
        :param sensors_root: The root directory for sensors (e.g. MP4 video files).
        :param ipc_compression: The IPC compression method, defaults to None.
        :param ipc_compression_level: The IPC compression level, defaults to None.
        :param sync_config: Configuration for deferred sync table construction, defaults to None.
        """
        self._dataset_converter_config = dataset_converter_config
        self._logs_root = Path(logs_root)
        self._sensors_root = Path(sensors_root)
        self._ipc_compression: Optional[Literal["lz4", "zstd"]] = ipc_compression
        self._ipc_compression_level: Optional[int] = ipc_compression_level
        self._sync_config: Optional[SyncConfig] = sync_config

        self._state: Optional[ArrowLogWriterState] = None

    # ------------------------------------------------------------------------------------------------------------------
    # Writer lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def _close_writers(self) -> None:
        """Close all open modality writers."""
        if self._state is not None:
            for writer in self._state.modality_writers.values():
                writer.close()
            self._state.modality_writers.clear()

    def reset(self, log_metadata: LogMetadata, deferred_sync: bool = False) -> bool:
        """Prepare the writer for a new log. Returns True if the log needs writing.

        Modality writers are initialized from the modality metadata embedded in *log_metadata*.

        :param log_metadata: Metadata for the log to write (includes all modality metadata).
        :param deferred_sync: If True, the sync table is built at close() from buffered timestamps.
        """
        assert self._state is None, "Log writer is already initialized. Call close() before reset()."

        log_dir: Path = self._logs_root / log_metadata.split / log_metadata.log_name
        sync_file_path = log_dir / "sync.arrow"

        if not sync_file_path.exists() or self._dataset_converter_config.force_log_conversion:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._state = ArrowLogWriterState(
                log_dir=log_dir,
                log_metadata=log_metadata,
                deferred_sync=deferred_sync,
            )
            for metadata in log_metadata.all_modality_metadatas:
                self._init_modality_writer(metadata)
            return True

        return False

    def _init_modality_writer(self, modality_metadata: BaseModalityMetadata) -> None:
        """Create the Arrow writer(s) for a single modality metadata entry."""
        assert self._state is not None, "Log writer state is not initialized."
        if isinstance(modality_metadata, EgoStateSE3Metadata):
            if self._dataset_converter_config.include_ego:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowEgoStateSE3Writer(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, BoxDetectionsSE3Metadata):
            if self._dataset_converter_config.include_box_detections:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowBoxDetectionsSE3Writer(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, TrafficLightDetectionsMetadata):
            if self._dataset_converter_config.include_traffic_lights:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowTrafficLightDetectionsWriter(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, PinholeCameraMetadata):
            if self._dataset_converter_config.include_pinhole_cameras:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowPinholeCameraWriter(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    data_codec=self._dataset_converter_config.pinhole_camera_store_option,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, FisheyeMEICameraMetadata):
            if self._dataset_converter_config.include_fisheye_mei_cameras:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowFisheyeMEICameraWriter(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    data_codec=self._dataset_converter_config.fisheye_mei_camera_store_option,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, (LidarMergedMetadata, LidarMetadata)):
            if self._dataset_converter_config.include_lidars:
                self._state.modality_writers[modality_metadata.modality_name] = ArrowLidarWriter(
                    log_dir=self._state.log_dir,
                    metadata=modality_metadata,
                    log_metadata=self._state.log_metadata,
                    lidar_store_option=self._dataset_converter_config.lidar_store_option,
                    lidar_point_cloud_codec=self._dataset_converter_config.lidar_point_cloud_codec,
                    lidar_point_feature_codec=self._dataset_converter_config.lidar_point_feature_codec,
                    ipc_compression=self._ipc_compression,
                    ipc_compression_level=self._ipc_compression_level,
                )

        elif isinstance(modality_metadata, CustomModalityMetadata):
            self._state.modality_writers[modality_metadata.modality_name] = ArrowCustomModalityWriter(
                log_dir=self._state.log_dir,
                metadata=modality_metadata,
                ipc_compression=self._ipc_compression,
                ipc_compression_level=self._ipc_compression_level,
            )

        else:
            raise ValueError(f"Unsupported modality metadata type: {type(modality_metadata)}")

    # ------------------------------------------------------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------------------------------------------------------

    # Modality data types that support lazy writer initialization (have a .metadata attribute)
    _LAZY_INIT_TYPES: tuple = ()  # Populated after imports; currently only CustomModality

    def _write_single_modality(self, modality_name: str, data: Any) -> Optional[int]:
        """Write a single modality and return its row index, or None if skipped."""
        assert self._state is not None
        writer = self._state.modality_writers.get(modality_name)
        if writer is None:
            if isinstance(data, self._LAZY_INIT_TYPES):
                self._init_modality_writer(data.metadata)
                writer = self._state.modality_writers.get(modality_name)
            if writer is None:
                return None

        row_idx = writer.row_count
        writer.write_modality(data)
        return row_idx

    def write_sync(self, frame: ParsedFrame) -> None:
        """Inherited, see superclass."""
        assert self._state is not None, "Log writer is not initialized. Call reset() first."

        # Unpack the ParsedFrame into modality name -> data pairs
        modality_items = _unpack_parsed_frame(frame)

        # Write each modality and collect row indices for the sync table
        sync_row_indices: Dict[str, List[int]] = {}
        for modality_name, data in modality_items:
            row_idx = self._write_single_modality(modality_name, data)
            if row_idx is not None:
                sync_row_indices[modality_name] = [row_idx]

        # Build the sync row
        frame_uuid = frame.uuid
        if frame_uuid is None:
            frame_uuid = create_deterministic_uuid(
                split=self._state.log_metadata.split,
                log_name=self._state.log_metadata.log_name,
                timestamp_us=frame.timestamp.time_us,
            )

        sync_writer = self._state.modality_writers.get("sync")
        if sync_writer is None:
            self._create_sync_writer(list(sync_row_indices.keys()))
            sync_writer = self._state.modality_writers["sync"]

        sync_data: Dict[str, Any] = {
            _SYNC_COL_UUID: [frame_uuid.bytes],
            _SYNC_COL_TIMESTAMP_US: [frame.timestamp.time_us],
        }
        for modality_name, row_indices in sync_row_indices.items():
            sync_data[modality_name] = [row_indices]
        sync_writer.write_batch(sync_data)

    def write_async(self, modality: ParsedModality, modality_metadata: BaseModalityMetadata) -> None:
        """Inherited, see superclass."""
        assert self._state is not None, "Log writer is not initialized. Call reset() first."
        self._write_single_modality(modality_metadata.modality_name, modality)

    # ------------------------------------------------------------------------------------------------------------------
    # Sync table helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _create_sync_writer(self, addon_modality_names: List[str]) -> None:
        """Create the sync.arrow writer with columns for uuid, timestamp, plus one list column per modality."""
        assert self._state is not None
        schema_fields: List[Tuple[str, pa.DataType]] = [
            (_SYNC_COL_UUID, _get_uuid_arrow_type()),
            (_SYNC_COL_TIMESTAMP_US, pa.int64()),
        ]
        for name in addon_modality_names:
            schema_fields.append((name, pa.list_(pa.int64())))

        schema = pa.schema(schema_fields)
        schema = add_metadata_to_arrow_schema(schema, self._state.log_metadata)

        sync_writer = BaseModalityWriter(
            file_path=self._state.log_dir / "sync.arrow",
            schema=schema,
            ipc_compression=self._ipc_compression,
            ipc_compression_level=self._ipc_compression_level,
            max_batch_size=1000,
        )
        self._state.modality_writers["sync"] = sync_writer

    def _build_deferred_sync_table(self) -> None:
        """Build the sync table by reading timestamps from the written Arrow files.

        After all modality writers have been closed, this method scans the log directory
        for ``*.arrow`` files, extracts timestamp columns (columns ending in
        ``timestamp_us``), and builds the sync table using the reference modality
        from :class:`SyncConfig`.

        Lidars are the only modality with two timestamp columns (``start_timestamp_us``
        and ``end_timestamp_us``). Which one to use is determined by the
        :attr:`SyncConfig.reference_column` when the lidar is the reference modality.
        For non-reference lidar addons, the first ``*timestamp_us`` column is used.
        """
        assert self._state is not None
        assert self._sync_config is not None, "SyncConfig is required for deferred sync."

        ref_modality_name = self._sync_config.reference_modality
        ref_timestamp_field = self._sync_config.reference_timestamp_field
        direction = self._sync_config.direction

        # Read timestamps from all written Arrow files
        timestamp_logs: Dict[str, List[Tuple[int, int]]] = {}  # modality_name -> [(row_idx, ts_us)]

        for arrow_path in sorted(self._state.log_dir.glob("*.arrow")):
            if arrow_path.name == "sync.arrow":
                continue

            modality_name = arrow_path.stem
            reader = pa.ipc.open_file(arrow_path)
            table = reader.read_all()

            # Find the timestamp column: use the specific field for the reference modality,
            # otherwise pick the first column ending in "timestamp_us".
            ts_col_name = None
            if modality_name == ref_modality_name:
                ts_col_name = f"{modality_name}.{ref_timestamp_field}"
            else:
                for col_name in table.column_names:
                    if col_name.endswith("timestamp_us"):
                        ts_col_name = col_name
                        break

            if ts_col_name is None or ts_col_name not in table.column_names:
                continue

            timestamps = table.column(ts_col_name).to_pylist()
            timestamp_logs[modality_name] = [(i, ts) for i, ts in enumerate(timestamps)]

        # Extract reference timestamps
        ref_entries = timestamp_logs.get(ref_modality_name, [])
        if not ref_entries:
            return

        ref_timestamps_us = [ts for _, ts in ref_entries]
        addon_names = list(timestamp_logs.keys())

        # Create and populate the sync writer
        self._create_sync_writer(addon_names)
        sync_writer = self._state.modality_writers["sync"]

        # Pre-extract sorted timestamp arrays for efficient bisect lookups
        addon_timestamps: Dict[str, List[int]] = {}
        for addon in addon_names:
            addon_timestamps[addon] = [ts for _, ts in timestamp_logs[addon]]

        # Build one sync row per reference timestamp
        for ref_idx, (_, ref_ts) in enumerate(ref_entries):
            sync_addon_data: Dict[str, List[int]] = {}

            for addon in addon_names:
                ts_list = addon_timestamps[addon]

                if direction == "forward":
                    # Interval: [ref_ts, next_ref_ts)
                    next_ts = ref_timestamps_us[ref_idx + 1] if ref_idx + 1 < len(ref_entries) else None
                    lo = bisect.bisect_left(ts_list, ref_ts)
                    hi = bisect.bisect_left(ts_list, next_ts) if next_ts is not None else len(ts_list)
                else:
                    # Interval: (prev_ref_ts, ref_ts]
                    prev_ts = ref_timestamps_us[ref_idx - 1] if ref_idx > 0 else None
                    lo = bisect.bisect_right(ts_list, prev_ts) if prev_ts is not None else 0
                    hi = bisect.bisect_right(ts_list, ref_ts)

                sync_addon_data[addon] = [timestamp_logs[addon][i][0] for i in range(lo, hi)]

            sync_uuid = create_deterministic_uuid(
                split=self._state.log_metadata.split,
                log_name=self._state.log_metadata.log_name,
                timestamp_us=ref_ts,
            )

            sync_data: Dict[str, Any] = {
                _SYNC_COL_UUID: [sync_uuid.bytes],
                _SYNC_COL_TIMESTAMP_US: [ref_ts],
            }
            for addon, row_indices in sync_addon_data.items():
                sync_data[addon] = [row_indices]
            sync_writer.write_batch(sync_data)

        sync_writer.close()

    # ------------------------------------------------------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def close(self) -> None:
        """Inherited, see superclass."""
        if self._state is not None:
            if self._state.deferred_sync:
                # Close modality writers first so Arrow files are finalized and readable
                self._close_writers()
                # Then read back the files to build the sync table
                self._build_deferred_sync_table()
            else:
                self._close_writers()

        self._state = None


# ------------------------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------------------------


def _unpack_parsed_frame(frame: ParsedFrame) -> List[Tuple[str, Any]]:
    """Unpack a ParsedFrame into a list of (modality_name, data) pairs."""
    items: List[Tuple[str, Any]] = []

    if frame.ego_state_se3 is not None:
        items.append(("ego_state_se3", frame.ego_state_se3))

    if frame.box_detections_se3 is not None:
        items.append(("box_detections_se3", frame.box_detections_se3))

    if frame.traffic_lights is not None:
        items.append(("traffic_light_detections", frame.traffic_lights))

    if frame.pinhole_cameras is not None:
        for cam in frame.pinhole_cameras:
            modality_name = f"pinhole_camera.{cam.camera_id.serialize()}"
            items.append((modality_name, cam))

    if frame.fisheye_mei_cameras is not None:
        for cam in frame.fisheye_mei_cameras:
            modality_name = f"fisheye_mei_camera.{cam.camera_id.serialize()}"
            items.append((modality_name, cam))

    if frame.lidars is not None:
        for lidar in frame.lidars:
            modality_name = f"lidar.{lidar.lidar_type.serialize()}"
            items.append((modality_name, lidar))

    if frame.custom_modalities is not None:
        for name, custom in frame.custom_modalities.items():
            modality_name = f"custom.{name}"
            items.append((modality_name, custom))

    return items
