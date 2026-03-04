import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.abstract_log_writer import AbstractLogWriter, CameraData, LidarData
from py123d.api.scene.arrow.utils.arrow_metadata_utils import (
    add_box_detection_metadata_to_arrow_schema,
    add_ego_metadata_to_arrow_schema,
    add_fisheye_mei_camera_metadatas_to_arrow_schema,
    add_lidar_metadatas_to_arrow_schema,
    add_log_metadata_to_arrow_schema,
    add_pinhole_camera_metadatas_to_arrow_schema,
)
from py123d.api.utils.arrow_schema import (
    BOX_DETECTIONS_SE3,
    CAMERA_STORE_TYPES,
    CUSTOM_MODALITY,
    EGO_STATE_SE3,
    FISHEYE_MEI,
    LIDAR,
    LIDAR_STORE_TYPES,
    PINHOLE_CAMERA,
    SYNC,
    TRAFFIC_LIGHTS,
)
from py123d.common.dataset_paths import get_dataset_paths
from py123d.common.utils.msgpack_utils import msgpack_encode_with_numpy
from py123d.common.utils.uuid_utils import create_deterministic_uuid
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.sensor_io.camera.jpeg_camera_io import (
    decode_image_from_jpeg_binary,
    encode_image_as_jpeg_binary,
    load_image_from_jpeg_file,
    load_jpeg_binary_from_jpeg_file,
)
from py123d.conversion.sensor_io.camera.mp4_camera_io import MP4Writer
from py123d.conversion.sensor_io.camera.png_camera_io import (
    encode_image_as_png_binary,
    load_image_from_png_file,
    load_png_binary_from_png_file,
)
from py123d.conversion.sensor_io.lidar.draco_lidar_io import encode_point_cloud_3d_as_draco_binary
from py123d.conversion.sensor_io.lidar.ipc_lidar_io import (
    encode_point_cloud_3d_as_ipc_binary,
    encode_point_cloud_features_as_ipc_binary,
)
from py123d.conversion.sensor_io.lidar.laz_lidar_io import encode_point_cloud_3d_as_laz_binary
from py123d.conversion.sensor_io.lidar.path_lidar_io import load_point_cloud_data_from_path
from py123d.datatypes import (
    BoxDetectionsSE3,
    CustomModality,
    EgoStateSE3,
    LidarID,
    LogMetadata,
    Timestamp,
    TrafficLightDetections,
)
from py123d.datatypes.detections.box_detection_label_metadata import BoxDetectionMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID, FisheyeMEICameraMetadata
from py123d.datatypes.sensors.lidar import LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID, PinholeCameraMetadata
from py123d.datatypes.vehicle_state.ego_metadata import EgoMetadata


def _get_logs_root() -> Path:
    logs_root = get_dataset_paths().py123d_logs_root
    assert logs_root is not None, "PY123D_DATA_ROOT must be set."
    return logs_root


def _get_sensors_root() -> Path:
    sensors_root = get_dataset_paths().py123d_sensors_root
    assert sensors_root is not None, "PY123D_DATA_ROOT must be set."
    return sensors_root


# ------------------------------------------------------------------------------------------------------------------
# Internal modality writer
# ------------------------------------------------------------------------------------------------------------------


class _ModalityWriter:
    """Manages a single Arrow IPC file for one modality."""

    def __init__(
        self,
        file_path: Path,
        schema: pa.Schema,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        def _get_compression() -> Optional[pa.Codec]:
            """Returns the IPC compression codec, or None if no compression is configured."""
            if ipc_compression is not None:
                return pa.Codec(ipc_compression, compression_level=ipc_compression_level)
            return None

        self._file_path = file_path
        self._schema = schema
        self._source = pa.OSFile(str(file_path), "wb")
        options = pa.ipc.IpcWriteOptions(compression=_get_compression())
        self._writer = pa.ipc.new_file(self._source, schema=schema, options=options)

    def write_batch(self, data: Dict[str, Any]) -> None:
        """Write a record batch from a dict of column name -> column values."""
        batch = pa.record_batch(data, schema=self._schema)
        self._writer.write_batch(batch)  # type: ignore

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._source is not None:
            self._source.close()
            self._source = None


# ------------------------------------------------------------------------------------------------------------------
# ArrowLogWriter (modular folder-per-log)
# ------------------------------------------------------------------------------------------------------------------


class ArrowLogWriter(AbstractLogWriter):
    """Log writer that stores each modality in a separate Arrow IPC file.

    Directory layout per log::

        {logs_root}/{split}/{log_name}/
            sync.arrow                        # reference timeline with offset+count refs
            ego_state_se3.arrow               # imu_se3, dynamic_state_se3, timestamp_us
            box_detections_se3.arrow          # per-detection rows
            traffic_lights.arrow              # per-traffic-light rows
            pinhole_camera.arrow              # all pinhole cameras row-wise (camera_id column)
            fisheye_mei.arrow                 # all fisheye MEI cameras row-wise (camera_id column)
            lidar.arrow                       # all lidars row-wise (lidar_id column)
            custom.{name}.arrow               # msgpack-encoded binary data, timestamp_us

    Use :meth:`write` for frame-wise synchronized writing (writes sync row + all modalities).
    Use individual ``write_{modality}`` methods for independent / async writing.
    """

    def __init__(
        self,
        logs_root: Optional[Union[str, Path]] = None,
        sensors_root: Optional[Union[str, Path]] = None,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        """Initializes the :class:`ArrowLogWriter`.

        :param logs_root: The root directory for logs, defaults to None
        :param sensors_root: The root directory for sensors (e.g. MP4 video files), defaults to None
        :param ipc_compression: The IPC compression method, defaults to None
        :param ipc_compression_level: The IPC compression level, defaults to None
        """
        self._logs_root = Path(logs_root) if logs_root is not None else _get_logs_root()
        self._sensors_root = Path(sensors_root) if sensors_root is not None else _get_sensors_root()
        self._ipc_compression: Optional[Literal["lz4", "zstd"]] = ipc_compression
        self._ipc_compression_level: Optional[int] = ipc_compression_level

        self._dataset_converter_config: Optional[DatasetConverterConfig] = None
        self._log_metadata: Optional[LogMetadata] = None
        self._lidar_metadatas: Optional[Dict[LidarID, LidarMetadata]] = None
        self._log_dir: Optional[Path] = None
        self._current_timestamp: Optional[Timestamp] = None

        self._modality_writers: Dict[str, _ModalityWriter] = {}
        self._row_counts: Dict[str, int] = {}
        self._pinhole_mp4_writers: Dict[str, MP4Writer] = {}
        self._fisheye_mei_mp4_writers: Dict[str, MP4Writer] = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Writer lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def _create_modality_writer(
        self,
        name: str,
        schema_dict: dict,
        schema_post_hook: Optional[Callable[[pa.Schema], pa.Schema]] = None,
    ) -> _ModalityWriter:
        """Create and register a :class:`_ModalityWriter` for one modality.

        :param name: Modality name used as the arrow file stem (e.g. ``sync``, ``ego_state_se3``).
        :param schema_dict: Column-name → Arrow-type mapping for the schema.
        :param schema_post_hook: Optional callable that receives the schema (with log metadata already added)
            and returns a schema with additional metadata embedded.
        :return: The created writer.
        """
        assert self._log_dir is not None
        assert self._log_metadata is not None
        file_path = self._log_dir / f"{name}.arrow"
        schema = add_log_metadata_to_arrow_schema(pa.schema(list(schema_dict.items())), self._log_metadata)
        if schema_post_hook is not None:
            schema = schema_post_hook(schema)
        writer = _ModalityWriter(file_path, schema, self._ipc_compression, self._ipc_compression_level)
        self._modality_writers[name] = writer
        return writer

    def _close_writers(self) -> None:
        """Close all open modality writers."""
        for writer in self._modality_writers.values():
            writer.close()
        self._modality_writers = {}

    def reset(
        self,
        dataset_converter_config: DatasetConverterConfig,
        log_metadata: LogMetadata,
        ego_metadata: Optional[EgoMetadata] = None,
        box_detection_metadata: Optional[BoxDetectionMetadata] = None,
        pinhole_camera_metadatas: Optional[Dict[PinholeCameraID, PinholeCameraMetadata]] = None,
        fisheye_mei_camera_metadatas: Optional[Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]] = None,
        lidar_metadatas: Optional[Dict[LidarID, LidarMetadata]] = None,
    ) -> bool:
        """Inherited, see superclass."""
        log_needs_writing: bool = False
        log_dir: Path = self._logs_root / log_metadata.split / log_metadata.log_name
        sync_file_path = log_dir / f"{SYNC.prefix()}.arrow"

        if not sync_file_path.exists() or dataset_converter_config.force_log_conversion:
            log_needs_writing = True

            # Close any previous writers
            self._close_writers()

            if log_dir.exists():
                shutil.rmtree(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            self._dataset_converter_config = dataset_converter_config
            self._log_metadata = log_metadata
            self._lidar_metadatas = lidar_metadatas or {}
            self._log_dir = log_dir

            # --- Create per-modality writers (each embeds its own metadata in the Arrow schema) ---
            self._create_modality_writer(SYNC.prefix(), SYNC.schema_dict())

            if dataset_converter_config.include_ego and ego_metadata is not None:
                self._create_modality_writer(
                    EGO_STATE_SE3.prefix(),
                    EGO_STATE_SE3.schema_dict(),
                    schema_post_hook=lambda s: add_ego_metadata_to_arrow_schema(s, ego_metadata),
                )

            if dataset_converter_config.include_box_detections and box_detection_metadata is not None:
                self._create_modality_writer(
                    BOX_DETECTIONS_SE3.prefix(),
                    BOX_DETECTIONS_SE3.schema_dict(),
                    schema_post_hook=lambda s: add_box_detection_metadata_to_arrow_schema(s, box_detection_metadata),
                )

            if dataset_converter_config.include_traffic_lights:
                self._create_modality_writer(TRAFFIC_LIGHTS.prefix(), TRAFFIC_LIGHTS.schema_dict())

            if dataset_converter_config.include_pinhole_cameras and pinhole_camera_metadatas:
                store_overrides = CAMERA_STORE_TYPES[dataset_converter_config.pinhole_camera_store_option]
                _pcam_metas = pinhole_camera_metadatas
                self._create_modality_writer(
                    PINHOLE_CAMERA.prefix(),
                    PINHOLE_CAMERA.schema_dict(type_overrides=store_overrides),
                    schema_post_hook=lambda s: add_pinhole_camera_metadatas_to_arrow_schema(s, _pcam_metas),
                )

            if dataset_converter_config.include_fisheye_mei_cameras and fisheye_mei_camera_metadatas:
                store_overrides = CAMERA_STORE_TYPES[dataset_converter_config.fisheye_mei_camera_store_option]
                _fcam_metas = fisheye_mei_camera_metadatas
                self._create_modality_writer(
                    FISHEYE_MEI.prefix(),
                    FISHEYE_MEI.schema_dict(type_overrides=store_overrides),
                    schema_post_hook=lambda s: add_fisheye_mei_camera_metadatas_to_arrow_schema(s, _fcam_metas),
                )

            if dataset_converter_config.include_lidars and len(self._lidar_metadatas) > 0:
                store_overrides = LIDAR_STORE_TYPES[dataset_converter_config.lidar_store_option]
                _lidar_metas = self._lidar_metadatas
                self._create_modality_writer(
                    LIDAR.prefix(),
                    LIDAR.schema_dict(type_overrides=store_overrides),
                    schema_post_hook=lambda s: add_lidar_metadatas_to_arrow_schema(s, _lidar_metas),
                )

            self._row_counts = {name: 0 for name in self._modality_writers}
            self._pinhole_mp4_writers = {}
            self._fisheye_mei_mp4_writers = {}

        return log_needs_writing

    # ------------------------------------------------------------------------------------------------------------------
    # Synchronized (frame-wise) write
    # ------------------------------------------------------------------------------------------------------------------

    def write(
        self,
        timestamp: Timestamp,
        uuid: Optional[uuid.UUID] = None,
        ego_state_se3: Optional[EgoStateSE3] = None,
        box_detections_se3: Optional[BoxDetectionsSE3] = None,
        traffic_lights: Optional[TrafficLightDetections] = None,
        pinhole_cameras: Optional[List[CameraData]] = None,
        fisheye_mei_cameras: Optional[List[CameraData]] = None,
        lidars: Optional[List[LidarData]] = None,
        custom_modalities: Optional[Dict[str, CustomModality]] = None,
    ) -> None:
        """Inherited, see superclass.

        Writes one sync row and dispatches each provided modality to its own writer.
        """
        assert self._dataset_converter_config is not None, "Log writer is not initialized. Call reset() first."
        assert self._log_metadata is not None, "Log writer is not initialized. Call reset() first."
        assert timestamp is not None, "Timestamp must be provided for writing."

        self._current_timestamp = timestamp

        if uuid is None:
            uuid = create_deterministic_uuid(
                split=self._log_metadata.split,
                log_name=self._log_metadata.log_name,
                timestamp_us=timestamp.time_us,
            )

        # 1. Record offsets BEFORE writing modality data
        ego_offset = self._row_counts.get(EGO_STATE_SE3.prefix(), 0)
        box_offset = self._row_counts.get(BOX_DETECTIONS_SE3.prefix(), 0)
        tl_offset = self._row_counts.get(TRAFFIC_LIGHTS.prefix(), 0)
        pcam_offset = self._row_counts.get(PINHOLE_CAMERA.prefix(), 0)
        fcam_offset = self._row_counts.get(FISHEYE_MEI.prefix(), 0)
        lidar_offset = self._row_counts.get(LIDAR.prefix(), 0)

        # 2. Write modality data (each increments _row_counts)
        ego_count = 0
        if ego_state_se3 is not None:
            self.write_ego_state_se3(ego_state_se3)
            ego_count = 1

        box_count = 0
        if box_detections_se3 is not None:
            self.write_box_detections_se3(box_detections_se3)
            box_count = 1

        tl_count = 0
        if traffic_lights is not None:
            self.write_traffic_lights(traffic_lights)
            tl_count = 1

        pcam_count = 0
        if pinhole_cameras is not None:
            for camera_data in pinhole_cameras:
                self.write_pinhole_camera(camera_data)
                pcam_count += 1

        fcam_count = 0
        if fisheye_mei_cameras is not None:
            for camera_data in fisheye_mei_cameras:
                self.write_fisheye_mei_camera(camera_data)
                fcam_count += 1

        lidar_count = 0
        if lidars is not None:
            for lidar_data in lidars:
                self.write_lidar(lidar_data)
                lidar_count += 1

        if custom_modalities is not None:
            self.write_custom_modalities(custom_modalities)

        # 3. Write sync row with offset+count references to all modalities
        sync_writer = self._modality_writers[SYNC.prefix()]
        sync_writer.write_batch(
            {
                SYNC.col("uuid"): [uuid.bytes],
                SYNC.col("timestamp_us"): [timestamp.time_us],
                SYNC.col("ego_state_se3_offset"): [ego_offset],
                SYNC.col("ego_state_se3_count"): [ego_count],
                SYNC.col("box_detections_se3_offset"): [box_offset],
                SYNC.col("box_detections_se3_count"): [box_count],
                SYNC.col("traffic_lights_offset"): [tl_offset],
                SYNC.col("traffic_lights_count"): [tl_count],
                SYNC.col("pinhole_camera_offset"): [pcam_offset],
                SYNC.col("pinhole_camera_count"): [pcam_count],
                SYNC.col("fisheye_mei_offset"): [fcam_offset],
                SYNC.col("fisheye_mei_count"): [fcam_count],
                SYNC.col("lidar_offset"): [lidar_offset],
                SYNC.col("lidar_count"): [lidar_count],
            }
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Individual modality writers (usable independently for async writing)
    # ------------------------------------------------------------------------------------------------------------------

    def write_ego_state_se3(self, ego_state_se3: EgoStateSE3) -> None:
        """Write a single ego-state observation to ``ego_state_se3.arrow``."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        if self._dataset_converter_config.include_ego:
            assert ego_state_se3.timestamp is not None, "EgoStateSE3 must have a timestamp for writing."

            writer = self._modality_writers[EGO_STATE_SE3.prefix()]
            writer.write_batch(
                {
                    EGO_STATE_SE3.col("imu_se3"): [ego_state_se3.imu_se3.array],
                    EGO_STATE_SE3.col("dynamic_state_se3"): [ego_state_se3.dynamic_state_se3],
                    EGO_STATE_SE3.col("timestamp_us"): [ego_state_se3.timestamp.time_us],
                }
            )
            self._row_counts[EGO_STATE_SE3.prefix()] += 1

    def write_box_detections_se3(self, box_detections_se3: BoxDetectionsSE3) -> None:
        """Write box detections to ``box_detections_se3.arrow`` (one row per detection)."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_box_detections:
            return

        writer = self._modality_writers[BOX_DETECTIONS_SE3.prefix()]

        bounding_box_se3_list = []
        tokens_list = []
        labels_list = []
        velocities_list = []
        num_lidar_points_list = []

        for det in box_detections_se3:
            bounding_box_se3_list.append(det.bounding_box_se3)
            tokens_list.append(det.metadata.track_token)
            labels_list.append(int(det.metadata.label))
            velocities_list.append(det.velocity_3d)
            num_lidar_points_list.append(det.metadata.num_lidar_points)

        if len(bounding_box_se3_list) > 0:
            writer.write_batch(
                {
                    BOX_DETECTIONS_SE3.col("bounding_box_se3"): [bounding_box_se3_list],
                    BOX_DETECTIONS_SE3.col("token"): [tokens_list],
                    BOX_DETECTIONS_SE3.col("label"): [labels_list],
                    BOX_DETECTIONS_SE3.col("velocity_3d"): [velocities_list],
                    BOX_DETECTIONS_SE3.col("num_lidar_points"): [num_lidar_points_list],
                }
            )
            self._row_counts[BOX_DETECTIONS_SE3.prefix()] += 1

    def write_traffic_lights(self, traffic_lights: TrafficLightDetections) -> None:
        """Write traffic lights to ``traffic_lights.arrow`` (one row per traffic light)."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._current_timestamp is not None, "Current timestamp is not set. Call write() with a timestamp first."
        if not self._dataset_converter_config.include_traffic_lights:
            return

        writer = self._modality_writers[TRAFFIC_LIGHTS.prefix()]
        timestamp_us = self._current_timestamp.time_us if self._current_timestamp is not None else 0

        lane_ids = []
        statuses = []
        timestamp_list = []

        for tl in traffic_lights:
            lane_ids.append(tl.lane_id)
            statuses.append(int(tl.status))
            timestamp_list.append(timestamp_us)

        if len(lane_ids) > 0:
            writer.write_batch(
                {
                    TRAFFIC_LIGHTS.col("lane_id"): [lane_ids],
                    TRAFFIC_LIGHTS.col("status"): [statuses],
                    TRAFFIC_LIGHTS.col("timestamp_us"): [timestamp_list],
                }
            )
            self._row_counts[TRAFFIC_LIGHTS.prefix()] += 1

    def write_pinhole_camera(self, camera_data: CameraData) -> None:
        """Write a single pinhole camera observation to ``pinhole_camera.arrow``."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_pinhole_cameras:
            return

        writer = self._modality_writers[PINHOLE_CAMERA.prefix()]

        store_option = self._dataset_converter_config.pinhole_camera_store_option
        data_value = self._get_camera_data_value(camera_data, store_option, self._pinhole_mp4_writers)
        timestamp_us = (
            camera_data.timestamp.time_us
            if camera_data.timestamp is not None
            else (self._current_timestamp.time_us if self._current_timestamp is not None else None)
        )

        writer.write_batch(
            {
                PINHOLE_CAMERA.col("camera_id"): [int(camera_data.camera_id)],
                PINHOLE_CAMERA.col("data"): [data_value],
                PINHOLE_CAMERA.col("state_se3"): [camera_data.extrinsic],
                PINHOLE_CAMERA.col("timestamp_us"): [timestamp_us],
            }
        )
        self._row_counts[PINHOLE_CAMERA.prefix()] += 1

    def write_fisheye_mei_camera(self, camera_data: CameraData) -> None:
        """Write a single fisheye MEI camera observation to ``fisheye_mei.arrow``."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_fisheye_mei_cameras:
            return

        writer = self._modality_writers[FISHEYE_MEI.prefix()]

        store_option = self._dataset_converter_config.fisheye_mei_camera_store_option
        data_value = self._get_camera_data_value(camera_data, store_option, self._fisheye_mei_mp4_writers)
        timestamp_us = (
            camera_data.timestamp.time_us
            if camera_data.timestamp is not None
            else (self._current_timestamp.time_us if self._current_timestamp is not None else None)
        )

        writer.write_batch(
            {
                FISHEYE_MEI.col("camera_id"): [int(camera_data.camera_id)],
                FISHEYE_MEI.col("data"): [data_value],
                FISHEYE_MEI.col("state_se3"): [camera_data.extrinsic],
                FISHEYE_MEI.col("timestamp_us"): [timestamp_us],
            }
        )
        self._row_counts[FISHEYE_MEI.prefix()] += 1

    def write_lidar(self, lidar_data: LidarData) -> None:
        """Write a single lidar observation to ``lidar.arrow``."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_lidars:
            return
        if not self._lidar_metadatas:
            return

        writer = self._modality_writers[LIDAR.prefix()]

        if self._dataset_converter_config.lidar_store_option == "path":
            data_path: Optional[str] = str(lidar_data.relative_path) if lidar_data.has_file_path else None
            writer.write_batch(
                {
                    LIDAR.col("lidar_id"): [int(lidar_data.lidar_type)],
                    LIDAR.col("data"): [data_path],
                    LIDAR.col("start_timestamp_us"): [lidar_data.start_timestamp.time_us],
                    LIDAR.col("end_timestamp_us"): [lidar_data.end_timestamp.time_us],
                }
            )
        elif self._dataset_converter_config.lidar_store_option == "binary":
            point_cloud_binary, features_binary = self._prepare_lidar_data(lidar_data)
            writer.write_batch(
                {
                    LIDAR.col("lidar_id"): [int(lidar_data.lidar_type)],
                    LIDAR.col("point_cloud_3d"): [point_cloud_binary],
                    LIDAR.col("point_cloud_features"): [features_binary],
                    LIDAR.col("start_timestamp_us"): [lidar_data.start_timestamp.time_us],
                    LIDAR.col("end_timestamp_us"): [lidar_data.end_timestamp.time_us],
                }
            )
        else:
            raise ValueError(f"Unsupported lidar store option: {self._dataset_converter_config.lidar_store_option}")
        self._row_counts[LIDAR.prefix()] += 1

    def write_custom_modalities(self, custom_modalities: Dict[str, CustomModality]) -> None:
        """Write custom modalities to ``custom.{name}.arrow`` (one file per named modality).

        Writers are created lazily on first encounter of each modality name.
        """
        assert self._log_metadata is not None, "Log writer is not initialized."

        for name, modality in custom_modalities.items():
            writer_key = CUSTOM_MODALITY.prefix(name)

            # Lazily create writer for this custom modality name
            if writer_key not in self._modality_writers:
                self._create_modality_writer(writer_key, CUSTOM_MODALITY.schema_dict(name))

            encoded_data = msgpack_encode_with_numpy(modality.data)
            writer = self._modality_writers[writer_key]
            writer.write_batch(
                {
                    CUSTOM_MODALITY.col("data", name): [encoded_data],
                    CUSTOM_MODALITY.col("timestamp_us", name): [modality.timestamp.time_us],
                }
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def close(self) -> None:
        """Inherited, see superclass."""
        self._close_writers()

        self._dataset_converter_config = None
        self._log_metadata = None
        self._lidar_metadatas = None
        self._log_dir = None
        self._current_timestamp = None
        self._row_counts = {}

        for mp4_writer in self._pinhole_mp4_writers.values():
            mp4_writer.close()
        self._pinhole_mp4_writers = {}
        for mp4_writer in self._fisheye_mei_mp4_writers.values():
            mp4_writer.close()
        self._fisheye_mei_mp4_writers = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _get_camera_data_value(
        self,
        camera_data: CameraData,
        store_option: Literal["path", "jpeg_binary", "png_binary", "mp4"],
        mp4_writers: Dict[str, MP4Writer],
    ) -> Union[str, bytes, int]:
        """Prepare a single camera's data value for the configured store option.

        :param camera_data: The camera observation.
        :param store_option: How to store the camera data.
        :param mp4_writers: Dict of MP4 writers (lazily populated).
        :return: The value to write into the arrow column.
        """
        if store_option == "path":
            assert camera_data.has_file_path, "Camera data must have a file path for path storage."
            return str(camera_data.relative_path)
        elif store_option == "jpeg_binary":
            return _get_jpeg_binary_from_camera_data(camera_data)
        elif store_option == "png_binary":
            return _get_png_binary_from_camera_data(camera_data)
        elif store_option == "mp4":
            assert self._log_metadata is not None
            camera_name = camera_data.camera_id.serialize()
            if camera_name not in mp4_writers:
                mp4_path = (
                    self._sensors_root / self._log_metadata.split / self._log_metadata.log_name / f"{camera_name}.mp4"
                )
                mp4_path.parent.mkdir(parents=True, exist_ok=True)
                frame_interval = self._log_metadata.timestep_seconds
                mp4_writers[camera_name] = MP4Writer(mp4_path, fps=1 / frame_interval)

            image = _get_numpy_image_from_camera_data(camera_data)
            return mp4_writers[camera_name].write_frame(image)
        else:
            raise ValueError(f"Unsupported camera store option: {store_option}")

    def _prepare_lidar_data(self, lidar_data: LidarData) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Load and/or encode the lidar data in binary for point cloud and features.

        :param lidar_data: Helper class referencing the lidar observation.
        :return: Tuple of (point_cloud_binary, point_cloud_features_binary)
        """
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."

        # 1. Load point cloud and point features
        point_cloud_3d: Optional[npt.NDArray] = None
        point_cloud_features: Optional[Dict[str, npt.NDArray]] = None
        if lidar_data.has_point_cloud_3d:
            point_cloud_3d = lidar_data.point_cloud_3d
            point_cloud_features = lidar_data.point_cloud_features
        elif lidar_data.has_file_path:
            point_cloud_3d, point_cloud_features = load_point_cloud_data_from_path(
                lidar_data.relative_path,  # type: ignore
                self._log_metadata,
                lidar_data.iteration,
                lidar_data.dataset_root,
                lidar_metadatas=self._lidar_metadatas,
            )
        else:
            raise ValueError("Lidar data must provide either point cloud data or a file path.")

        # 2. Compress point clouds with target codec
        point_cloud_3d_output: Optional[bytes] = None
        if point_cloud_3d is not None:
            codec = self._dataset_converter_config.lidar_point_cloud_codec
            if codec == "draco":
                point_cloud_3d_output = encode_point_cloud_3d_as_draco_binary(point_cloud_3d)
            elif codec == "laz":
                point_cloud_3d_output = encode_point_cloud_3d_as_laz_binary(point_cloud_3d)
            elif codec == "ipc":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec=None)
            elif codec == "ipc_zstd":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec="zstd")
            elif codec == "ipc_lz4":
                point_cloud_3d_output = encode_point_cloud_3d_as_ipc_binary(point_cloud_3d, codec="lz4")
            else:
                raise NotImplementedError(f"Unsupported lidar point cloud codec: {codec}")

        # 3. Compress point cloud features with target codec, if specified
        point_cloud_feature_output: Optional[bytes] = None
        feature_codec = self._dataset_converter_config.lidar_point_feature_codec
        if feature_codec is not None and point_cloud_features is not None:
            if feature_codec == "ipc":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(point_cloud_features, codec=None)
            elif feature_codec == "ipc_zstd":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(
                    point_cloud_features, codec="zstd"
                )
            elif feature_codec == "ipc_lz4":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(
                    point_cloud_features, codec="lz4"
                )
            else:
                raise NotImplementedError(f"Unsupported lidar point feature codec: {feature_codec}")

        return point_cloud_3d_output, point_cloud_feature_output


# ------------------------------------------------------------------------------------------------------------------
# Camera data helpers (module-level)
# ------------------------------------------------------------------------------------------------------------------


def _get_jpeg_binary_from_camera_data(camera_data: CameraData) -> bytes:
    if camera_data.has_jpeg_binary:
        return camera_data.jpeg_binary  # type: ignore
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        return load_jpeg_binary_from_jpeg_file(absolute_path)
    elif camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        numpy_image = load_image_from_png_file(absolute_path)
        return encode_image_as_jpeg_binary(numpy_image)
    elif camera_data.has_numpy_image:
        return encode_image_as_jpeg_binary(camera_data.numpy_image)  # type: ignore[arg-type]
    else:
        raise NotImplementedError("Camera data must provide jpeg_binary, numpy_image, or file path for binary storage.")


def _get_png_binary_from_camera_data(camera_data: CameraData) -> bytes:
    if camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        return load_png_binary_from_png_file(absolute_path)
    elif camera_data.has_numpy_image:
        return encode_image_as_png_binary(camera_data.numpy_image)  # type: ignore[arg-type]
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        numpy_image = load_image_from_jpeg_file(absolute_path)
        return encode_image_as_png_binary(numpy_image)
    elif camera_data.has_jpeg_binary:
        numpy_image = decode_image_from_jpeg_binary(camera_data.jpeg_binary)  # type: ignore[arg-type]
        return encode_image_as_png_binary(numpy_image)
    else:
        raise NotImplementedError("Camera data must provide png_binary, numpy_image, or file path for binary storage.")


def _get_numpy_image_from_camera_data(camera_data: CameraData) -> np.ndarray:
    if camera_data.has_numpy_image:
        return camera_data.numpy_image  # type: ignore
    elif camera_data.has_jpeg_binary:
        return decode_image_from_jpeg_binary(camera_data.jpeg_binary)  # type: ignore
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        return load_image_from_jpeg_file(absolute_path)
    elif camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        return load_image_from_png_file(absolute_path)
    else:
        raise NotImplementedError("Camera data must provide numpy_image, jpeg_binary, or file path for numpy image.")
