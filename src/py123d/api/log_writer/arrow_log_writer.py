import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.common.dataset_paths import get_dataset_paths
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
    EgoStateSE3,
    LidarID,
    LogMetadata,
    Timestamp,
    TrafficLights,
)
from py123d.api.log_writer.abstract_log_writer import AbstractLogWriter, CameraData, LidarData
from py123d.api.scene.arrow.utils.arrow_metadata_utils import add_log_metadata_to_arrow_schema
from py123d.api.utils.arrow_schema import (
    BOX_DETECTIONS_SE3_NAME,
    BOX_DETECTIONS_SE3_SCHEMA_DICT,
    EGO_STATE_SE3_NAME,
    EGO_STATE_SE3_SCHEMA_DICT,
    FCAM_BINARY_SCHEMA_DICT,
    FCAM_INT_SCHEMA_DICT,
    FCAM_NAME,
    FCAM_STRING_SCHEMA_DICT,
    LIDAR_BINARY_SCHEMA_DICT,
    LIDAR_NAME,
    LIDAR_STRING_SCHEMA_DICT,
    PCAM_BINARY_SCHEMA_DICT,
    PCAM_INT_SCHEMA_DICT,
    PCAM_NAME,
    PCAM_STRING_SCHEMA_DICT,
    SYNC_NAME,
    SYNC_SCHEMA_DICT,
    TRAFFIC_LIGHTS_NAME,
    TRAFFIC_LIGHTS_SCHEMA_DICT,
)


def _get_logs_root() -> Path:
    logs_root = get_dataset_paths().py123d_logs_root
    assert logs_root is not None, "PY123D_DATA_ROOT must be set."
    return logs_root


def _get_sensors_root() -> Path:
    sensors_root = get_dataset_paths().py123d_sensors_root
    assert sensors_root is not None, "PY123D_DATA_ROOT must be set."
    return sensors_root


def _get_pcam_schema_dict(cam_name: str, store_option: Literal["path", "jpeg_binary", "png_binary", "mp4"]) -> dict:
    """Returns the pinhole camera schema dict for the given store option."""
    if store_option == "path":
        return PCAM_STRING_SCHEMA_DICT(cam_name)
    elif store_option in {"jpeg_binary", "png_binary"}:
        return PCAM_BINARY_SCHEMA_DICT(cam_name)
    elif store_option == "mp4":
        return PCAM_INT_SCHEMA_DICT(cam_name)
    else:
        raise ValueError(f"Unsupported pinhole camera store option: {store_option}")


def _get_fcam_schema_dict(cam_name: str, store_option: Literal["path", "jpeg_binary", "png_binary", "mp4"]) -> dict:
    """Returns the fisheye MEI camera schema dict for the given store option."""
    if store_option == "path":
        return FCAM_STRING_SCHEMA_DICT(cam_name)
    elif store_option in {"jpeg_binary", "png_binary"}:
        return FCAM_BINARY_SCHEMA_DICT(cam_name)
    elif store_option == "mp4":
        return FCAM_INT_SCHEMA_DICT(cam_name)
    else:
        raise ValueError(f"Unsupported fisheye MEI camera store option: {store_option}")


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
            sync.arrow                        # reference timeline (uuid + timestamp_us)
            ego_state_se3.arrow               # imu_se3, dynamic_state_se3, timestamp_us
            box_detections_se3.arrow          # per-detection rows
            traffic_lights.arrow              # per-traffic-light rows
            pinhole_camera.{name}.arrow       # data, state_se3, timestamp_us
            fisheye_mei.{name}.arrow          # data, state_se3, timestamp_us
            lidar.{name}.arrow                # point cloud data, timestamps

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
        self._log_dir: Optional[Path] = None
        self._current_timestamp: Optional[Timestamp] = None

        self._modality_writers: Dict[str, _ModalityWriter] = {}
        self._pinhole_mp4_writers: Dict[str, MP4Writer] = {}
        self._fisheye_mei_mp4_writers: Dict[str, MP4Writer] = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Writer lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def _create_modality_writer(self, name: str, schema_dict: dict) -> _ModalityWriter:
        """Create and register a :class:`_ModalityWriter` for one modality.

        :param name: Modality name used as the arrow file stem (e.g. ``sync``, ``ego_state_se3``).
        :param schema_dict: Column-name → Arrow-type mapping for the schema.
        :return: The created writer.
        """
        assert self._log_dir is not None
        assert self._log_metadata is not None
        file_path = self._log_dir / f"{name}.arrow"
        schema = add_log_metadata_to_arrow_schema(pa.schema(list(schema_dict.items())), self._log_metadata)
        writer = _ModalityWriter(file_path, schema, self._ipc_compression, self._ipc_compression_level)
        self._modality_writers[name] = writer
        return writer

    def _close_writers(self) -> None:
        """Close all open modality writers."""
        for writer in self._modality_writers.values():
            writer.close()
        self._modality_writers = {}

    def reset(self, dataset_converter_config: DatasetConverterConfig, log_metadata: LogMetadata) -> bool:
        """Inherited, see superclass."""
        log_needs_writing: bool = False
        log_dir: Path = self._logs_root / log_metadata.split / log_metadata.log_name

        sync_file_path = log_dir / f"{SYNC_NAME}.arrow"
        if not sync_file_path.exists() or dataset_converter_config.force_log_conversion:
            log_needs_writing = True

            # Close any previous writers
            self._close_writers()

            if log_dir.exists():
                shutil.rmtree(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            self._dataset_converter_config = dataset_converter_config
            self._log_metadata = log_metadata
            self._log_dir = log_dir

            # --- Create per-modality writers ---
            self._create_modality_writer(SYNC_NAME, SYNC_SCHEMA_DICT)

            if dataset_converter_config.include_ego:
                self._create_modality_writer(EGO_STATE_SE3_NAME, EGO_STATE_SE3_SCHEMA_DICT)

            if dataset_converter_config.include_box_detections:
                self._create_modality_writer(BOX_DETECTIONS_SE3_NAME, BOX_DETECTIONS_SE3_SCHEMA_DICT)

            if dataset_converter_config.include_traffic_lights:
                self._create_modality_writer(TRAFFIC_LIGHTS_NAME, TRAFFIC_LIGHTS_SCHEMA_DICT)

            if dataset_converter_config.include_pinhole_cameras:
                for cam_type in log_metadata.pinhole_camera_metadata.keys():
                    cam_name = cam_type.serialize()
                    schema_dict = _get_pcam_schema_dict(cam_name, dataset_converter_config.pinhole_camera_store_option)
                    self._create_modality_writer(PCAM_NAME(cam_name), schema_dict)

            if dataset_converter_config.include_fisheye_mei_cameras:
                for cam_type in log_metadata.fisheye_mei_camera_metadata.keys():
                    cam_name = cam_type.serialize()
                    schema_dict = _get_fcam_schema_dict(
                        cam_name, dataset_converter_config.fisheye_mei_camera_store_option
                    )
                    self._create_modality_writer(FCAM_NAME(cam_name), schema_dict)

            if dataset_converter_config.include_lidars and len(log_metadata.lidar_metadata) > 0:
                lidar_name = LidarID.LIDAR_MERGED.serialize()
                if dataset_converter_config.lidar_store_option == "path":
                    schema_dict = LIDAR_STRING_SCHEMA_DICT(lidar_name)
                elif dataset_converter_config.lidar_store_option == "binary":
                    schema_dict = LIDAR_BINARY_SCHEMA_DICT(lidar_name)
                else:
                    raise ValueError(f"Unsupported lidar store option: {dataset_converter_config.lidar_store_option}")
                self._create_modality_writer(LIDAR_NAME(lidar_name), schema_dict)

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
        traffic_lights: Optional[TrafficLights] = None,
        pinhole_cameras: Optional[List[CameraData]] = None,
        fisheye_mei_cameras: Optional[List[CameraData]] = None,
        lidar: Optional[LidarData] = None,
    ) -> None:
        """Inherited, see superclass.

        Writes one sync row and dispatches each provided modality to its own writer.
        """
        assert self._dataset_converter_config is not None, "Log writer is not initialized. Call reset() first."
        assert self._log_metadata is not None, "Log writer is not initialized. Call reset() first."

        self._current_timestamp = timestamp

        if uuid is None:
            uuid = create_deterministic_uuid(
                split=self._log_metadata.split,
                log_name=self._log_metadata.log_name,
                timestamp_us=timestamp.time_us,
            )

        # Write sync row (reference timeline)
        sync_writer = self._modality_writers[SYNC_NAME]
        sync_writer.write_batch({
            f"{SYNC_NAME}.uuid": [uuid.bytes],
            f"{SYNC_NAME}.timestamp_us": [timestamp.time_us],
        })

        # Dispatch to per-modality writers
        if ego_state_se3 is not None:
            self.write_ego_state_se3(ego_state_se3)

        if box_detections_se3 is not None:
            self.write_box_detections_se3(box_detections_se3)

        if traffic_lights is not None:
            self.write_traffic_lights(traffic_lights)

        if pinhole_cameras is not None:
            for camera_data in pinhole_cameras:
                self.write_pinhole_camera(camera_data)

        if fisheye_mei_cameras is not None:
            for camera_data in fisheye_mei_cameras:
                self.write_fisheye_mei_camera(camera_data)

        if lidar is not None:
            self.write_lidar(lidar)

    # ------------------------------------------------------------------------------------------------------------------
    # Individual modality writers (usable independently for async writing)
    # ------------------------------------------------------------------------------------------------------------------

    def write_ego_state_se3(self, ego_state_se3: EgoStateSE3) -> None:
        """Write a single ego-state observation to ``ego_state_se3.arrow``."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_ego:
            return

        writer = self._modality_writers[EGO_STATE_SE3_NAME]
        n = EGO_STATE_SE3_NAME
        timestamp_us = self._current_timestamp.time_us if self._current_timestamp is not None else 0

        writer.write_batch({
            f"{n}.imu_se3": [ego_state_se3.imu_se3.array],
            f"{n}.dynamic_state_se3": [ego_state_se3.dynamic_state_se3],
            f"{n}.timestamp_us": [timestamp_us],
        })

    def write_box_detections_se3(self, box_detections_se3: BoxDetectionsSE3) -> None:
        """Write box detections to ``box_detections_se3.arrow`` (one row per detection)."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_box_detections:
            return

        writer = self._modality_writers[BOX_DETECTIONS_SE3_NAME]
        n = BOX_DETECTIONS_SE3_NAME
        timestamp_us = self._current_timestamp.time_us if self._current_timestamp is not None else 0

        bounding_box_se3_list = []
        tokens_list = []
        labels_list = []
        velocities_list = []
        num_lidar_points_list = []
        timestamps_list = []

        for det in box_detections_se3:
            bounding_box_se3_list.append(det.bounding_box_se3)
            tokens_list.append(det.metadata.track_token)
            labels_list.append(int(det.metadata.label))
            velocities_list.append(det.velocity_3d)
            num_lidar_points_list.append(det.metadata.num_lidar_points)
            timestamps_list.append(timestamp_us)

        if len(bounding_box_se3_list) > 0:
            writer.write_batch({
                f"{n}.bounding_box_se3": [bounding_box_se3_list],
                f"{n}.token": [tokens_list],
                f"{n}.label": [labels_list],
                f"{n}.velocity_3d": [velocities_list],
                f"{n}.num_lidar_points": [num_lidar_points_list],
                f"{n}.timestamp_us": [timestamps_list],
            })

    def write_traffic_lights(self, traffic_lights: TrafficLights) -> None:
        """Write traffic lights to ``traffic_lights.arrow`` (one row per traffic light)."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_traffic_lights:
            return

        writer = self._modality_writers[TRAFFIC_LIGHTS_NAME]
        n = TRAFFIC_LIGHTS_NAME
        timestamp_us = self._current_timestamp.time_us if self._current_timestamp is not None else 0

        lane_ids = []
        statuses = []
        timestamp_list = []

        for tl in traffic_lights:
            lane_ids.append(tl.lane_id)
            statuses.append(int(tl.status))
            timestamp_list.append(timestamp_us)

        if len(lane_ids) > 0:
            writer.write_batch({
                f"{n}.lane_id": [lane_ids],
                f"{n}.status": [statuses],
                f"{n}.timestamp_us": [timestamp_list],
            })

    def write_pinhole_camera(self, camera_data: CameraData) -> None:
        """Write a single pinhole camera observation to ``pinhole_camera.{name}.arrow``."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_pinhole_cameras:
            return

        cam_name = camera_data.camera_id.serialize()
        writer_name = PCAM_NAME(cam_name)
        writer = self._modality_writers[writer_name]

        store_option = self._dataset_converter_config.pinhole_camera_store_option
        data_value = self._get_camera_data_value(camera_data, store_option, self._pinhole_mp4_writers)
        timestamp_us = (
            camera_data.timestamp.time_us
            if camera_data.timestamp is not None
            else (self._current_timestamp.time_us if self._current_timestamp is not None else None)
        )

        writer.write_batch({
            f"{writer_name}.data": [data_value],
            f"{writer_name}.state_se3": [camera_data.extrinsic],
            f"{writer_name}.timestamp_us": [timestamp_us],
        })

    def write_fisheye_mei_camera(self, camera_data: CameraData) -> None:
        """Write a single fisheye MEI camera observation to ``fisheye_mei.{name}.arrow``."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_fisheye_mei_cameras:
            return

        cam_name = camera_data.camera_id.serialize()
        writer_name = FCAM_NAME(cam_name)
        writer = self._modality_writers[writer_name]

        store_option = self._dataset_converter_config.fisheye_mei_camera_store_option
        data_value = self._get_camera_data_value(camera_data, store_option, self._fisheye_mei_mp4_writers)
        timestamp_us = (
            camera_data.timestamp.time_us
            if camera_data.timestamp is not None
            else (self._current_timestamp.time_us if self._current_timestamp is not None else None)
        )

        writer.write_batch({
            f"{writer_name}.data": [data_value],
            f"{writer_name}.state_se3": [camera_data.extrinsic],
            f"{writer_name}.timestamp_us": [timestamp_us],
        })

    def write_lidar(self, lidar_data: LidarData) -> None:
        """Write a single lidar observation to ``lidar.{name}.arrow``."""
        assert self._dataset_converter_config is not None, "Log writer is not initialized."
        assert self._log_metadata is not None, "Log writer is not initialized."
        if not self._dataset_converter_config.include_lidars:
            return
        if len(self._log_metadata.lidar_metadata) == 0:
            return

        lidar_name = LidarID.LIDAR_MERGED.serialize()
        writer_name = LIDAR_NAME(lidar_name)
        writer = self._modality_writers[writer_name]

        timestamp_us = (
            lidar_data.timestamp.time_us
            if lidar_data.timestamp is not None
            else (self._current_timestamp.time_us if self._current_timestamp is not None else 0)
        )

        if self._dataset_converter_config.lidar_store_option == "path":
            data_path: Optional[str] = str(lidar_data.relative_path) if lidar_data.has_file_path else None
            writer.write_batch({
                f"{writer_name}.data": [data_path],
                f"{writer_name}.start_timestamp_us": [timestamp_us],
                f"{writer_name}.end_timestamp_us": [timestamp_us],
            })
        elif self._dataset_converter_config.lidar_store_option == "binary":
            point_cloud_binary, features_binary = self._prepare_lidar_data(lidar_data)
            writer.write_batch({
                f"{writer_name}.point_cloud_3d": [point_cloud_binary],
                f"{writer_name}.point_cloud_features": [features_binary],
                f"{writer_name}.start_timestamp_us": [timestamp_us],
                f"{writer_name}.end_timestamp_us": [timestamp_us],
            })
        else:
            raise ValueError(f"Unsupported lidar store option: {self._dataset_converter_config.lidar_store_option}")

    def write_aux_dict(self, aux_dict: Dict[str, Union[str, int, float, bool]]) -> None:
        """Write auxiliary data. Not yet implemented."""
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def close(self) -> None:
        """Inherited, see superclass."""
        self._close_writers()

        self._dataset_converter_config = None
        self._log_metadata = None
        self._log_dir = None
        self._current_timestamp = None

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
        return decode_image_from_jpeg_binary(camera_data.jpeg_binary)
    elif camera_data.has_jpeg_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        return load_image_from_jpeg_file(absolute_path)
    elif camera_data.has_png_file_path:
        absolute_path = Path(camera_data.dataset_root) / camera_data.relative_path  # type: ignore
        return load_image_from_png_file(absolute_path)
    else:
        raise NotImplementedError("Camera data must provide numpy_image, jpeg_binary, or file path for numpy image.")
