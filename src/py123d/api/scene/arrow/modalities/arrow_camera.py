from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import (
    ARRIVAL_TIME_FIELD,
    add_arrival_time_to_row,
    all_columns_in_schema,
    get_arrival_time_field,
    read_arrival_time_column,
    read_arrival_timestamp,
)
from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.io.camera.jpeg_camera_io import (
    decode_image_from_jpeg_binary,
    encode_image_as_jpeg_binary,
    is_jpeg_binary,
    load_image_from_jpeg_file,
    load_jpeg_binary_from_jpeg_file,
)
from py123d.common.io.camera.mp4_camera_io import MP4Writer, get_mp4_reader_from_path
from py123d.common.io.camera.png_camera_io import (
    decode_image_from_png_binary,
    decode_label_map_from_png_binary,
    encode_image_as_png_binary,
    encode_label_map_as_png_binary,
    is_png_binary,
    load_image_from_png_file,
    load_png_binary_from_png_file,
)
from py123d.common.runtime import get_dataset_paths
from py123d.datatypes.modalities.base_modality import (
    BaseModality,
    BaseModalityMetadata,
    ModalityType,
    get_modality_key,
)
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, Camera, CameraChannelType
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.geometry_index import PoseSE3Index
from py123d.geometry.pose import PoseSE3
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.geometry.utils.rotation_utils import slerp_quaternion_arrays
from py123d.parser.base_dataset_parser import ParsedCamera

# ------------------------------------------------------------------------------------------------------------------
# Writers
# ------------------------------------------------------------------------------------------------------------------

CAMERA_CODEC_PA_DTYPES = {
    "path": pa.string(),
    "jpeg_binary": pa.binary(),
    "png_binary": pa.binary(),
    "label_png": pa.binary(),
    "mp4": pa.int32(),
}

CAMERA_CODEC_MAX_BATCH_SIZES = {
    "path": 1000,
    "jpeg_binary": 10,
    "png_binary": 10,
    "label_png": 10,
    "mp4": 1000,
}


class ArrowCameraWriter(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        camera_codec: Literal["path", "jpeg_binary", "png_binary", "label_png", "mp4"] = "path",
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, BaseCameraMetadata), f"Expected BaseCameraMetadata subclass, got {type(metadata)}"
        assert camera_codec in CAMERA_CODEC_PA_DTYPES, f"Unsupported camera codec: {camera_codec}"

        self._metadata = metadata
        self._camera_codec = camera_codec
        self._log_dir = log_dir
        self._mp4_writer: Optional[MP4Writer] = None

        data_type = CAMERA_CODEC_PA_DTYPES[camera_codec]
        max_batch_size = CAMERA_CODEC_MAX_BATCH_SIZES[camera_codec]

        file_path = log_dir / f"{metadata.modality_key}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_key}.timestamp_us", pa.int64()),
                (f"{metadata.modality_key}.data", data_type),
                (f"{metadata.modality_key}.camera_to_global_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
                # Per-frame exposure normalization gain (see Camera.exposure_factor); null
                # for datasets that do not provide it.
                (f"{metadata.modality_key}.exposure_factor", pa.float32()),
            ]
        )
        if metadata.has_arrival_time:
            schema = schema.append(pa.field(*get_arrival_time_field(metadata.modality_key)))
        schema = add_metadata_to_arrow_schema(schema, metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=max_batch_size,
        )

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, (ParsedCamera, Camera)), f"Expected ParsedCamera or Camera, got {type(modality)}"
        if self._camera_codec == "jpeg_binary":
            data: Union[str, bytes, int] = _get_jpeg_binary_from_camera_modality(modality)
        elif self._camera_codec == "png_binary":
            data = _get_png_binary_from_camera_modality(modality)
        elif self._camera_codec == "label_png":
            data = _get_label_png_binary_from_camera_modality(modality)
        elif self._camera_codec == "mp4":
            image = _get_numpy_image_from_camera_modality(modality)
            if self._mp4_writer is None:
                mp4_path = self._log_dir / f"{self._metadata.modality_key}.mp4"
                self._mp4_writer = MP4Writer(mp4_path)
            data = self._mp4_writer.write_frame(image)
        elif self._camera_codec == "path":
            assert isinstance(modality, ParsedCamera), (
                f"Path codec requires ParsedCamera with file path, got {type(modality)}"
            )
            assert modality.has_file_path, "ParsedCamera must have a file path for path codec."
            data = str(modality.relative_path)
        else:
            raise NotImplementedError(f"Unsupported camera codec: {self._camera_codec}")

        row: Dict[str, Any] = {
            f"{self._metadata.modality_key}.timestamp_us": [modality.timestamp.time_us],
            f"{self._metadata.modality_key}.data": [data],
            f"{self._metadata.modality_key}.camera_to_global_se3": [modality.camera_to_global_se3],
            # None where a dataset stores the pose implicitly, to be composed on read from
            # ego_state_se3 and the camera extrinsic (see _camera_to_global_from_ego).
            f"{self._metadata.modality_key}.exposure_factor": [modality.exposure_factor],
        }
        add_arrival_time_to_row(row, self._metadata, modality)
        self.write_batch(row)

    def close(self) -> None:
        if self._mp4_writer is not None:
            self._mp4_writer.close()
            self._mp4_writer = None
        super().close()


# ------------------------------------------------------------------------------------------------------------------
# Writer Helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_jpeg_binary_from_camera_modality(camera_data: Union[ParsedCamera, Camera]) -> bytes:
    if isinstance(camera_data, ParsedCamera):
        if camera_data.has_byte_string:
            byte_string = camera_data._byte_string
            assert byte_string is not None
            if is_jpeg_binary(byte_string):
                return byte_string
            elif is_png_binary(byte_string):
                return encode_image_as_jpeg_binary(decode_image_from_png_binary(byte_string))
            else:
                raise ValueError("ParsedCamera byte_string is neither JPEG nor PNG.")
        elif camera_data.has_jpeg_file_path:
            absolute_path = Path(camera_data._dataset_root) / camera_data.relative_path  # type: ignore
            return load_jpeg_binary_from_jpeg_file(absolute_path)
        elif camera_data.has_png_file_path:
            absolute_path = Path(camera_data._dataset_root) / camera_data.relative_path  # type: ignore
            numpy_image = load_image_from_png_file(absolute_path)
            return encode_image_as_jpeg_binary(numpy_image)
        else:
            raise NotImplementedError("ParsedCamera must provide byte_string or file path for jpeg_binary codec.")
    elif isinstance(camera_data, Camera):
        return encode_image_as_jpeg_binary(camera_data.image)
    else:
        raise NotImplementedError(f"Unsupported camera type for jpeg_binary codec: {type(camera_data)}")


def _get_png_binary_from_camera_modality(camera_data: Union[ParsedCamera, Camera]) -> bytes:
    if isinstance(camera_data, ParsedCamera):
        if camera_data.has_byte_string:
            byte_string = camera_data._byte_string
            assert byte_string is not None
            if is_png_binary(byte_string):
                return byte_string
            elif is_jpeg_binary(byte_string):
                return encode_image_as_png_binary(decode_image_from_jpeg_binary(byte_string))
            else:
                raise ValueError("ParsedCamera byte_string is neither JPEG nor PNG.")
        elif camera_data.has_png_file_path:
            absolute_path = Path(camera_data._dataset_root) / camera_data.relative_path  # type: ignore
            return load_png_binary_from_png_file(absolute_path)
        elif camera_data.has_jpeg_file_path:
            absolute_path = Path(camera_data._dataset_root) / camera_data.relative_path  # type: ignore
            numpy_image = load_image_from_jpeg_file(absolute_path)
            return encode_image_as_png_binary(numpy_image)
        else:
            raise NotImplementedError("ParsedCamera must provide byte_string or file path for png_binary codec.")
    elif isinstance(camera_data, Camera):
        return encode_image_as_png_binary(camera_data.image)
    else:
        raise NotImplementedError(f"Unsupported camera type for png_binary codec: {type(camera_data)}")


def _get_label_png_binary_from_camera_modality(camera_data: Union[ParsedCamera, Camera]) -> bytes:
    """Encode a semantic camera (single-channel integer label map) as lossless PNG binary.

    A :class:`Camera` carries the label map directly in ``image``. A :class:`ParsedCamera` is
    expected to already provide a PNG-encoded label map (byte string or ``.png`` file).
    """
    if isinstance(camera_data, Camera):
        return encode_label_map_as_png_binary(camera_data.image)
    elif isinstance(camera_data, ParsedCamera):
        if camera_data.has_byte_string:
            byte_string = camera_data._byte_string
            assert byte_string is not None and is_png_binary(byte_string), (
                "label_png codec requires the ParsedCamera byte_string to be a PNG-encoded label map."
            )
            return byte_string
        elif camera_data.has_png_file_path:
            absolute_path = Path(camera_data._dataset_root) / camera_data.relative_path  # type: ignore
            return load_png_binary_from_png_file(absolute_path)
        else:
            raise NotImplementedError("label_png codec requires a Camera image or a PNG ParsedCamera.")
    else:
        raise NotImplementedError(f"Unsupported camera type for label_png codec: {type(camera_data)}")


def _get_numpy_image_from_camera_modality(camera_data: Union[ParsedCamera, Camera]) -> np.ndarray:
    """Extract an RGB numpy image from a camera modality for MP4 encoding."""
    if isinstance(camera_data, Camera):
        return camera_data.image
    elif isinstance(camera_data, ParsedCamera):
        if camera_data.has_byte_string:
            byte_string = camera_data._byte_string
            assert byte_string is not None
            if is_jpeg_binary(byte_string):
                return decode_image_from_jpeg_binary(byte_string)
            elif is_png_binary(byte_string):
                return decode_image_from_png_binary(byte_string)
            else:
                raise ValueError("ParsedCamera byte_string is neither JPEG nor PNG.")
        elif camera_data.has_jpeg_file_path:
            absolute_path = Path(camera_data._dataset_root) / camera_data.relative_path  # type: ignore
            return load_image_from_jpeg_file(absolute_path)
        elif camera_data.has_png_file_path:
            absolute_path = Path(camera_data._dataset_root) / camera_data.relative_path  # type: ignore
            return load_image_from_png_file(absolute_path)
        else:
            raise NotImplementedError("ParsedCamera must provide byte_string or file path for mp4 codec.")
    else:
        raise NotImplementedError(f"Unsupported camera type for mp4 codec: {type(camera_data)}")


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowCameraReader(ArrowBaseModalityReader):
    """Stateless reader for camera data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        scale: Optional[int] = None,
        log_dir: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Camera]:
        assert isinstance(metadata, BaseCameraMetadata)
        return _deserialize_camera(table, index, metadata, dataset, scale=scale, log_dir=log_dir)

    @staticmethod
    def read_column_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        column: str,
        dataset: str,
        deserialize: bool = False,
        scale: Optional[int] = None,
        log_dir: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Any]:
        if column == ARRIVAL_TIME_FIELD:
            return read_arrival_time_column(table, index, metadata.modality_key, deserialize)
        column_at_iteration: Optional[Any] = None
        full_column_name = f"{metadata.modality_key}.{column}"
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()
        if column == "camera_to_global_se3" and column_at_iteration is None:
            # Stored implicitly; see _camera_to_global_from_ego. Returned in the same shape the
            # column would have had, so a caller that asked for the raw value still gets one.
            timestamp_column = f"{metadata.modality_key}.timestamp_us"
            timestamp_us = table[timestamp_column][index].as_py() if timestamp_column in table.column_names else None
            assert isinstance(metadata, BaseCameraMetadata)
            pose = _camera_to_global_from_ego(log_dir, metadata, timestamp_us)
            return pose if (deserialize and pose is not None) else (None if pose is None else pose.tolist())
        if deserialize and column_at_iteration is not None:
            if column == "data":
                column_at_iteration = _deserialize_data_column(
                    data=column_at_iteration,
                    dataset=dataset,
                    scale=scale,
                    log_dir=log_dir,
                    modality_key=metadata.modality_key,
                    channel_type=metadata.channel_type,
                )
            elif column == "camera_to_global_se3":
                column_at_iteration = PoseSE3.from_list(column_at_iteration)
            elif column == "timestamp_us":
                column_at_iteration = Timestamp.from_us(column_at_iteration)
        return column_at_iteration


# ------------------------------------------------------------------------------------------------------------------
# Camera pose from the ego trajectory
# ------------------------------------------------------------------------------------------------------------------

# ``camera_to_global_se3`` is ego_pose composed with the camera's own extrinsic, so it is the one
# per-frame field that a change to the ego trajectory invalidates. A dataset may therefore leave
# it null and have it composed here instead, which turns re-estimating a trajectory from a
# rewrite of every camera table into a rewrite of ``ego_state_se3.arrow`` alone -- on the Kesai
# logs, 7 MB rather than 38 GB. Nothing else changes: lidar and radar points are stored in the
# IMU frame and are unaffected by the ego pose, and a log that carries the column keeps using it,
# so every dataset written before this reads back exactly as it did.


def _ego_state_key() -> str:
    """The modality key the ego trajectory is stored under, from the enum rather than a literal."""
    return get_modality_key(ModalityType.EGO_STATE_SE3)


@lru_cache(maxsize=8)
def _ego_trajectory(log_dir_str: str) -> Optional[Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]]:
    """The log's ego poses as arrays, cached per log directory.

    :param log_dir_str: The log directory, as a string so it can be a cache key.
    :return: Tuple of (timestamps in microseconds, (N, 7) poses), or None if the log has no ego states.
    """
    path = Path(log_dir_str) / f"{_ego_state_key()}.arrow"
    if not path.exists():
        return None
    table = get_lru_cached_arrow_table(path)
    timestamp_column = f"{_ego_state_key()}.timestamp_us"
    pose_column = f"{_ego_state_key()}.imu_se3"
    if timestamp_column not in table.column_names or pose_column not in table.column_names:
        return None
    timestamps = table[timestamp_column].to_numpy().astype(np.int64)
    poses = np.asarray(table[pose_column].to_pylist(), dtype=np.float64)
    if not len(timestamps) or poses.shape[-1] != len(PoseSE3Index):
        return None
    return timestamps, poses


def _ego_pose_at(timestamps_us: npt.NDArray[np.int64], poses: npt.NDArray[np.float64], timestamp_us: int) -> PoseSE3:
    """The ego pose at an arbitrary time, interpolated and clamped to the trajectory's range.

    Interpolated rather than snapped: the ego states are 100 Hz and a camera frame falls between
    them, so taking the nearest would displace the pose by up to 5 ms of travel -- 15 cm at
    30 m/s, which is the same order as the trajectory's own accuracy.
    """
    if len(timestamps_us) == 1:
        return PoseSE3.from_list(poses[0].tolist())
    clamped = int(np.clip(timestamp_us, timestamps_us[0], timestamps_us[-1]))
    upper = int(np.clip(np.searchsorted(timestamps_us, clamped, side="right"), 1, len(timestamps_us) - 1))
    lower = upper - 1
    span = float(timestamps_us[upper] - timestamps_us[lower])
    ratio = 0.0 if span <= 0.0 else (clamped - timestamps_us[lower]) / span

    interpolated = np.empty(len(PoseSE3Index), dtype=np.float64)
    interpolated[PoseSE3Index.XYZ] = (
        poses[lower, PoseSE3Index.XYZ] + (poses[upper, PoseSE3Index.XYZ] - poses[lower, PoseSE3Index.XYZ]) * ratio
    )
    interpolated[PoseSE3Index.QUATERNION] = slerp_quaternion_arrays(
        poses[lower, PoseSE3Index.QUATERNION], poses[upper, PoseSE3Index.QUATERNION], np.array(ratio)
    )
    return PoseSE3.from_list(interpolated.tolist())


def _camera_to_global_from_ego(
    log_dir: Optional[Path], metadata: BaseCameraMetadata, timestamp_us: Optional[int]
) -> Optional[PoseSE3]:
    """Compose the camera's world pose from the log's ego trajectory and the camera extrinsic.

    :param log_dir: The log directory; without it the trajectory cannot be found.
    :param metadata: The camera metadata, carrying ``camera_to_imu_se3``.
    :param timestamp_us: The frame time to place the camera at.
    :return: The camera-to-global pose, or None when the log carries no ego trajectory.
    """
    if log_dir is None or timestamp_us is None:
        return None
    trajectory = _ego_trajectory(str(log_dir))
    if trajectory is None:
        return None
    timestamps_us, poses = trajectory
    return rel_to_abs_se3(origin=_ego_pose_at(timestamps_us, poses, timestamp_us), pose_se3=metadata.camera_to_imu_se3)


# ------------------------------------------------------------------------------------------------------------------
# Reader Internals
# ------------------------------------------------------------------------------------------------------------------


def _deserialize_camera(
    arrow_table: pa.Table,
    index: int,
    camera_metadata: BaseCameraMetadata,
    dataset: str,
    scale: Optional[int] = None,
    log_dir: Optional[Path] = None,
) -> Optional[Camera]:
    """Deserialize a camera observation from Arrow table columns at the given row index."""
    modality_key = camera_metadata.modality_key

    camera_data_column = f"{modality_key}.data"
    camera_extrinsic_column = f"{modality_key}.camera_to_global_se3"
    camera_timestamp_column = f"{modality_key}.timestamp_us"

    if not all_columns_in_schema(arrow_table, [camera_data_column, camera_extrinsic_column, camera_timestamp_column]):
        return None

    table_data = arrow_table[camera_data_column][index].as_py()
    camera_to_global_se3_data = arrow_table[camera_extrinsic_column][index].as_py()
    timestamp_data = arrow_table[camera_timestamp_column][index].as_py()

    # Optional column; absent in logs written before it was added.
    exposure_factor_column = f"{modality_key}.exposure_factor"
    exposure_factor = (
        arrow_table[exposure_factor_column][index].as_py()
        if exposure_factor_column in arrow_table.column_names
        else None
    )

    if table_data is None:
        return None
    # A null pose means the dataset stores it implicitly: compose it from the ego trajectory and
    # the camera's own extrinsic. A dataset that wrote the column keeps its stored value, so
    # every log written before this reads back byte for byte as it did.
    if camera_to_global_se3_data is None:
        camera_to_global_se3 = _camera_to_global_from_ego(log_dir, camera_metadata, timestamp_data)
        if camera_to_global_se3 is None:
            return None
    else:
        camera_to_global_se3 = PoseSE3.from_list(camera_to_global_se3_data)
    image = _deserialize_data_column(
        data=table_data,
        dataset=dataset,
        scale=scale,
        log_dir=log_dir,
        modality_key=modality_key,
        channel_type=camera_metadata.channel_type,
    )
    assert image is not None, "Failed to load camera image from Arrow table data."
    return Camera(
        metadata=camera_metadata,
        image=image,
        camera_to_global_se3=camera_to_global_se3,
        timestamp=Timestamp.from_us(timestamp_data),
        exposure_factor=exposure_factor,
        arrival_timestamp=read_arrival_timestamp(arrow_table, index, modality_key),
    )


def _deserialize_data_column(
    data: Union[str, bytes, int],
    dataset: str,
    scale: Optional[int] = None,
    log_dir: Optional[Path] = None,
    modality_key: Optional[str] = None,
    channel_type: CameraChannelType = CameraChannelType.RGB,
) -> Optional[Any]:
    image: Optional[np.ndarray] = None
    # Segmentation cameras (semantic class-id or panoptic/instance) and depth cameras (quantized metric
    # depth) both store a single-channel integer raster; decode it without colour conversion and resample
    # with nearest-neighbour so the raw integer values are preserved exactly — bilinear would blend class
    # ids into nonexistent classes, and depth across occlusion boundaries into nonexistent surfaces.
    if channel_type in (CameraChannelType.SEMANTIC, CameraChannelType.INSTANCE, CameraChannelType.DEPTH):
        if isinstance(data, bytes):
            image = decode_label_map_from_png_binary(data, scale=scale)
        elif isinstance(data, str):
            sensor_root = get_dataset_paths().get_sensor_root(dataset)
            assert sensor_root is not None, f"Dataset path for sensor loading not found for dataset: {dataset}"
            full_label_path = Path(sensor_root) / data
            assert full_label_path.exists(), f"{channel_type.name} camera file not found: {full_label_path}"
            image = decode_label_map_from_png_binary(load_png_binary_from_png_file(full_label_path), scale=scale)
        else:
            raise NotImplementedError(
                f"{channel_type.name} camera data must be PNG bytes or a file path, got {type(data)}."
            )
    elif isinstance(data, str):
        sensor_root = get_dataset_paths().get_sensor_root(dataset)
        assert sensor_root is not None, f"Dataset path for sensor loading not found for dataset: {dataset}"
        full_image_path = Path(sensor_root) / data
        assert full_image_path.exists(), f"Camera file not found: {full_image_path}"
        image = load_image_from_jpeg_file(full_image_path, scale=scale)
    elif isinstance(data, bytes):
        if is_jpeg_binary(data):
            image = decode_image_from_jpeg_binary(data, scale=scale)
        elif is_png_binary(data):
            image = decode_image_from_png_binary(data, scale=scale)
        else:
            raise ValueError("Camera binary data is neither in JPEG nor PNG format.")
    elif isinstance(data, int):
        assert log_dir is not None, "log_dir is required for MP4 frame index deserialization."
        assert modality_key is not None, "modality_key is required for MP4 frame index deserialization."
        mp4_path = str(log_dir / f"{modality_key}.mp4")
        reader = get_mp4_reader_from_path(mp4_path)
        image = reader.get_frame(data)
        if image is not None and scale is not None and scale > 1:
            h, w = image.shape[:2]
            image = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    else:
        raise NotImplementedError(
            f"Only string file paths, bytes, or int frame indices are supported for camera data, got {type(data)}"
        )
    return image
