from pathlib import Path
from typing import Any, Literal, Optional, Union

import cv2
import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema
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
    encode_image_as_png_binary,
    is_png_binary,
    load_image_from_png_file,
    load_png_binary_from_png_file,
)
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata, Camera
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.geometry_index import PoseSE3Index
from py123d.geometry.pose import PoseSE3
from py123d.parser.base_dataset_parser import ParsedCamera
from py123d.script.utils.dataset_path_utils import get_dataset_paths

# ------------------------------------------------------------------------------------------------------------------
# Writers
# ------------------------------------------------------------------------------------------------------------------

CAMERA_CODEC_PA_DTYPES = {
    "path": pa.string(),
    "jpeg_binary": pa.binary(),
    "png_binary": pa.binary(),
    "mp4": pa.int32(),
}

CAMERA_CODEC_MAX_BATCH_SIZES = {
    "path": 1000,
    "jpeg_binary": 10,
    "png_binary": 10,
    "mp4": 1000,
}


class ArrowCameraWriter(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        camera_codec: Literal["path", "jpeg_binary", "png_binary", "mp4"] = "path",
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, BaseCameraMetadata), f"Expected BaseCameraMetadata subclass, got {type(metadata)}"
        assert camera_codec in {"path", "jpeg_binary", "png_binary", "mp4"}, f"Unsupported camera codec: {camera_codec}"

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
            ]
        )
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

        self.write_batch(
            {
                f"{self._metadata.modality_key}.timestamp_us": [modality.timestamp.time_us],
                f"{self._metadata.modality_key}.data": [data],
                f"{self._metadata.modality_key}.camera_to_global_se3": [modality.camera_to_global_se3],
            }
        )

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
        column_at_iteration: Optional[Any] = None
        full_column_name = f"{metadata.modality_key}.{column}"
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()
        if deserialize and column_at_iteration is not None:
            if column == "data":
                column_at_iteration = _deserialize_data_column(
                    data=column_at_iteration,
                    dataset=dataset,
                    scale=scale,
                    log_dir=log_dir,
                    modality_key=metadata.modality_key,
                )
            elif column == "camera_to_global_se3":
                column_at_iteration = PoseSE3.from_list(column_at_iteration)
            elif column == "timestamp_us":
                column_at_iteration = Timestamp.from_us(column_at_iteration)
        return column_at_iteration


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

    if table_data is None or camera_to_global_se3_data is None:
        return None
    image = _deserialize_data_column(
        data=table_data,
        dataset=dataset,
        scale=scale,
        log_dir=log_dir,
        modality_key=modality_key,
    )
    camera_to_global_se3 = PoseSE3.from_list(camera_to_global_se3_data)
    assert image is not None, "Failed to load camera image from Arrow table data."
    return Camera(
        metadata=camera_metadata,
        image=image,
        camera_to_global_se3=camera_to_global_se3,
        timestamp=Timestamp.from_us(timestamp_data),
    )


def _deserialize_data_column(
    data: Union[str, bytes, int],
    dataset: str,
    scale: Optional[int] = None,
    log_dir: Optional[Path] = None,
    modality_key: Optional[str] = None,
) -> Optional[Any]:
    image: Optional[np.ndarray] = None
    if isinstance(data, str):
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
