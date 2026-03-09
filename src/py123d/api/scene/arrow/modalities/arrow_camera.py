from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.io.camera.jpeg_camera_io import (
    decode_image_from_jpeg_binary,
    encode_image_as_jpeg_binary,
    is_jpeg_binary,
    load_image_from_jpeg_file,
    load_jpeg_binary_from_jpeg_file,
)
from py123d.common.io.camera.mp4_camera_io import get_mp4_reader_from_path
from py123d.common.io.camera.png_camera_io import (
    decode_image_from_png_binary,
    encode_image_as_png_binary,
    is_png_binary,
    load_image_from_png_file,
    load_png_binary_from_png_file,
)
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraID, FisheyeMEICameraMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraID, PinholeCameraMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.geometry_index import PoseSE3Index
from py123d.geometry.pose import PoseSE3
from py123d.parser.abstract_dataset_parser import ParsedCamera
from py123d.script.utils.dataset_path_utils import get_dataset_paths


class ArrowPinholeCameraWriter(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: PinholeCameraMetadata,
        data_codec: Literal["path", "jpeg_binary", "png_binary"] = "path",
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, PinholeCameraMetadata), f"Expected PinholeCameraMetadata, got {type(metadata)}"
        assert data_codec in {"path", "jpeg_binary", "png_binary"}, f"Unsupported data codec: {data_codec}"

        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name
        self._data_codec = data_codec

        data_type = pa.binary() if data_codec in {"jpeg_binary", "png_binary"} else pa.string()
        max_batch_size = 10 if data_codec in {"jpeg_binary", "png_binary"} else 1000

        file_path = log_dir / f"{metadata.modality_name}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_name}.timestamp_us", pa.int64()),
                (f"{metadata.modality_name}.data", data_type),
                (f"{metadata.modality_name}.state_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
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

    def write_modality(self, camera_data: ParsedCamera):
        assert isinstance(camera_data, ParsedCamera), f"Expected CameraData, got {type(camera_data)}"
        if self._data_codec == "jpeg_binary":
            data = _get_jpeg_binary_from_camera_data(camera_data)
        elif self._data_codec == "png_binary":
            data = _get_png_binary_from_camera_data(camera_data)
        else:
            data = str(camera_data.relative_path)
        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [camera_data.timestamp.time_us],
                f"{self._modality_name}.data": [data],
                f"{self._modality_name}.state_se3": [camera_data.extrinsic],
            }
        )


class ArrowFisheyeMEICameraWriter(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: FisheyeMEICameraMetadata,
        data_codec: Literal["path", "jpeg_binary", "png_binary"] = "path",
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, FisheyeMEICameraMetadata), (
            f"Expected FisheyeMEICameraMetadata, got {type(metadata)}"
        )
        assert data_codec in {"path", "jpeg_binary", "png_binary"}, f"Unsupported data codec: {data_codec}"

        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name
        self._data_codec = data_codec

        data_type = pa.binary() if data_codec in {"jpeg_binary", "png_binary"} else pa.string()
        max_batch_size = 10 if data_codec in {"jpeg_binary", "png_binary"} else 1000

        file_path = log_dir / f"{metadata.modality_name}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_name}.timestamp_us", pa.int64()),
                (f"{metadata.modality_name}.data", data_type),
                (f"{metadata.modality_name}.state_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
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

    def write_modality(self, camera_data: ParsedCamera):
        assert isinstance(camera_data, ParsedCamera), f"Expected CameraData, got {type(camera_data)}"
        if self._data_codec == "jpeg_binary":
            data = _get_jpeg_binary_from_camera_data(camera_data)
        elif self._data_codec == "png_binary":
            data = _get_png_binary_from_camera_data(camera_data)
        else:
            data = str(camera_data.relative_path)
        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [camera_data.timestamp.time_us],
                f"{self._modality_name}.data": [data],
                f"{self._modality_name}.state_se3": [camera_data.extrinsic],
            }
        )


def _get_jpeg_binary_from_camera_data(camera_data: ParsedCamera) -> bytes:
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


def _get_png_binary_from_camera_data(camera_data: ParsedCamera) -> bytes:
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


def get_camera_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    camera_id: Union[PinholeCameraID, FisheyeMEICameraID],
    camera_metadata: Union[PinholeCameraMetadata, FisheyeMEICameraMetadata],
    log_metadata: LogMetadata,
) -> Optional[Union[PinholeCamera, FisheyeMEICamera]]:
    """Builds a camera object from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the camera data.
    :param index: The index to extract the camera data from.
    :param camera_id: The ID of the camera to build (Pinhole or FisheyeMEI).
    :param camera_metadata: The camera metadata (intrinsics, distortion, etc.).
    :param log_metadata: Metadata about the log (used for dataset path resolution and MP4 reading).
    :raises ValueError: If the camera data format is unsupported.
    :raises NotImplementedError: If the camera data type is not supported.
    :return: The constructed camera object, or None if not available.
    """

    assert isinstance(camera_id, (PinholeCameraID, FisheyeMEICameraID)), (
        f"camera_id must be PinholeCameraID or FisheyeMEICameraID, got {type(camera_id)}"
    )

    camera: Optional[Union[PinholeCamera, FisheyeMEICamera]] = None

    is_pinhole = isinstance(camera_id, PinholeCameraID)
    modality_prefix = "pinhole_camera" if is_pinhole else "fisheye_mei_camera"
    instance = camera_id.serialize()

    camera_data_column = f"{modality_prefix}.{instance}.data"
    camera_extrinsic_column = f"{modality_prefix}.{instance}.state_se3"
    camera_timestamp_column = f"{modality_prefix}.{instance}.timestamp_us"

    if all_columns_in_schema(arrow_table, [camera_data_column, camera_extrinsic_column, camera_timestamp_column]):
        table_data = arrow_table[camera_data_column][index].as_py()
        extrinsic_data = arrow_table[camera_extrinsic_column][index].as_py()
        timestamp_data = arrow_table[camera_timestamp_column][index].as_py()

        if table_data is not None and extrinsic_data is not None:
            extrinsic = PoseSE3.from_list(extrinsic_data)
            image: Optional[npt.NDArray[np.uint8]] = None

            if isinstance(table_data, str):
                sensor_root = get_dataset_paths().get_sensor_root(log_metadata.dataset)
                assert sensor_root is not None, (
                    f"Dataset path for sensor loading not found for dataset: {log_metadata.dataset}"
                )
                full_image_path = Path(sensor_root) / table_data
                assert full_image_path.exists(), f"Camera file not found: {full_image_path}"

                image = load_image_from_jpeg_file(full_image_path)
            elif isinstance(table_data, bytes):
                if is_jpeg_binary(table_data):
                    image = decode_image_from_jpeg_binary(table_data)
                elif is_png_binary(table_data):
                    image = decode_image_from_png_binary(table_data)
                else:
                    raise ValueError("Camera binary data is neither in JPEG nor PNG format.")

            elif isinstance(table_data, int):
                image = _unoptimized_demo_mp4_read(log_metadata, camera_id.serialize(), table_data)
            else:
                raise NotImplementedError(
                    f"Only string file paths, bytes, or int frame indices are supported for camera data, "
                    f"got {type(table_data)}"
                )

            assert image is not None, "Failed to load camera image from Arrow table data."
            if is_pinhole:
                camera = PinholeCamera(
                    metadata=camera_metadata,  # type: ignore[arg-type]
                    image=image,
                    extrinsic=extrinsic,
                    timestamp=Timestamp.from_us(timestamp_data),
                )
            else:
                camera = FisheyeMEICamera(
                    metadata=camera_metadata,  # type: ignore[arg-type]
                    image=image,
                    extrinsic=extrinsic,
                    timestamp=Timestamp.from_us(timestamp_data),
                )

    return camera


def _unoptimized_demo_mp4_read(log_metadata: LogMetadata, camera_name: str, frame_index: int) -> Optional[np.ndarray]:
    """Reads a frame from an MP4 file for demonstration purposes. This features is not optimized for performance.

    :param log_metadata: The metadata of the log containing the MP4 file.
    :param camera_name: The name of the camera whose MP4 file is to be read.
    :param frame_index: The index of the frame to read from the MP4 file.
    :return: The image frame as a numpy array, or None if the file does not exist.
    """
    image: Optional[npt.NDArray[np.uint8]] = None

    py123d_sensor_root = get_dataset_paths().py123d_sensors_root
    assert py123d_sensor_root is not None, "PY123D_DATA_ROOT must be set for MP4 reading."
    mp4_path = py123d_sensor_root / log_metadata.split / log_metadata.log_name / f"{camera_name}.mp4"
    if mp4_path.exists():
        reader = get_mp4_reader_from_path(str(mp4_path))
        image = reader.get_frame(frame_index)

    return image


def get_camera_timestamp_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    camera_id: Union[PinholeCameraID, FisheyeMEICameraID],
) -> Optional[Timestamp]:
    """Gets the camera timestamp from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the camera timestamp data.
    :param index: The index to extract the camera timestamp from.
    :param camera_id: The type of camera (Pinhole or FisheyeMEI).
    :return: The camera timestamp at the given index, or None if not available.
    """

    assert isinstance(camera_id, (PinholeCameraID, FisheyeMEICameraID)), (
        f"The argument 'camera_id' must be PinholeCameraID or FisheyeMEICameraID, got {type(camera_id)}"
    )

    camera_timestamp: Optional[Timestamp] = None
    is_pinhole = isinstance(camera_id, PinholeCameraID)
    modality_prefix = "pinhole_camera" if is_pinhole else "fisheye_mei_camera"
    instance = camera_id.serialize()
    camera_timestamp_column = f"{modality_prefix}.{instance}.timestamp_us"

    if camera_timestamp_column in arrow_table.schema.names:
        timestamp_data = arrow_table[camera_timestamp_column][index].as_py()
        if timestamp_data is not None:
            camera_timestamp = Timestamp.from_us(timestamp_data)

    return camera_timestamp
