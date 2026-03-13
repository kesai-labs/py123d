from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.sync_utils import (
    get_all_modality_timestamps,
    get_first_sync_index,
    get_modality_table,
)
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema
from py123d.api.scene.scene_metadata import SceneMetadata
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
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICamera, FisheyeMEICameraID, FisheyeMEICameraMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCamera, PinholeCameraID, PinholeCameraMetadata
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
}

CAMERA_CODEC_MAX_BATCH_SIZES = {
    "path": 1000,
    "jpeg_binary": 10,
    "png_binary": 10,
}


class ArrowCameraWriter(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        camera_codec: Literal["path", "jpeg_binary", "png_binary"] = "path",
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, (PinholeCameraMetadata, FisheyeMEICameraMetadata)), (
            f"Expected PinholeCameraMetadata or FisheyeMEICameraMetadata, got {type(metadata)}"
        )
        assert camera_codec in {"path", "jpeg_binary", "png_binary"}, f"Unsupported camera codec: {camera_codec}"

        self._metadata = metadata
        self._camera_codec = camera_codec

        data_type = CAMERA_CODEC_PA_DTYPES[camera_codec]
        max_batch_size = CAMERA_CODEC_MAX_BATCH_SIZES[camera_codec]

        file_path = log_dir / f"{metadata.modality_key}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_key}.timestamp_us", pa.int64()),
                (f"{metadata.modality_key}.data", data_type),
                (f"{metadata.modality_key}.pose_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
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
        assert isinstance(modality, (ParsedCamera, PinholeCamera, FisheyeMEICamera)), (
            f"Expected ParsedCamera, PinholeCamera, or FisheyeMEICamera, got {type(modality)}"
        )
        if self._camera_codec == "jpeg_binary":
            data = _get_jpeg_binary_from_camera_modality(modality)
        elif self._camera_codec == "png_binary":
            data = _get_png_binary_from_camera_modality(modality)
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
                f"{self._metadata.modality_key}.pose_se3": [modality.extrinsic],
            }
        )


# ------------------------------------------------------------------------------------------------------------------
# Writer Helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_jpeg_binary_from_camera_modality(camera_data: Union[ParsedCamera, PinholeCamera, FisheyeMEICamera]) -> bytes:
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
    elif isinstance(camera_data, (PinholeCamera, FisheyeMEICamera)):
        return encode_image_as_jpeg_binary(camera_data.image)
    else:
        raise NotImplementedError(f"Unsupported camera type for jpeg_binary codec: {type(camera_data)}")


def _get_png_binary_from_camera_modality(camera_data: Union[ParsedCamera, PinholeCamera, FisheyeMEICamera]) -> bytes:
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
    elif isinstance(camera_data, (PinholeCamera, FisheyeMEICamera)):
        return encode_image_as_png_binary(camera_data.image)
    else:
        raise NotImplementedError(f"Unsupported camera type for png_binary codec: {type(camera_data)}")


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowCameraReader:
    """Stateless reader for pinhole and fisheye MEI camera data from Arrow tables."""

    @staticmethod
    def read_pinhole_at_iteration(
        log_dir: Path,
        sync_table: pa.Table,
        table_index: int,
        camera_id: PinholeCameraID,
        camera_metadata: Optional[PinholeCameraMetadata],
        log_metadata: LogMetadata,
    ) -> Optional[PinholeCamera]:
        """Read a pinhole camera observation at a specific sync table index.

        :param log_dir: Path to the log directory.
        :param sync_table: The sync Arrow table.
        :param table_index: The resolved sync table index.
        :param camera_id: The pinhole camera ID.
        :param camera_metadata: The camera intrinsics / metadata.
        :param log_metadata: Log metadata (for dataset path resolution).
        :return: The pinhole camera, or None if unavailable.
        """
        return _read_camera_at_iteration(log_dir, sync_table, table_index, camera_id, camera_metadata, log_metadata)  # type: ignore[return-value]

    @staticmethod
    def read_fisheye_mei_at_iteration(
        log_dir: Path,
        sync_table: pa.Table,
        table_index: int,
        camera_id: FisheyeMEICameraID,
        camera_metadata: Optional[FisheyeMEICameraMetadata],
        log_metadata: LogMetadata,
    ) -> Optional[FisheyeMEICamera]:
        """Read a fisheye MEI camera observation at a specific sync table index.

        :param log_dir: Path to the log directory.
        :param sync_table: The sync Arrow table.
        :param table_index: The resolved sync table index.
        :param camera_id: The fisheye MEI camera ID.
        :param camera_metadata: The camera intrinsics / metadata.
        :param log_metadata: Log metadata (for dataset path resolution).
        :return: The fisheye MEI camera, or None if unavailable.
        """
        return _read_camera_at_iteration(log_dir, sync_table, table_index, camera_id, camera_metadata, log_metadata)  # type: ignore[return-value]

    @staticmethod
    def read_all_pinhole_timestamps(
        log_dir: Path,
        sync_table: pa.Table,
        scene_metadata: SceneMetadata,
        camera_id: PinholeCameraID,
    ) -> List[Timestamp]:
        """Read all timestamps for a pinhole camera within the scene range."""
        instance = camera_id.serialize()
        modality_key = f"pinhole_camera.{instance}"
        return get_all_modality_timestamps(
            log_dir, sync_table, scene_metadata, modality_key, f"{modality_key}.timestamp_us"
        )

    @staticmethod
    def read_all_fisheye_mei_timestamps(
        log_dir: Path,
        sync_table: pa.Table,
        scene_metadata: SceneMetadata,
        camera_id: FisheyeMEICameraID,
    ) -> List[Timestamp]:
        """Read all timestamps for a fisheye MEI camera within the scene range."""
        instance = camera_id.serialize()
        modality_key = f"fisheye_mei_camera.{instance}"
        return get_all_modality_timestamps(
            log_dir, sync_table, scene_metadata, modality_key, f"{modality_key}.timestamp_us"
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader Internals
# ------------------------------------------------------------------------------------------------------------------


def _read_camera_at_iteration(
    log_dir: Path,
    sync_table: pa.Table,
    table_index: int,
    camera_id: Union[PinholeCameraID, FisheyeMEICameraID],
    camera_metadata: Union[PinholeCameraMetadata, FisheyeMEICameraMetadata, None],
    log_metadata: LogMetadata,
) -> Optional[Union[PinholeCamera, FisheyeMEICamera]]:
    """Shared implementation for reading any camera type at a given iteration."""
    if camera_metadata is None:
        return None

    camera_instance = camera_id.serialize()
    is_pinhole = isinstance(camera_id, PinholeCameraID)
    modality_prefix = "pinhole_camera" if is_pinhole else "fisheye_mei_camera"
    modality_key = f"{modality_prefix}.{camera_instance}"

    cam_table = get_modality_table(log_dir, modality_key)
    if cam_table is None:
        return None

    row_idx = get_first_sync_index(sync_table, modality_key, table_index)
    if row_idx is None:
        return None

    return _deserialize_camera(cam_table, row_idx, camera_id, camera_metadata, log_metadata)


def _deserialize_camera(
    arrow_table: pa.Table,
    index: int,
    camera_id: Union[PinholeCameraID, FisheyeMEICameraID],
    camera_metadata: Union[PinholeCameraMetadata, FisheyeMEICameraMetadata],
    log_metadata: LogMetadata,
) -> Optional[Union[PinholeCamera, FisheyeMEICamera]]:
    """Deserialize a camera observation from Arrow table columns at the given row index."""
    is_pinhole = isinstance(camera_id, PinholeCameraID)
    modality_prefix = "pinhole_camera" if is_pinhole else "fisheye_mei_camera"
    instance = camera_id.serialize()

    camera_data_column = f"{modality_prefix}.{instance}.data"
    camera_extrinsic_column = f"{modality_prefix}.{instance}.pose_se3"
    camera_timestamp_column = f"{modality_prefix}.{instance}.timestamp_us"

    if not all_columns_in_schema(arrow_table, [camera_data_column, camera_extrinsic_column, camera_timestamp_column]):
        return None

    table_data = arrow_table[camera_data_column][index].as_py()
    extrinsic_data = arrow_table[camera_extrinsic_column][index].as_py()
    timestamp_data = arrow_table[camera_timestamp_column][index].as_py()

    if table_data is None or extrinsic_data is None:
        return None

    extrinsic = PoseSE3.from_list(extrinsic_data)
    image: Optional[npt.NDArray[np.uint8]] = None

    if isinstance(table_data, str):
        sensor_root = get_dataset_paths().get_sensor_root(log_metadata.dataset)
        assert sensor_root is not None, f"Dataset path for sensor loading not found for dataset: {log_metadata.dataset}"
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
            f"Only string file paths, bytes, or int frame indices are supported for camera data, got {type(table_data)}"
        )

    assert image is not None, "Failed to load camera image from Arrow table data."
    if is_pinhole:
        return PinholeCamera(
            metadata=camera_metadata,  # type: ignore[arg-type]
            image=image,
            extrinsic=extrinsic,
            timestamp=Timestamp.from_us(timestamp_data),
        )
    else:
        return FisheyeMEICamera(
            metadata=camera_metadata,  # type: ignore[arg-type]
            image=image,
            extrinsic=extrinsic,
            timestamp=Timestamp.from_us(timestamp_data),
        )


def _unoptimized_demo_mp4_read(log_metadata: LogMetadata, camera_name: str, frame_index: int) -> Optional[np.ndarray]:
    """Reads a frame from an MP4 file for demonstration purposes. This feature is not optimized for performance."""
    image: Optional[npt.NDArray[np.uint8]] = None
    py123d_sensor_root = get_dataset_paths().py123d_sensors_root
    assert py123d_sensor_root is not None, "PY123D_DATA_ROOT must be set for MP4 reading."
    mp4_path = py123d_sensor_root / log_metadata.split / log_metadata.log_name / f"{camera_name}.mp4"
    if mp4_path.exists():
        reader = get_mp4_reader_from_path(str(mp4_path))
        image = reader.get_frame(frame_index)
    return image
