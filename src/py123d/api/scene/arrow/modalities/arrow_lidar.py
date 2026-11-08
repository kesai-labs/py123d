import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import (
    ARRIVAL_TIME_FIELD,
    add_arrival_time_to_row,
    get_arrival_time_field,
    read_arrival_time_column,
    read_arrival_timestamp,
)
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.io.lidar.path_lidar_io import load_point_cloud_data_from_path
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig
from py123d.common.io.lidar.point_cloud_codec_registry import (
    CODEC_METADATA_KEY,
    decode_point_cloud,
    encode_codec_metadata,
    encode_point_cloud,
    point_cloud_codec_metadata,
    point_cloud_codec_names,
)
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarID, LidarMergedMetadata, LidarMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.parser.base_dataset_parser import ParsedLidar


class ArrowLidarWriter(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: Union[LidarMetadata, LidarMergedMetadata],
        log_metadata: LogMetadata,
        lidar_store_option: Literal["path", "binary"],
        lidar_codec: Optional[str],
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
        lidar_codec_config: Optional[PointCloudCodecConfig] = None,
        encode_threads: int = 1,
    ) -> None:
        """Initializes the lidar writer.

        :param encode_threads: Number of threads that encode binary blobs in parallel; the rows are written in
            their original order. 1 encodes in the calling thread.
        """
        assert isinstance(metadata, (LidarMetadata, LidarMergedMetadata)), (
            f"Expected LidarMetadata or LidarMergedMetadata, got {type(metadata)}"
        )
        assert lidar_store_option in {"path", "binary"}, f"Unsupported lidar store option: {lidar_store_option}"
        if lidar_store_option == "binary":
            assert lidar_codec in point_cloud_codec_names("lidar"), f"Unsupported lidar codec: {lidar_codec}"
        assert encode_threads >= 1, f"encode_threads must be >= 1, got {encode_threads}."

        self._modality_metadata = metadata
        self._modality_key = metadata.modality_key
        self._log_metadata = log_metadata

        self._lidar_store_option = lidar_store_option
        self._lidar_codec = lidar_codec
        self._lidar_codec_config = lidar_codec_config if lidar_codec_config is not None else PointCloudCodecConfig()
        lidar_metadatas = dict(metadata) if isinstance(metadata, LidarMergedMetadata) else {metadata.lidar_id: metadata}
        self._sensor_poses = {
            int(lidar_id.value): lidar_metadata.lidar_to_imu_se3.tolist()
            for lidar_id, lidar_metadata in lidar_metadatas.items()
        }
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: Deque[Tuple["Future[Optional[bytes]]", Dict[str, Any]]] = deque()
        self._max_pending = 2 * encode_threads
        if lidar_store_option == "binary" and encode_threads > 1:
            self._executor = ThreadPoolExecutor(max_workers=encode_threads, thread_name_prefix="lidar-encode")

        file_path = log_dir / f"{metadata.modality_key}.arrow"

        schema_list = [
            (f"{metadata.modality_key}.timestamp_us", pa.int64()),
            (f"{metadata.modality_key}.end_timestamp_us", pa.int64()),
        ]
        if lidar_store_option == "binary":
            schema_list.append((f"{metadata.modality_key}.data", pa.binary()))
        elif lidar_store_option == "path":
            schema_list.append((f"{metadata.modality_key}.data", pa.string()))
            # Generic JSON extras forwarded to the point-cloud loader at API read time (e.g. nuScenes
            # carries the keyframe's panoptic-label relative path here). Nullable; unused datasets
            # write None. In "binary" mode the extras are consumed at conversion, so no column is added.
            schema_list.append((f"{metadata.modality_key}.kwargs", pa.string()))
        else:
            raise ValueError(f"Unsupported lidar store option: {lidar_store_option}")

        if metadata.has_arrival_time:
            schema_list.append(get_arrival_time_field(metadata.modality_key))
        schema = add_metadata_to_arrow_schema(pa.schema(schema_list), metadata)
        if lidar_store_option == "binary" and lidar_codec == "kedec":
            description = point_cloud_codec_metadata(lidar_codec, self._lidar_codec_config)
            schema = schema.with_metadata({**schema.metadata, CODEC_METADATA_KEY: encode_codec_metadata(description)})
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    @property
    def row_count(self) -> int:
        """Returns the total number of rows written, including buffered rows and rows still being encoded."""
        return self._row_count + len(self._pending)

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, (ParsedLidar, Lidar)), f"Expected ParsedLidar or Lidar, got {type(modality)}"

        if isinstance(modality, ParsedLidar):
            timestamp_us = modality.timestamp.time_us
            end_timestamp_us = modality.end_timestamp.time_us
        else:
            timestamp_us = modality.timestamp.time_us
            end_timestamp_us = modality.timestamp_end.time_us

        batch: Dict[str, Union[List[int], List[Optional[str]], List[Optional[bytes]]]] = {
            f"{self._modality_key}.timestamp_us": [timestamp_us],
            f"{self._modality_key}.end_timestamp_us": [end_timestamp_us],
        }

        if self._lidar_store_option == "path":
            assert isinstance(modality, ParsedLidar), "Path store option requires ParsedLidar with file path."
            data_path: Optional[str] = str(modality._relative_path) if modality._relative_path is not None else None
            batch[f"{self._modality_key}.data"] = [data_path]
            load_kwargs = modality._load_kwargs
            batch[f"{self._modality_key}.kwargs"] = [json.dumps(load_kwargs) if load_kwargs else None]

        elif self._lidar_store_option == "binary":
            add_arrival_time_to_row(batch, self._modality_metadata, modality)
            if self._executor is not None:
                self._pending.append((self._executor.submit(self._prepare_lidar_data, modality), batch))
                while len(self._pending) >= self._max_pending:
                    self._write_oldest_pending()
                return
            batch[f"{self._modality_key}.data"] = [self._prepare_lidar_data(modality)]
            self.write_batch(batch)
            return

        add_arrival_time_to_row(batch, self._modality_metadata, modality)
        self.write_batch(batch)

    def _write_oldest_pending(self) -> None:
        """Wait for the oldest blob that is being encoded and write its row."""
        future, batch = self._pending[0]
        data_binary = future.result()
        self._pending.popleft()
        batch[f"{self._modality_key}.data"] = [data_binary]
        self.write_batch(batch)

    def close(self) -> None:
        """Write the rows still being encoded, stop the encoding threads and close the file.

        If an encoding failed, the rows before it are written, the file is closed and the error is raised.
        """
        try:
            while self._pending:
                self._write_oldest_pending()
        finally:
            if self._executor is not None:
                for future, _ in self._pending:
                    future.cancel()
                self._executor.shutdown(wait=True)
                self._executor = None
            self._pending.clear()
            super().close()

    def _prepare_lidar_data(self, modality: Union[ParsedLidar, Lidar]) -> Optional[bytes]:
        """Load and encode the lidar data (xyz + features) into a single binary blob.

        :param modality: The lidar modality data (ParsedLidar or Lidar).
        :return: Encoded binary blob, or None if no point cloud data.
        """
        # 1. Load point cloud and features.
        point_cloud_3d: Optional[npt.NDArray] = None
        point_cloud_features: Optional[Dict[str, npt.NDArray]] = None
        if isinstance(modality, Lidar):
            point_cloud_3d = modality.point_cloud_3d
            point_cloud_features = modality.point_cloud_features
        elif isinstance(modality, ParsedLidar):
            assert modality._dataset_root is not None and modality._relative_path is not None, (
                "ParsedLidar must have dataset_root and relative_path for binary codec."
            )
            lidar_metadatas = (
                dict(self._modality_metadata)
                if isinstance(self._modality_metadata, LidarMergedMetadata)
                else {self._modality_metadata.lidar_id: self._modality_metadata}
            )
            point_cloud_3d, point_cloud_features = load_point_cloud_data_from_path(
                modality._relative_path,
                self._log_metadata.dataset,
                modality._iteration,
                modality._dataset_root,
                lidar_metadatas=lidar_metadatas,
                load_kwargs=modality._load_kwargs,
            )
        else:
            raise ValueError(f"Unsupported lidar modality type: {type(modality)}")

        # 2. Encode xyz + features together with the target codec.
        if point_cloud_3d is None:
            return None

        assert self._lidar_codec is not None, "The binary store option requires a lidar codec."
        return encode_point_cloud(
            self._lidar_codec,
            point_cloud_3d,
            point_cloud_features,
            config=self._lidar_codec_config,
            sensor_poses=self._sensor_poses,
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowLidarReader(ArrowBaseModalityReader):
    """Stateless reader for lidar data from Arrow tables.

    Always reads from the merged lidar table. An optional ``lidar_id`` kwarg controls
    whether to return the full merged cloud or filter to an individual sensor.
    """

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[Lidar]:
        assert isinstance(metadata, (LidarMetadata, LidarMergedMetadata))
        modality_key = metadata.modality_key
        lidar_metadatas = dict(metadata) if isinstance(metadata, LidarMergedMetadata) else {metadata.lidar_id: metadata}
        lidar_id = kwargs.get(
            "lidar_id", LidarID.LIDAR_MERGED if isinstance(metadata, LidarMergedMetadata) else metadata.lidar_id
        )
        return _deserialize_lidar(table, index, lidar_id, modality_key, lidar_metadatas, dataset)

    @staticmethod
    def read_column_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        column: str,
        dataset: str,
        deserialize: bool = False,
        **kwargs,
    ) -> Optional[Any]:
        """For lidar reader, we only support reading the full point cloud data column as binary or path."""
        if column == ARRIVAL_TIME_FIELD:
            return read_arrival_time_column(table, index, metadata.modality_key, deserialize)
        assert isinstance(metadata, (LidarMetadata, LidarMergedMetadata))
        full_column_name = f"{metadata.modality_key}.{column}"
        column_at_iteration: Optional[Any] = None
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()
        if deserialize and column_at_iteration is not None:
            if column in {"timestamp_us", "end_timestamp_us"}:
                column_at_iteration = Timestamp.from_us(column_at_iteration)  # type: ignore
            elif column == "data":
                lidar_metadatas = (
                    dict(metadata) if isinstance(metadata, LidarMergedMetadata) else {metadata.lidar_id: metadata}
                )
                load_kwargs = _read_load_kwargs(table, index, metadata.modality_key)
                column_at_iteration = _deserialize_pcs(column_at_iteration, dataset, lidar_metadatas, load_kwargs)  # type: ignore
        return column_at_iteration


# ------------------------------------------------------------------------------------------------------------------
# Reader Internals
# ------------------------------------------------------------------------------------------------------------------


def _deserialize_lidar(
    arrow_table: pa.Table,
    index: int,
    lidar_id: LidarID,
    modality_key: str,
    lidar_metadatas: Dict[LidarID, LidarMetadata],
    dataset: str,
) -> Optional[Lidar]:
    """Deserialize a lidar observation from Arrow table columns at the given row index."""
    point_cloud_3d: Optional[np.ndarray] = None
    point_cloud_feature: Optional[Dict[str, np.ndarray]] = None

    ts_col = f"{modality_key}.timestamp_us"
    end_ts_col = f"{modality_key}.end_timestamp_us"
    data_col = f"{modality_key}.data"

    # Read timestamps
    timestamp_us = arrow_table[ts_col][index].as_py() if ts_col in arrow_table.schema.names else None
    end_timestamp_us = arrow_table[end_ts_col][index].as_py() if end_ts_col in arrow_table.schema.names else None
    if timestamp_us is None or end_timestamp_us is None:
        return None
    timestamp = Timestamp.from_us(timestamp_us)
    timestamp_end = Timestamp.from_us(end_timestamp_us)
    arrival_timestamp = read_arrival_timestamp(arrow_table, index, modality_key)

    if data_col in arrow_table.schema.names:
        lidar_data = arrow_table[data_col][index].as_py()
        if lidar_data is not None:
            load_kwargs = _read_load_kwargs(arrow_table, index, modality_key)
            point_cloud_3d, point_cloud_feature = _deserialize_pcs(lidar_data, dataset, lidar_metadatas, load_kwargs)

    if point_cloud_3d is None:
        return None

    if lidar_id != LidarID.LIDAR_MERGED:
        # A per-sensor lidar file holds exactly one sensor's points; no per-point IDS
        # filtering is needed (or present). Only a merged file requires the IDS split.
        if set(lidar_metadatas) == {lidar_id}:
            return Lidar(
                timestamp=timestamp,
                timestamp_end=timestamp_end,
                metadata=lidar_metadatas[lidar_id],
                point_cloud_3d=point_cloud_3d,
                point_cloud_features=point_cloud_feature,
                arrival_timestamp=arrival_timestamp,
            )
        if point_cloud_feature is not None and LidarFeature.IDS.serialize() in point_cloud_feature:
            mask = point_cloud_feature[LidarFeature.IDS.serialize()] == int(lidar_id.value)
            point_cloud_feature = {key: value[mask] for key, value in point_cloud_feature.items()}
            point_cloud_3d = point_cloud_3d[mask]
            return Lidar(
                timestamp=timestamp,
                timestamp_end=timestamp_end,
                metadata=lidar_metadatas[lidar_id],
                point_cloud_3d=point_cloud_3d,
                point_cloud_features=point_cloud_feature,
                arrival_timestamp=arrival_timestamp,
            )
        return None

    return Lidar(
        timestamp=timestamp,
        timestamp_end=timestamp_end,
        metadata=LidarMergedMetadata(lidar_metadata_dict=lidar_metadatas),
        point_cloud_3d=point_cloud_3d,
        point_cloud_features=point_cloud_feature,
        arrival_timestamp=arrival_timestamp,
    )


def _read_load_kwargs(table: pa.Table, index: int, modality_key: str) -> Optional[Dict[str, Any]]:
    """Reads the optional generic loader-kwargs JSON for a "path"-stored lidar row, if the column exists.

    The ``{modality_key}.kwargs`` column is only present for logs written with the "path" store option;
    older logs predate it, so its absence (or a null cell) is treated as "no extras".
    """
    load_kwargs: Optional[Dict[str, Any]] = None
    kwargs_col = f"{modality_key}.kwargs"
    if kwargs_col in table.schema.names:
        raw_kwargs = table[kwargs_col][index].as_py()
        if raw_kwargs is not None:
            load_kwargs = json.loads(raw_kwargs)
    return load_kwargs


def _deserialize_pcs(
    lidar_data: Union[bytes, str],
    dataset: str,
    lidar_metadatas: Optional[Dict[LidarID, LidarMetadata]],
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    if isinstance(lidar_data, str):
        point_cloud_3d, point_cloud_feature = load_point_cloud_data_from_path(
            relative_path=lidar_data,
            dataset=dataset,
            index=None,
            lidar_metadatas=lidar_metadatas,
            load_kwargs=load_kwargs,
        )
    elif isinstance(lidar_data, bytes):
        point_cloud_3d, point_cloud_feature = _decode_lidar_binary(lidar_data)
    else:
        raise ValueError(f"Unsupported lidar data type: {type(lidar_data)}")

    return point_cloud_3d, point_cloud_feature


def _decode_lidar_binary(blob: bytes) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    """Auto-detect codec and decode a lidar binary blob into xyz + features."""
    return decode_point_cloud(blob)
