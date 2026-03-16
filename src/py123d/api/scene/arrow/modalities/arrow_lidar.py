from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.io.lidar.draco_lidar_io import (
    encode_point_cloud_as_draco_binary,
    is_draco_binary,
    load_point_cloud_from_draco_binary,
)
from py123d.common.io.lidar.ipc_lidar_io import (
    encode_point_cloud_as_ipc_binary,
    is_ipc_binary,
    load_point_cloud_from_ipc_binary,
)
from py123d.common.io.lidar.laz_lidar_io import (
    encode_point_cloud_as_laz_binary,
    is_laz_binary,
    load_point_cloud_from_laz_binary,
)
from py123d.common.io.lidar.path_lidar_io import load_point_cloud_data_from_path
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
        lidar_codec: Optional[Literal["laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"]],
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, (LidarMetadata, LidarMergedMetadata)), (
            f"Expected LidarMetadata or LidarMergedMetadata, got {type(metadata)}"
        )
        assert lidar_store_option in {"path", "binary"}, f"Unsupported lidar store option: {lidar_store_option}"

        self._modality_metadata = metadata
        self._modality_key = metadata.modality_key
        self._log_metadata = log_metadata

        self._lidar_store_option = lidar_store_option
        self._lidar_codec = lidar_codec

        file_path = log_dir / f"{metadata.modality_key}.arrow"

        schema_list = [
            (f"{metadata.modality_key}.timestamp_us", pa.int64()),
            (f"{metadata.modality_key}.end_timestamp_us", pa.int64()),
        ]
        if lidar_store_option == "binary":
            schema_list.append((f"{metadata.modality_key}.data", pa.binary()))
        elif lidar_store_option == "path":
            schema_list.append((f"{metadata.modality_key}.data", pa.string()))
        else:
            raise ValueError(f"Unsupported lidar store option: {lidar_store_option}")

        schema = add_metadata_to_arrow_schema(pa.schema(schema_list), metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

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

        elif self._lidar_store_option == "binary":
            data_binary = self._prepare_lidar_data(modality)
            batch[f"{self._modality_key}.data"] = [data_binary]

        self.write_batch(batch)

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
                dict(self._modality_metadata) if isinstance(self._modality_metadata, LidarMergedMetadata) else None
            )
            point_cloud_3d, point_cloud_features = load_point_cloud_data_from_path(
                modality._relative_path,
                self._log_metadata.dataset,
                modality._iteration,
                modality._dataset_root,
                lidar_metadatas=lidar_metadatas,
            )
        else:
            raise ValueError(f"Unsupported lidar modality type: {type(modality)}")

        # 2. Encode xyz + features together with the target codec.
        if point_cloud_3d is None:
            return None

        codec = self._lidar_codec
        if codec == "draco":
            return encode_point_cloud_as_draco_binary(point_cloud_3d, point_cloud_features)
        elif codec == "laz":
            return encode_point_cloud_as_laz_binary(point_cloud_3d, point_cloud_features)
        elif codec == "ipc":
            return encode_point_cloud_as_ipc_binary(point_cloud_3d, point_cloud_features, codec=None)
        elif codec == "ipc_zstd":
            return encode_point_cloud_as_ipc_binary(point_cloud_3d, point_cloud_features, codec="zstd")
        elif codec == "ipc_lz4":
            return encode_point_cloud_as_ipc_binary(point_cloud_3d, point_cloud_features, codec="lz4")
        else:
            raise NotImplementedError(f"Unsupported lidar codec: {codec}")


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
                column_at_iteration = _deserialize_pcs(column_at_iteration, dataset, lidar_metadatas)  # type: ignore
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

    if data_col in arrow_table.schema.names:
        lidar_data = arrow_table[data_col][index].as_py()
        if lidar_data is not None:
            point_cloud_3d, point_cloud_feature = _deserialize_pcs(lidar_data, dataset, lidar_metadatas)

    if point_cloud_3d is None:
        return None

    if lidar_id != LidarID.LIDAR_MERGED:
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
            )
        return None

    return Lidar(
        timestamp=timestamp,
        timestamp_end=timestamp_end,
        metadata=LidarMergedMetadata(lidar_metadata_dict=lidar_metadatas),
        point_cloud_3d=point_cloud_3d,
        point_cloud_features=point_cloud_feature,
    )


def _deserialize_pcs(
    lidar_data: Union[bytes, str],
    dataset: str,
    lidar_metadatas: Optional[Dict[LidarID, LidarMetadata]],
) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    if isinstance(lidar_data, str):
        point_cloud_3d, point_cloud_feature = load_point_cloud_data_from_path(
            relative_path=lidar_data,
            dataset=dataset,
            index=None,
            lidar_metadatas=lidar_metadatas,
        )
    elif isinstance(lidar_data, bytes):
        point_cloud_3d, point_cloud_feature = _decode_lidar_binary(lidar_data)
    else:
        raise ValueError(f"Unsupported lidar data type: {type(lidar_data)}")

    return point_cloud_3d, point_cloud_feature


def _decode_lidar_binary(blob: bytes) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    """Auto-detect codec and decode a lidar binary blob into xyz + features."""
    if is_draco_binary(blob):
        return load_point_cloud_from_draco_binary(blob)
    elif is_laz_binary(blob):
        return load_point_cloud_from_laz_binary(blob)
    elif is_ipc_binary(blob):
        return load_point_cloud_from_ipc_binary(blob)
    else:
        raise ValueError("Unknown lidar binary format (not Draco, LAZ, or IPC).")
