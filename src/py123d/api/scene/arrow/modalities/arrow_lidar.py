from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.io.lidar.draco_lidar_io import (
    encode_point_cloud_3d_as_draco_binary,
    is_draco_binary,
    load_point_cloud_3d_from_draco_binary,
)
from py123d.common.io.lidar.ipc_lidar_io import (
    encode_point_cloud_3d_as_ipc_binary,
    encode_point_cloud_features_as_ipc_binary,
    is_ipc_binary,
    load_point_cloud_3d_from_ipc_binary,
    load_point_cloud_features_from_ipc_binary,
)
from py123d.common.io.lidar.laz_lidar_io import (
    encode_point_cloud_3d_as_laz_binary,
    is_laz_binary,
    load_point_cloud_3d_from_laz_binary,
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
        lidar_point_cloud_codec: Optional[Literal["laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"]],
        lidar_point_feature_codec: Optional[Literal["ipc_zstd", "ipc_lz4", "ipc"]],  # None drops features.
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
        self._lidar_point_cloud_codec = lidar_point_cloud_codec
        self._lidar_point_feature_codec = lidar_point_feature_codec

        file_path = log_dir / f"{metadata.modality_key}.arrow"

        schema_list = [
            (f"{metadata.modality_key}.start_timestamp_us", pa.int64()),
            (f"{metadata.modality_key}.end_timestamp_us", pa.int64()),
        ]
        if lidar_store_option == "binary":
            schema_list.append((f"{metadata.modality_key}.point_cloud_3d", pa.binary()))
            if lidar_point_feature_codec:
                schema_list.append((f"{metadata.modality_key}.point_cloud_features", pa.binary()))
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
            start_timestamp_us = modality.timestamp.time_us
            end_timestamp_us = modality.end_timestamp.time_us
        else:
            start_timestamp_us = modality.timestamp.time_us
            end_timestamp_us = modality.timestamp_end.time_us

        batch: Dict[str, Union[List[int], List[Optional[str]], List[Optional[bytes]]]] = {
            f"{self._modality_key}.start_timestamp_us": [start_timestamp_us],
            f"{self._modality_key}.end_timestamp_us": [end_timestamp_us],
        }

        if self._lidar_store_option == "path":
            assert isinstance(modality, ParsedLidar), "Path store option requires ParsedLidar with file path."
            data_path: Optional[str] = str(modality._relative_path) if modality._relative_path is not None else None
            batch[f"{self._modality_key}.data"] = [data_path]

        elif self._lidar_store_option == "binary":
            point_cloud_binary, features_binary = self._prepare_lidar_data(modality)
            batch[f"{self._modality_key}.point_cloud_3d"] = [point_cloud_binary]
            if self._lidar_point_feature_codec:
                batch[f"{self._modality_key}.point_cloud_features"] = [features_binary]

        self.write_batch(batch)

    def _prepare_lidar_data(self, modality: Union[ParsedLidar, Lidar]) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Load and/or encode the lidar data in binary for point cloud and features.

        :param modality: The lidar modality data (ParsedLidar or Lidar).
        :return: Tuple of (point_cloud_binary, point_cloud_features_binary)
        """
        # 1. Load point cloud and point features
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

        # 2. Compress point clouds with target codec
        point_cloud_3d_output: Optional[bytes] = None
        if point_cloud_3d is not None:
            codec = self._lidar_point_cloud_codec
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
        if self._lidar_point_feature_codec is not None and point_cloud_features is not None:
            if self._lidar_point_feature_codec == "ipc":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(point_cloud_features, codec=None)
            elif self._lidar_point_feature_codec == "ipc_zstd":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(
                    point_cloud_features, codec="zstd"
                )
            elif self._lidar_point_feature_codec == "ipc_lz4":
                point_cloud_feature_output = encode_point_cloud_features_as_ipc_binary(
                    point_cloud_features, codec="lz4"
                )
            else:
                raise NotImplementedError(f"Unsupported lidar point feature codec: {self._lidar_point_feature_codec}")

        return point_cloud_3d_output, point_cloud_feature_output


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowLidarReader(ArrowBaseModalityReader):
    """Stateless reader for lidar data from Arrow tables.

    When called via the common interface (``read_at_iteration``), reads directly from the
    Arrow table identified by ``metadata.modality_key``. No cross-table branching.

    For higher-level logic (merged vs individual, on-the-fly merging), see
    :meth:`ArrowSceneAPI.get_lidar_at_iteration`.
    """

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
    ) -> Optional[Lidar]:
        assert isinstance(metadata, (LidarMetadata, LidarMergedMetadata))
        modality_key = metadata.modality_key
        lidar_metadatas = dict(metadata) if isinstance(metadata, LidarMergedMetadata) else {metadata.lidar_id: metadata}
        lidar_id = LidarID.LIDAR_MERGED if isinstance(metadata, LidarMergedMetadata) else metadata.lidar_id
        return _deserialize_lidar(table, index, lidar_id, modality_key, lidar_metadatas, dataset)


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

    start_ts_col = f"{modality_key}.start_timestamp_us"
    end_ts_col = f"{modality_key}.end_timestamp_us"
    data_col = f"{modality_key}.data"
    pc3d_col = f"{modality_key}.point_cloud_3d"
    pcf_col = f"{modality_key}.point_cloud_features"

    # Read timestamps
    start_timestamp_us = arrow_table[start_ts_col][index].as_py() if start_ts_col in arrow_table.schema.names else None
    end_timestamp_us = arrow_table[end_ts_col][index].as_py() if end_ts_col in arrow_table.schema.names else None
    if start_timestamp_us is None or end_timestamp_us is None:
        return None
    timestamp = Timestamp.from_us(start_timestamp_us)
    timestamp_end = Timestamp.from_us(end_timestamp_us)

    if data_col in arrow_table.schema.names:
        # 1. Load lidar sweep from origin dataset using a relative file path.
        lidar_data = arrow_table[data_col][index].as_py()
        if lidar_data is not None:
            assert isinstance(lidar_data, str), f"Lidar path data must be a string file path, got {type(lidar_data)}"
            point_cloud_3d, point_cloud_feature = load_point_cloud_data_from_path(
                relative_path=lidar_data,
                dataset=dataset,
                index=index,
                lidar_metadatas=lidar_metadatas,
            )

    elif pc3d_col in arrow_table.schema.names:
        # 2.1 Loading the lidar xyz point cloud from blob in the Arrow table.
        lidar_data = arrow_table[pc3d_col][index].as_py()
        if lidar_data is not None:
            if is_draco_binary(lidar_data):
                point_cloud_3d = load_point_cloud_3d_from_draco_binary(lidar_data)
            elif is_laz_binary(lidar_data):
                point_cloud_3d = load_point_cloud_3d_from_laz_binary(lidar_data)
            elif is_ipc_binary(lidar_data):
                point_cloud_3d = load_point_cloud_3d_from_ipc_binary(lidar_data)

        # 2.2 Load lidar features from blob in the Arrow table, if available.
        if pcf_col in arrow_table.schema.names:
            lidar_point_cloud_feature_data = arrow_table[pcf_col][index].as_py()
            if lidar_point_cloud_feature_data is not None:
                if is_ipc_binary(lidar_point_cloud_feature_data):
                    point_cloud_feature = load_point_cloud_features_from_ipc_binary(lidar_point_cloud_feature_data)

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
