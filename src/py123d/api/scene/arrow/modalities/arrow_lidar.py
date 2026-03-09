from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
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
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarID, LidarMergedMetadata, LidarMetadata
from py123d.geometry.pose import PoseSE3
from py123d.parser.abstract_dataset_parser import ParsedLidar


class ArrowLidarWriter(BaseModalityWriter):
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
        self._modality_name = metadata.modality_name
        self._log_metadata = log_metadata

        self._lidar_store_option = lidar_store_option
        self._lidar_point_cloud_codec = lidar_point_cloud_codec
        self._lidar_point_feature_codec = lidar_point_feature_codec

        file_path = log_dir / f"{metadata.modality_name}.arrow"

        schema_list = [
            (f"{metadata.modality_name}.start_timestamp_us", pa.int64()),
            (f"{metadata.modality_name}.end_timestamp_us", pa.int64()),
        ]
        if lidar_store_option == "binary":
            schema_list.append((f"{metadata.modality_name}.point_cloud_3d", pa.binary()))
            if lidar_point_feature_codec:
                schema_list.append((f"{metadata.modality_name}.point_cloud_features", pa.binary()))
        elif lidar_store_option == "path":
            schema_list.append((f"{metadata.modality_name}.data", pa.string()))
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

    def write_modality(self, lidar_data: ParsedLidar) -> None:
        batch: Dict[str, Union[List[int], List[Optional[str]], List[Optional[bytes]]]] = {
            f"{self._modality_name}.start_timestamp_us": [lidar_data.start_timestamp.time_us],
            f"{self._modality_name}.end_timestamp_us": [lidar_data.end_timestamp.time_us],
        }

        if self._lidar_store_option == "path":
            data_path: Optional[str] = str(lidar_data.relative_path) if lidar_data.has_file_path else None
            batch[f"{self._modality_name}.data"] = [data_path]

        elif self._lidar_store_option == "binary":
            point_cloud_binary, features_binary = self._prepare_lidar_data(lidar_data)
            batch[f"{self._modality_name}.point_cloud_3d"] = [point_cloud_binary]
            if self._lidar_point_feature_codec:
                batch[f"{self._modality_name}.point_cloud_features"] = [features_binary]

        self.write_batch(batch)

    def _prepare_lidar_data(self, lidar_data: ParsedLidar) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Load and/or encode the lidar data in binary for point cloud and features.

        :param lidar_data: Helper class referencing the lidar observation.
        :return: Tuple of (point_cloud_binary, point_cloud_features_binary)
        """
        # 1. Load point cloud and point features
        point_cloud_3d: Optional[npt.NDArray] = None
        point_cloud_features: Optional[Dict[str, npt.NDArray]] = None
        if lidar_data.has_point_cloud_3d:
            point_cloud_3d = lidar_data.point_cloud_3d
            point_cloud_features = lidar_data.point_cloud_features
        elif lidar_data.has_file_path:
            lidar_metadatas = (
                dict(self._modality_metadata) if isinstance(self._modality_metadata, LidarMergedMetadata) else None
            )
            point_cloud_3d, point_cloud_features = load_point_cloud_data_from_path(
                lidar_data.relative_path,  # type: ignore
                self._log_metadata,
                lidar_data.iteration,
                lidar_data.dataset_root,
                lidar_metadatas=lidar_metadatas,
            )
        else:
            raise ValueError("Lidar data must provide either point cloud data or a file path.")

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


def get_lidar_from_arrow_table(
    arrow_table: pa.Table,
    index: int,
    lidar_type: LidarID,
    lidar_instance: str,
    lidar_metadatas: Dict[LidarID, LidarMetadata],
    log_metadata: LogMetadata,
) -> Optional[Lidar]:
    """Builds a Lidar object from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the Lidar data.
    :param index: The index to extract the Lidar data from.
    :param lidar_type: The type of Lidar to build.
    :param lidar_instance: The lidar instance name for parametric column lookup.
    :param lidar_metadatas: Per-sensor lidar metadata dict.
    :param log_metadata: Metadata about the log (used for dataset path resolution).
    :raises ValueError: If the Lidar data format is unsupported.
    :raises NotImplementedError: If the Lidar data type is not supported.
    :return: The constructed Lidar object, or None if not available.
    """
    point_cloud_3d: Optional[np.ndarray] = None
    point_cloud_feature: Optional[Dict[str, np.ndarray]] = None

    data_col = f"lidar.{lidar_instance}.data"
    pc3d_col = f"lidar.{lidar_instance}.point_cloud_3d"
    pcf_col = f"lidar.{lidar_instance}.point_cloud_features"

    if data_col in arrow_table.schema.names:
        # 1. Load lidar sweep from origin dataset using a relative file path.
        lidar_data = arrow_table[data_col][index].as_py()
        if lidar_data is not None:
            assert isinstance(lidar_data, str), f"Lidar path data must be a string file path, got {type(lidar_data)}"
            point_cloud_3d, point_cloud_feature = load_point_cloud_data_from_path(
                relative_path=lidar_data,
                log_metadata=log_metadata,
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

    lidar: Optional[Lidar] = None
    if point_cloud_3d is not None:
        if lidar_type != LidarID.LIDAR_MERGED:
            if point_cloud_feature is not None and LidarFeature.IDS.serialize() in point_cloud_feature.keys():
                mask = point_cloud_feature[LidarFeature.IDS.serialize()] == int(lidar_type.value)
                point_cloud_feature = {key: value[mask] for key, value in point_cloud_feature.items()}
                point_cloud_3d = point_cloud_3d[mask]
                lidar = Lidar(
                    metadata=lidar_metadatas[lidar_type],
                    point_cloud_3d=point_cloud_3d,
                    point_cloud_features=point_cloud_feature,
                )
        else:
            lidar = Lidar(
                metadata=LidarMetadata(
                    lidar_name=LidarID.LIDAR_MERGED.serialize(),
                    lidar_id=LidarID.LIDAR_MERGED,
                    lidar_to_imu_se3=PoseSE3.identity(),
                ),
                point_cloud_3d=point_cloud_3d,
                point_cloud_features=point_cloud_feature,
            )

    return lidar
