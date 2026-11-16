import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig
from py123d.common.io.radar.path_radar_io import load_radar_point_cloud_data_from_path
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.sensors.radar import Radar, RadarFeature, RadarID, RadarMergedMetadata, RadarMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.parser.base_dataset_parser import ParsedRadar


class ArrowRadarWriter(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: Union[RadarMetadata, RadarMergedMetadata],
        log_metadata: LogMetadata,
        radar_store_option: Literal["path", "binary"],
        radar_codec: Optional[Literal["laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"]],
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
        radar_codec_config: Optional[PointCloudCodecConfig] = None,
    ) -> None:
        assert isinstance(metadata, (RadarMetadata, RadarMergedMetadata)), (
            f"Expected RadarMetadata or RadarMergedMetadata, got {type(metadata)}"
        )
        assert radar_store_option in {"path", "binary"}, f"Unsupported radar store option: {radar_store_option}"

        self._modality_metadata = metadata
        self._modality_key = metadata.modality_key
        self._log_metadata = log_metadata

        self._radar_store_option = radar_store_option
        self._radar_codec = radar_codec
        self._radar_codec_config = radar_codec_config if radar_codec_config is not None else PointCloudCodecConfig()

        file_path = log_dir / f"{metadata.modality_key}.arrow"

        schema_list = [
            (f"{metadata.modality_key}.timestamp_us", pa.int64()),
        ]
        if radar_store_option == "binary":
            schema_list.append((f"{metadata.modality_key}.data", pa.binary()))
        elif radar_store_option == "path":
            schema_list.append((f"{metadata.modality_key}.data", pa.string()))
            # Generic JSON extras forwarded to the point-cloud loader at API read time (mirrors lidar). Nullable.
            schema_list.append((f"{metadata.modality_key}.kwargs", pa.string()))
        else:
            raise ValueError(f"Unsupported radar store option: {radar_store_option}")

        if metadata.has_arrival_time:
            schema_list.append(get_arrival_time_field(metadata.modality_key))
        schema = add_metadata_to_arrow_schema(pa.schema(schema_list), metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, (ParsedRadar, Radar)), f"Expected ParsedRadar or Radar, got {type(modality)}"

        timestamp_us = modality.timestamp.time_us

        batch: Dict[str, Union[List[int], List[Optional[str]], List[Optional[bytes]]]] = {
            f"{self._modality_key}.timestamp_us": [timestamp_us],
        }

        if self._radar_store_option == "path":
            assert isinstance(modality, ParsedRadar), "Path store option requires ParsedRadar with file path."
            data_path: Optional[str] = str(modality._relative_path) if modality._relative_path is not None else None
            batch[f"{self._modality_key}.data"] = [data_path]
            load_kwargs = modality._load_kwargs
            batch[f"{self._modality_key}.kwargs"] = [json.dumps(load_kwargs) if load_kwargs else None]

        elif self._radar_store_option == "binary":
            data_binary = self._prepare_radar_data(modality)
            batch[f"{self._modality_key}.data"] = [data_binary]

        add_arrival_time_to_row(batch, self._modality_metadata, modality)
        self.write_batch(batch)

    def _prepare_radar_data(self, modality: Union[ParsedRadar, Radar]) -> Optional[bytes]:
        """Load and encode the radar data (xyz + features) into a single binary blob.

        :param modality: The radar modality data (ParsedRadar or Radar).
        :return: Encoded binary blob, or None if no point cloud data.
        """
        # 1. Load point cloud and features.
        point_cloud_3d: Optional[npt.NDArray] = None
        point_cloud_features: Optional[Dict[str, npt.NDArray]] = None
        if isinstance(modality, Radar):
            point_cloud_3d = modality.point_cloud_3d
            point_cloud_features = modality.point_cloud_features
        elif isinstance(modality, ParsedRadar):
            assert modality._dataset_root is not None and modality._relative_path is not None, (
                "ParsedRadar must have dataset_root and relative_path for binary codec."
            )
            radar_metadatas = (
                dict(self._modality_metadata) if isinstance(self._modality_metadata, RadarMergedMetadata) else None
            )
            point_cloud_3d, point_cloud_features = load_radar_point_cloud_data_from_path(
                modality._relative_path,
                self._log_metadata.dataset,
                modality._iteration,
                modality._dataset_root,
                radar_metadatas=radar_metadatas,
                load_kwargs=modality._load_kwargs,
            )
        else:
            raise ValueError(f"Unsupported radar modality type: {type(modality)}")

        # 2. Encode xyz + features together with the target codec.
        if point_cloud_3d is None:
            return None

        codec = self._radar_codec
        config = self._radar_codec_config
        if codec == "draco":
            return encode_point_cloud_as_draco_binary(point_cloud_3d, point_cloud_features, config=config)
        elif codec == "laz":
            return encode_point_cloud_as_laz_binary(point_cloud_3d, point_cloud_features, config=config)
        elif codec == "ipc":
            return encode_point_cloud_as_ipc_binary(point_cloud_3d, point_cloud_features, codec=None)
        elif codec == "ipc_zstd":
            return encode_point_cloud_as_ipc_binary(point_cloud_3d, point_cloud_features, codec="zstd")
        elif codec == "ipc_lz4":
            return encode_point_cloud_as_ipc_binary(point_cloud_3d, point_cloud_features, codec="lz4")
        else:
            raise NotImplementedError(f"Unsupported radar codec: {codec}")


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowRadarReader(ArrowBaseModalityReader):
    """Stateless reader for radar data from Arrow tables.

    Always reads from the merged radar table. An optional ``radar_id`` kwarg controls
    whether to return the full merged cloud or filter to an individual sensor.
    """

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[Radar]:
        assert isinstance(metadata, (RadarMetadata, RadarMergedMetadata))
        modality_key = metadata.modality_key
        radar_metadatas = dict(metadata) if isinstance(metadata, RadarMergedMetadata) else {metadata.radar_id: metadata}
        radar_id = kwargs.get(
            "radar_id", RadarID.RADAR_MERGED if isinstance(metadata, RadarMergedMetadata) else metadata.radar_id
        )
        return _deserialize_radar(table, index, radar_id, modality_key, radar_metadatas, dataset)

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
        """For radar reader, we only support reading the full point cloud data column as binary or path."""
        if column == ARRIVAL_TIME_FIELD:
            return read_arrival_time_column(table, index, metadata.modality_key, deserialize)
        assert isinstance(metadata, (RadarMetadata, RadarMergedMetadata))
        full_column_name = f"{metadata.modality_key}.{column}"
        column_at_iteration: Optional[Any] = None
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()
        if deserialize and column_at_iteration is not None:
            if column == "timestamp_us":
                column_at_iteration = Timestamp.from_us(column_at_iteration)  # type: ignore
            elif column == "data":
                radar_metadatas = (
                    dict(metadata) if isinstance(metadata, RadarMergedMetadata) else {metadata.radar_id: metadata}
                )
                load_kwargs = _read_load_kwargs(table, index, metadata.modality_key)
                column_at_iteration = _deserialize_pcs(column_at_iteration, dataset, radar_metadatas, load_kwargs)  # type: ignore
        return column_at_iteration


# ------------------------------------------------------------------------------------------------------------------
# Reader Internals
# ------------------------------------------------------------------------------------------------------------------


def _deserialize_radar(
    arrow_table: pa.Table,
    index: int,
    radar_id: RadarID,
    modality_key: str,
    radar_metadatas: Dict[RadarID, RadarMetadata],
    dataset: str,
) -> Optional[Radar]:
    """Deserialize a radar observation from Arrow table columns at the given row index."""
    point_cloud_3d: Optional[np.ndarray] = None
    point_cloud_feature: Optional[Dict[str, np.ndarray]] = None

    ts_col = f"{modality_key}.timestamp_us"
    data_col = f"{modality_key}.data"

    # Read the (single, snapshot) timestamp.
    timestamp_us = arrow_table[ts_col][index].as_py() if ts_col in arrow_table.schema.names else None
    if timestamp_us is None:
        return None
    timestamp = Timestamp.from_us(timestamp_us)
    arrival_timestamp = read_arrival_timestamp(arrow_table, index, modality_key)

    if data_col in arrow_table.schema.names:
        radar_data = arrow_table[data_col][index].as_py()
        if radar_data is not None:
            load_kwargs = _read_load_kwargs(arrow_table, index, modality_key)
            point_cloud_3d, point_cloud_feature = _deserialize_pcs(radar_data, dataset, radar_metadatas, load_kwargs)

    if point_cloud_3d is None:
        return None

    if radar_id != RadarID.RADAR_MERGED:
        # A per-sensor radar file holds exactly one sensor's points; no per-point IDS
        # filtering is needed. Only a merged file requires the IDS split.
        if set(radar_metadatas) == {radar_id}:
            return Radar(
                timestamp=timestamp,
                metadata=radar_metadatas[radar_id],
                point_cloud_3d=point_cloud_3d,
                point_cloud_features=point_cloud_feature,
                arrival_timestamp=arrival_timestamp,
            )
        if point_cloud_feature is not None and RadarFeature.IDS.serialize() in point_cloud_feature:
            mask = point_cloud_feature[RadarFeature.IDS.serialize()] == int(radar_id.value)
            point_cloud_feature = {key: value[mask] for key, value in point_cloud_feature.items()}
            point_cloud_3d = point_cloud_3d[mask]
            return Radar(
                timestamp=timestamp,
                metadata=radar_metadatas[radar_id],
                point_cloud_3d=point_cloud_3d,
                point_cloud_features=point_cloud_feature,
                arrival_timestamp=arrival_timestamp,
            )
        return None

    return Radar(
        timestamp=timestamp,
        metadata=RadarMergedMetadata(radar_metadata_dict=radar_metadatas),
        point_cloud_3d=point_cloud_3d,
        point_cloud_features=point_cloud_feature,
        arrival_timestamp=arrival_timestamp,
    )


def _read_load_kwargs(table: pa.Table, index: int, modality_key: str) -> Optional[Dict[str, Any]]:
    """Reads the optional generic loader-kwargs JSON for a "path"-stored radar row, if the column exists."""
    load_kwargs: Optional[Dict[str, Any]] = None
    kwargs_col = f"{modality_key}.kwargs"
    if kwargs_col in table.schema.names:
        raw_kwargs = table[kwargs_col][index].as_py()
        if raw_kwargs is not None:
            load_kwargs = json.loads(raw_kwargs)
    return load_kwargs


def _deserialize_pcs(
    radar_data: Union[bytes, str],
    dataset: str,
    radar_metadatas: Optional[Dict[RadarID, RadarMetadata]],
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    if isinstance(radar_data, str):
        point_cloud_3d, point_cloud_feature = load_radar_point_cloud_data_from_path(
            relative_path=radar_data,
            dataset=dataset,
            index=None,
            radar_metadatas=radar_metadatas,
            load_kwargs=load_kwargs,
        )
    elif isinstance(radar_data, bytes):
        point_cloud_3d, point_cloud_feature = _decode_radar_binary(radar_data)
    else:
        raise ValueError(f"Unsupported radar data type: {type(radar_data)}")

    return point_cloud_3d, point_cloud_feature


def _decode_radar_binary(blob: bytes) -> Tuple[npt.NDArray[np.float32], Optional[Dict[str, npt.NDArray]]]:
    """Auto-detect codec and decode a radar binary blob into xyz + features."""
    if is_draco_binary(blob):
        return load_point_cloud_from_draco_binary(blob)
    elif is_laz_binary(blob):
        return load_point_cloud_from_laz_binary(blob)
    elif is_ipc_binary(blob):
        return load_point_cloud_from_ipc_binary(blob)
    else:
        raise ValueError("Unknown radar binary format (not Draco, LAZ, or IPC).")
