from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
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
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.sensors.magnetometer import Magnetometer, MagnetometerMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.geometry_index import Vector3DIndex
from py123d.geometry.vector import Vector3D

_COVARIANCE_SIZE: int = 9

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowMagnetometerWriter(ArrowBaseModalityWriter):
    """Writes magnetometer measurements to an Arrow IPC file.

    The magnetic field is always present; the covariance is a nullable column, stored as null
    when a measurement lacks it.
    """

    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, MagnetometerMetadata), f"Expected MagnetometerMetadata, got {type(metadata)}"

        self._metadata = metadata
        self._key = metadata.modality_key

        schema = pa.schema(
            [
                (f"{self._key}.timestamp_us", pa.int64()),
                (f"{self._key}.magnetic_field", pa.list_(pa.float64(), len(Vector3DIndex))),
                (f"{self._key}.magnetic_field_covariance", pa.list_(pa.float64(), _COVARIANCE_SIZE)),
            ]
        )
        if metadata.has_arrival_time:
            schema = schema.append(pa.field(*get_arrival_time_field(self._key)))
        schema = add_metadata_to_arrow_schema(schema, metadata)
        super().__init__(
            file_path=log_dir / f"{self._key}.arrow",
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, Magnetometer), f"Expected Magnetometer, got {type(modality)}"
        covariance = modality.magnetic_field_covariance
        row: dict = {
            f"{self._key}.timestamp_us": [modality.timestamp.time_us],
            f"{self._key}.magnetic_field": [modality.magnetic_field.array],
            f"{self._key}.magnetic_field_covariance": [covariance if covariance is not None else None],
        }
        add_arrival_time_to_row(row, self._metadata, modality)
        self.write_batch(row)


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowMagnetometerReader(ArrowBaseModalityReader):
    """Stateless reader for magnetometer data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[Magnetometer]:
        assert isinstance(metadata, MagnetometerMetadata), f"Expected MagnetometerMetadata, got {type(metadata)}"
        key = metadata.modality_key
        covariance = table[f"{key}.magnetic_field_covariance"][index].as_py()
        return Magnetometer(
            timestamp=Timestamp.from_us(table[f"{key}.timestamp_us"][index].as_py()),
            metadata=metadata,
            magnetic_field=Vector3D(*table[f"{key}.magnetic_field"][index].as_py()),
            magnetic_field_covariance=np.asarray(covariance, dtype=np.float64) if covariance is not None else None,
            arrival_timestamp=read_arrival_timestamp(table, index, key),
        )

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
        if column == ARRIVAL_TIME_FIELD:
            return read_arrival_time_column(table, index, metadata.modality_key, deserialize)
        full_column_name = f"{metadata.modality_key}.{column}"
        if full_column_name not in table.column_names:
            raise ValueError(
                f"Column '{full_column_name}' not found in Arrow table for modality '{metadata.modality_key}'"
            )
        value = table[full_column_name][index].as_py()
        if deserialize and value is not None:
            if column == "timestamp_us":
                value = Timestamp.from_us(value)
            elif column == "magnetic_field":
                value = Vector3D(*value)
        return value
