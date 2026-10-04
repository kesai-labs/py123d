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
from py123d.datatypes.sensors.imu import Imu, ImuMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.geometry_index import QuaternionIndex, Vector3DIndex
from py123d.geometry.rotation import Quaternion
from py123d.geometry.vector import Vector3D

_COVARIANCE_SIZE: int = 9

_COVARIANCE_FIELDS = (
    "orientation_covariance",
    "angular_velocity_covariance",
    "linear_acceleration_covariance",
)

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowImuWriter(ArrowBaseModalityWriter):
    """Writes IMU measurements to an Arrow IPC file.

    The schema is derived from the metadata's :attr:`~py123d.datatypes.sensors.imu.ImuMetadata.has_orientation`
    and :attr:`~py123d.datatypes.sensors.imu.ImuMetadata.has_covariances` flags: columns for fields the
    sensor does not provide are omitted from the file entirely.
    """

    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, ImuMetadata), f"Expected ImuMetadata, got {type(metadata)}"

        self._metadata = metadata
        self._key = metadata.modality_key

        fields = [
            (f"{self._key}.timestamp_us", pa.int64()),
            (f"{self._key}.angular_velocity", pa.list_(pa.float64(), len(Vector3DIndex))),
            (f"{self._key}.linear_acceleration", pa.list_(pa.float64(), len(Vector3DIndex))),
        ]
        if metadata.has_orientation:
            fields.append((f"{self._key}.orientation", pa.list_(pa.float64(), len(QuaternionIndex))))
        if metadata.has_covariances:
            for name in _COVARIANCE_FIELDS:
                fields.append((f"{self._key}.{name}", pa.list_(pa.float64(), _COVARIANCE_SIZE)))

        if metadata.has_arrival_time:
            fields.append(get_arrival_time_field(self._key))
        schema = add_metadata_to_arrow_schema(pa.schema(fields), metadata)
        super().__init__(
            file_path=log_dir / f"{self._key}.arrow",
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, Imu), f"Expected Imu, got {type(modality)}"
        row: dict = {
            f"{self._key}.timestamp_us": [modality.timestamp.time_us],
            f"{self._key}.angular_velocity": [modality.angular_velocity.array],
            f"{self._key}.linear_acceleration": [modality.linear_acceleration.array],
        }
        if self._metadata.has_orientation:
            orientation = modality.orientation
            row[f"{self._key}.orientation"] = [orientation.array if orientation is not None else None]
        else:
            assert modality.orientation is None, (
                "Measurement carries an orientation but ImuMetadata.has_orientation is False; "
                "the value would be silently dropped."
            )
        if self._metadata.has_covariances:
            for name in _COVARIANCE_FIELDS:
                value = getattr(modality, name)
                row[f"{self._key}.{name}"] = [value if value is not None else None]
        else:
            for name in _COVARIANCE_FIELDS:
                assert getattr(modality, name) is None, (
                    f"Measurement carries {name} but ImuMetadata.has_covariances is False; "
                    "the value would be silently dropped."
                )
        add_arrival_time_to_row(row, self._metadata, modality)
        self.write_batch(row)


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowImuReader(ArrowBaseModalityReader):
    """Stateless reader for IMU data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[Imu]:
        assert isinstance(metadata, ImuMetadata), f"Expected ImuMetadata, got {type(metadata)}"
        key = metadata.modality_key

        def _optional_array(column: str) -> Optional[np.ndarray]:
            full_name = f"{key}.{column}"
            if full_name not in table.column_names:
                return None
            value = table[full_name][index].as_py()
            return np.asarray(value, dtype=np.float64) if value is not None else None

        orientation_array = _optional_array("orientation")
        return Imu(
            timestamp=Timestamp.from_us(table[f"{key}.timestamp_us"][index].as_py()),
            metadata=metadata,
            angular_velocity=Vector3D(*table[f"{key}.angular_velocity"][index].as_py()),
            linear_acceleration=Vector3D(*table[f"{key}.linear_acceleration"][index].as_py()),
            orientation=Quaternion(*orientation_array) if orientation_array is not None else None,
            orientation_covariance=_optional_array("orientation_covariance"),
            angular_velocity_covariance=_optional_array("angular_velocity_covariance"),
            linear_acceleration_covariance=_optional_array("linear_acceleration_covariance"),
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
            # Columns can be legitimately absent by schema design (has_orientation /
            # has_covariances flags); return None like the sibling readers do.
            return None
        value = table[full_column_name][index].as_py()
        if deserialize and value is not None:
            if column == "timestamp_us":
                value = Timestamp.from_us(value)
            elif column in ("angular_velocity", "linear_acceleration"):
                value = Vector3D(*value)
            elif column == "orientation":
                value = Quaternion(*value)
        return value
