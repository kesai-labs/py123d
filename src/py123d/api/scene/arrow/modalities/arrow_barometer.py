from pathlib import Path
from typing import Any, Literal, Optional

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
from py123d.datatypes.sensors.barometer import Barometer, BarometerMetadata
from py123d.datatypes.time.timestamp import Timestamp

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowBarometerWriter(ArrowBaseModalityWriter):
    """Writes barometer measurements to an Arrow IPC file.

    Pressure is always present; the derived altitude and environmental readings are nullable
    columns, stored as null when a measurement lacks them.
    """

    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, BarometerMetadata), f"Expected BarometerMetadata, got {type(metadata)}"

        self._metadata = metadata
        self._key = metadata.modality_key

        schema = pa.schema(
            [
                (f"{self._key}.timestamp_us", pa.int64()),
                (f"{self._key}.pressure", pa.float64()),
                (f"{self._key}.msl_altitude", pa.float64()),
                (f"{self._key}.temperature", pa.float64()),
                (f"{self._key}.humidity", pa.float64()),
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
        assert isinstance(modality, Barometer), f"Expected Barometer, got {type(modality)}"
        row: dict = {
            f"{self._key}.timestamp_us": [modality.timestamp.time_us],
            f"{self._key}.pressure": [modality.pressure],
            f"{self._key}.msl_altitude": [modality.msl_altitude],
            f"{self._key}.temperature": [modality.temperature],
            f"{self._key}.humidity": [modality.humidity],
        }
        add_arrival_time_to_row(row, self._metadata, modality)
        self.write_batch(row)


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowBarometerReader(ArrowBaseModalityReader):
    """Stateless reader for barometer data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[Barometer]:
        assert isinstance(metadata, BarometerMetadata), f"Expected BarometerMetadata, got {type(metadata)}"
        key = metadata.modality_key
        return Barometer(
            timestamp=Timestamp.from_us(table[f"{key}.timestamp_us"][index].as_py()),
            metadata=metadata,
            pressure=table[f"{key}.pressure"][index].as_py(),
            msl_altitude=table[f"{key}.msl_altitude"][index].as_py(),
            temperature=table[f"{key}.temperature"][index].as_py(),
            humidity=table[f"{key}.humidity"][index].as_py(),
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
        if deserialize and value is not None and column == "timestamp_us":
            value = Timestamp.from_us(value)
        return value
