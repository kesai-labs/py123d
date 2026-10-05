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
from py123d.datatypes.sensors.gnss import Gnss, GnssMetadata
from py123d.datatypes.time.timestamp import Timestamp

_LLA_SIZE: int = 3
_COVARIANCE_SIZE: int = 9
_VELOCITY_SIZE: int = 3

# Written only when GnssMetadata.has_solution_quality is set, so logs converted before these
# fields existed keep their original schema and still load.
_SOLUTION_QUALITY_FIELDS: tuple = (
    ("num_satellites", pa.int32()),
    ("fix_type", pa.int32()),
    ("horizontal_accuracy", pa.float64()),
    ("vertical_accuracy", pa.float64()),
    ("position_dop", pa.float64()),
    ("velocity_ned", pa.list_(pa.float64(), _VELOCITY_SIZE)),
    ("speed_accuracy", pa.float64()),
)

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowGnssWriter(ArrowBaseModalityWriter):
    """Writes GNSS fixes to an Arrow IPC file.

    Geodetic coordinates are stored as one fixed-size (lat, lon, alt) column; the NavSatFix
    quality fields are nullable columns, so fixes lacking them are stored as null.
    """

    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, GnssMetadata), f"Expected GnssMetadata, got {type(metadata)}"

        self._metadata = metadata
        self._key = metadata.modality_key

        fields = [
            (f"{self._key}.timestamp_us", pa.int64()),
            (f"{self._key}.lla", pa.list_(pa.float64(), _LLA_SIZE)),
            (f"{self._key}.position_covariance", pa.list_(pa.float64(), _COVARIANCE_SIZE)),
            (f"{self._key}.position_covariance_type", pa.int32()),
            (f"{self._key}.status", pa.int32()),
            (f"{self._key}.service", pa.int32()),
        ]
        if metadata.has_solution_quality:
            fields.extend((f"{self._key}.{name}", dtype) for name, dtype in _SOLUTION_QUALITY_FIELDS)

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
        assert isinstance(modality, Gnss), f"Expected Gnss, got {type(modality)}"
        covariance = modality.position_covariance
        row: dict = {
            f"{self._key}.timestamp_us": [modality.timestamp.time_us],
            f"{self._key}.lla": [modality.lla],
            f"{self._key}.position_covariance": [covariance if covariance is not None else None],
            f"{self._key}.position_covariance_type": [modality.position_covariance_type],
            f"{self._key}.status": [modality.status],
            f"{self._key}.service": [modality.service],
        }
        if self._metadata.has_solution_quality:
            for name, _ in _SOLUTION_QUALITY_FIELDS:
                row[f"{self._key}.{name}"] = [getattr(modality, name)]
        else:
            for name, _ in _SOLUTION_QUALITY_FIELDS:
                assert getattr(modality, name) is None, (
                    f"Fix carries {name} but GnssMetadata.has_solution_quality is False; "
                    "the value would be silently dropped."
                )
        add_arrival_time_to_row(row, self._metadata, modality)
        self.write_batch(row)


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowGnssReader(ArrowBaseModalityReader):
    """Stateless reader for GNSS data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[Gnss]:
        assert isinstance(metadata, GnssMetadata), f"Expected GnssMetadata, got {type(metadata)}"
        key = metadata.modality_key

        def _optional(column: str) -> Optional[Any]:
            # Absent by schema design for logs converted without the solution-quality fields.
            full_name = f"{key}.{column}"
            if full_name not in table.column_names:
                return None
            return table[full_name][index].as_py()

        lla = table[f"{key}.lla"][index].as_py()
        covariance = table[f"{key}.position_covariance"][index].as_py()
        velocity_ned = _optional("velocity_ned")
        return Gnss(
            timestamp=Timestamp.from_us(table[f"{key}.timestamp_us"][index].as_py()),
            metadata=metadata,
            latitude=lla[0],
            longitude=lla[1],
            altitude=lla[2],
            position_covariance=np.asarray(covariance, dtype=np.float64) if covariance is not None else None,
            position_covariance_type=table[f"{key}.position_covariance_type"][index].as_py(),
            status=table[f"{key}.status"][index].as_py(),
            service=table[f"{key}.service"][index].as_py(),
            num_satellites=_optional("num_satellites"),
            fix_type=_optional("fix_type"),
            horizontal_accuracy=_optional("horizontal_accuracy"),
            vertical_accuracy=_optional("vertical_accuracy"),
            position_dop=_optional("position_dop"),
            velocity_ned=np.asarray(velocity_ned, dtype=np.float64) if velocity_ned is not None else None,
            speed_accuracy=_optional("speed_accuracy"),
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
            # Columns can be legitimately absent by schema design (has_solution_quality flag);
            # return None like the sibling readers do.
            return None
        value = table[full_column_name][index].as_py()
        if deserialize and value is not None and column == "timestamp_us":
            value = Timestamp.from_us(value)
        return value
