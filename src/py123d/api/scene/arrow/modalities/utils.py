from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.common.utils.mixin import ArrayMixin
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.time.timestamp import Timestamp

ARRIVAL_TIME_FIELD: str = "arrival_us"
"""Field name of the optional per-row arrival time, stored as ``<modality_key>.arrival_us`` in microseconds
(see :attr:`~py123d.datatypes.BaseModality.arrival_timestamp`). The name must not end in ``timestamp_us``:
the sync table construction, the timestamp-order check and the scene API take the first column with that
suffix as the measurement time."""


def get_optional_array_mixin(data: Optional[Union[List, npt.NDArray]], cls: Type[ArrayMixin]) -> Optional[ArrayMixin]:
    """Builds an optional ArrayMixin if data is provided.

    :param data: The data to convert into an ArrayMixin.
    :param cls: The ArrayMixin class to instantiate.
    :raises ValueError: If the data type is unsupported.
    :return: The instantiated ArrayMixin, or None if data is None.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return cls.from_list(data)
    elif isinstance(data, np.ndarray):
        return cls.from_array(data, copy=False)
    else:
        raise ValueError(f"Unsupported data type for ArrayMixin conversion: {type(data)}")


def all_columns_in_schema(arrow_table: pa.Table, columns: List[str]) -> bool:
    """Checks if all specified columns are present in the Arrow table schema.

    :param arrow_table: The Arrow table to check.
    :param columns: The list of column names to check for.
    :return: True if all columns are present, False otherwise.
    """
    return all(column in arrow_table.schema.names for column in columns)


def get_arrival_time_field(modality_key: str) -> Tuple[str, pa.DataType]:
    """Returns the schema field of the arrival-time column, for writers whose metadata sets ``has_arrival_time``.

    :param modality_key: The modality key, e.g. ``"imu"``.
    :return: The (column name, Arrow type) pair. Nullable int64 microseconds.
    """
    return (f"{modality_key}.{ARRIVAL_TIME_FIELD}", pa.int64())


def add_arrival_time_to_row(row: Dict[str, Any], metadata: BaseModalityMetadata, modality: BaseModality) -> None:
    """Adds the modality's arrival time to a row about to be written, as the metadata declares.

    With ``has_arrival_time`` set, the column receives the arrival time, or null for a row without one.
    Without it, the row is left unchanged, so the file keeps the schema it had before the column existed.

    :param row: The row being written, mapping column names to single-element lists.
    :param metadata: The writer's modality metadata.
    :param modality: The modality being written.
    :raises AssertionError: If the modality carries an arrival time but the metadata does not declare
        ``has_arrival_time``; the value would otherwise be dropped without notice.
    """
    arrival_timestamp = modality.arrival_timestamp
    if metadata.has_arrival_time:
        row[f"{metadata.modality_key}.{ARRIVAL_TIME_FIELD}"] = [
            arrival_timestamp.time_us if arrival_timestamp is not None else None
        ]
    else:
        assert arrival_timestamp is None, (
            f"Modality '{metadata.modality_key}' carries an arrival time but its metadata's has_arrival_time is "
            "False; the value would be silently dropped."
        )


def read_arrival_time_column(
    table: pa.Table, index: int, modality_key: str, deserialize: bool = True
) -> Optional[Union[int, Timestamp]]:
    """Reads the arrival time of one row, if the file stores it.

    :param table: The Arrow modality table.
    :param index: The row index.
    :param modality_key: The modality key.
    :param deserialize: If True, return a :class:`~py123d.datatypes.Timestamp`, else the raw microseconds.
    :return: The arrival time, or None if the file has no arrival-time column (every log written without
        ``has_arrival_time``) or the row has none.
    """
    column = f"{modality_key}.{ARRIVAL_TIME_FIELD}"
    if column not in table.column_names:
        return None
    value = table[column][index].as_py()
    if value is None or not deserialize:
        return value
    return Timestamp.from_us(value)


def read_arrival_timestamp(table: pa.Table, index: int, modality_key: str) -> Optional[Timestamp]:
    """Reads the arrival time of one row as a :class:`~py123d.datatypes.Timestamp`, or None if not stored."""
    value = read_arrival_time_column(table, index, modality_key, deserialize=True)
    assert value is None or isinstance(value, Timestamp)
    return value
