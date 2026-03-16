import pyarrow as pa

from py123d.datatypes.time.timestamp import Timestamp


def get_timestamp_from_arrow_table(arrow_table: pa.Table, index: int) -> Timestamp:
    """Builds a :class:`~py123d.datatypes.time.Timestamp` from an Arrow table at a given index.

    :param arrow_table: The Arrow table containing the timestamp data.
    :param index: The index to extract the timestamp from.
    :return: The Timestamp at the given index.
    """
    assert "sync.timestamp_us" in arrow_table.schema.names, "Timestamp column not found in Arrow table."
    return Timestamp.from_us(arrow_table["sync.timestamp_us"][index].as_py())
