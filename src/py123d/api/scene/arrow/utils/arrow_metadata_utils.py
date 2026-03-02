import json
from pathlib import Path
from typing import Union

import pyarrow as pa

from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table
from py123d.datatypes.metadata import LogMetadata, MapMetadata
from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata


def get_log_metadata_from_arrow_file(arrow_file_path: Union[Path, str]) -> LogMetadata:
    """Gets the log metadata from an Arrow file."""
    table = get_lru_cached_arrow_table(arrow_file_path)
    return get_log_metadata_from_arrow_table(table)


def get_log_metadata_from_arrow_table(arrow_table: pa.Table) -> LogMetadata:
    """Gets the log metadata from an Arrow table."""
    return get_log_metadata_from_arrow_schema(arrow_table.schema)


def get_log_metadata_from_arrow_schema(arrow_schema: pa.Schema) -> LogMetadata:
    """Gets the log metadata from an Arrow schema."""
    return LogMetadata.from_dict(json.loads(arrow_schema.metadata[b"log_metadata"].decode()))


def get_map_metadata_from_arrow_table(arrow_table: pa.Table) -> MapMetadata:
    """Gets the map metadata from an Arrow table."""
    return get_map_metadata_from_arrow_schema(arrow_table.schema)


def get_map_metadata_from_arrow_schema(arrow_schema: pa.Schema) -> MapMetadata:
    """Gets the map metadata from an Arrow schema."""
    return MapMetadata.from_dict(json.loads(arrow_schema.metadata[b"map_metadata"].decode()))


def add_log_metadata_to_arrow_schema(schema: pa.schema, log_metadata: LogMetadata) -> pa.schema:
    """Adds log metadata to an Arrow schema."""
    schema = schema.with_metadata({"log_metadata": json.dumps(log_metadata.to_dict())})
    return schema


def add_metadata_to_arrow_schema(schema: pa.schema, metadata: AbstractMetadata) -> pa.schema:
    """Adds metadata to an Arrow schema."""
    schema = schema.with_metadata({"metadata": json.dumps(metadata.to_dict())})
    return schema
