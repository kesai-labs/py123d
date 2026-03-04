import json
from typing import TypeVar

import pyarrow as pa

from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata

T = TypeVar("T", bound=AbstractMetadata)

# _LOG_METADATA_KEY = b"log_metadata"
_METADATA_KEY = b"metadata"


def get_metadata_from_arrow_schema(
    arrow_schema: pa.Schema,
    metadata_class: type[T],
    modality_key: bytes = _METADATA_KEY,
) -> T:
    """Gets metadata for a specific modality from an Arrow schema."""

    deserialized_metadata = None
    if modality_key in arrow_schema.metadata:
        deserialized_metadata = metadata_class.from_dict(json.loads(arrow_schema.metadata[modality_key].decode()))

    try:
        assert deserialized_metadata is not None, (
            f"Metadata for modality key '{modality_key.decode()}' not found in Arrow schema."
        )
    except AssertionError as e:
        available_keys = [k.decode() for k in arrow_schema.metadata.keys()] if arrow_schema.metadata else []
        raise ValueError(f"{str(e)} Available metadata keys: {available_keys}") from e
    return deserialized_metadata


def add_metadata_to_arrow_schema(
    schema: pa.Schema,
    metadata: AbstractMetadata,
    modality_key: bytes = _METADATA_KEY,
) -> pa.Schema:
    """Adds metadata for a specific modality to an Arrow schema."""
    existing = dict(schema.metadata) if schema.metadata else {}
    existing[modality_key] = json.dumps(metadata.to_dict()).encode()
    return schema.with_metadata(existing)
