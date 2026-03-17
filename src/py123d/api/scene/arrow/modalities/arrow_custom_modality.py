from pathlib import Path
from typing import Any, Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.utils.msgpack_utils import msgpack_decode_with_numpy, msgpack_encode_with_numpy
from py123d.datatypes.custom.custom_modality import CustomModality, CustomModalityMetadata
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.time.timestamp import Timestamp

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowCustomModalityWriter(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: CustomModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, CustomModalityMetadata), f"Expected CustomModalityMetadata, got {type(metadata)}"

        self._modality_metadata = metadata
        self._modality_key = metadata.modality_key

        file_path = log_dir / f"{metadata.modality_key}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_key}.timestamp_us", pa.int64()),
                (f"{metadata.modality_key}.data", pa.binary()),
            ]
        )
        schema = add_metadata_to_arrow_schema(schema, metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, CustomModality), f"Expected CustomModality, got {type(modality)}"
        encoded_data = msgpack_encode_with_numpy(modality.data)
        self.write_batch(
            {
                f"{self._modality_key}.timestamp_us": [modality.timestamp.time_us],
                f"{self._modality_key}.data": [encoded_data],
            }
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowCustomModalityReader(ArrowBaseModalityReader):
    """Stateless reader for custom modality data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[CustomModality]:
        assert isinstance(metadata, CustomModalityMetadata), f"Expected CustomModalityMetadata, got {type(metadata)}"
        _modality_key = metadata.modality_key
        timestamp_us: int = table[f"{_modality_key}.timestamp_us"][index].as_py()
        encoded_data: bytes = table[f"{_modality_key}.data"][index].as_py()
        data = msgpack_decode_with_numpy(encoded_data)
        return CustomModality(
            data=data,
            metadata=metadata,
            timestamp=Timestamp.from_us(timestamp_us),
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
        assert isinstance(metadata, CustomModalityMetadata), f"Expected CustomModalityMetadata, got {type(metadata)}"
        full_column_name = f"{metadata.modality_key}.{column}"

        column_at_iteration: Optional[Any] = None
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()
            if deserialize and column == "data":
                column_at_iteration = msgpack_decode_with_numpy(column_at_iteration)  # type: ignore
            elif deserialize and column == "timestamp_us":
                column_at_iteration = Timestamp.from_us(column_at_iteration)  # type: ignore

        return column_at_iteration
