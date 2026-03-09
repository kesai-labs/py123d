from pathlib import Path
from typing import Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.utils.msgpack_utils import msgpack_encode_with_numpy
from py123d.datatypes.custom.custom_modality import CustomModality, CustomModalityMetadata


class ArrowCustomModalityWriter(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: CustomModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, CustomModalityMetadata), f"Expected CustomModalityMetadata, got {type(metadata)}"

        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name

        file_path = log_dir / f"{metadata.modality_name}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_name}.timestamp_us", pa.int64()),
                (f"{metadata.modality_name}.data", pa.binary()),
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

    def write_modality(self, custom_modality: CustomModality) -> None:
        assert isinstance(custom_modality, CustomModality), f"Expected CustomModality, got {type(custom_modality)}"
        encoded_data = msgpack_encode_with_numpy(custom_modality.data)
        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [custom_modality.timestamp.time_us],
                f"{self._modality_name}.data": [encoded_data],
            }
        )


def get_custom_modality_from_arrow_table(
    modality_table: pa.Table, index: int, modality_metadata: CustomModalityMetadata
) -> Optional[CustomModality]:
    pass
