from pathlib import Path
from typing import List, Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.modalities.sync_utils import (
    get_all_modality_timestamps,
    get_modality_table,
)
from py123d.api.scene.scene_metadata import SceneMetadata
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.common.utils.msgpack_utils import msgpack_decode_with_numpy, msgpack_encode_with_numpy
from py123d.datatypes.custom.custom_modality import CustomModality, CustomModalityMetadata
from py123d.datatypes.time.timestamp import Timestamp

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowCustomModalityReader:
    """Stateless reader for custom modality data from Arrow tables."""

    @staticmethod
    def read_at_iteration(
        log_dir: Path,
        table_index: int,
        name: str,
    ) -> Optional[CustomModality]:
        """Read a custom modality at a specific table index.

        Custom modalities are indexed directly by table index (not via sync table),
        since each row maps 1:1 to a sync iteration.

        :param log_dir: Path to the log directory.
        :param table_index: The resolved table index.
        :param name: The custom modality name (e.g. ``"route"``, ``"predictions"``).
        :return: The custom modality, or None if unavailable.
        """
        modality_prefix = f"custom.{name}"
        table = get_modality_table(log_dir, modality_prefix)
        if table is None:
            return None
        encoded_data: bytes = table[f"{modality_prefix}.data"][table_index].as_py()
        timestamp_us: int = table[f"{modality_prefix}.timestamp_us"][table_index].as_py()
        data = msgpack_decode_with_numpy(encoded_data)
        return CustomModality(data=data, timestamp=Timestamp.from_us(timestamp_us))  # type: ignore

    @staticmethod
    def read_all_timestamps(
        log_dir: Path,
        sync_table: pa.Table,
        scene_metadata: SceneMetadata,
        name: str,
    ) -> List[Timestamp]:
        """Read all timestamps for a custom modality within the scene range.

        :param log_dir: Path to the log directory.
        :param sync_table: The sync Arrow table.
        :param scene_metadata: Scene metadata defining the iteration range.
        :param name: The custom modality name.
        :return: All custom modality timestamps in the scene, ordered by time.
        """
        modality_name = f"custom.{name}"
        return get_all_modality_timestamps(
            log_dir,
            sync_table,
            scene_metadata,
            modality_name,
            f"{modality_name}.timestamp_us",
        )
