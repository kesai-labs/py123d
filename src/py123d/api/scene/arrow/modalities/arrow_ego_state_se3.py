from pathlib import Path
from typing import Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema, get_optional_array_mixin
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3, DynamicStateSE3Index
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry.geometry_index import PoseSE3Index
from py123d.geometry.pose import PoseSE3

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowEgoStateSE3Writer(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, EgoStateSE3Metadata), f"Expected EgoStateSE3Metadata, got {type(metadata)}"

        self._metadata = metadata

        file_path = log_dir / f"{metadata.modality_key}.arrow"
        schema = pa.schema(
            [
                (f"{self._metadata.modality_key}.timestamp_us", pa.int64()),
                (f"{self._metadata.modality_key}.imu_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
                (f"{self._metadata.modality_key}.dynamic_state_se3", pa.list_(pa.float64(), len(DynamicStateSE3Index))),
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
        assert isinstance(modality, EgoStateSE3), f"Expected EgoStateSE3, got {type(modality)}"
        self.write_batch(
            {
                f"{self._metadata.modality_key}.timestamp_us": [modality.timestamp.time_us],
                f"{self._metadata.modality_key}.imu_se3": [modality.imu_se3],
                f"{self._metadata.modality_key}.dynamic_state_se3": [modality.dynamic_state_se3],
            }
        )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowEgoStateSE3Reader(ArrowBaseModalityReader):
    """Stateless reader for ego state SE3 data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
    ) -> Optional[EgoStateSE3]:
        assert isinstance(metadata, EgoStateSE3Metadata)
        return _deserialize_ego_state_se3(table, index, metadata)


def _deserialize_ego_state_se3(
    modality_table: pa.Table,
    index: int,
    metadata: EgoStateSE3Metadata,
) -> Optional[EgoStateSE3]:
    """Deserialize an ego state from Arrow table columns at the given row index."""

    modality_key = metadata.modality_key
    ego_columns = [
        f"{modality_key}.imu_se3",
        f"{modality_key}.dynamic_state_se3",
        f"{modality_key}.timestamp_us",
    ]
    if not all_columns_in_schema(modality_table, ego_columns):
        return None
    timestamp = Timestamp.from_us(modality_table[f"{modality_key}.timestamp_us"][index].as_py())
    imu_se3 = PoseSE3.from_list(modality_table[f"{modality_key}.imu_se3"][index].as_py())
    dynamic_state_se3 = get_optional_array_mixin(
        data=modality_table[f"{modality_key}.dynamic_state_se3"][index].as_py(),
        cls=DynamicStateSE3,
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        metadata=metadata,
        dynamic_state_se3=dynamic_state_se3,  # type: ignore
        timestamp=timestamp,
    )
