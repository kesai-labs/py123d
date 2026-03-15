from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

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
        **kwargs,
    ) -> Optional[EgoStateSE3]:
        assert isinstance(metadata, EgoStateSE3Metadata)
        return _deserialize_ego_state_se3(table, index, metadata)

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
        """Return a single column value from the ego state Arrow table at a given row index.

        :param index: The row index in the Arrow table.
        :param table: The Arrow modality table.
        :param metadata: The modality metadata.
        :param column: The field name (e.g. ``"imu_se3"``, ``"timestamp_us"``).
        :param deserialize: If True, deserialize the value to its domain type.
        :return: The column value, or None if the column is not present.
        """
        full_column_name = f"{metadata.modality_key}.{column}"
        column_at_iteration: Optional[Any] = None
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()
            if deserialize and column in EGO_STATE_SE3_DESERIALIZE_FUNC:
                column_at_iteration = EGO_STATE_SE3_DESERIALIZE_FUNC[column](column_at_iteration)
        else:
            raise ValueError(
                f"Column '{full_column_name}' not found in Arrow table for modality '{metadata.modality_key}'"
            )

        return column_at_iteration


EGO_STATE_SE3_DESERIALIZE_FUNC: Dict[str, Callable[[Any], Any]] = {
    "imu_se3": PoseSE3.from_list,
    "dynamic_state_se3": lambda v: get_optional_array_mixin(data=v, cls=DynamicStateSE3),
    "timestamp_us": Timestamp.from_us,
}


def _deserialize_ego_state_se3(
    modality_table: pa.Table,
    index: int,
    metadata: EgoStateSE3Metadata,
) -> Optional[EgoStateSE3]:
    """Deserialize an ego state from Arrow table columns at the given row index."""

    modality_key = metadata.modality_key
    ego_columns = [f"{modality_key}.{field}" for field in EGO_STATE_SE3_DESERIALIZE_FUNC.keys()]
    if not all_columns_in_schema(modality_table, ego_columns):
        return None

    timestamp = EGO_STATE_SE3_DESERIALIZE_FUNC["timestamp_us"](
        modality_table[f"{modality_key}.timestamp_us"][index].as_py()
    )
    imu_se3 = EGO_STATE_SE3_DESERIALIZE_FUNC["imu_se3"](modality_table[f"{modality_key}.imu_se3"][index].as_py())
    dynamic_state_se3 = EGO_STATE_SE3_DESERIALIZE_FUNC["dynamic_state_se3"](
        modality_table[f"{modality_key}.dynamic_state_se3"][index].as_py()
    )
    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        metadata=metadata,
        dynamic_state_se3=dynamic_state_se3,  # type: ignore
        timestamp=timestamp,
    )
